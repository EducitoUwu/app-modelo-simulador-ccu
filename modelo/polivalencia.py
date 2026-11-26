"""Polivalencia de roles en el patio: cómo cambia la dotación con la ayuda de los chequeadores y con un gruero-despachador polivalente (modelo de capacidad + simulación).

Pedido del usuario (2026-11-19), a partir de la conversación con el jefe de operaciones: el cuello de botella del patio está en el REGISTRO de la información (escaneo con el equipo móvil),
no en bajar el pallet. Hoy hay «doble puesto» en la recepción (el gruero baja y el chequeador escanea y traslada). Propuesta: un gruero-despachador polivalente con competencias digitales
hace la recepción completa y el despacho (carga de camión y retorno de fletero); los grueros puros hacen el almacenamiento y el resto.

La recepción de un pallet se parte en TRES pasos (misma entidad, en serie):
  rec_descarga  bajar el pallet del camión            30,2 s  (planilla: tiempo del gruero cuando escanea el planificador, mezcla de camiones del log; `capacity.seg_recepcion_por_pallet(1.0)`)
  rec_escaneo   registrar el pallet en el WMS          14 s si lo hace un chequeador o el polivalente (planilla: camiones grandes con planificador, 390,5 s ÷ 27,9 pallets, N = 6; decisión del usuario:
                                                       el mismo para todos los camiones) · 30 s si lo hace el gruero (DECISIÓN DEL USUARIO, 2026-11-22; la planilla mide
                                                       21 s en camiones chicos y 26 s en grandes, ≈ 24 s ponderado, N = 12)
  rec_traslado  llevar el pallet del patio a STREC     58 s    (dato del equipo, 2026-11-19: medido entre 45 y 65 s; se fija 58 s por ahora). Hoy NO es un proceso del modelo de capacidad.

ESCENARIOS (mismas réplicas y semillas en todos: los muestreos no dependen del escenario):
  E0  AS-IS procedimiento: el gruero hace todo (bajar, escanear a 30 s y trasladar). Sin chequeadores (el procedimiento dice que la recepción es del gruero).
  E1  práctica con chequeador: el gruero baja; el chequeador escanea (14 s) y traslada con su equipo móvil (sin ocupar una de las 6 grúas). Chequeadores reales: A 2 + 1 intermedio, B 2, C 3.
  E2  polivalente: el gruero-despachador hace la recepción completa (escaneo 14 s), el traslado, la carga de camión y el retorno de fletero del gruero; el gruero puro, todo lo demás
      (almacenamiento, reabastos, recogida de pallets completos, traslado de mixtos y la carga para el CL, que por ahora no se le asigna al polivalente). Sin chequeadores.
  E2b gruero integral: todos los grueros tienen las competencias del polivalente (escaneo 14 s) y hacen todo. Sin chequeadores. Muestra el efecto de juntar a todos en un solo grupo.

LÍMITES [declarados]: el escaneo va DESPUÉS de la descarga (en la operación puede ir en paralelo); el intermedio cuenta en el turno A (regla de `dotacion_historica`); los chequeadores no tienen
otras tareas en el modelo (si las tienen, su disponibilidad es menor); el costo por persona es el dato del jefe de operaciones («dos palitos brutos» ≈ $2 M al mes), por confirmar.
Los operarios no cambian entre escenarios: en la simulación se dejan holgados (A 4, B 4, C 20) para que el resultado sea solo del patio.

USO: .venv/bin/python -m modelo.polivalencia [--replicas 300] [--confirmar 1000] [--procesos 4] [--solo-analitico]
Salida: salidas/polivalencia/{analitico.csv, simulacion.csv, resumen.md}
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from modelo import capacity, config, fit_params
from modelo import llegadas_camiones as lc
from modelo import simulacion_dia as sd
from modelo import simulacion_turno_c as sc

SAL = Path(__file__).resolve().parent.parent / "salidas" / "polivalencia"
TURNOS = ("A", "B", "C")
DEMANDAS = {"baja": 15913, "media": 22681, "alta": 33051, "saturada": 37273, "p99": 41892}      # escenarios del estimador (821 jornadas 2024-sep 2026)
DEMANDAS_SIM = ("baja", "media", "alta")

DESCARGA_S = {"valor": None, "fuente": "planilla: tiempo del gruero cuando escanea el planificador (mezcla de camiones del log), capacity.seg_recepcion_por_pallet(1.0)"}
ESCANEO_RAPIDO_S = {"valor": 14.0, "fuente": "planilla: camiones grandes con planificador, 390,5 s / 27,9 pallets (N = 6); decisión del usuario: mismo valor para todos los camiones"}
ESCANEO_GRUERO_S = {"valor": 30.0, "fuente": "DECISIÓN DEL USUARIO (2026-11-22): 30 s, algo sobre lo medido (planilla: 21 s en camiones chicos y 26 s en grandes, ≈ 24 s ponderado, N = 12)"}
ESCANEO_GRUERO_MEDIDO_S = {"valor": None, "fuente": "lo que usa el modelo de capacidad: 47,4 s de gruero solo − 30,2 s de descarga = 17,2 s (planilla: 21 a 27 s por pallet, N = 12)"}
TRASLADO_S = {"valor": 58.0, "fuente": "dato del equipo (2026-11-19): medido entre 45 y 65 s por pallet; fijo en 58 s por ahora"}
CV = {"rec_escaneo": 0.25, "rec_traslado": 0.10}          # escaneo: CV del planificador en camiones grandes (3,4 / 14,4 s); traslado: rango 45-65 s ≈ ±2 desvíos de 5 s
CHEQUEADORES = {"A": 3, "B": 2, "C": 3}                   # dato del usuario: 2 en A, 2 en B, 1 intermedio (cuenta en A) y 3 en C
COSTO_PERSONA_MES = {"valor": 2_000_000, "fuente": "jefe de operaciones: «dos palitos brutos» por gruero al mes (por confirmar)"}
OPERARIOS = {"A": 4, "B": 4, "C": 20}                     # holgados: el resultado es solo del patio
PASOS_REC = ("rec_descarga", "rec_escaneo", "rec_traslado")
DESPACHO = ("carga_camion", "retorno_fletero_gruero")

for _p in PASOS_REC:                                      # prioridad del patio (1), igual que la recepción
    sc.PRIORIDAD.setdefault(_p, sc.SUPUESTOS["prioridad_patio"]["valor"])


def _usa_grua(paso: str) -> bool:
    return "Grúas" in capacity.USO_DE_EQUIPO.get(paso, {})


def escenarios() -> dict:
    """{nombre: {"escaneo_s", "rol" (paso -> (rol, usa_grua)), "chequeadores", "descripcion"}}. Los procesos de gruero que no figuran quedan con el gruero."""
    g, ch, po = "gruero", "chequeador", "polivalente"
    return {
        "E0": {"escaneo_s": ESCANEO_GRUERO_S["valor"], "chequeadores": False, "descripcion": "AS-IS procedimiento: el gruero baja, escanea (30 s) y traslada",
               "rol": {"rec_descarga": (g, True), "rec_escaneo": (g, True), "rec_traslado": (g, True)}},
        "E1": {"escaneo_s": ESCANEO_RAPIDO_S["valor"], "chequeadores": True, "descripcion": "práctica: el gruero baja; el chequeador escanea (14 s) y traslada con su equipo móvil",
               "rol": {"rec_descarga": (g, True), "rec_escaneo": (ch, False), "rec_traslado": (ch, False)}},
        "E2": {"escaneo_s": ESCANEO_RAPIDO_S["valor"], "chequeadores": False, "descripcion": "polivalente: recepción completa (14 s), traslado, carga de camión y retorno de fletero; gruero puro el resto",
               "rol": {**{p: (po, True) for p in PASOS_REC}, **{p: (po, _usa_grua(p)) for p in DESPACHO}}},
        "E2b": {"escaneo_s": ESCANEO_RAPIDO_S["valor"], "chequeadores": False, "descripcion": "gruero integral: todos con competencias digitales (14 s) y hacen todo",
                "rol": {"rec_descarga": (g, True), "rec_escaneo": (g, True), "rec_traslado": (g, True)}},
    }


def _tiempos_rec(escaneo_s: float) -> dict:
    return {"rec_descarga": DESCARGA_S["valor"], "rec_escaneo": escaneo_s, "rec_traslado": TRASLADO_S["valor"]}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Modelo de capacidad (analítico): horas de trabajo por rol y turno ÷ (horas del turno × utilización)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def analitico(demanda: float, esc: dict, escaneo_gruero_s: float | None = None) -> dict:
    """{turno: {rol: personas necesarias (fraccional)}} + pallets recibidos. La recepción del modelo de capacidad (47,4 s de gruero solo) se reemplaza por los tres pasos."""
    aj = capacity.Ajustes()
    e = capacity.estimar(demanda, aj)
    H = e["duracion_h"]
    ef = aj.eficiencia_resto
    out = {}
    t_esc = esc["escaneo_s"] if escaneo_gruero_s is None else escaneo_gruero_s
    t_rec = _tiempos_rec(t_esc)
    for t in TURNOS:
        roles = {"Gruero": 0.0, "Chequeador": 0.0, "Polivalente": 0.0}
        pal = 0.0
        for d in e["detalle"]:
            if d["turno"] != t or d["rol"] != "Gruero":
                continue
            if d["proceso"] == "recepcion_directo_stage":
                pal = d["volumen"]
                continue
            if d["proceso"] == "traslado_stage":          # va dentro de los tres pasos de la recepción
                continue
            rol = esc["rol"].get(d["proceso"], ("gruero", True))[0]
            roles[rol.capitalize()] += d["horas_persona"] / H
        for p in PASOS_REC:
            roles[esc["rol"][p][0].capitalize()] += pal * t_rec[p] / 3600 / ef / H
        out[t] = {"pallets": pal, **roles}
    return out


def tabla_analitica() -> list[dict]:
    filas = []
    for nom, D in DEMANDAS.items():
        for en, esc in escenarios().items():
            for variante, eg in (("escaneo del gruero 30 s (usuario)", None), ("escaneo del gruero medido 17,2 s", ESCANEO_GRUERO_MEDIDO_S["valor"])):
                if variante.startswith("escaneo del gruero medido") and en != "E0":
                    continue
                a = analitico(D, esc, eg)
                f = {"demanda": nom, "cajas": D, "escenario": en, "variante": variante}
                tot = 0
                for t in TURNOS:
                    for rol in ("Gruero", "Chequeador", "Polivalente"):
                        f[f"{rol}_{t}"] = round(a[t][rol], 2)
                        n = math.ceil(a[t][rol] - 1e-9) if a[t][rol] > 1e-9 else 0
                        if rol == "Chequeador" and esc["chequeadores"]:
                            n = CHEQUEADORES[t]                      # los chequeadores son los reales: su trabajo se informa, su dotación no se dimensiona
                        f[f"{rol}_{t}_entero"] = n
                        tot += n
                f["personas_patio"] = tot
                f["grueros_mas_polivalentes"] = sum(f[f"{r}_{t}_entero"] for t in TURNOS for r in ("Gruero", "Polivalente"))
                filas.append(f)
    return filas


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
_P: dict = {}


def _lognormal_unitario(cv: float):
    s = math.sqrt(math.log(1 + cv * cv))

    def f(n, rng):
        return rng.lognormal(-s * s / 2, s, size=n)
    return f


def preparar(demanda: float, esc: dict, base: dict) -> dict:
    """Copia de la preparación del día con la recepción partida en tres pasos. Los muestreos consumen los mismos números aleatorios en todos los escenarios."""
    aj = capacity.Ajustes()
    preps = {}
    t_rec = _tiempos_rec(esc["escaneo_s"])
    for t in sd.TURNOS:
        p = copy.copy(base[t])
        p["entidades"] = {k: dict(v) for k, v in base[t]["entidades"].items()}
        p["muestreadores"] = dict(base[t]["muestreadores"])
        p["eta"] = dict(base[t]["eta"])
        if "recepcion_directo_stage" in p["entidades"]:
            f_rec, _ = base[t]["muestreadores"]["recepcion_directo_stage"]
            k_rec = base[t]["escala"].get("recepcion_directo_stage", 1.0) * DESCARGA_S["valor"] / capacity.seg_recepcion_por_pallet(0.0)
            u_esc, u_tr = _lognormal_unitario(CV["rec_escaneo"]), _lognormal_unitario(CV["rec_traslado"])
            p["muestreadores"]["rec_descarga"] = (lambda n, rng, f=f_rec, k=k_rec: f(n, rng) * k, f"recepción (gruero) reescalada a {DESCARGA_S['valor']:.1f} s")
            p["muestreadores"]["rec_escaneo"] = (lambda n, rng, u=u_esc, m=t_rec["rec_escaneo"]: u(n, rng) * m, f"lognormal media {t_rec['rec_escaneo']:.0f} s, CV {CV['rec_escaneo']}")
            p["muestreadores"]["rec_traslado"] = (lambda n, rng, u=u_tr, m=t_rec["rec_traslado"]: u(n, rng) * m, f"lognormal media {t_rec['rec_traslado']:.0f} s, CV {CV['rec_traslado']}")
            p["entidades"]["recepcion_directo_stage"]["pasos"] = list(PASOS_REC)
            for q in PASOS_REC:
                p["eta"][q] = aj.eficiencia_resto
        preps[t] = p
    return preps


def _trab(a):
    clave, dot, sem = a
    nivel, en = clave
    g = lc.GeneradorLlegadas(DEMANDAS[nivel], "rec_carga", horas=sc.HORAS_TURNO)
    rol = escenarios()[en]["rol"]
    return [sd._replica_dia(_P[clave], dot, sc.SUPUESTOS["gruas"]["valor"], s, llegadas=g, traspaso=sd.TRASPASO_TOL, roles_paso=rol) for s in sem]


def _correr(pool, clave, dot, R):
    base = sc.SUPUESTOS["semilla_base"]["valor"]
    blq = [list(range(base + i, base + min(R, i + 25))) for i in range(0, R, 25)]
    return [x for p in pool.map(_trab, [(clave, dot, b) for b in blq]) for x in p]


def _evaluar(res: list, dot: dict) -> dict:
    """% de réplicas en que cumple cada (turno, rol que decide) y el día, con el criterio de `simulacion_dia.agregar` (traspaso ≤ 5% de las horas del rol en el turno que recibe)."""
    tol = sd.TRASPASO_TOL
    ok, por_rol = [], {}
    for t in sd.TURNOS:
        roles = res[0]["por_turno"][t].get("roles_criterio", sd.CRITERIO_ROLES[t])
        sig = sd.SIGUIENTE[t]
        for ro in roles:
            v = []
            for r in res:
                x = r["por_turno"][t]
                if sig is None or (t, ro) in sd.SIN_TRASPASO:
                    v.append(x["t_fin_rol"][ro] <= sd.FIN[t] + 1e-9)
                else:
                    v.append(x["traspaso_h"][ro] <= tol * max(1, dot[sig].get(ro.capitalize(), 0)) * sc.HORAS_TURNO + 1e-9)
            por_rol[(t, ro)] = np.array(v)
    dia = np.all(np.array(list(por_rol.values())), axis=0)
    return {"pct_dia": float(dia.mean()), "pct_rol": {k: float(v.mean()) for k, v in por_rol.items()}}


def _dot(esc: dict, n: dict) -> dict:
    d = {t: {"Operario": OPERARIOS[t], "Gruero": n.get((t, "Gruero"), 1)} for t in sd.TURNOS}
    for t in sd.TURNOS:
        if esc["chequeadores"]:
            d[t]["Chequeador"] = CHEQUEADORES[t]
        if any(r == "polivalente" for r, _ in esc["rol"].values()):
            d[t]["Polivalente"] = n.get((t, "Polivalente"), 1)
    return d


def buscar(pool, nivel: str, en: str, R: int, R_conf: int, log) -> dict:
    """Mínimo de la simulación: parte del entero del modelo de capacidad; sube 1 al rol que menos cumple hasta llegar al 95%; después baja 1 a cada rol mientras siga cumpliendo; confirma con R_conf."""
    esc = escenarios()[en]
    clave = (nivel, en)
    a = analitico(DEMANDAS[nivel], esc)
    tam = [(t, "Gruero") for t in TURNOS] + ([(t, "Polivalente") for t in TURNOS] if "Polivalente" in _dot(esc, {})["A"] else [])
    n = {k: max(1, math.ceil(a[k[0]][k[1]] - 1e-9)) for k in tam}
    cache = {}

    def ev(n_, RR):
        key = (tuple(sorted(n_.items())), RR)
        if key not in cache:
            d = _dot(esc, n_)
            res = _correr(pool, clave, d, RR)
            cache[key] = (_evaluar(res, d), sd.agregar(res, d, 0.95))
            log(f"  {nivel} {en} R={RR} " + " ".join(f"{r[0]}{t}={v}" for (t, r), v in sorted(n_.items())) + f" -> día {100 * cache[key][0]['pct_dia']:.1f}%")
        return cache[key]

    def subir(n_, RR):
        for _ in range(12):
            e, _ = ev(n_, RR)
            if e["pct_dia"] >= 0.95 - 1e-9:
                return n_
            cand = [k for k in tam if (k[0], k[1].lower()) in e["pct_rol"]]
            k = min(cand, key=lambda k: (e["pct_rol"][(k[0], k[1].lower())], -n_[k]))
            n_ = {**n_, k: n_[k] + 1}
        return n_

    n = subir(n, R)
    for k in tam:                                           # bajar mientras siga cumpliendo
        while n[k] > 1:
            m = {**n, k: n[k] - 1}
            if ev(m, R)[0]["pct_dia"] >= 0.95 - 1e-9:
                n = m
            else:
                break
    n = subir(n, R_conf)
    e, ag = ev(n, R_conf)
    d = _dot(esc, n)
    return {"n": n, "dot": d, "eval": e, "agregado": ag, "analitico": a}


def _fila_sim(nivel, en, b) -> dict:
    d, ag, e = b["dot"], b["agregado"], b["eval"]
    f = {"demanda": nivel, "cajas": DEMANDAS[nivel], "escenario": en, "replicas": ag["replicas"], "pct_dia_viable": round(100 * e["pct_dia"], 1)}
    tot = 0
    for t in TURNOS:
        for rol in ("Gruero", "Chequeador", "Polivalente"):
            if rol in d[t]:
                f[f"{rol}_{t}"] = d[t][rol]
                tot += d[t][rol]
        T = ag["turnos"][t]
        f[f"uso_grueros_{t}"] = round(T["util_gr"], 3)
        for k, v in T.get("util_x", {}).items():
            f[f"uso_{k}_{t}"] = round(v, 3)
        f[f"traspaso_h_{t}"] = round(sum(v for k, v in T["traspaso_h"].items() if k != "operario"), 2)
    f["personas_patio"] = tot
    f["uso_gruas_24h"] = round(ag["util_grua"], 3)
    return f


def resumen(filas_a: list[dict], filas_s: list[dict]) -> str:
    c = COSTO_PERSONA_MES["valor"]
    L = ["# Polivalencia en el patio: chequeadores y gruero-despachador", "",
         f"Generado por `modelo/polivalencia.py` ({time.strftime('%Y-%m-%d')}). Escrito para el equipo del proyecto. Opción aparte: no cambia el modelo de capacidad ni la simulación por defecto.", "",
         "## Parámetros", "",
         f"- Descarga (gruero): {DESCARGA_S['valor']:.1f} s por pallet. {DESCARGA_S['fuente']}.",
         f"- Escaneo rápido (chequeador o polivalente): {ESCANEO_RAPIDO_S['valor']:.0f} s. {ESCANEO_RAPIDO_S['fuente']}.",
         f"- Escaneo del gruero: {ESCANEO_GRUERO_S['valor']:.0f} s. {ESCANEO_GRUERO_S['fuente']}. Variante con el medido: {ESCANEO_GRUERO_MEDIDO_S['valor']:.1f} s.",
         f"- Traslado patio → STREC: {TRASLADO_S['valor']:.0f} s. {TRASLADO_S['fuente']}.",
         f"- Chequeadores reales: A {CHEQUEADORES['A']} (2 + 1 intermedio), B {CHEQUEADORES['B']}, C {CHEQUEADORES['C']}. Costo por persona: ${c:,.0f} al mes ({COSTO_PERSONA_MES['fuente']}).".replace(",", "."),
         "", "## Escenarios", ""] + [f"- **{k}**: {v['descripcion']}." for k, v in escenarios().items()]
    L += ["", "## Modelo de capacidad (personas necesarias, entero entre paréntesis)", "",
          "| Demanda | Escenario | Gruero A/B/C | Chequeador A/B/C (trabajo; dotación real) | Polivalente A/B/C | Personas del patio |", "|---|---|---|---|---|---|"]
    for f in filas_a:
        fx = lambda r: " / ".join(f"{f[f'{r}_{t}']:.2f} ({f[f'{r}_{t}_entero']})" for t in TURNOS)
        L.append(f"| {f['demanda']} ({f['cajas']:,}) | {f['escenario']}{' — ' + f['variante'] if 'medido' in f['variante'] else ''} | {fx('Gruero')} | {fx('Chequeador')} | {fx('Polivalente')} | {f['personas_patio']} |".replace(",", "."))
    if filas_s:
        L += ["", "## Simulación (mínimo que cumple en ≥ 95% de las réplicas; traspaso 5%, llegadas de camiones del log)", "",
              "| Demanda | Escenario | Gruero A/B/C | Chequeador A/B/C | Polivalente A/B/C | Personas del patio | % días viables | Ahorro vs E1 (personas) | Ahorro vs E1 ($ al año) |", "|---|---|---|---|---|---|---|---|---|"]
        e1 = {f["demanda"]: f["personas_patio"] for f in filas_s if f["escenario"] == "E1"}
        for f in filas_s:
            fx = lambda r: " / ".join(str(f.get(f"{r}_{t}", "–")) for t in TURNOS)
            ah = e1.get(f["demanda"], f["personas_patio"]) - f["personas_patio"]
            L.append(f"| {f['demanda']} | {f['escenario']} | {fx('Gruero')} | {fx('Chequeador')} | {fx('Polivalente')} | {f['personas_patio']} | {f['pct_dia_viable']} | {ah} | {ah * c * 12:,.0f} |".replace(",", "."))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replicas", type=int, default=300)
    ap.add_argument("--confirmar", type=int, default=config.N_REPLICAS)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--solo-analitico", action="store_true")
    a = ap.parse_args(argv)
    DESCARGA_S["valor"] = capacity.seg_recepcion_por_pallet(1.0)
    ESCANEO_GRUERO_MEDIDO_S["valor"] = capacity.seg_recepcion_por_pallet(0.0) - DESCARGA_S["valor"]
    SAL.mkdir(parents=True, exist_ok=True)
    fa = tabla_analitica()
    with open(SAL / "analitico.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fa[0]))
        w.writeheader(); w.writerows(fa)
    fs = []
    if not a.solo_analitico:
        t0 = time.time()
        aj = capacity.Ajustes()
        sd.configurar_ventana(aj.horas_efectivas())
        params = fit_params.cargar()
        for nivel in DEMANDAS_SIM:
            base, _ = sd.preparar_dia(DEMANDAS[nivel], params, aj)
            for en, esc in escenarios().items():
                _P[(nivel, en)] = preparar(DEMANDAS[nivel], esc, base)
        log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min]{m}", flush=True)
        with mp.get_context("fork").Pool(a.procesos) as pool:
            for nivel in DEMANDAS_SIM:
                for en in escenarios():
                    b = buscar(pool, nivel, en, a.replicas, a.confirmar, log)
                    fs.append(_fila_sim(nivel, en, b))
                    with open(SAL / "simulacion.csv", "w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for f in fs for k in f)))
                        w.writeheader(); w.writerows(fs)
        log(" listo")
    (SAL / "resumen.md").write_text(resumen(fa, fs), encoding="utf-8")
    print((SAL / "resumen.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
