"""Grilla de escenarios de apoyo en el patio: 2 personas de apoyo por turno (chequeadores y planificadores = un solo rol, decisión del usuario 2026-11-20), cada una en un modo.

MODOS DE CADA PERSONA (los mismos en los tres turnos):
  N     no ayuda en el patio (hace su trabajo de oficina; no se cuenta en el patio)
  A50   apoya el 50% de cada hora: escanea (14 s) y traslada a STREC (58 s, con su equipo móvil, sin ocupar una de las 6 grúas)
  A100  apoya todo el turno
  P50   polivalente el 50% de cada hora: además de escanear y trasladar, baja pallets (con grúa), carga camiones y hace el retorno de fletero del gruero
        («solo patio», decisión del usuario: el almacenamiento y los reabastos siguen siendo del gruero)
  P100  polivalente todo el turno
Combinaciones de 2 personas sin orden: 15 (N+N es la línea base: el gruero hace todo). Demandas: baja, media y alta → 45 búsquedas.

QUIÉN TOMA CADA PASO: el primero que quede libre entre los elegibles (`simulacion_dia._replica_dia(pools=...)`); a igual instante, el primero de la lista.
  rec_descarga   gruero (grúa) · polivalentes (grúa)                                 30,2 s
  rec_escaneo    apoyos y polivalentes (14 s, sin grúa) · gruero (30 s, con grúa)   el gruero escanea si nadie de apoyo está libre
  rec_traslado   apoyos y polivalentes (sin grúa) · gruero (grúa)                    58 s
  carga_camion, retorno_fletero_gruero   gruero · polivalentes
La disponibilidad parcial va al final de cada hora (p. ej. con 50%: libre los primeros 30 min de cada hora) y nadie de apoyo trabaja después de su hora de salida.
Lo que no alcanza a empezar se traspasa como trabajo del gruero. Deciden el turno los grueros (y los operarios del C, holgados: A 4, B 4, C 20).

BÚSQUEDA del mínimo de grueros por turno (el apoyo queda fijo): parte del mayor entre el entero del modelo de capacidad de la línea base menos lo que aporta el apoyo (cota baja: 1)
y el mínimo de la demanda anterior (MONÓTONO: con más demanda no puede bajar); sube 1 al turno que menos cumple hasta el 95% (300 réplicas), baja mientras siga cumpliendo, y confirma con 1.000.
Mismas semillas en todas las combinaciones (los muestreos no dependen del escenario).

USO: .venv/bin/python -m modelo.polivalencia_grilla [--replicas 300] [--confirmar 1000] [--procesos 4] [--demandas baja,media,alta] [--combos N+N,A100+A100]
Salida: salidas/polivalencia/grilla.csv y salidas/polivalencia/grilla.md (se reescriben a medida que avanza; una búsqueda ya hecha se reutiliza de grilla_cache.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import time

import numpy as np

from modelo import capacity, config, fit_params
from modelo import llegadas_camiones as lc
from modelo import polivalencia as pv
from modelo import simulacion_dia as sd
from modelo import simulacion_turno_c as sc

MODOS = {"N": (None, 0.0), "A50": ("A", 0.5), "A100": ("A", 1.0), "P50": ("P", 0.5), "P100": ("P", 1.0)}
COMBOS = [f"{a}+{b}" for a, b in itertools.combinations_with_replacement(MODOS, 2)]
TURNOS = ("A", "B", "C")
SAL = pv.SAL
CACHE = SAL / "grilla_cache.jsonl"
_P: dict = {}
_POOLS: dict = {}
UMBRAL = 0.95                                             # % mínimo de días viables (se cambia con --umbral)


def pools(combo: str) -> dict:
    """Personas de apoyo y elegibles por paso, iguales en los tres turnos."""
    esc_rap, esc_gr = pv.ESCANEO_RAPIDO_S["valor"], pv.ESCANEO_GRUERO_S["valor"]
    ug_ret = pv._usa_grua("retorno_fletero_gruero")
    personas, pasos = {}, {}
    for t in sd.TURNOS:
        per = {}
        for k, m in enumerate(combo.split("+"), 1):
            tipo, f = MODOS[m]
            if tipo:
                per[f"{tipo.lower()}{k}"] = f
        ap = [n for n in per if n.startswith("a")]
        po = [n for n in per if n.startswith("p")]
        personas[t] = per
        pasos[t] = {"rec_descarga": [("gruero", True, 1.0)] + [(n, True, 1.0) for n in po],
                    "rec_escaneo": [(n, False, esc_rap) for n in ap + po] + [("gruero", True, esc_gr)],
                    "rec_traslado": [(n, False, 1.0) for n in ap + po] + [("gruero", True, 1.0)],
                    "carga_camion": [("gruero", True, 1.0)] + [(n, True, 1.0) for n in po],
                    "retorno_fletero_gruero": [("gruero", ug_ret, 1.0)] + [(n, ug_ret, 1.0) for n in po]}
    return {"personas": personas, "pasos": pasos}


def analitico(demanda: float, combo: str) -> dict:
    """Grueros necesarios por turno (fraccional) si el apoyo absorbe primero lo que puede: escaneo y traslado (apoyo y polivalente) y descarga, carga y retorno (polivalente),
    hasta sus horas disponibles (fracción × 7,17 h × utilización del resto, 0,80). Es solo el punto de partida de la búsqueda."""
    a = pv.analitico(demanda, pv.escenarios()["E0"])          # el gruero hace todo, escaneo 30 s
    aj = capacity.Ajustes()
    H, ef = aj.horas_efectivas(), aj.eficiencia_resto
    e = capacity.estimar(demanda, aj)
    out = {}
    for t in TURNOS:
        pal = a[t]["pallets"]
        per = pools(combo)["personas"][t]
        cap = sum(f for f in per.values()) * H                  # horas de trabajo disponibles del apoyo (antes de dividir por la utilización)
        ap_h = pal * (pv.ESCANEO_RAPIDO_S["valor"] + pv.TRASLADO_S["valor"]) / 3600 / ef
        ahorro_esc = pal * (pv.ESCANEO_GRUERO_S["valor"] + pv.TRASLADO_S["valor"]) / 3600 / ef
        hecho = min(cap, ap_h) / ap_h if ap_h > 0 else 0.0
        libre = max(0.0, cap - ap_h * hecho)
        horas_g = a[t]["Gruero"] * H - ahorro_esc * hecho
        if any(n.startswith("p") for n in per) and libre > 0:
            poli = [d["horas_persona"] for d in e["detalle"] if d["turno"] == t and d["rol"] == "Gruero" and d["proceso"] in pv.DESPACHO]
            poli.append(pal * pv.DESCARGA_S["valor"] / 3600 / ef)
            horas_g -= min(libre, sum(poli))
        out[t] = max(0.0, horas_g) / H
    return out


def _trab(a):
    nivel, combo, dot, sem = a
    g = lc.GeneradorLlegadas(pv.DEMANDAS[nivel], "rec_carga", horas=sc.HORAS_TURNO)
    return [sd._replica_dia(_P[nivel], dot, sc.SUPUESTOS["gruas"]["valor"], s, llegadas=g, traspaso=sd.TRASPASO_TOL, pools=_POOLS[combo]) for s in sem]


def _correr(pool, nivel, combo, dot, R):
    base = sc.SUPUESTOS["semilla_base"]["valor"]
    blq = [list(range(base + i, base + min(R, i + 25))) for i in range(0, R, 25)]
    return [x for p in pool.map(_trab, [(nivel, combo, dot, b) for b in blq]) for x in p]


def _dot(n: dict) -> dict:
    return {t: {"Operario": pv.OPERARIOS[t], "Gruero": n[t]} for t in sd.TURNOS}


def simular(demanda: float, combo: str, grueros: dict | None, replicas: int, umbral: float | None = None, params=None, progreso=None,
            dot: dict | None = None, ajustes=None) -> dict:
    """Una combinación a cualquier demanda, en serie (la usa el dashboard). Mismos supuestos que la grilla: descarga con planificador, ventana de 7 h 10, llegadas del log,
    traspaso 5% y semillas semilla_base + r. Sin `dot`, operarios holgados (A 4, B 4, C 20: el resultado es solo del patio, como la grilla); con `dot` = {turno: {"Operario", "Gruero"}}
    se simula esa dotación completa (el día entero). `ajustes` = los del estimador (por defecto, los del modelo). Devuelve además `agregado` (salida de `simulacion_dia.agregar`)."""
    umbral = sc.SUPUESTOS["umbral_viabilidad"]["valor"] if umbral is None else umbral
    aj = capacity.Ajustes() if ajustes is None else ajustes
    pv.DESCARGA_S["valor"] = capacity.seg_recepcion_por_pallet(1.0)
    sd.configurar_ventana(aj.horas_efectivas())
    base, _ = sd.preparar_dia(demanda, params or fit_params.cargar(), aj)
    preps = pv.preparar(demanda, {"escaneo_s": 1.0, "rol": {}}, base)
    if dot is not None:
        d = {t: {"Operario": int(dot[t]["Operario"]), "Gruero": int(dot[t]["Gruero"])} for t in sd.TURNOS}
        grueros = {t: d[t]["Gruero"] for t in TURNOS}
    else:
        d = _dot(grueros)
    pool_ = pools(combo)
    g = lc.GeneradorLlegadas(demanda, "rec_carga", horas=sc.HORAS_TURNO)
    s0 = sc.SUPUESTOS["semilla_base"]["valor"]
    res = []
    for i in range(replicas):
        res.append(sd._replica_dia(preps, d, sc.SUPUESTOS["gruas"]["valor"], s0 + i, llegadas=g, traspaso=sd.TRASPASO_TOL, pools=pool_))
        if progreso:
            progreso(i + 1, replicas)
    e, ag = pv._evaluar(res, d), sd.agregar(res, d, umbral)
    per = pool_["personas"]
    uso_ap = {t: (float(np.mean([np.mean([r["por_turno"][t]["util_apoyo"][nom] for r in res]) / f for nom, f in per[t].items()])) if per[t] else None) for t in TURNOS}
    return {"demanda": demanda, "combo": combo, "grueros": dict(grueros), "replicas": replicas, "umbral": umbral, "pct_dia": e["pct_dia"], "viable": e["pct_dia"] >= umbral - 1e-9,
            "pct_turno": {t: e["pct_rol"][(t, "gruero")] for t in TURNOS}, "uso_grueros": {t: ag["turnos"][t]["util_gr"] for t in TURNOS},
            "traspaso_h": {t: ag["turnos"][t]["traspaso_h"]["gruero"] for t in TURNOS}, "uso_apoyo": uso_ap, "uso_gruas_24h": ag["util_grua"], "agregado": ag}


def referencia(combo: str) -> list[dict]:
    """Mínimos ya validados de la grilla (1.000 réplicas) para la combinación, con cada umbral que se haya corrido (grilla.csv = 95%, grilla90.csv = 90%)."""
    out = []
    for archivo, umb in (("grilla.csv", 95), ("grilla90.csv", 90)):
        f = SAL / archivo
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["combinacion"] == combo:
                    out.append({"umbral": umb, **r})
    return out


def minimos_validados() -> dict:
    """Dotaciones mínimas ya validadas con 1.000 réplicas y umbral 90%, {(combo o None, cajas): {turno: {"Operario", "Gruero"}}}: el modelo base (combo None) desde
    `salidas/simulacion_final/escenarios.csv` (última fila «mínimo simulación» viable de cada demanda) y los TO-BE desde `salidas/polivalencia/grilla90.csv` (grueros; los operarios
    del TO-BE son los del modelo base, porque el apoyo no toca el trabajo de los operarios: verificado con 1 polivalente en demanda media, 13 y 20 operarios en C dan el mismo 92,0% con 200 réplicas)."""
    out = {}
    f = SAL.parent / "simulacion_final" / "escenarios.csv"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["tipo"] == "mínimo simulación" and r["viable"] == "True":
                    g, o = [int(x) for x in r["grueros_A_B_C"].split("/")], [int(x) for x in r["operarios_A_B_C"].split("/")]
                    out[(None, int(r["cajas"]))] = {t: {"Operario": o[i], "Gruero": g[i]} for i, t in enumerate(TURNOS)}
    f = SAL / "grilla90.csv"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                base = out.get((None, int(r["cajas"])))
                if base:
                    out[(r["combinacion"], int(r["cajas"]))] = {t: {"Operario": base[t]["Operario"], "Gruero": int(r[f"grueros_{t}"])} for t in TURNOS}
    return out


def buscar(pool, nivel: str, combo: str, piso: dict, R: int, R_conf: int, log, inicio: dict | None = None) -> dict:
    a = analitico(pv.DEMANDAS[nivel], combo)
    n = {t: max(1, piso.get(t, 1), math.ceil(a[t] - 1e-9)) for t in TURNOS} if inicio is None else {t: max(1, piso.get(t, 1), inicio[t]) for t in TURNOS}
    cache = {}

    def ev(n_, RR):
        key = (tuple(sorted(n_.items())), RR)
        if key not in cache:
            d = _dot(n_)
            res = _correr(pool, nivel, combo, d, RR)
            cache[key] = (pv._evaluar(res, d), sd.agregar(res, d, UMBRAL), res)
            log(f"  {nivel} {combo} R={RR} G A/B/C={n_['A']}/{n_['B']}/{n_['C']} -> día {100 * cache[key][0]['pct_dia']:.1f}%")
        return cache[key]

    def subir(n_, RR):
        for _ in range(12):
            e = ev(n_, RR)[0]
            if e["pct_dia"] >= UMBRAL - 1e-9:
                return n_
            t = min(TURNOS, key=lambda t: (e["pct_rol"][(t, "gruero")], -n_[t]))
            n_ = {**n_, t: n_[t] + 1}
        return n_

    n = subir(n, R)
    for t in TURNOS:
        while n[t] > max(1, piso.get(t, 1)):
            m = {**n, t: n[t] - 1}
            if ev(m, R)[0]["pct_dia"] >= UMBRAL - 1e-9:
                n = m
            else:
                break
    n = subir(n, R_conf)
    e, ag, res = ev(n, R_conf)
    per = _POOLS[combo]["personas"]
    util = {t: {nom: float(np.mean([r["por_turno"][t]["util_apoyo"][nom] for r in res])) / f for nom, f in per[t].items()} for t in TURNOS}
    return {"demanda": nivel, "cajas": pv.DEMANDAS[nivel], "combo": combo, "grueros": n, "pct_dia": e["pct_dia"], "replicas": ag["replicas"],
            "uso_grueros": {t: ag["turnos"][t]["util_gr"] for t in TURNOS}, "uso_apoyo": util, "traspaso_h": {t: ag["turnos"][t]["traspaso_h"]["gruero"] for t in TURNOS},
            "uso_gruas_24h": ag["util_grua"], "analitico": a}


def fila(b: dict) -> dict:
    per = _POOLS[b["combo"]]["personas"]["A"]
    f = {"demanda": b["demanda"], "cajas": b["cajas"], "combinacion": b["combo"], "replicas": b["replicas"], "pct_dia_viable": round(100 * b["pct_dia"], 1)}
    for t in TURNOS:
        f[f"grueros_{t}"] = b["grueros"][t]
        f[f"uso_grueros_{t}"] = round(b["uso_grueros"][t], 3)
        f[f"traspaso_h_{t}"] = round(b["traspaso_h"][t], 2)
        f[f"uso_apoyo_{t}"] = round(float(np.mean(list(b["uso_apoyo"][t].values()))), 3) if b["uso_apoyo"][t] else ""
    f["grueros_total"] = sum(b["grueros"].values())
    f["apoyo_personas_turno"] = len(per)
    f["apoyo_personas_dia"] = 3 * len(per)
    f["apoyo_equivalentes_dia"] = round(3 * sum(per.values()), 1)
    f["personas_patio_dia"] = f["grueros_total"] + f["apoyo_personas_dia"]
    f["equivalentes_patio_dia"] = round(f["grueros_total"] + f["apoyo_equivalentes_dia"], 1)
    f["uso_gruas_24h"] = round(b["uso_gruas_24h"], 3)
    return f


def _mil(x: float) -> str:
    """Entero con separador de miles «·» (se cambia a punto al final de la fila, después de pasar los decimales a coma)."""
    return f"{x:,.0f}".replace(",", "·")


def informe(filas: list[dict]) -> str:
    c = pv.COSTO_PERSONA_MES["valor"]
    base = {f["demanda"]: f for f in filas if f["combinacion"] == "N+N"}
    L = ["# Grilla de apoyo en el patio (2 personas de apoyo por turno)", "", f"Generado por `modelo/polivalencia_grilla.py` ({time.strftime('%Y-%m-%d')}). Modos: N no ayuda · A apoya (escanea y traslada) · "
         f"P polivalente (además baja, carga y retorno de fletero) · 50 / 100 = % de cada hora. Grueros mínimos con ≥ {round(100 * UMBRAL)}% de días viables (confirmado con "
         f"{filas[0]['replicas'] if filas else '–'} réplicas). Equivalentes = grueros + apoyo × fracción. Ahorro contra N+N (el gruero hace todo), a ${_mil(c).replace('·', '.')} por persona al mes.", "",
         "| Demanda | Combinación | Grueros A/B/C | Grueros | Apoyo (personas/día) | Equivalentes/día | Uso del apoyo A/B/C | % días viables | Grueros ahorrados | Equivalentes ahorrados | $ al año (equivalentes) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in filas:
        b = base.get(f["demanda"])
        dg = b["grueros_total"] - f["grueros_total"] if b else ""
        de = round(b["equivalentes_patio_dia"] - f["equivalentes_patio_dia"], 1) if b else ""
        us = " / ".join(f"{100 * f[f'uso_apoyo_{t}']:.0f}%" if f[f"uso_apoyo_{t}"] != "" else "–" for t in TURNOS)
        L.append(f"| {f['demanda']} | {f['combinacion']} | {f['grueros_A']}/{f['grueros_B']}/{f['grueros_C']} | {f['grueros_total']} | {f['apoyo_personas_dia']} | {f['equivalentes_patio_dia']} | {us} | "
                 f"{f['pct_dia_viable']} | {dg} | {de} | {_mil(de * c * 12) if de != '' else ''} |".replace(".", ",").replace("·", "."))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replicas", type=int, default=300)
    ap.add_argument("--confirmar", type=int, default=config.N_REPLICAS)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--demandas", default="baja,media,alta")
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--salida", default="grilla")
    ap.add_argument("--umbral", type=float, default=0.95, help="fracción mínima de días viables (0,95 por defecto)")
    ap.add_argument("--desde", default=None, help="parte de los mínimos de otra corrida (su caché), p. ej. grilla; sirve para un umbral más bajo")
    a = ap.parse_args(argv)
    global UMBRAL
    UMBRAL = a.umbral
    pv.DESCARGA_S["valor"] = capacity.seg_recepcion_por_pallet(1.0)
    aj = capacity.Ajustes()
    sd.configurar_ventana(aj.horas_efectivas())
    params = fit_params.cargar()
    niveles = a.demandas.split(",")
    combos = a.combos.split(",")
    unidad = {"escaneo_s": 1.0, "rol": {}}                      # el escaneo se muestrea en unidades: su duración la pone el factor del que lo toma (14 o 30 s)
    for nivel in niveles:
        base, _ = sd.preparar_dia(pv.DEMANDAS[nivel], params, aj)
        _P[nivel] = pv.preparar(pv.DEMANDAS[nivel], unidad, base)
    for cb in combos:
        _POOLS[cb] = pools(cb)
    SAL.mkdir(parents=True, exist_ok=True)
    cache_f = SAL / f"{a.salida}_cache.jsonl"
    hechos = {}
    if cache_f.exists():
        for ln in cache_f.read_text(encoding="utf-8").splitlines():
            x = json.loads(ln)
            if x.get("replicas") == a.confirmar:
                hechos[(x["demanda"], x["combo"])] = x
    inicios = {}
    if a.desde:
        for ln in (SAL / f"{a.desde}_cache.jsonl").read_text(encoding="utf-8").splitlines():
            x = json.loads(ln)
            inicios[(x["demanda"], x["combo"])] = x["grueros"]
    t0 = time.time()
    log = lambda m: print(f"[{(time.time() - t0) / 60:6.1f} min]{m}", flush=True)
    filas = []
    with mp.get_context("fork").Pool(a.procesos) as pool:
        for cb in combos:
            piso = {}
            for nivel in niveles:                              # demandas en orden creciente: el mínimo no puede bajar al subir la demanda
                b = hechos.get((nivel, cb))
                if b is None:
                    b = buscar(pool, nivel, cb, piso, a.replicas, a.confirmar, log, inicios.get((nivel, cb)))
                    b["grueros"] = {k: int(v) for k, v in b["grueros"].items()}
                    with open(cache_f, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(b, default=float) + "\n")
                piso = dict(b["grueros"])
                filas.append(fila(b))
                filas.sort(key=lambda f: (niveles.index(f["demanda"]), combos.index(f["combinacion"])))
                with open(SAL / f"{a.salida}.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(filas[0]))
                    w.writeheader(); w.writerows(filas)
                (SAL / f"{a.salida}.md").write_text(informe(filas), encoding="utf-8")
    log(" listo")
    print(informe(filas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
