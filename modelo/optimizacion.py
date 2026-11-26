"""CAPA DE OPTIMIZACIÓN (MILP, PuLP + CBC) sobre el estimador de capacidad (2026-10-23).

Formulación: docs_modelo/formulacion_modelo_final.tex. No cambia la lógica ni los valores de capacity.py: toma de `capacity.estimar()` las horas de presencia que pide cada
tarea, W[k,s] = V[k,s] · τ_k / ρ_k (la columna `horas_persona` del detalle), y el tiempo disponible por persona H·η = 8 h − charla − colación (`Ajustes.horas_efectivas`).

Variables (por escenario):  n[r,s] ∈ Z≥0 (personas del rol r en el turno s);  y[k,r,s] ≥ 0 (horas de presencia del rol r en la tarea k).
Objetivo:                   min Σ_r Σ_s n[r,s]
Restricciones:
  (1) cumplimiento     Σ_r y[k,r,s] = W[k,s]                               ∀ k, s
  (2) capacidad        Σ_k y[k,r,s] ≤ n[r,s] · H·η                         ∀ r, s
  (3) grúas            Σ_k g_k Σ_r y[k,r,s] ≤ G_s · H·η                    ∀ s
  (4) elegibilidad     y[k,r,s] ≤ E[k,r] · W[k,s]                          ∀ k, r, s
  (5) mínimo y tope    F[r,s] ≤ n[r,s] ≤ R[r,s]                            ∀ r, s
Si (5) hace infactible el escenario, se resuelve sin la cota superior y se informa la brecha n − R. Si además (3) lo hace infactible (las horas de grúa que pide el
turno superan G·H·η), se resuelve sin (3) y se informa el exceso de horas de grúa.

TERCER ROL «planificador-gruero» (2026-10-26, OPCIONAL: `pg=None` no cambia nada). pg = {"n": {turno: personas}, "vel": velocidad relativa, "nucleo": procesos que atiende con toda su presencia (más la fracción
  `frac_nucleo[turno]` del turno), "ayuda": procesos en que solo ayuda hasta `f_ayuda` de su presencia}. Son personas FIJAS (no variable de decisión): variables y[k,'PG',s] ≥ 0 solo para los procesos elegibles,
  (2') Σ_k y[k,PG,s] ≤ n·H·η·vel·frac_nucleo   y   Σ_{k∈ayuda} y[k,PG,s] ≤ n·H·η·vel·f_ayuda;  (1) y (3) incluyen sus horas. `factor_paso` multiplica las horas de presencia de un proceso (recepción TO-BE).
Salidas: salidas/optimizacion/resultados_optimizacion.csv y equivalencia_optimizacion.csv.  USO: .venv/bin/python -m modelo.optimizacion
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import pulp

from modelo import capacity, config, fit_params

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "salidas" / "optimizacion"
TURNOS = config.TURNOS
ROLES = config.ROLES_OPTIMIZACION
EPS = 1e-9


def escenarios() -> dict[str, float]:
    """Los mismos escenarios del paquete de documentación (exportar_documentacion.escenarios)."""
    from modelo import exportar_documentacion
    return exportar_documentacion.escenarios()


def carga(demanda: float, ajustes: capacity.Ajustes | None = None, parametros=None) -> dict:
    """W[k,s] (horas de presencia por tarea y turno), el rol de cada tarea, H·η y la dotación fraccional N[r,s] del estimador. Todo sale de capacity.estimar()."""
    aj = ajustes or capacity.Ajustes()
    est = capacity.estimar(demanda, aj, parametros=parametros)
    W, rol_de = {}, {}
    for x in est["detalle"]:
        if x["horas_persona"] > EPS:
            W[(x["proceso"], x["turno"])] = W.get((x["proceso"], x["turno"]), 0.0) + x["horas_persona"]
            rol_de[x["proceso"]] = x["rol"]
    N = {(f["turno"], f["recurso"]): f["necesarios"] for f in est["filas"] if f["tipo"] == "rol"}
    return {"W": W, "rol_de": rol_de, "H_eta": aj.horas_efectivas(), "N": N, "est": est}


def resolver(demanda: float, ajustes: capacity.Ajustes | None = None, parametros=None, con_tope: bool = True, con_gruas: bool = True,
             tope: dict | None = None, elegibilidad: dict | None = None, uso_grua: dict | None = None, gruas: dict | None = None, tope_modo: str = "turno",
             pg: dict | None = None, factor_paso: dict | None = None) -> dict:
    """Resuelve un escenario. Si es infactible con el tope, lo resuelve sin la cota superior; si sigue infactible, también sin el límite de grúas.
    `tope_modo`: "turno" = n[r,s] ≤ R[r,s] (restricción 5 del .tex); "total" = Σ_s n[r,s] ≤ Σ_s R[r,s] (regla del proyecto de dotación total neutra: se puede mover
    gente entre turnos dentro de un rol, sin sumar personas)."""
    c = carga(demanda, ajustes, parametros)
    if factor_paso:
        c = {**c, "W": {k: w * factor_paso.get(k[0], 1.0) for k, w in c["W"].items()}}
    E = elegibilidad or config.elegibilidad()
    g = uso_grua or config.uso_grua()
    G = gruas or config.GRUAS_POR_TURNO
    R = tope or {**config.TOPE_DOTACION, **{(t, r): n for t, r, n in (ajustes.dotacion_real if ajustes else ())}}      # dotación actual editada en el dashboard (si la hay) manda sobre el tope por defecto
    intentos = [(con_tope, con_gruas)]
    if con_tope:
        intentos.append((False, con_gruas))
    if con_gruas:
        intentos.append((False, False))
    for tope_on, gruas_on in intentos:
        r = _resolver_una_vez(c, E, g, G, R, tope_on, gruas_on, tope_modo, pg)
        if r["estado"] == "Optimal":
            break
    r.update({"demanda": demanda, "tope_activo_pedido": con_tope, "gruas_pedidas": con_gruas, "tope_relajado": con_tope and not r["con_tope"],
              "gruas_relajadas": con_gruas and not r["con_gruas"], "N": c["N"], "H_eta": c["H_eta"], "tope": R})
    r["tope_modo"] = tope_modo
    r["brecha"] = {k: max(0, r["n"][k] - R[k]) for k in r["n"]} if r["estado"] == "Optimal" else {}
    r["horas_grua"] = {s: sum(g.get(k, 0.0) * w for (k, t), w in c["W"].items() if t == s) for s in TURNOS}
    r["horas_grua_disponibles"] = {s: G[s] * c["H_eta"] for s in TURNOS}
    r["horas_operario_conocidas"] = {s: sum(w for (k, t), w in c["W"].items() if t == s and c["rol_de"][k] == "Operario") for s in TURNOS}
    return r


def _resolver_una_vez(c: dict, E: dict, g: dict, G: dict, R: dict, con_tope: bool, con_gruas: bool, tope_modo: str = "turno", pg: dict | None = None) -> dict:
    W, He = c["W"], c["H_eta"]
    p = pulp.LpProblem("dotacion", pulp.LpMinimize)
    n = {(s, r): pulp.LpVariable(f"n_{r}_{s}", lowBound=0, cat="Integer") for s in TURNOS for r in ROLES}
    y = {(k, r, s): pulp.LpVariable(f"y_{k}_{r}_{s}", lowBound=0) for (k, s) in W for r in ROLES}
    p += pulp.lpSum(n.values())                                                                     # objetivo
    y3 = {}
    if pg:
        elegibles = set(pg["nucleo"]) | set(pg["ayuda"])
        y3 = {(k, s): pulp.LpVariable(f"ypg_{k}_{s}", lowBound=0) for (k, s) in W if k in elegibles and pg["n"].get(s, 0) > 0}
        for s in TURNOS:
            cap = pg["n"].get(s, 0) * He * pg["vel"]
            p += pulp.lpSum(v for (k, t), v in y3.items() if t == s) <= cap * pg["frac_nucleo"].get(s, 1.0), f"c2pg_capacidad_{s}"
            if pg["ayuda"]:
                p += pulp.lpSum(v for (k, t), v in y3.items() if t == s and k in pg["ayuda"] and k not in pg["nucleo"]) <= cap * pg["f_ayuda"], f"c2pg_ayuda_{s}"
    for (k, s), w in W.items():
        p += pulp.lpSum(y[(k, r, s)] for r in ROLES) + (y3[(k, s)] if (k, s) in y3 else 0) == w, f"c1_cumplimiento_{k}_{s}"               # (1)
        for r in ROLES:
            p += y[(k, r, s)] <= E.get((k, r), 0) * w, f"c4_elegibilidad_{k}_{r}_{s}"               # (4)
    for s in TURNOS:
        for r in ROLES:
            p += pulp.lpSum(y[(k, r, t)] for (k, r2, t) in y if r2 == r and t == s) <= n[(s, r)] * He, f"c2_capacidad_{r}_{s}"   # (2)
            F = 1 if any(E.get((k, r), 0) and w > EPS for (k, t), w in W.items() if t == s) else 0
            p += n[(s, r)] >= F, f"c5_minimo_{r}_{s}"                                               # (5) mínimo
            if con_tope and tope_modo == "turno":
                p += n[(s, r)] <= R[(s, r)], f"c5_tope_{r}_{s}"                                       # (5) tope por turno
        if con_gruas:
            p += (pulp.lpSum(g.get(k, 0.0) * y[(k, r, t)] for (k, r, t) in y if t == s)
                  + pulp.lpSum(g.get(k, 0.0) * v for (k, t), v in y3.items() if t == s)) <= G[s] * He, f"c3_gruas_{s}"   # (3)
    if con_tope and tope_modo == "total":
        for r in ROLES:
            p += pulp.lpSum(n[(s, r)] for s in TURNOS) <= sum(R[(s, r)] for s in TURNOS), f"c5b_tope_total_{r}"   # (5b) dotación total neutra por rol
    t0 = time.perf_counter()
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    dt = time.perf_counter() - t0
    estado = pulp.LpStatus[p.status]
    nv = {k: int(round(v.value())) for k, v in n.items()} if estado == "Optimal" else {}
    holgura_grua = {}
    if estado == "Optimal" and con_gruas:
        for s in TURNOS:
            holgura_grua[s] = p.constraints[f"c3_gruas_{s}"].slack
    return {"estado": estado, "n": nv, "con_tope": con_tope, "con_gruas": con_gruas, "segundos": dt, "objetivo": sum(nv.values()) if nv else None, "holgura_grua": holgura_grua}


def equivalencia(demanda: float, ajustes=None, parametros=None) -> dict:
    """Con grúas sin limitar y sin tope, n debe ser ceil(N) del estimador (cada tarea tiene un solo rol)."""
    r = resolver(demanda, ajustes, parametros, con_tope=False, con_gruas=False)
    filas = []
    for k, Nk in r["N"].items():
        esperado = math.ceil(Nk - 1e-9)
        filas.append({"turno": k[0], "rol": k[1], "N": Nk, "ceil_N": esperado, "n_solver": r["n"].get(k), "igual": r["n"].get(k) == esperado})
    return {"estado": r["estado"], "filas": filas, "todo_igual": all(f["igual"] for f in filas)}


def correr_todos(parametros=None) -> tuple[list[dict], list[dict]]:
    params = parametros or fit_params.cargar()
    res, eq = [], []
    for nombre, D in escenarios().items():
        r = resolver(D, parametros=params)
        for (s, rol), Nk in r["N"].items():
            res.append({"escenario": nombre, "demanda_cajas": round(D), "turno": s, "rol": rol, "N_fraccional": round(Nk, 3), "n_optimo": r["n"].get((s, rol)),
                        "tope_R": r["tope"][(s, rol)], "brecha": r["brecha"].get((s, rol)), "estado_solver": r["estado"], "tope_relajado": r["tope_relajado"],
                        "gruas_relajadas": r["gruas_relajadas"], "horas_grua_turno": round(r["horas_grua"][s], 2), "horas_grua_disponibles": round(r["horas_grua_disponibles"][s], 2),
                        "segundos_solver": round(r["segundos"], 3)})
        e = equivalencia(D, parametros=params)
        for f in e["filas"]:
            eq.append({"escenario": nombre, "demanda_cajas": round(D), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}})
    return res, eq


def _escribir(ruta: Path, filas: list[dict]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    res, eq = correr_todos()
    _escribir(SALIDA / "resultados_optimizacion.csv", res)
    _escribir(SALIDA / "equivalencia_optimizacion.csv", eq)
    print(f"{len(res)} filas de resultados; equivalencia n = ceil(N): {sum(e['igual'] for e in eq)} de {len(eq)} -> {SALIDA.relative_to(RAIZ)}")
