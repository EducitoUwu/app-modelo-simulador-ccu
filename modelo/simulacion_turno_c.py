"""Simulación SIMPLE del turno C (00:00 a 08:00) para probar si la dotación que calcula el modelo de capacidad alcanza.

QUÉ ES (y qué NO es)
  Un solo turno, un solo script. Cada tarea del turno C es una entidad (cada línea de picking, cada pallet, cada trailer...) que compite por operarios, grueros y grúas
  con `simpy.PriorityResource`. TODO el trabajo del turno está disponible desde t = 0 (el picking primero: prioridad 0; el resto con menor prioridad). No hay olas, ni
  llegadas por hora, ni pausas, ni contención con otros turnos. Es una prueba de suficiencia de la dotación del modelo, NO una calibración: si muestra que el modelo se queda
  corto SOLO se reporta; no cambia la dotación ni ningún parámetro. No modifica ningún otro archivo del modelo (solo lee `capacity`, `config`, `fit_params`).

DE DÓNDE SALE CADA INSUMO (nada numérico escrito a mano fuera del diccionario SUPUESTOS)
  Volúmenes del turno C ........ capacity.por_turno("C", demanda): cuotas y niveles del régimen vigente (octubre) y picking/carga según la demanda del escenario.
  Dotación de operarios/grueros . capacity.dotacion_por_turno("C", demanda): personas-equivalente del modelo, redondeadas HACIA ARRIBA.
  Grúas ......................... config.equipos_disponibles()["Grúas"] (planilla).
  Quién hace qué ................ capacity.ROL_POR_PROCESO y capacity.USO_DE_EQUIPO (el reabasto de cajas pide gruero+grúa para bajar el pallet y luego operario).
  Tiempos de servicio ........... fit_params (parametros.json): se MUESTREAN de las distribuciones ajustadas con la misma regla que `capacity.tiempo_estandar`
                                  (picking: mezcla por tipo de línea ponderada por N; carga y recepción: suma de pasos; reabasto de pallet: mezcla ALMAC/ALMPIC; el resto: ciclo_total).
  Eficiencia η .................. capacity.EFICIENCIA_*: cada tiempo muestreado se divide entre η (tiempo no productivo), igual que el modelo.
  Prioridades ................... picking 0; reabasto de pallet y de cajas las del WMS (config.config_wms); el resto una más baja [hipótesis].

LÍMITES [declarados]
  - El modelo calcula con 7 horas efectivas por turno y 0,85 de utilización; esta simulación corre las 8 horas completas y aplica solo η: la dotación del modelo trae ~1 h de holgura.
  - Las tareas no se interrumpen entre sí (sin expropiación) y el picking no pide grúa (el modelo no se la asigna).
  - Volúmenes fraccionarios (p. ej. 0,5 camiones de recepción): parte entera + 1 con probabilidad igual a la parte fraccionaria [hipótesis], para conservar el valor esperado.
  - Las esperas en cola salen enormes por construcción: todo el trabajo está disponible en t = 0, así que lo que se mide es el tamaño de la cola, no una demora de servicio.

USO:  .venv/bin/python -m modelo.simulacion_turno_c [--replicas N]      Salida: tabla en pantalla y salidas/limpio/simulacion_turno_c.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import simpy

from modelo import capacity, config, fit_params

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "salidas" / "limpio" / "simulacion_turno_c.csv"

TURNO = "C"
_H = config.HORARIO_TURNO[TURNO]
HORAS_TURNO = float((_H[1] - _H[0]) % 24 or 24)          # 8 h: 00:00 a 08:00 (config.HORARIO_TURNO)
SEG_POR_HORA = 3600.0

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SUPUESTOS: todo valor numérico que no sale del modelo está aquí, con su fuente y su estado.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
_METODOS = {m.id: m.prioridad for m in config.config_wms().metodos_reabasto}
SUPUESTOS = {
    "horas_turno": {"valor": HORAS_TURNO, "fuente": "config.HORARIO_TURNO['C'] (00:00 a 08:00)", "estado": "verificado"},
    "eficiencia": {"valor": capacity.EFICIENCIA_SUPUESTA, "fuente": "capacity.EFICIENCIA_SUPUESTA (utilización objetivo del equipo, 2026-10-13); antes 0,848 medido en picking",
                   "estado": "decisión del equipo (el pedido decía 0,848: el modelo hoy usa este valor)"},
    "gruas": {"valor": max(config.GRUAS_POR_TURNO.values()), "fuente": "config.GRUAS_POR_TURNO: 6 grúas por turno, confirmado por el usuario el 2026-10-24 (la planilla de terreno contó 4 a las 15:30)",
              "estado": "confirmado por el usuario"},
    "replicas": {"valor": config.N_REPLICAS, "fuente": "config.N_REPLICAS (analisis_replicas.py, 2026-11-20; antes 10 por la regla de las ~2 min)", "estado": "decisión del usuario"},
    "semilla_base": {"valor": 20261014, "fuente": "semilla fija arbitraria; réplica r usa semilla_base + r (las mismas réplicas en todos los escenarios)", "estado": "convención"},
    "umbral_viabilidad": {"valor": 0.90, "fuente": "p_min: viable si ≥ 90% de las réplicas cumple (decisión del usuario, 2026-11-23, tras la grilla de apoyo con 95% y 90%: en alta no cambia ningún mínimo, en baja y media baja 1 gruero en 12 de 30 casos; antes 0,95 desde el 2026-11-20 y 0,90 antes)", "estado": "decisión del usuario"},
    "percentil_picking": {"valor": 0.99, "fuente": "hora en que se completa el 99% de las líneas de picking (misma definición que analisis_picking_noche)", "estado": "verificado"},
    "prioridad_picking": {"valor": 0, "fuente": "decisión del usuario (2026-11-17): 0 = picking y recogida de pallets completos (el picking del gruero); simpy: menor número = mayor prioridad", "estado": "decisión del usuario"},
    "prioridad_patio": {"valor": 1, "fuente": "decisión del usuario (2026-11-17): 1 = patio: carga de camión, recepción y carga para el CL", "estado": "decisión del usuario"},
    "prioridad_reabasto": {"valor": 2, "fuente": "decisión del usuario (2026-11-17): 2 = reabastos (pallet y cajas) y traslado de pallets mixtos; antes 1 y 2 según el método del WMS (REAPROPALLET, REAPROCAJA)", "estado": "decisión del usuario"},
    "prioridad_almacenamiento": {"valor": 3, "fuente": "decisión del usuario (2026-11-17): 3 = almacenamiento", "estado": "decisión del usuario"},
    "prioridad_resto": {"valor": 4, "fuente": "decisión del usuario (2026-11-17): 4 = mermas y retorno de fletero (lo último); también las otras actividades", "estado": "decisión del usuario"},
    "muestras_chequeo": {"valor": 200_000, "fuente": "tamaño de la muestra para comparar el tiempo muestreado con el tiempo estándar del modelo", "estado": "convención"},
}
PRIORIDAD = {"picking": SUPUESTOS["prioridad_picking"]["valor"], "recogida_pallets": SUPUESTOS["prioridad_picking"]["valor"],
             "carga_camion": SUPUESTOS["prioridad_patio"]["valor"], "recepcion_directo_stage": SUPUESTOS["prioridad_patio"]["valor"], "carga_cl": SUPUESTOS["prioridad_patio"]["valor"],
             "traslado_stage": SUPUESTOS["prioridad_patio"]["valor"],
             "reabasto_pallet": SUPUESTOS["prioridad_reabasto"]["valor"], "reabasto_caja": SUPUESTOS["prioridad_reabasto"]["valor"], "reabasto_caja_grua": SUPUESTOS["prioridad_reabasto"]["valor"],
             "almacenamiento": SUPUESTOS["prioridad_almacenamiento"]["valor"],
             "traslado_mixtos": SUPUESTOS["prioridad_reabasto"]["valor"],          # traslado de mixtos: misma prioridad que los reabastos (decisión del usuario, 2026-11-17)
             "merma": SUPUESTOS["prioridad_resto"]["valor"], "retorno_fletero": SUPUESTOS["prioridad_resto"]["valor"], "retorno_fletero_gruero": SUPUESTOS["prioridad_resto"]["valor"]}
PRIO_RESTO = SUPUESTOS["prioridad_resto"]["valor"]

# El reabasto de cajas del modelo son DOS procesos con el mismo volumen (la parte del gruero y la del operario): en la simulación es UNA entidad con dos pasos, el gruero primero
# (baja el pallet, con grúa) y después el operario (repone).
ENTIDAD_COMPUESTA = {"reabasto_caja": ["reabasto_caja_grua", "reabasto_caja"], "recepcion_directo_stage": ["recepcion_directo_stage", "traslado_stage"]}
AVISOS: list[str] = []
# Sensibilidad opcional (--media-del-modelo): reescala cada muestra para que la media de la distribución ajustada sea la MEDIA OBSERVADA del paso (la que usa el modelo de capacidad).
# Por defecto NO se hace (se muestrea la distribución tal cual). Existe porque las distribuciones ajustadas tienen media menor que la observada (picking −5%).
AJUSTAR_MEDIA = {"activo": False}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Tiempos de servicio: muestreo de las distribuciones ajustadas (misma regla que capacity.tiempo_estandar)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _muestrear_paso(p: fit_params.Parametro, n: int, rng: np.random.Generator) -> np.ndarray:
    """n tiempos (s) del paso `p`. Distribución ajustada si existe; si no, lognormal con la media y el CV OBSERVADOS (aviso)."""
    if p.distribucion in fit_params.CANDIDATAS and p.parametros:
        d = fit_params.CANDIDATAS[p.distribucion]
        # parametros.json guarda (forma, loc, escala) con los nombres s/a/loc/scale; en weibull_min la forma se llama «s» aunque scipy la llama «c»: se pasa por posición
        forma, loc, escala = (p.parametros[k] for k in list(p.parametros)[:3])
        x = np.asarray(d.rvs(forma, loc=loc, scale=escala, size=n, random_state=rng), dtype=float)
        if AJUSTAR_MEDIA["activo"]:
            x = x * (p.media_s / float(d.mean(forma, loc=loc, scale=escala)))
        return x
    AVISOS.append(f"{p.proceso}.{p.paso} [{p.turno}]: sin distribución ajustada ('{p.distribucion}', N={p.n}): lognormal con la media {p.media_s:.0f} s y el CV observado")
    if p.media_s <= 0 or p.desvio_s <= 0:
        return np.full(n, p.media_s)
    cv2 = (p.desvio_s / p.media_s) ** 2
    sig2 = math.log(1 + cv2)
    return rng.lognormal(math.log(p.media_s) - sig2 / 2, math.sqrt(sig2), size=n)


# TOPE DE LAS COLAS (2026-10-21). Las distribuciones ajustadas (lognormal, gamma…) extrapolan muy por encima de lo observado: una carga de camión salía hasta en 900 min y un reabasto de pallet
# en 280 min. Una sola de esas muestras al final del turno lo deja fuera de ventana, y la simulación rechazaba días por un evento que nunca se midió. Regla declarada: cada duración se
# recorta en el percentil 99 de su propia distribución y se reescala para que la MEDIA no cambie (el tiempo estándar del modelo se conserva; solo se quita la cola extrema).
TOPE_COLA = {"activo": True, "percentil": 0.99, "muestras": 50_000, "semilla": 20261021, "estado": "regla declarada, sin medición de la cola"}


def _muestreador(proceso: str, parametros) -> tuple[callable, str]:
    f, desc = _muestreador_sin_tope(proceso, parametros)
    if not TOPE_COLA["activo"]:
        return f, desc
    x = f(TOPE_COLA["muestras"], np.random.default_rng(TOPE_COLA["semilla"]))
    if len(x) == 0 or float(np.mean(x)) <= 0:
        return f, desc
    tope = float(np.quantile(x, TOPE_COLA["percentil"]))
    k = float(np.mean(x)) / float(np.mean(np.minimum(x, tope)))
    return (lambda n, rng, _f=f, _t=tope, _k=k: np.minimum(_f(n, rng), _t) * _k), f"{desc}; recortado en el p{100 * TOPE_COLA['percentil']:.0f} ({tope:.0f} s) con la media conservada (×{k:.3f})"


def _muestreador_sin_tope(proceso: str, parametros) -> tuple[callable, str]:
    """PICKING se simula por LISTAS: el tiempo de una lista es la suma de los tiempos de sus líneas (k líneas muestreadas de la distribución observada de líneas por lista,
    reescalada para que la media sea `capacity.LINEAS_POR_LISTA`). El resto de los procesos: ver `_muestreador_base`."""
    f, desc = _muestreador_base(proceso, parametros)
    if proceso != "picking":
        base = capacity._tiempo_estandar_base(capacity.ALIAS_TIEMPO.get(proceso, proceso), TURNO, parametros)[0]
        objetivo = capacity.tiempo_estandar(proceso, TURNO, parametros)[0]
        if base <= 0 and objetivo > 0:            # proceso sin distribución medida (p. ej. carga para el CL): lognormal con CV 0,5 alrededor del tiempo del modelo [supuesto]
            sg = float(np.sqrt(np.log(1 + 0.5 ** 2)))
            return (lambda n, rng, _m=objetivo, _s=sg: rng.lognormal(np.log(_m) - _s * _s / 2, _s, size=n)), f"sin distribución medida: lognormal de media {objetivo:.0f} s y CV 0,5 [supuesto]"
        if base > 0 and objetivo > 0 and abs(objetivo / base - 1) > 1e-6:          # recepción: de tiempo por camión a tiempo por pallet
            k = objetivo / base
            return (lambda n, rng, _f=f, _k=k: _f(n, rng) * _k), f"{desc}; reescalado ×{k:.3f} a {objetivo:.0f} s por unidad"
        return f, desc
    from modelo import listas_picking
    r = listas_picking.cargar_resumen()
    if not r:
        return f, desc + " (por línea: falta listas_picking_resumen.json)"
    vals = np.array(r["dist_lineas"]["valores"], dtype=int)
    p = np.array(r["dist_lineas"]["cuentas"], dtype=float)
    p /= p.sum()
    ajuste = capacity.LINEAS_POR_LISTA / float((vals * p).sum())
    base_lista = capacity._tiempo_estandar_base("picking", TURNO, parametros)[0] * capacity.LINEAS_POR_LISTA
    objetivo = capacity.tiempo_estandar("picking", TURNO, parametros)[0]
    if base_lista > 0 and objetivo > 0:
        ajuste *= objetivo / base_lista             # tiempo por lista fijado (Diana, 6,92 min)

    def g(n, rng):
        k = rng.choice(vals, size=n, p=p)
        tot = f(int(k.sum()), rng)
        ini = np.concatenate(([0], np.cumsum(k)[:-1]))
        return np.add.reduceat(tot, ini) * ajuste
    return g, f"lista = suma de sus líneas ({capacity.LINEAS_POR_LISTA:.2f} líneas por lista en promedio); línea: {desc}"


def _muestreador_base(proceso: str, parametros) -> tuple[callable, str]:
    """(f(n, rng) -> segundos SIN dividir por η, descripción de la regla). Reproduce capacity.tiempo_estandar: reabasto de pallet por origen, `ciclo_total` si existe,
    mezcla ponderada por N (picking) o suma de pasos (carga, recepción)."""
    proceso = capacity.ALIAS_TIEMPO.get(proceso, proceso)
    ps = [p for p in parametros if p.proceso == proceso and p.turno == TURNO]
    por_paso = {p.paso: p for p in ps}
    if proceso == "reabasto_pallet" and "ciclo_total_almac" in por_paso and "ciclo_total_almpic" in por_paso:
        w = capacity.ORIGEN_REABASTO_PALLET
        a, b = por_paso["ciclo_total_almac"], por_paso["ciclo_total_almpic"]

        def f(n, rng):
            es_a = rng.random(n) < w["ALMAC"]
            return np.where(es_a, _muestrear_paso(a, n, rng), _muestrear_paso(b, n, rng))
        return f, f"mezcla por origen (ALMAC {w['ALMAC']:.1%} / ALMPIC {w['ALMPIC']:.1%})"
    if "ciclo_total" in por_paso:
        p = por_paso["ciclo_total"]
        return (lambda n, rng: _muestrear_paso(p, n, rng)), f"ciclo_total ({p.distribucion}, N={p.n}, turno_real={p.turno_real}, {p.estado})"
    pasos = [p for p in ps if p.paso != "ciclo_total"]
    if capacity.AGREGACION.get(proceso, "suma") == "mezcla":
        pesos = np.array([p.n for p in pasos], dtype=float)
        pesos = pesos / pesos.sum() if pesos.sum() > 0 else np.full(len(pasos), 1 / len(pasos))

        def f(n, rng):
            k = rng.choice(len(pasos), size=n, p=pesos)
            out = np.empty(n)
            for i, p in enumerate(pasos):
                m = k == i
                if m.any():
                    out[m] = _muestrear_paso(p, int(m.sum()), rng)
            return out
        return f, "mezcla por tipo de línea ponderada por N: " + ", ".join(f"{p.paso} {w:.0%}" for p, w in zip(pasos, pesos))
    return (lambda n, rng: sum(_muestrear_paso(p, n, rng) for p in pasos)), "suma de pasos: " + ", ".join(p.paso for p in pasos)


def _eta(proceso: str) -> float:
    if proceso == "picking":
        return capacity.Ajustes().eficiencia_picking_efectiva()
    return capacity.EFICIENCIA_MEDIDA.get(proceso, capacity.EFICIENCIA_SUPUESTA)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Escenario: volúmenes, pasos por entidad y dotación del modelo
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def preparar(demanda: float, parametros, ajustes=None) -> dict:
    """Volúmenes, pasos y dotación del modelo para una demanda. Devuelve {'entidades': {nombre: {...}}, 'dotacion': {...}, 'tiempos': {...}}."""
    reqs = {r.proceso: r for r in capacity._resolver_turno(TURNO, demanda, None, parametros, None, ajustes)[0]}
    entidades = {}
    consumidos = set()
    for nombre, partes in ENTIDAD_COMPUESTA.items():
        if all(p in reqs for p in partes):
            entidades[nombre] = {"volumen": reqs[nombre].volumen, "pasos": partes}
            consumidos |= set(partes)
    for proceso, r in reqs.items():
        if proceso not in consumidos:
            entidades[proceso] = {"volumen": r.volumen, "pasos": [proceso]}
    muestreadores = {}
    for ent in entidades.values():
        for paso in ent["pasos"]:
            if paso not in muestreadores:
                muestreadores[paso] = _muestreador(paso, parametros)
    dot = {d.recurso: d for d in capacity.dotacion_por_turno(TURNO, demanda, None, parametros, ajustes=ajustes)}
    # ajustes del usuario (dashboard): tiempo editado => se reescala la distribución muestreada; utilización editada => η por paso
    escala, eta = {}, {}
    if ajustes is not None:
        for paso in muestreadores:
            t_ov = ajustes.tiempo(paso)
            if t_ov is not None:
                base = capacity.tiempo_estandar(paso, TURNO, parametros)[0]
                if base > 0:
                    escala[paso] = t_ov / base
            eta[paso] = ajustes.eficiencia_picking_efectiva() if paso == "picking" else ajustes.eficiencia_resto
    if ajustes is not None and getattr(ajustes, "recepcion_por_camion", False) and "recepcion_directo_stage" in muestreadores and ajustes.tiempo("recepcion_directo_stage") is None:
        base_r = capacity.tiempo_estandar("recepcion_directo_stage", TURNO, parametros)[0]      # recepción por tipo de camión: el tiempo por pallet del turno sale del requerimiento
        if base_r > 0 and "recepcion_directo_stage" in reqs:
            escala["recepcion_directo_stage"] = reqs["recepcion_directo_stage"].tiempo_estandar_s / base_r
    extras = {}
    if ajustes is not None:
        for rol in ("Operario", "Gruero"):
            tr, ot = ajustes.traslado_min(rol, TURNO), ajustes.otras_horas(rol, TURNO)
            if tr > 0 or ot > 0:
                extras[rol] = {"traslado_min": tr, "otras_h": ot, "eta": ajustes.eficiencia_resto}
    return {"demanda": demanda, "extras": extras, "escala": escala, "eta": eta, "entidades": entidades, "muestreadores": muestreadores, "reqs": reqs,
            "modelo": {"Operario": dot["Operario"].necesarios, "Gruero": dot["Gruero"].necesarios}}


def dotacion_redondeada(prep: dict) -> dict:
    """Operarios y grueros que entrega el modelo, redondeados HACIA ARRIBA (un resto mínimo de punto flotante no cuenta como persona extra)."""
    return {rol: max(1, math.ceil(n - 1e-9)) for rol, n in prep["modelo"].items()}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Una réplica
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _replica(prep: dict, n_op: int, n_gr: int, n_grua: int, semilla: int) -> dict:
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    rec = {"operario": simpy.PriorityResource(env, capacity=n_op), "gruero": simpy.PriorityResource(env, capacity=n_gr), "grua": simpy.PriorityResource(env, capacity=n_grua)}
    T = HORAS_TURNO
    espera = defaultdict(float)      # horas de espera acumuladas por recurso
    pedidos = defaultdict(int)
    ocupado = defaultdict(float)     # horas ocupadas dentro de [0, T]
    fin: dict[str, list[float]] = defaultdict(list)

    def usar(nombre, prio, dur_h):
        t0 = env.now
        with rec[nombre].request(priority=prio) as rq:
            yield rq
            espera[nombre] += env.now - t0
            pedidos[nombre] += 1
            ini = env.now
            yield env.timeout(dur_h)
        ocupado[nombre] += max(0.0, min(env.now, T) - min(ini, T))

    def con_gruero_y_grua(prio, dur_h):
        """Gruero primero y después grúa (el mismo orden en todas las tareas: no hay bloqueo mutuo). Ambos se ocupan durante todo el tiempo de la tarea."""
        t0 = env.now
        with rec["gruero"].request(priority=prio) as rg:
            yield rg
            espera["gruero"] += env.now - t0
            pedidos["gruero"] += 1
            t1 = env.now
            with rec["grua"].request(priority=prio) as ru:
                yield ru
                espera["grua"] += env.now - t1
                pedidos["grua"] += 1
                ini = env.now
                yield env.timeout(dur_h)
        d = max(0.0, min(env.now, T) - min(ini, T))
        ocupado["gruero"] += d
        ocupado["grua"] += d

    def entidad(nombre, pasos_t):
        for paso, prio, dur_h in pasos_t:
            rol = capacity.ROL_POR_PROCESO[paso]
            if rol == "Gruero" and "Grúas" in capacity.USO_DE_EQUIPO.get(paso, {}):
                yield from con_gruero_y_grua(prio, dur_h)
            elif rol == "Gruero":
                yield from usar("gruero", prio, dur_h)
            else:
                yield from usar("operario", prio, dur_h)
        fin[nombre].append(env.now)

    total = {}
    for nombre, ent in prep["entidades"].items():
        v = ent["volumen"]
        n = int(math.floor(v)) + int(rng.random() < (v - math.floor(v)))
        total[nombre] = n
        if n == 0:
            continue
        tiempos = {}
        for paso in ent["pasos"]:
            f, _ = prep["muestreadores"][paso]
            tiempos[paso] = f(n, rng) * prep.get("escala", {}).get(paso, 1.0) / prep.get("eta", {}).get(paso, _eta(paso)) / SEG_POR_HORA
        for i in range(n):
            pasos_t = [(paso, PRIORIDAD.get(paso, PRIO_RESTO), float(tiempos[paso][i])) for paso in ent["pasos"]]
            env.process(entidad(nombre, pasos_t))
    env.run()
    q = SUPUESTOS["percentil_picking"]["valor"]
    pk = np.sort(np.array(fin.get("picking", [0.0])))
    t99 = float(pk[max(0, math.ceil(q * len(pk)) - 1)]) if len(pk) else 0.0
    todos = [t for v in fin.values() for t in v]
    ts_todas = np.sort(np.array(todos)) if todos else np.array([0.0])
    t99_todas = float(ts_todas[max(0, math.ceil(q * len(ts_todas)) - 1)])      # hora en que se completa el 99% de TODAS las tareas (útil donde no hay picking)
    backlog = {nombre: int(sum(1 for t in fin.get(nombre, []) if t > T)) for nombre in total}
    return {"t99_picking": t99, "t99_todas": t99_todas, "t_fin": max(todos) if todos else 0.0, "backlog": backlog, "n_tareas": total,
            "espera_h": {k: (espera[k] / pedidos[k] if pedidos[k] else 0.0) for k in rec}, "util": {k: ocupado[k] / (rec[k].capacity * T) for k in rec}}


def simular(prep: dict, n_op: int, n_gr: int, n_grua: int, replicas: int) -> dict:
    base = SUPUESTOS["semilla_base"]["valor"]
    res = [_replica(prep, n_op, n_gr, n_grua, base + r) for r in range(replicas)]
    T = HORAS_TURNO
    t99 = np.array([r["t99_picking"] for r in res])
    tf = np.array([r["t_fin"] for r in res])
    ok = [r["t_fin"] <= T and sum(r["backlog"].values()) == 0 for r in res]
    procs = sorted({k for r in res for k in r["backlog"]})
    return {"operarios": n_op, "grueros": n_gr, "gruas": n_grua, "replicas": replicas,
            "t99_media": float(t99.mean()), "t99_p90": float(np.percentile(t99, 90)), "t99todas_media": float(np.mean([r["t99_todas"] for r in res])),
            "t99todas_p90": float(np.percentile([r["t99_todas"] for r in res], 90)), "tfin_media": float(tf.mean()), "tfin_p90": float(np.percentile(tf, 90)),
            "backlog": {p: float(np.mean([r["backlog"].get(p, 0) for r in res])) for p in procs},
            "pct_ok": float(np.mean(ok)), "viable": float(np.mean(ok)) >= SUPUESTOS["umbral_viabilidad"]["valor"],
            "espera_min": {k: float(np.mean([r["espera_h"][k] for r in res])) * 60 for k in ("operario", "gruero", "grua")},
            "util": {k: float(np.mean([r["util"][k] for r in res])) for k in ("operario", "gruero", "grua")}}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def hhmm(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def chequeo_insumos(prep: dict) -> list[dict]:
    """Compara el tiempo medio MUESTREADO (ya dividido por η) con el tiempo estándar del modelo / η, para ver que la simulación usa los mismos tiempos."""
    rng = np.random.default_rng(SUPUESTOS["semilla_base"]["valor"])
    n = SUPUESTOS["muestras_chequeo"]["valor"]
    out = []
    for paso, (f, regla) in prep["muestreadores"].items():
        mod = capacity.tiempo_estandar(paso, TURNO, fit_params.cargar())[0]
        sim = float(f(n, rng).mean())
        out.append({"proceso": paso, "modelo_s": mod, "muestreado_s": sim, "dif_pct": 100 * (sim / mod - 1) if mod else float("nan"), "regla": regla})
    return out


def fila_csv(nombre: str, demanda: float, r: dict, procesos: list[str]) -> dict:
    fila = {"escenario": nombre, "demanda_cajas": round(demanda), "operarios": r["operarios"], "grueros": r["grueros"], "gruas": r["gruas"], "replicas": r["replicas"],
            "hora_99pct_picking_media": hhmm(r["t99_media"]), "hora_99pct_picking_p90": hhmm(r["t99_p90"]), "hora_fin_todo_media": hhmm(r["tfin_media"]),
            "hora_fin_todo_p90": hhmm(r["tfin_p90"])}
    for p in procesos:
        fila[f"backlog_08h_{p}"] = round(r["backlog"].get(p, 0.0), 2)
    fila.update({"pct_replicas_terminan_antes_de_08h": round(100 * r["pct_ok"], 1), "viable": "sí" if r["viable"] else "NO",
                 "espera_media_min_operario": round(r["espera_min"]["operario"], 1), "espera_media_min_gruero": round(r["espera_min"]["gruero"], 1),
                 "espera_media_min_grua": round(r["espera_min"]["grua"], 1), "utilizacion_operarios": round(r["util"]["operario"], 3),
                 "utilizacion_grueros": round(r["util"]["gruero"], 3), "utilizacion_gruas": round(r["util"]["grua"], 3)})
    return fila


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replicas", type=int, default=SUPUESTOS["replicas"]["valor"])
    ap.add_argument("--media-del-modelo", action="store_true", help="sensibilidad: reescala cada muestra a la media observada del paso (guarda otro CSV)")
    a = ap.parse_args(argv)
    AJUSTAR_MEDIA["activo"] = a.media_del_modelo
    salida = SALIDA.with_name("simulacion_turno_c_media_del_modelo.csv") if a.media_del_modelo else SALIDA
    t0 = time.time()
    params = fit_params.cargar()
    escenarios = capacity.escenarios_estimador()
    if not escenarios:
        print("Falta demanda_diaria.csv (correr modelo.sales_loader).")
        return 1
    n_grua = SUPUESTOS["gruas"]["valor"]
    print(f"SIMULACIÓN DEL TURNO C · {a.replicas} réplicas por escenario · semilla base {SUPUESTOS['semilla_base']['valor']} · η = {SUPUESTOS['eficiencia']['valor']:g} · grúas = {n_grua} "
          f"[{SUPUESTOS['gruas']['estado']}]")
    print("Supuestos:")
    for k, v in SUPUESTOS.items():
        print(f"  - {k}: {v['valor']}  ({v['fuente']}; {v['estado']})")
    filas, resultados = [], {}
    prep0 = None
    for nombre, dem in escenarios.items():
        prep = preparar(dem, params)
        prep0 = prep0 or prep
        dot = dotacion_redondeada(prep)
        resultados[nombre] = (dem, prep, simular(prep, dot["Operario"], dot["Gruero"], n_grua, a.replicas))
    procesos = sorted({p for _, _, r in resultados.values() for p in r["backlog"]})
    # chequeo de cordura: p50 con la dotación programada real de la semana actual del roster
    dem50 = escenarios["p50"]
    prep50 = resultados["p50"][1]
    op_real, gr_real = config.personas_turno(TURNO, "Operario"), config.personas_turno(TURNO, "Gruero")
    real = simular(prep50, op_real, gr_real, n_grua, a.replicas)
    resultados["p50 con dotación programada real"] = (dem50, prep50, real)

    print("\nCHEQUEO DE INSUMOS (p50): tiempo medio muestreado vs tiempo estándar del modelo (antes de dividir por η)")
    for c in chequeo_insumos(resultados["p50"][1]):
        print(f"  {c['proceso']:<22s} modelo {c['modelo_s']:8.1f} s · muestreado {c['muestreado_s']:8.1f} s ({c['dif_pct']:+.1f}%) · {c['regla']}")
    print("\nVOLÚMENES DEL TURNO C (p50, tareas): " + ", ".join(f"{k} {v['volumen']:.2f}" for k, v in resultados["p50"][1]["entidades"].items()))

    print("\nRESULTADOS (horas desde las 00:00; backlog = tareas sin terminar a las 08:00, media por réplica)")
    cab = f"{'escenario':<34s}{'cajas':>7s}{'op':>4s}{'gr':>4s}{'t99 picking':>12s}{'p90':>7s}{'fin todo':>10s}{'p90':>7s}{'%<08h':>7s}{'viable':>8s}{'util op':>9s}{'util gr':>9s}{'util grúa':>10s}"
    print(cab)
    for nombre, (dem, prep, r) in resultados.items():
        print(f"{nombre:<34s}{dem:>7.0f}{r['operarios']:>4d}{r['grueros']:>4d}{hhmm(r['t99_media']):>12s}{hhmm(r['t99_p90']):>7s}{hhmm(r['tfin_media']):>10s}{hhmm(r['tfin_p90']):>7s}"
              f"{100 * r['pct_ok']:>6.0f}%{('sí' if r['viable'] else 'NO'):>8s}{100 * r['util']['operario']:>8.0f}%{100 * r['util']['gruero']:>8.0f}%{100 * r['util']['grua']:>9.0f}%")
        filas.append(fila_csv(nombre, dem, r, procesos))
    print("\nBACKLOG a las 08:00 (tareas pendientes por proceso, media por réplica) y ESPERA media en cola (min por pedido)")
    for nombre, (dem, prep, r) in resultados.items():
        bl = ", ".join(f"{p} {r['backlog'].get(p, 0):.1f}" for p in procesos if r["backlog"].get(p, 0) > 0) or "sin backlog"
        print(f"  {nombre:<34s} backlog: {bl} | espera: operario {r['espera_min']['operario']:.0f} · gruero {r['espera_min']['gruero']:.0f} · grúa {r['espera_min']['grua']:.0f} min")

    pred = capacity.hora_termino_picking(dem50)
    print("\nCHEQUEO DE CORDURA: p50 con la dotación programada real (" f"{op_real} operarios, {gr_real} grueros)")
    if pred:
        print(f"  simulación: 99% del picking a las {hhmm(real['t99_media'])} (p90 {hhmm(real['t99_p90'])}) · análisis de hora de término: {hhmm(pred['hora'])} "
              f"(intervalo de predicción 95%: {hhmm(pred['hora_min'])} a {hhmm(pred['hora_max'])}) · diferencia {60 * (real['t99_media'] - pred['hora']):+.0f} min "
              f"{'[dentro del intervalo]' if pred['hora_min'] <= real['t99_media'] <= pred['hora_max'] else '[FUERA del intervalo]'}")
    else:
        print("  no se pudo leer la predicción (falta picking_referencias_estimador.json)")
    if AVISOS:
        print("\nAVISOS:")
        for m in sorted(set(AVISOS)):
            print("  -", m)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    campos = list(filas[0].keys())
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado: {salida}  ·  tiempo de corrida: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
