"""Simulación SIMPLE del DÍA COMPLETO (24 h: turno C 00-08, A 08-16, B 16-24) con la dotación que calcula el modelo de capacidad para cada turno.

QUÉ AGREGA RESPECTO DE LOS TRES SCRIPTS POR TURNO (`simulacion_turno_c/a/b.py`, de los que reutiliza toda la maquinaria: muestreo de distribuciones, tiempos ÷ η, prioridades, semillas)
  - UN SOLO RELOJ de 24 h y UNA SOLA bolsa de grúas (4) compartida por los tres turnos: es lo único que los acopla (la contención entre turnos que pide el Charter).
  - Cada turno tiene su propia dotación (operarios y grueros del modelo, redondeados hacia arriba), y todo el trabajo del turno está disponible desde su hora de inicio (00:00, 08:00, 16:00).
  - El trabajo que no se alcanza a terminar en la ventana del turno NO desaparece: la gente de ese turno lo sigue haciendo después de su hora de salida. Eso es SOBRETIEMPO y es la medida
    del derrame: hora de término de cada turno, horas-persona de sobretiempo y tareas pendientes al cierre de cada ventana. Lo que se derrama compite por las grúas con el turno siguiente.
  - Lo no iniciado al cierre de la ventana pasa al turno siguiente (versión única, ver SIGUIENTE y TRASPASO_TOL), salvo el trabajo de los operarios del C (picking), que lo terminan los mismos pickers (SIN_TRASPASO); la tarea ya iniciada termina. Recepción y carga A llegan según el log del WMS (`llegadas_camiones`), no hay refuerzos ni cambios de dotación durante el día. [hipótesis, ver límites]

CRITERIO DE VIABILIDAD (propuesto, a confirmar con el tutor): viable si ≥ 90% de las réplicas termina el trabajo de CADA turno dentro de su ventana, sin sobretiempo ni pendientes.
LÍMITES [declarados]: los de los scripts por turno (tareas sin interrupción, sin pausas, 8 h completas contra las 7 h × 0,85 del modelo, volúmenes fraccionarios al azar, tiempos de B copiados de A, el reabasto de
  cajas de A depende de UNA hoja de planificadores, picking sin grúa). Además: no hay unión de colas entre turnos ni stock de picking (el reabasto del día no cambia lo que hay en las posiciones).

USO:  .venv/bin/python -m modelo.simulacion_dia [--replicas N] [--media-del-modelo]      Salida: tabla en pantalla y salidas/limpio/simulacion_dia.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict

import numpy as np
import simpy

from modelo import capacity, config, fit_params
from modelo import simulacion_turno_c as sc

TURNOS = ("C", "A", "B")                                          # orden cronológico del día
INICIO = {t: float(config.HORARIO_TURNO[t][0]) for t in TURNOS}   # 0, 8, 16
FIN = {t: INICIO[t] + sc.HORAS_TURNO for t in TURNOS}             # 8, 16, 24
SALIDA = sc.SALIDA.with_name("simulacion_dia.csv")
REPLICAS = {"valor": config.N_REPLICAS, "fuente": "config.N_REPLICAS (analisis_replicas.py, 2026-11-20; antes 10)", "estado": "decisión del usuario"}


# TRASLADOS Y OTRAS ACTIVIDADES en la simulación. La distribución es INVENTADA por pedido del usuario (2026-10-15): no hay medición. Se reparten como interrupciones a lo largo de la ventana del turno:
#   traslados  : trozos lognormales de media 5 min (CV 0,5), prioridad máxima (0); el total por turno = minutos por persona × personas del turno; la duración es tiempo real (no se divide por η).
#   otras act. : trozos lognormales de media 20 min (CV 0,8), prioridad de las tareas de menor prioridad; el total = horas de trabajo editadas; la duración se divide por η (igual que en el modelo analítico).
# Los instantes de llegada son uniformes dentro de la ventana del turno.
EXTRAS = {"traslado_trozo_min": 5.0, "traslado_cv": 0.5, "otras_trozo_min": 20.0, "otras_cv": 0.8, "estado": "inventado por pedido del usuario; sin medición"}


# QUÉ ROLES DECIDEN SI UN TURNO TERMINA A TIEMPO (decisión del usuario, 2026-10-21): en A y B no se rechaza un día por los operarios (su dotación real está subestimada en las
# fuentes y sus tareas —reabasto de cajas, retorno de fletero, merma— se pueden mover dentro del día); en C cuentan todos, porque el picking nocturno es el problema central.
# El sobretiempo y los pendientes de los operarios de A y B se siguen informando, solo no rechazan el día.
CRITERIO_ROLES = {"C": ("operario", "gruero"), "A": ("gruero",), "B": ("gruero",)}
# TRASPASO AL TURNO SIGUIENTE (2026-11-10; opción DESACTIVADA por defecto: traspaso=None reproduce la simulación de siempre). Con traspaso = tol (fracción), la tarea que al cierre de la ventana
# todavía no empezó pasa a la cola del turno siguiente (C→A, A→B) y espera a su inicio; la que ya empezó termina. B no traspasa (el siguiente sería el C del otro día). El turno cumple si lo que
# suelta, en los roles que lo deciden, no pasa de `tol` × las horas de trabajo de esos roles en el turno que lo recibe. `tol` es una propuesta a confirmar con el tutor.
SIGUIENTE = {"C": "A", "A": "B", "B": None}
# EXCEPCIÓN VALIDADA CON EL LOG (2026-11-11): el trabajo de los operarios del turno C (picking) NO pasa al A: lo terminan los mismos pickers de la noche en sobretiempo (el 92% de los picks
# de la mañana del 16-sep los hizo gente de esa noche). Traspasar el picking al A empeoró la validación contra las 22 noches de septiembre (sesgo +8 → +40 min; el 15-sep 10:25 → 18:30
# contra 11:04 real). Para este rol el turno cumple si termina dentro de su ventana.
SIN_TRASPASO = {("C", "operario")}
NBIN = 32                  # horas del reloj de la simulación que se registran para el uso por hora (el B puede terminar pasada la hora 24 con el sobretiempo)
_ROL_DOT = {"operario": "Operario", "gruero": "Gruero"}


def _rol_paso(paso: str) -> str:
    if paso.startswith(("traslados_", "otras_")):
        return paso.rsplit("_", 1)[1]
    return "gruero" if capacity.ROL_POR_PROCESO.get(paso) == "Gruero" else "operario"


def _trozos(rng, total_h: float, media_min: float, cv: float) -> np.ndarray:
    """Duraciones (horas) de trozos lognormales que suman ≈ total_h (el número de trozos es aleatorio por redondeo estocástico)."""
    if total_h <= 0:
        return np.zeros(0)
    k_f = total_h * 60.0 / media_min
    k = int(math.floor(k_f)) + int(rng.random() < (k_f - math.floor(k_f)))
    if k == 0:
        return np.zeros(0)
    media_h = total_h / k
    sg = math.sqrt(math.log(1 + cv * cv))
    return rng.lognormal(math.log(media_h) - sg * sg / 2, sg, size=k)


def reloj(h: float) -> str:
    return sc.hhmm(h)


# APOYO DE PLANIFICADORES (2026-10-25, opción DESACTIVADA por defecto: apoyo=None y factor_paso=None reproducen la simulación de siempre, con las mismas semillas y los mismos números aleatorios).
#   apoyo = {turno: {"n": personas, "frac": fracción del turno disponible, "vel": velocidad relativa a un gruero, "modo": "inicio" | "fin"}}.
#   Las personas de apoyo comparten la cola de tareas elegibles con los grueros (la toma el primero que quede libre) y las mismas grúas. Su disponibilidad es una ventana: los primeros
#   frac·T horas del turno ("inicio") o los últimos ("fin"); fuera de ella un proceso las mantiene ocupadas. No consumen números aleatorios.
#   factor_paso = {paso: multiplicador de la duración} (p. ej. recepción TO-BE); determinista, tampoco consume números aleatorios.
PASOS_APOYO = ("recepcion_directo_stage", "almacenamiento", "retorno_fletero_gruero")      # por defecto del apoyo (informe v1); cada apoyo puede traer su propia lista en "pasos"
# PLANIFICADOR-GRUERO (2026-10-26, ESCENARIO 2 de RESULTADO_ROLES_V2; opción DESACTIVADA por defecto: pg=None). pg = {turno: {"n": K, "f": fracción de ayuda, "vel": velocidad relativa}}.
#   Los K planificadores que pasan a gruero atienden PRIMERO la recepción entera y el retorno de fletero del gruero (NUCLEO_PG, con prioridad sobre los demás pedidos del pool). Con el tiempo que les sobra
#   ayudan en los demás procesos de gruero hasta un PRESUPUESTO de K·f·T horas por turno (f = 0: solo núcleo). Comparten las grúas con todos y no consumen números aleatorios.
NUCLEO_PG = ("recepcion_directo_stage", "retorno_fletero_gruero")


def _replica_dia(preps: dict, dot: dict, n_grua: int, semilla: int, apoyo: dict | None = None, factor_paso: dict | None = None, pg: dict | None = None, llegadas=None, traspaso: float | None = None, traza: list | None = None,
                roles_paso: dict | None = None, pools: dict | None = None) -> dict:
    """`traza` (opcional, no cambia resultados): si es una lista, se le agrega un registro por evento (reloj, evento, entidad, estado y lista de eventos futuros).
    `roles_paso` (opcional, ROLES EXTRA; None reproduce la simulación de siempre): {paso: (rol, usa_grua)} asigna un paso a un rol cualquiera (p. ej. "chequeador", "polivalente") con su propia
    dotación en dot[turno][Rol]. Los roles extra deciden el turno como los grueros (traspasan C→A y A→B con la misma tolerancia).
    `pools` (opcional, PERSONAS DE APOYO; None reproduce la simulación de siempre): {"personas": {turno: {nombre: fracción disponible}},
    "pasos": {turno: {paso: [(rol o nombre de persona, usa_grua, factor de duración), ...]}}}. La tarea la toma el primero que quede libre entre los elegibles (a igual instante, el
    primero de la lista) y dura su muestra × el factor del que la toma. Cada persona de apoyo es un recurso de capacidad 1 y está disponible solo una fracción de cada hora de su turno
    (el resto, y después de su hora de salida, la ocupa otra labor). Lo que no alcanza a empezar se traspasa como trabajo del GRUERO (el rol que se dimensiona)."""
    rng = np.random.default_rng(semilla)
    llegadas_r = llegadas.replica(np.random.default_rng([semilla, 99])) if llegadas is not None else None      # generador PROPIO: no consume los números aleatorios de la simulación
    env = simpy.Environment()
    grua = simpy.PriorityResource(env, capacity=n_grua)
    rec = {t: {"operario": simpy.PriorityResource(env, capacity=dot[t]["Operario"]), "gruero": simpy.PriorityResource(env, capacity=dot[t]["Gruero"])} for t in TURNOS}
    roles_paso = roles_paso or {}
    roles_x = tuple(dict.fromkeys(r_ for r_, _ in roles_paso.values() if r_ not in ("operario", "gruero")))      # roles extra, en orden de aparición
    for t in TURNOS:
        for r_ in roles_x:
            rec[t][r_] = simpy.PriorityResource(env, capacity=max(1, int(dot[t].get(r_.capitalize(), 0))))
    roles_t = ("operario", "gruero") + roles_x

    pools = pools or {}
    pool_pasos = pools.get("pasos", {})
    pool_rec = {}
    for t, per in pools.get("personas", {}).items():
        for nom in per:
            rec[t][nom] = pool_rec[(t, nom)] = simpy.PriorityResource(env, capacity=1)

    def rp(paso):                                   # rol que ejecuta el paso
        if any(paso in v for v in pool_pasos.values()):
            return "gruero"
        return roles_paso[paso][0] if paso in roles_paso else _rol_paso(paso)

    crit = {t: CRITERIO_ROLES[t] + roles_x for t in TURNOS}
    espera = defaultdict(float)
    pedidos = defaultdict(int)
    dentro = defaultdict(float)      # horas ocupadas dentro de la ventana del turno, por (turno, rol)
    extra = defaultdict(float)       # horas ocupadas DESPUÉS de la hora de salida (sobretiempo), por turno
    gru_ocupada = [0.0]
    apoyo_h = defaultdict(float)     # horas de trabajo de las personas de apoyo, dentro y fuera de la ventana
    horas_x = defaultdict(float)     # horas de las personas extra por (turno, etiqueta, paso): etiqueta "apoyo", "pg_nucleo" o "pg_ayuda"
    fin: dict[tuple, list[float]] = defaultdict(list)
    fin_carry: dict[tuple, list[float]] = defaultdict(list)    # entidades que pasaron (algún paso) al turno siguiente: se informan aparte
    fin_paso: dict[tuple, list[float]] = defaultdict(list)     # término de cada paso, para el criterio por rol (CRITERIO_ROLES)

    fel: dict = {}                                 # lista de eventos futuros, solo para la traza: clave -> (tiempo, evento, entidad)

    def _fel(clave, tiempo=None, evento=None, eid=None):
        if traza is None:
            return
        if tiempo is None:
            fel.pop(clave, None)
        else:
            fel[clave] = (float(tiempo), evento, eid)

    def _reg(evento, eid, t, rol=None):
        if traza is None:
            return
        estado = {f"Q_{ro}_{tt}": len(rec[tt][ro].queue) for tt in TURNOS for ro in ("operario", "gruero")}
        estado.update({f"S_{ro}_{tt}": rec[tt][ro].capacity - rec[tt][ro].count for tt in TURNOS for ro in ("operario", "gruero")})
        estado.update({"Q_grua": len(grua.queue), "G": grua.capacity - grua.count, "P_operario": sum(carry[(tt, "operario")] for tt in TURNOS),
                       "P_gruero": sum(carry[(tt, "gruero")] for tt in TURNOS)})
        traza.append({"reloj": float(env.now), "evento": evento, "entidad": eid, "turno": t, "rol": rol, **estado,
                      "fel": sorted(fel.values(), key=lambda x: (x[0], str(x[2])))})

    uso_h = defaultdict(lambda: np.zeros(NBIN))   # horas-persona ocupadas en cada hora del reloj de la simulación, por (turno que ejecuta, rol): operario y gruero

    def reparte(t, rol, ini, fin_):
        a, b = INICIO[t], FIN[t]
        d_in = max(0.0, min(fin_, b) - max(ini, a))
        dentro[(t, rol)] += d_in
        extra[t] += max(0.0, (fin_ - ini) - d_in)
        if rol in ("operario", "gruero"):
            v = uso_h[(t, rol)]
            for k in range(int(ini), min(int(fin_), NBIN - 1) + 1):
                v[k] += max(0.0, min(fin_, k + 1.0) - max(ini, float(k)))

    carry = defaultdict(float)       # TRASPASO: horas-persona de tareas que NO alcanzaron a empezar dentro de la ventana y pasan al turno siguiente, por (turno que las suelta, rol)
    carry_n = defaultdict(int)

    def esperar_o_traspasar(t, cola, rq, rol, dur_h, ctx, eid=None):
        """Espera la persona en el turno `t`. Con traspaso, si al cierre de la ventana sigue en cola se cancela, se anota en `carry` y se pide en el turno siguiente (desde su inicio).
        Devuelve el turno que finalmente la atiende. B no traspasa (el siguiente sería el C del día próximo, fuera del día simulado): sigue como sobretiempo."""
        if ctx is None or traspaso is None or SIGUIENTE[t] is None or (t, rol) in SIN_TRASPASO:
            yield rq
            return t
        _fel(('cierre', eid), max(env.now, FIN[t]), 'Cierre', eid)
        got = yield rq | env.timeout(max(0.0, FIN[t] - env.now))
        _fel(('cierre', eid))
        if rq in got:
            return t
        if rq.triggered:
            cola.release(rq)
        else:
            rq.cancel()
        carry[(t, rol)] += dur_h
        carry_n[(t, rol)] += 1
        ctx["carried"] = True
        t2 = SIGUIENTE[t]
        ctx["t"] = t2
        _fel(('arribo', eid), max(env.now, INICIO[t2]), 'Arribo', eid)
        _reg("Cierre: traspaso", eid, t, rol)
        if INICIO[t2] > env.now:
            yield env.timeout(INICIO[t2] - env.now)
        _fel(('arribo', eid))
        _reg("Arribo (traspasada)", eid, t2, rol)
        return t2

    def usar(t, rol, prio, dur_h, ctx=None, eid=None):
        t0 = env.now
        rq = rec[t][rol].request(priority=prio)
        t = yield from esperar_o_traspasar(t, rec[t][rol], rq, rol, dur_h, ctx, eid)
        if ctx is not None and ctx.get("carried") and rq not in rec[t][rol].users:     # traspasada: pedir en el turno nuevo
            rq = rec[t][rol].request(priority=prio)
            yield rq
        espera[rol] += env.now - t0
        pedidos[rol] += 1
        ini = env.now
        _fel(('termino', eid), env.now + dur_h, 'Término', eid)
        _reg("Comienzo servicio", eid, t, rol)
        yield env.timeout(dur_h)
        rec[t][rol].release(rq)
        reparte(t, rol, ini, env.now)
        _fel(('termino', eid))
        _reg("Término servicio", eid, t, rol)

    def gruero_y_grua(t, prio, dur_h, ctx=None, eid=None, rol="gruero"):
        t0 = env.now
        rg = rec[t][rol].request(priority=prio)
        t = yield from esperar_o_traspasar(t, rec[t][rol], rg, rol, dur_h, ctx, eid)
        if ctx is not None and ctx.get("carried") and rg not in rec[t][rol].users:
            rg = rec[t][rol].request(priority=prio)
            yield rg
        espera[rol] += env.now - t0
        pedidos[rol] += 1
        t1 = env.now
        with grua.request(priority=prio) as ru:
            _reg("Toma de gruero", eid, t, rol)              # ya pidió la grúa: si no hay libre, figura en su cola
            yield ru
            espera["grua"] += env.now - t1
            pedidos["grua"] += 1
            ini = env.now
            _fel(('termino', eid), env.now + dur_h, 'Término', eid)
            _reg("Comienzo servicio (grúa)", eid, t, rol)
            yield env.timeout(dur_h)
        rec[t][rol].release(rg)
        reparte(t, rol, ini, env.now)
        gru_ocupada[0] += env.now - ini
        _fel(('termino', eid))
        _reg("Término servicio", eid, t, rol)

    apoyo_rec = {}
    for t_, a_ in (apoyo or {}).items():
        if a_ and a_.get("n", 0) > 0:
            apoyo_rec[t_] = simpy.PriorityResource(env, capacity=int(a_["n"]))

    def fuera_de_ventana(t_, a_):                          # mantiene ocupadas a las personas de apoyo cuando no están disponibles
        T = sc.HORAS_TURNO
        tramos = [(INICIO[t_] + a_["frac"] * T, 48.0)] if a_.get("modo", "inicio") == "inicio" else [(INICIO[t_], (1 - a_["frac"]) * T), (INICIO[t_] + T, 48.0)]
        for desde, dur in tramos:
            if dur > 0:
                yield env.timeout(desde - env.now) if desde > env.now else env.timeout(0)
                for _ in range(int(a_["n"])):
                    env.process(ocupar(t_, dur))
                yield env.timeout(dur)

    def ocupar(t_, dur):
        with apoyo_rec[t_].request(priority=-10) as rq:
            yield rq
            yield env.timeout(dur)

    pg_rec = {t_: simpy.PriorityResource(env, capacity=int(a_["n"])) for t_, a_ in (pg or {}).items() if a_ and a_.get("n", 0) > 0}
    presup = {t_: pg[t_]["n"] * pg[t_]["f"] * sc.HORAS_TURNO for t_ in pg_rec}          # horas de ayuda fuera del núcleo que le quedan a los planificadores-gruero del turno

    def gruero_o_extra(t, paso, prio, dur_h, pool, vel, tag, nucleo=True):
        """El primero libre entre un gruero y una persona extra (apoyo o planificador-gruero) toma la tarea; después, la grúa si el proceso la usa.
        Fuera del núcleo, un planificador-gruero solo la toma si le queda presupuesto de ayuda."""
        usa_grua = "Grúas" in capacity.USO_DE_EQUIPO.get(paso, {})
        t0 = env.now
        rg = rec[t]["gruero"].request(priority=prio)
        ra = pool.request(priority=prio - (100 if tag == "pg" and nucleo else 0))
        d_ap = dur_h / vel
        got = yield rg | ra
        ap = rg not in got                                 # gana el gruero si ambos quedan libres a la vez
        if ap and tag == "pg" and not nucleo and presup[t] < d_ap - 1e-12:
            pool.release(ra)                               # sin presupuesto de ayuda: devuelve la persona y espera a un gruero
            ap = False
            yield rg
        else:
            perdedor, cola_perdedor = (ra, pool) if not ap else (rg, rec[t]["gruero"])
            if perdedor.triggered:
                cola_perdedor.release(perdedor)
            else:
                perdedor.cancel()
        cola = pool if ap else rec[t]["gruero"]
        if ap and tag == "pg" and not nucleo:
            presup[t] -= d_ap
        espera["gruero"] += env.now - t0
        pedidos["gruero"] += 1
        d = d_ap if ap else dur_h
        if usa_grua:
            t1 = env.now
            with grua.request(priority=prio) as ru:
                yield ru
                espera["grua"] += env.now - t1
                pedidos["grua"] += 1
                ini = env.now
                yield env.timeout(d)
            gru_ocupada[0] += env.now - ini
        else:
            ini = env.now
            yield env.timeout(d)
        cola.release(ra if ap else rg)
        reparte(t, tag if ap else "gruero", ini, env.now)
        if ap:
            horas_x[(t, tag if tag == "apoyo" else ("pg_nucleo" if nucleo else "pg_ayuda"), paso)] += env.now - ini
            if tag == "apoyo":
                apoyo_h[t] += env.now - ini

    def no_disponible(t_, nom, f_):
        """La persona de apoyo trabaja en el patio solo la fracción f_ de cada hora de su turno; el resto de la hora (al final) y después de la salida, otra labor la ocupa."""
        r_ = pool_rec[(t_, nom)]
        k = 0
        while f_ < 1.0 - 1e-9 and INICIO[t_] + k < FIN[t_] - 1e-9:
            desde = INICIO[t_] + k + f_
            hasta = min(INICIO[t_] + k + 1.0, FIN[t_])
            if desde < hasta:
                if desde > env.now:
                    yield env.timeout(desde - env.now)
                with r_.request(priority=-100) as rq:
                    yield rq
                    if hasta > env.now:
                        yield env.timeout(hasta - env.now)
            k += 1
        if FIN[t_] > env.now:
            yield env.timeout(FIN[t_] - env.now)
        with r_.request(priority=-100) as rq:          # después de su hora de salida no toma tareas del patio
            yield rq
            yield env.timeout(1e6)

    def pool_usar(t, paso, prio, dur_h, ctx, eid=None):
        """La tarea la toma el primero libre entre los elegibles del turno (`pools`); con traspaso, si al cierre nadie la empezó pasa al turno siguiente como trabajo del gruero."""
        opciones = pool_pasos[t][paso]
        reqs = [(ro, ug, f, rec[t][ro].request(priority=prio)) for ro, ug, f in opciones]
        t0 = env.now
        puede = ctx is not None and traspaso is not None and SIGUIENTE[t] is not None and not ctx.get("carried")
        evs = [x[3] for x in reqs] + ([env.timeout(max(0.0, FIN[t] - env.now))] if puede else [])
        yield env.any_of(evs)
        gana = next((x for x in reqs if x[3].triggered), None)
        for x in reqs:
            if x is gana:
                continue
            if x[3].triggered:
                rec[t][x[0]].release(x[3])
            else:
                x[3].cancel()
        if gana is None:                                   # traspaso: nadie la empezó en la ventana
            carry[(t, "gruero")] += dur_h
            carry_n[(t, "gruero")] += 1
            ctx["carried"] = True
            t2 = SIGUIENTE[t]
            ctx["t"] = t2
            if INICIO[t2] > env.now:
                yield env.timeout(INICIO[t2] - env.now)
            yield from pool_usar(t2, paso, prio, dur_h, ctx, eid)
            return
        ro, ug, f, rq = gana
        espera[ro if ro in ("operario", "gruero") else "gruero"] += env.now - t0
        pedidos["gruero"] += 1
        d = dur_h * f
        if ug:
            t1 = env.now
            with grua.request(priority=prio) as ru:
                yield ru
                espera["grua"] += env.now - t1
                pedidos["grua"] += 1
                ini = env.now
                yield env.timeout(d)
            gru_ocupada[0] += env.now - ini
        else:
            ini = env.now
            yield env.timeout(d)
        rec[t][ro].release(rq)
        reparte(t, ro, ini, env.now)

    def entidad(t, nombre, pasos_t, t_ini=None, eid=None):
        yield env.timeout(INICIO[t] if t_ini is None else max(float(t_ini), INICIO[t]))      # el trabajo del turno aparece al inicio de su ventana (o a su hora de llegada, con `llegadas`)
        ctx = {"t": t, "carried": False} if traspaso is not None else None
        _fel(('arribo', eid))
        _reg("Arribo", eid, t)
        for k_paso, (paso, prio, dur_h) in enumerate(pasos_t):
            tc = ctx["t"] if ctx else t
            if k_paso > 0:
                _reg("Arribo", eid, tc)
            if paso in pool_pasos.get(tc, {}) or (ctx is None and paso in pool_pasos.get(t, {})):
                yield from pool_usar(tc, paso, prio, dur_h, ctx, eid)
            elif paso in roles_paso:
                rol_x, ug = roles_paso[paso]
                if ug:
                    yield from gruero_y_grua(tc, prio, dur_h, ctx, eid, rol=rol_x)
                else:
                    yield from usar(tc, rol_x, prio, dur_h, ctx, eid)
            elif t in apoyo_rec and paso in apoyo[t].get("pasos", PASOS_APOYO):
                yield from gruero_o_extra(t, paso, prio, dur_h, apoyo_rec[t], apoyo[t].get("vel", 1.0), "apoyo")
            elif t in pg_rec and capacity.ROL_POR_PROCESO[paso] == "Gruero" and (paso in NUCLEO_PG or presup[t] > 1e-12):
                yield from gruero_o_extra(t, paso, prio, dur_h, pg_rec[t], pg[t].get("vel", 1.0), "pg", paso in NUCLEO_PG)
            elif capacity.ROL_POR_PROCESO[paso] == "Gruero" and "Grúas" in capacity.USO_DE_EQUIPO.get(paso, {}):
                yield from gruero_y_grua(tc, prio, dur_h, ctx, eid)
            elif capacity.ROL_POR_PROCESO[paso] == "Gruero":
                yield from usar(tc, "gruero", prio, dur_h, ctx, eid)
            else:
                yield from usar(tc, "operario", prio, dur_h, ctx, eid)
            if not (ctx and ctx["carried"]):
                fin_paso[(t, paso)].append(env.now)
        if ctx and ctx["carried"]:
            fin_carry[(t, nombre)].append(env.now)
        else:
            fin[(t, nombre)].append(env.now)

    total = {}
    for t in TURNOS:
        for nombre, ent in preps[t]["entidades"].items():
            v = ent["volumen"]
            n = int(math.floor(v)) + int(rng.random() < (v - math.floor(v)))
            total[(t, nombre)] = n
            if n == 0:
                continue
            tiempos = {}
            for paso in ent["pasos"]:
                f, _ = preps[t]["muestreadores"][paso]
                tiempos[paso] = f(n, rng) * preps[t].get("escala", {}).get(paso, 1.0) / preps[t].get("eta", {}).get(paso, sc._eta(paso)) / sc.SEG_POR_HORA * (factor_paso or {}).get(paso, 1.0)
            ini_ent = llegadas_r.tiempos(t, nombre, n) if llegadas_r is not None else None
            for i in range(n):
                eid = f"{nombre}#{t}{i + 1}"
                _fel(('arribo', eid), INICIO[t] if ini_ent is None else max(float(ini_ent[i]), INICIO[t]), 'Arribo', eid)
                env.process(entidad(t, nombre, [(paso, sc.PRIORIDAD.get(paso, sc.PRIO_RESTO), float(tiempos[paso][i])) for paso in ent["pasos"]], None if ini_ent is None else ini_ent[i], eid))
    def interrupcion(t, rol, llega, prio, dur_h, nombre):
        yield env.timeout(llega)
        yield from usar(t, rol, prio, dur_h)
        fin[(t, nombre)].append(env.now)
        fin_paso[(t, nombre)].append(env.now)

    for t in TURNOS:                      # traslados y otras actividades editados (distribución inventada: ver EXTRAS)
        for rol_n, rol in (("Operario", "operario"), ("Gruero", "gruero")):
            ex = preps[t].get("extras", {}).get(rol_n)
            if not ex:
                continue
            for nombre, durs, prio in ((f"traslados_{rol}", _trozos(rng, dot[t][rol_n] * ex["traslado_min"] / 60.0, EXTRAS["traslado_trozo_min"], EXTRAS["traslado_cv"]), 0),
                                       (f"otras_{rol}", _trozos(rng, ex["otras_h"], EXTRAS["otras_trozo_min"], EXTRAS["otras_cv"]) / ex["eta"], sc.PRIO_RESTO)):
                total[(t, nombre)] = len(durs)
                for d_ in durs:
                    env.process(interrupcion(t, rol, INICIO[t] + rng.random() * sc.HORAS_TURNO, prio, float(d_), nombre))
    for t_, per in pools.get("personas", {}).items():          # disponibilidad de las personas de apoyo
        for nom, f_ in per.items():
            env.process(no_disponible(t_, nom, f_))
    for t_, a_ in (apoyo or {}).items():
        if t_ in apoyo_rec and (a_["frac"] < 1.0 or a_.get("modo") == "fin"):
            env.process(fuera_de_ventana(t_, a_))
    _reg("Run", None, None)
    env.run()
    _reg("Fin (lista de eventos vacía)", None, None)
    q = sc.SUPUESTOS["percentil_picking"]["valor"]
    pk = np.sort(np.array(fin.get(("C", "picking"), []) + fin_carry.get(("C", "picking"), []) or [0.0]))
    por_turno = {}
    for t in TURNOS:
        ts = [x for (tt, _), v in fin.items() if tt == t for x in v]
        backlog = {nombre: int(sum(1 for x in fin.get((t, nombre), []) if x > FIN[t] + 1e-9)) for (tt, nombre) in total if tt == t}
        crit_x = [x for (tt, paso), v in fin_paso.items() if tt == t and rp(paso) in crit[t] for x in v]
        backlog_crit = {paso: int(sum(1 for x in v if x > FIN[t] + 1e-9)) for (tt, paso), v in fin_paso.items() if tt == t and rp(paso) in crit[t]}
        por_rol = {rol: max([x for (tt, paso), v in fin_paso.items() if tt == t and rp(paso) == rol for x in v], default=INICIO[t]) for rol in roles_t}
        por_turno[t] = {"t_fin": max(ts) if ts else INICIO[t], "backlog": backlog, "sobretiempo_h": extra[t], "t_fin_criterio": max(crit_x) if crit_x else INICIO[t],
                        "backlog_criterio": backlog_crit, "t_fin_rol": por_rol,
                        "util_op": dentro[(t, "operario")] / (dot[t]["Operario"] * sc.HORAS_TURNO), "util_gr": dentro[(t, "gruero")] / (dot[t]["Gruero"] * sc.HORAS_TURNO),
                        "horas_apoyo": apoyo_h[t],
                        "traspaso_h": {rol: carry[(t, rol)] for rol in roles_t}, "traspaso_n": {rol: carry_n[(t, rol)] for rol in roles_t},
                        "horas_apoyo_paso": {pa: horas_x[(t, "apoyo", pa)] for (tt, tg, pa) in list(horas_x) if tt == t and tg == "apoyo"},
                        "horas_pg_nucleo": sum(v for (tt, tg, _), v in horas_x.items() if tt == t and tg == "pg_nucleo"),
                        "horas_pg_ayuda": sum(v for (tt, tg, _), v in horas_x.items() if tt == t and tg == "pg_ayuda")}
        if pool_rec:
            por_turno[t]["util_apoyo"] = {nom: dentro[(t, nom)] / sc.HORAS_TURNO for (tt, nom) in pool_rec if tt == t}
        if roles_x:
            por_turno[t]["roles_criterio"] = crit[t]
            por_turno[t]["util_x"] = {r_: dentro[(t, r_)] / (rec[t][r_].capacity * sc.HORAS_TURNO) for r_ in roles_x}
    return {"t99_picking": float(pk[max(0, math.ceil(q * len(pk)) - 1)]), "por_turno": por_turno, "uso_hora": {f"{t}|{ro}": uso_h[(t, ro)].tolist() for t in TURNOS for ro in ("operario", "gruero")},
            "t_fin_dia": max([v["t_fin"] for v in por_turno.values()] + [x for v in fin_carry.values() for x in v]), "util_grua": gru_ocupada[0] / (n_grua * 24.0),
            "espera_min": {k: 60 * espera[k] / pedidos[k] if pedidos[k] else 0.0 for k in ("operario", "gruero", "grua")}, "traspaso_tol": traspaso}


TRASPASO_TOL = 0.05        # versión única (decisión del usuario, 2026-11-10): el trabajo no iniciado pasa al turno siguiente; tolerancia propuesta, a confirmar con el tutor (con 10% la dotación mínima no cambia)


def simular_dia(preps: dict, dot: dict, n_grua: int, replicas: int, umbral: float | None = None, llegadas="auto", traspaso: float | None = TRASPASO_TOL,
                demanda: float | None = None, charla_min: float = 20.0, colacion_min: float = 30.0, progreso=None) -> dict:
    """VERSIÓN ÚNICA: traspaso al turno siguiente (tolerancia `TRASPASO_TOL`) y llegadas de camiones de recepción y carga A ajustadas al log del WMS (`llegadas_camiones`), que se activan cuando se
    pasa la `demanda` del día. `llegadas=None` y `traspaso=None` reproducen la simulación anterior (la usa `validacion_simulacion`, que valida el turno C contra noches históricas)."""
    if isinstance(llegadas, str) and llegadas == "auto":
        if demanda is None:
            llegadas = None
        else:
            from modelo import llegadas_camiones as lc
            llegadas = lc.GeneradorLlegadas(float(demanda), "rec_carga", charla_min=charla_min, colacion_min=colacion_min, horas=sc.HORAS_TURNO)
    base = sc.SUPUESTOS["semilla_base"]["valor"]
    res = []
    for r in range(replicas):
        res.append(_replica_dia(preps, dot, n_grua, base + r, llegadas=llegadas, traspaso=traspaso))
        if progreso is not None:
            progreso(r + 1, replicas)
    return agregar(res, dot, umbral)


def agregar(res: list[dict], dot: dict, umbral: float | None = None) -> dict:
    """Resumen de un conjunto de réplicas (las de `simular_dia` o las que corre en paralelo `modelo.sensibilidad_simulacion`, con las mismas semillas semilla_base + r)."""
    replicas = len(res)
    out = {"replicas": replicas, "dot": dot, "t99_picking": float(np.mean([r["t99_picking"] for r in res])), "t99_picking_p90": float(np.percentile([r["t99_picking"] for r in res], 90)),
           "util_grua": float(np.mean([r["util_grua"] for r in res])), "t_fin_dia": float(np.mean([r["t_fin_dia"] for r in res])),
           "t_fin_dia_p90": float(np.percentile([r["t_fin_dia"] for r in res], 90)),
           "espera_min": {k: float(np.mean([r["espera_min"][k] for r in res])) for k in ("operario", "gruero", "grua")}, "turnos": {}}
    ok_dia = []
    for t in TURNOS:
        tf = np.array([r["por_turno"][t]["t_fin"] for r in res])
        tol = res[0].get("traspaso_tol")
        roles_c = res[0]["por_turno"][t].get("roles_criterio", CRITERIO_ROLES[t])      # con roles extra (roles_paso) también deciden
        roles_r = tuple(res[0]["por_turno"][t]["traspaso_h"])
        if tol is None:
            ok_t = [r["por_turno"][t]["t_fin_criterio"] <= FIN[t] + 1e-9 and sum(r["por_turno"][t]["backlog_criterio"].values()) == 0 for r in res]
        else:     # con traspaso, por rol que decide: si el rol traspasa, lo que suelta no pasa de `tol` × las horas de ese rol en el turno que lo recibe; si no traspasa (B, operarios del C), termina en su ventana
            sig = SIGUIENTE[t]

            def _ok_rol(r, ro):
                if sig is None or (t, ro) in SIN_TRASPASO:
                    return r["por_turno"][t]["t_fin_rol"][ro] <= FIN[t] + 1e-9
                return r["por_turno"][t]["traspaso_h"][ro] <= tol * max(1, dot[sig].get(_ROL_DOT.get(ro, ro.capitalize()), 0)) * sc.HORAS_TURNO + 1e-9
            ok_t = [all(_ok_rol(r, ro) for ro in roles_c) for r in res]
        ok_todo = [r["por_turno"][t]["t_fin"] <= FIN[t] + 1e-9 and sum(r["por_turno"][t]["backlog"].values()) == 0 for r in res]
        procs = sorted({p for r in res for p in r["por_turno"][t]["backlog"]})
        out["turnos"][t] = {"fin_media": float(tf.mean()), "fin_p90": float(np.percentile(tf, 90)),
                            "fin_criterio_p90": float(np.percentile([r["por_turno"][t]["t_fin_criterio"] for r in res], 90)), "pct_en_ventana": float(np.mean(ok_t)), "pct_en_ventana_todos": float(np.mean(ok_todo)), "roles_criterio": roles_c,
                            "fin_rol": {rol: float(np.mean([r["por_turno"][t]["t_fin_rol"][rol] for r in res])) for rol in roles_r},
                            "pct_rol": {rol: float(np.mean([r["por_turno"][t]["t_fin_rol"][rol] <= FIN[t] + 1e-9 for r in res])) for rol in roles_r},
                            "sobretiempo_h": float(np.mean([r["por_turno"][t]["sobretiempo_h"] for r in res])),
                            "backlog": {p: float(np.mean([r["por_turno"][t]["backlog"].get(p, 0) for r in res])) for p in procs},
                            "util_op": float(np.mean([r["por_turno"][t]["util_op"] for r in res])), "util_gr": float(np.mean([r["por_turno"][t]["util_gr"] for r in res])),
                            "fin_gruero_p90": float(np.percentile([r["por_turno"][t]["t_fin_rol"]["gruero"] for r in res], 90)),
                            "horas_apoyo": float(np.mean([r["por_turno"][t]["horas_apoyo"] for r in res])),
                            "traspaso_h": {ro: float(np.mean([r["por_turno"][t]["traspaso_h"][ro] for r in res])) for ro in roles_r},
                            "traspaso_h_p90": {ro: float(np.percentile([r["por_turno"][t]["traspaso_h"][ro] for r in res], 90)) for ro in roles_r},
                            "traspaso_n": {ro: float(np.mean([r["por_turno"][t]["traspaso_n"][ro] for r in res])) for ro in roles_r},
                            "horas_apoyo_paso": {pa: float(np.mean([r["por_turno"][t]["horas_apoyo_paso"].get(pa, 0.0) for r in res])) for pa in sorted({k for r in res for k in r["por_turno"][t]["horas_apoyo_paso"]})},
                            "horas_pg_nucleo": float(np.mean([r["por_turno"][t]["horas_pg_nucleo"] for r in res])),
                            "horas_pg_ayuda": float(np.mean([r["por_turno"][t]["horas_pg_ayuda"] for r in res]))}
        if "util_x" in res[0]["por_turno"][t]:
            out["turnos"][t]["util_x"] = {k: float(np.mean([r["por_turno"][t]["util_x"][k] for r in res])) for k in res[0]["por_turno"][t]["util_x"]}
            out["turnos"][t]["ok_rol"] = {ro: float(np.mean([_ok_rol(r, ro) for r in res])) if tol is not None else None for ro in roles_c}
        ok_dia.append(ok_t)
    todo = [all(x) for x in zip(*ok_dia)]
    out["pct_dia_ok"] = float(np.mean(todo))
    out["umbral"] = sc.SUPUESTOS["umbral_viabilidad"]["valor"] if umbral is None else umbral
    out["traspaso_tol"] = res[0].get("traspaso_tol")
    out["t99_lista"] = [float(r["t99_picking"]) for r in res]                    # una hora por réplica (reloj de la simulación): histograma de la hora en que termina el picking
    if "uso_hora" in res[0]:                                                       # uso medio y p90 por hora = horas-persona ocupadas ÷ personas del rol (1 = todos ocupados toda la hora)
        out["uso_hora"] = {t: {ro: {"media": (np.mean([r["uso_hora"][f"{t}|{ro}"] for r in res], axis=0) / max(1, dot[t][ro.capitalize()])).tolist(),
                                    "p90": (np.percentile([r["uso_hora"][f"{t}|{ro}"] for r in res], 90, axis=0) / max(1, dot[t][ro.capitalize()])).tolist()} for ro in ("operario", "gruero")} for t in TURNOS}
    out["viable"] = out["pct_dia_ok"] >= out["umbral"] - 1e-9
    return out


def configurar_ventana(horas: float) -> None:
    """Ventana de trabajo EFECTIVA de cada turno (horas). El reloj de la simulación es de horas efectivas: la hora real se obtiene con `hora_real`."""
    sc.HORAS_TURNO = float(horas)
    for t in TURNOS:
        FIN[t] = INICIO[t] + float(horas)


def hora_real(t: str, h_sim: float, charla_min: float, colacion_min: float, horas: float) -> str:
    """Convierte una hora del reloj efectivo a hora de reloj: suma la charla de inicio y, si ya pasó la mitad de la ventana, la colación."""
    rel = h_sim - INICIO[t]
    return reloj(INICIO[t] + charla_min / 60 + rel + (colacion_min / 60 if rel > horas / 2 else 0.0))


def hora_real_picking(t: str, h_sim: float, ajustes) -> str:
    """Hora de reloj de un instante del PICKING: los pickers empiezan tras su charla (10 min observados) y su tiempo simulado se estira a presencia (la colación va dentro de la
    utilización medida), en vez de sumar charla de 20 min y colación como `hora_real`. Es la conversión que se validó contra las 22 noches de septiembre."""
    h = ajustes.horas_efectivas()
    if not ajustes.colacion_en_utilizacion_picking:           # criterio anterior: mismo reloj que el resto
        return hora_real(t, h_sim, ajustes.charla_min, ajustes.colacion_min, h)
    k = ajustes.horas_picking() / h if h else 1.0
    return reloj(INICIO[t] + ajustes.charla_picking_min / 60 + (h_sim - INICIO[t]) * k)


def preparar_dia(demanda: float, params, ajustes=None) -> tuple[dict, dict]:
    """(preparación por turno, dotación redondeada del modelo por turno). Cambia sc.TURNO para reutilizar `preparar` de simulacion_turno_c."""
    preps, dot = {}, {}
    for t in TURNOS:
        sc.TURNO = t
        preps[t] = sc.preparar(demanda, params, ajustes)
        dot[t] = sc.dotacion_redondeada(preps[t])
    return preps, dot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replicas", type=int, default=REPLICAS["valor"])
    ap.add_argument("--media-del-modelo", action="store_true", help="sensibilidad: reescala cada muestra a la media observada del paso (guarda otro CSV)")
    a = ap.parse_args(argv)
    sc.AJUSTAR_MEDIA["activo"] = a.media_del_modelo
    salida = SALIDA.with_name("simulacion_dia_media_del_modelo.csv") if a.media_del_modelo else SALIDA
    t0 = time.time()
    params = fit_params.cargar()
    escenarios = capacity.escenarios_estimador()
    n_grua = sc.SUPUESTOS["gruas"]["valor"]
    print(f"SIMULACIÓN DEL DÍA COMPLETO (C 00-08, A 08-16, B 16-24) · {a.replicas} réplicas por escenario · semilla base {sc.SUPUESTOS['semilla_base']['valor']} · η = {sc.SUPUESTOS['eficiencia']['valor']:g} · "
          f"grúas compartidas = {n_grua} [{sc.SUPUESTOS['gruas']['estado']}]")
    resultados = {}
    for nombre, dem in escenarios.items():
        preps, dot = preparar_dia(dem, params)
        resultados[nombre] = (dem, simular_dia(preps, dot, n_grua, a.replicas, demanda=dem))
    preps50, _ = preparar_dia(escenarios["p50"], params)
    real = {t: {"Operario": max(1, config.personas_turno(t, "Operario")), "Gruero": max(1, config.personas_turno(t, "Gruero"))} for t in TURNOS}
    resultados["p50 con dotación programada real"] = (escenarios["p50"], simular_dia(preps50, real, n_grua, a.replicas, demanda=escenarios["p50"]))
    sc.TURNO = "C"
    print("Dotación por turno (operarios/grueros): " + " · ".join(f"{t}: " + "/".join(f"{resultados['p50'][1]['dot'][t][r]}" for r in ('Operario', 'Gruero')) for t in TURNOS)
          + "  [modelo, p50]  |  programada: " + " · ".join(f"{t}: {real[t]['Operario']}/{real[t]['Gruero']}" for t in TURNOS)
          + f"  (B tiene {config.personas_turno('B', 'Operario')} operarios en el ODS: se simula con 1)")

    print("\nRESULTADOS: hora de término de cada turno (hora de reloj; pasar de la hora de salida = sobretiempo/derrame), % de réplicas dentro de su ventana y viabilidad del día")
    print(f"{'escenario':<34s}{'cajas':>7s} | {'C fin':>6s}{'%':>5s} | {'A fin':>6s}{'%':>5s} | {'B fin':>6s}{'%':>5s} | {'día ok':>7s}{'viable':>8s}{'sobret. h-p':>12s}{'util grúa 24h':>14s}")
    filas = []
    for nombre, (dem, r) in resultados.items():
        tt = r["turnos"]
        so = sum(tt[t]["sobretiempo_h"] for t in TURNOS)
        print(f"{nombre:<34s}{dem:>7.0f} | {reloj(tt['C']['fin_media']):>6s}{100 * tt['C']['pct_en_ventana']:>4.0f}% | {reloj(tt['A']['fin_media']):>6s}{100 * tt['A']['pct_en_ventana']:>4.0f}% | "
              f"{reloj(tt['B']['fin_media']):>6s}{100 * tt['B']['pct_en_ventana']:>4.0f}% | {100 * r['pct_dia_ok']:>6.0f}%{('sí' if r['viable'] else 'NO'):>8s}{so:>12.2f}{100 * r['util_grua']:>13.0f}%")
        fila = {"escenario": nombre, "demanda_cajas": round(dem), "replicas": r["replicas"], "hora_99pct_picking_C_media": reloj(r["t99_picking"]), "hora_99pct_picking_C_p90": reloj(r["t99_picking_p90"]),
                "hora_fin_dia_media": reloj(r["t_fin_dia"]), "hora_fin_dia_p90": reloj(r["t_fin_dia_p90"])}
        for t in TURNOS:
            x = tt[t]
            fila.update({f"operarios_{t}": r["dot"][t]["Operario"], f"grueros_{t}": r["dot"][t]["Gruero"], f"hora_fin_{t}_media": reloj(x["fin_media"]), f"hora_fin_{t}_p90": reloj(x["fin_p90"]),
                         f"pct_replicas_en_ventana_{t}": round(100 * x["pct_en_ventana"], 1), f"sobretiempo_horas_persona_{t}": round(x["sobretiempo_h"], 2),
                         f"pendientes_al_cierre_{t}": round(sum(x["backlog"].values()), 2), f"utilizacion_operarios_{t}": round(x["util_op"], 3), f"utilizacion_grueros_{t}": round(x["util_gr"], 3)})
        fila.update({"pct_replicas_dia_completo_en_ventana": round(100 * r["pct_dia_ok"], 1), "viable": "sí" if r["viable"] else "NO", "utilizacion_gruas_24h": round(r["util_grua"], 3),
                     "espera_media_min_operario": round(r["espera_min"]["operario"], 1), "espera_media_min_gruero": round(r["espera_min"]["gruero"], 1),
                     "espera_media_min_grua": round(r["espera_min"]["grua"], 1)})
        filas.append(fila)
    print("\nDETALLE POR TURNO (p50 con la dotación del modelo y con la programada): sobretiempo en horas-persona, pendientes al cierre de la ventana y utilización dentro de la ventana")
    for nombre in ("p50", "p50 con dotación programada real"):
        dem, r = resultados[nombre]
        print(f"  {nombre}:")
        for t in TURNOS:
            x = r["turnos"][t]
            bl = ", ".join(f"{p} {v:.1f}" for p, v in x["backlog"].items() if v > 0) or "sin pendientes"
            print(f"    turno {t}: fin {reloj(x['fin_media'])} (p90 {reloj(x['fin_p90'])}) · sobretiempo {x['sobretiempo_h']:.2f} h-persona · pendientes al cierre: {bl} · uso operarios {100 * x['util_op']:.0f}% · uso grueros {100 * x['util_gr']:.0f}%")
    if sc.AVISOS:
        print("\nAVISOS:")
        for m in sorted(set(sc.AVISOS)):
            print("  -", m)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado: {salida}  ·  tiempo de corrida: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
