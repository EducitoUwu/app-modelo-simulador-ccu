"""Punto único de configuración: roles, turnos, dotación real, equipos y el umbral de
criticidad de recurso. Ningún módulo debe declarar estas cifras por su cuenta — todo
lo que necesita una fuente (planilla, WMS, ODS) se lee aquí desde los cargadores
(`dotacion.py`, `wms_config.py`, `fit_params.py`), nunca escrito a mano.

Umbral de criticidad de recurso (regla de rigor del CLAUDE.md: uno solo, justificado,
usado por todos los módulos):
  Un recurso se considera CRÍTICO cuando su utilización en el intervalo supera
  UMBRAL_UTILIZACION_CRITICA. Se elige 0.85 como punto de partida [supuesto, PENDIENTE
  de confirmar con Ernesto/Milton junto con el objetivo de Fill Rate — ver
  CLAUDE.md > Brechas de datos, ítem 7].

  ESTADO (2026-10-05): RESERVADO para `metrics.py` y la simulación, donde se mide la utilización REAL
  (tiempo ocupado / tiempo disponible, con colas y contención). `capacity.py` NO calcula U = N/R ni marca
  roles "críticos": su criterio operativo es la brecha (N > R). No usarlo en capacity.py hasta que exista
  esa medición, porque N ya incluye el factor 1/η (horas de presencia) y compararlo con 0.85 mezclaría
  eficiencia (η, que hoy vale 0,848) con utilización crítica (ρ*).

  Justificación: la aproximación de Kingman para una cola G/G/1 (Kingman, 1961), Wq ≈ ρ/(1−ρ)·(ca²+cs²)/2·E[S],
  muestra que la espera crece de forma desproporcionada con la ocupación (ρ/(1−ρ) = 5,7 en 0,85 y 19 en 0,95).
  M/M/c queda solo como referencia: la grúa es compartida con prioridades y viola sus supuestos. No reemplaza
  una meta de negocio explícita.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

from . import dotacion, wms_config

UMBRAL_UTILIZACION_CRITICA = 0.85   # ver docstring: PENDIENTE de validar con el tutor/Ernesto

TURNOS = dotacion.TURNOS                 # ("A", "B", "C")
HORARIO_TURNO = dotacion.HORARIO_TURNO   # {"A": (8, 16), ...}
HORAS_POR_TURNO = 8
# HORAS EFECTIVAS (decisión del equipo, 2026-10-13): el turno dura 8 h pero NO se trabajan todas (colación, charlas, tiempos muertos). Las personas necesarias se
# calculan sobre 7 h, y de esas 7 h solo se usa el 85% (utilización), de modo que N = horas de trabajo / (7 x 0,85) = horas de trabajo / 5,95 por persona.
# Antes: 8 h x 0,848 (la eficiencia medida en picking = horas de gap / horas de presencia del log) = 6,78 h por persona. Es una DECISIÓN del equipo, no una medición:
# si las pausas ya estuvieran dentro del 0,848 medido en picking, el criterio nuevo las descuenta dos veces (declarado).
# [ACTUALIZADO 2026-10-21, parámetros finales acordados; `modelo/parametros_diana.py`]: 8 h − 30 min de colación − 20 min de charla = 7 h 10 min por persona,
# × utilización: picking 0,644 (la de Diana, medida por picker en el log) y 0,7 en el resto.
from . import parametros_diana as _pd
HORAS_TURNO_NOMINAL = 8.0            # horas del turno en el reloj
HORAS_EFECTIVAS_POR_TURNO = _pd.HORAS_EFECTIVAS          # 7 h 10 min
UTILIZACION_OBJETIVO = _pd.UTILIZACION_RESTO             # 0,8 desde el 2026-11-24 (antes 0,7; todo salvo picking, supuesto)
UTILIZACION_PICKING = _pd.valores()["util_picking_pond"]   # 0,649 ponderado por listas (2026-10-21; el promedio simple 0,644 sobreestimaba el picking, ver validacion_simulacion)

# Tabla de equivalencia de roles: planilla de terreno <-> ficha de proyecto de la
# empresa <-> WMS. [inferido, PENDIENTE de confirmar — CLAUDE.md > Brechas, "equivalencia
# de roles"]. La ficha pide "grueros, pickers, reponedores"; el WMS no tiene un rol
# "reponedor" (es una función repartida entre Gruero y Operario, ver wms_config.py).
EQUIVALENCIA_ROLES = {
    "Gruero": "gruero",                       # opera grúa: recepción, carga, almacenamiento, reabasto de pallet
    "Operario": "reponedor_caja",             # reabasto de cajas (rama manual, ver Charter Fig. 4.3)
    "Externos": "reponedor_caja_apoyo",       # refuerzo externo de reabasto/picking según el turno
    "Planificador": "planificador",
    "Chequeador": "chequeador",
}

# Roles que requieren grúa para su tarea principal (comparten el mismo recurso físico).
ROLES_QUE_USAN_GRUA = ("Gruero",)


@dataclass(frozen=True)
class ProcesoEnTurno:
    """Si un proceso ocurre en un turno, y con qué respaldo.

    `ocurre`: True / False / **None = no se sabe** (se trata como incertidumbre, nunca
    como "no ocurre": un proceso que no se sabe si ocurre deja el resultado incompleto).
    `estado`: "confirmado" (lo dijo alguien del equipo o lo prueba un archivo) o
    "pendiente" (hay evidencia pero falta que lo valide Víctor, o no hay evidencia).
    `volumen_fijo`: unidades por turno declaradas directamente (p. ej. camiones por noche)
    en vez de derivarlas de la demanda. None = se deriva de la demanda con su driver.
    """
    ocurre: bool | None
    estado: str
    fuente: str
    volumen_fijo: float | None = None
    unidad: str | None = None
    nota: str = ""
    n_volumen: int | None = None
    # `n_volumen`: de cuántas observaciones (noches/días contados) sale `volumen_fijo`. `capacity` etiqueta el
    # volumen con las mismas reglas que fit_params (`fit_params.estado_por_n`): 10 noches = preliminar. None = no se
    # sabe de cuántas sale (se trata como insuficiente).


_DAVID = "confirmado por David (equipo de terreno)"
_LOG = "exports WMS de septiembre-2026 (docs/wms actividades septiembre/)"

# ---------------------------------------------------------------------------------------
# DOS CALIBRACIONES DISTINTAS, A PROPÓSITO (decisión del usuario, 2026-10-02). No mezclar.
#
# (1) PRÁCTICA DE TRABAJO — qué proceso ocurre en qué turno, y cuánto de su volumen cae en
#     cada uno. Es una decisión de organización del CD, y cambió el 22-sep-2026: desde esa
#     fecha el reabasto de palés pasó de A/B/C 48/37/15% a 2/8/90% y el de cajas de 0% a
#     56% en turno C. Lo que describe al CD HOY es el régimen posterior (`PROCESOS_POR_TURNO`,
#     `capacity.CUOTAS_TURNO`). El período anterior (4 a 21-sep, que incluye la semana peak de
#     Fiestas Patrias) queda en `PROCESOS_POR_TURNO_HISTORICO` como histórico/atípico: sirve
#     para comparar, no describe el estado actual.
#
# (2) VOLUMEN DE DEMANDA — cuántas cajas hay que mover por día. Es un fenómeno del mercado,
#     no una decisión del CD; lo que se necesita es VARIEDAD (peak y días normales) para
#     poder correlacionarlo con las ventas. Por eso usa septiembre COMPLETO, sin recortar la
#     semana de Fiestas Patrias (ver `capacity.demanda_septiembre`).
#
# Por qué son distintas: recortar la demanda al régimen posterior dejaría solo 8 días
# operados de volumen casi constante (17-22k cajas) y mataría la correlación; usar el mes
# completo para la práctica mezclaría dos formas de organizar el trabajo que no coexisten.
# ---------------------------------------------------------------------------------------
REGIMEN_VIGENTE_DESDE = "2026-09-22"
# RÉGIMEN VIGENTE = LA PRÁCTICA MÁS RECIENTE (2026-10-13): 2 a 10-oct-2026 (7 días operados completos del 2 al 9-oct y el turno C del 10-oct), de los exports WMS de
# octubre. Reemplaza al «vigente» de versiones anteriores (22-sep a 1-oct, que ahora es el régimen INTERMEDIO): en octubre la práctica siguió cambiando
# (reabasto de cajas 56% -> 94% en C; recepción y almacenamiento del turno A al B). Los números de cada proceso salen de `calibracion.py` y se recalculan solos.
REGIMEN_ACTUAL_DESDE = "2026-10-02"
_VIG = "régimen vigente = octubre (2 al 9-oct completos + noche del 10-oct)"
_LOGO = "exports WMS de octubre-2026 (docs/wms actividades octubre hasta el 11/)"
_LOGM = "movimientos de inventario parcial sep-oct (docs/wms actividades octubre hasta el 11/; modelo/movimientos_parciales.py)"

# Carga NOCTURNA: no depende del régimen (21 a 25 trailers por noche desde el 22-sep), así que se cuentan todas las noches desde el 22-sep (17 noches).
from . import calibracion as _cal
_CARGA_C, _N_CARGA_C = _cal.carga_noche_c() if _cal.disponible() else (22.4, 10)

# Qué procesos ocurren en cada turno HOY. Lo que no tiene fuente clara está marcado
# "pendiente": nada de esto se asume en silencio. Procesos modelados: los 6 con tiempos
# medidos. Retornos de fleteros, mermas y staging no tienen tiempos ni hoja de planilla
# con datos, así que no están acá (ver CLAUDE.md > Brechas).
# Participación de cada turno (octubre, tareas por día): ver `calibracion.resumen_texto()`.
PROCESOS_POR_TURNO: dict[str, dict[str, ProcesoEnTurno]] = {
    "C": {
        "picking": ProcesoEnTurno(True, "confirmado", f"{_DAVID}; {_LOG}: 99,99% de las líneas, 22-sep a 1-oct (los exports de octubre NO traen picking)"),
        "reabasto_pallet": ProcesoEnTurno(
            True, "confirmado", f"{_DAVID}; {_LOGO}: 78% de los palés, {_VIG}",
            nota="Antes del 22-sep era 15%; 22-sep a 1-oct 91%; octubre 78% (muy irregular: de 1 a 152 por noche). Ver PROCESOS_POR_TURNO_HISTORICO."),
        "reabasto_caja": ProcesoEnTurno(
            True, "confirmado", f"{_DAVID}; {_LOGO}: 94% de los retiros de caja, {_VIG}",
            nota="Antes del 22-sep era 0%; 22-sep a 1-oct 57%; octubre 94%. Frecuencia por log; el tiempo viene de la planilla oficial (ciclo medido, N=30)."),
        "carga_camion": ProcesoEnTurno(
            True, "confirmado",
            f"{_LOG} y {_LOGO}: {_CARGA_C:.1f} trailers por noche (CONTEO DIRECTO de TRL distintos; media de {_N_CARGA_C} noches, 22-sep a 10-oct)",
            volumen_fijo=round(_CARGA_C, 2), unidad="camiones", n_volumen=_N_CARGA_C,
            nota="Conteo directo, no estimación: los exports de carga traen el número de trailer en todas las líneas despachadas. Rango 21-25 por noche desde el 22-sep "
                 "(la carga nocturna no cambió con el régimen). Reemplaza al '~5' de David y a la estimación por proxy (líneas/162,2), que subestimaba 7-11%. "
                 "Confirmación exacta de David/Víctor sigue PENDIENTE."),
        "recepcion_directo_stage": ProcesoEnTurno(
            True, "confirmado", f"{_LOGO}: 5% de los camiones estimados (0,5 por noche), {_VIG}",
            nota="Casi desapareció de la noche (en 22-sep a 1-oct eran 0,8 por noche; en septiembre antes del 22 ≈0). El log no distingue 'directo a stage' del "
                 "resto de la recepción: la cifra es de recepción en general."),
        "almacenamiento": ProcesoEnTurno(
            True, "pendiente", f"{_LOGO}: 8% de las tareas (16,6 por noche; 4 de 8 noches con 30 a 40 tareas); David dijo que prácticamente no ocurre",
            nota="CAMBIO (2026-10-13): antes False (<4%, confirmación de David). En octubre el log da 30 a 40 tareas en 4 de 8 noches (2, 5, 6 y 7-oct). Se pasa a True/'pendiente': "
                 "confirmar con Víctor si el almacenamiento nocturno es nuevo. Aporte a la dotación de C: pequeño (≈0,06 grueros)."),
        "retorno_fletero": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 0,8 tareas por noche (1% del proceso), {_VIG}",
            nota="Casi no ocurre de noche (2 noches con actividad de 8). Rol por confirmar con Víctor: se asigna a Operario."),
        "merma": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 24,9 tareas por noche (69% del proceso), {_VIG}",
            nota="Se concentra de 04:00 a 06:00. Rol por confirmar con Víctor: se asigna a Operario (99,5% de las tareas de C las hacen cuentas personales)."),
    },
    "A": {
        # David: "todos los procesos, es el turno de mayor actividad general".
        "picking": ProcesoEnTurno(
            True, "confirmado", f"{_DAVID}; {_LOG}: 0 líneas en A desde el 22-sep (octubre sin dato de picking)",
            nota="Solo aparece como derrame cuando la noche no cabe en 8 h (14-16-sep, 21-ago). Se deja True "
                 "(confirmado por David) con cuota ≈0; el resultado no cambia."),
        "reabasto_pallet": ProcesoEnTurno(True, "confirmado", f"{_DAVID}; {_LOGO}: 20% de los palés, {_VIG}",
                                          nota="Antes del 22-sep era 48%; 22-sep a 1-oct 2%; octubre 20% (irregular: 62 palés el 5-oct y 27 el 8-oct, ≈0 otros días)."),
        "reabasto_caja": ProcesoEnTurno(True, "confirmado", f"{_DAVID}; {_LOGO}: 5% de los retiros de caja (1,1 por día), {_VIG}",
                                        nota="Antes del 22-sep era 81%; 22-sep a 1-oct 40%; octubre 5%. Lo que ejecuta A además sale de las hojas de los planificadores (fuera del WMS)."),
        "recepcion_directo_stage": ProcesoEnTurno(True, "confirmado", f"{_DAVID}; {_LOGO}: 42% de los camiones estimados, {_VIG}"),
        "carga_camion": ProcesoEnTurno(
            True, "confirmado",
            "ventas de septiembre (columna CARGA, vueltas >= 2) y log WMS (trailers del turno A) concuerdan: "
            "NIVEL ≈14 trailers por día (ventas 14,1-14,9; log 13,5 en 22-sep a 1-oct y 13,6 en octubre)",
            nota="Decisión del usuario (2026-10-04): se confirma el NIVEL, no un cambio de práctica — la diferencia "
                 "entre regímenes del turno A no es significativa (Mann-Whitney p=0,28 con los datos de septiembre). El volumen NO es fijo: las vueltas >= 2 se modelan como función del facturado "
                 "(capacity.DRIVERS['carga_camion']: vueltas>=2 = -14,77 + 1,295 por cada 1.000 cajas facturadas; R² 0,95, sigma 1,8 camiones, n=25 días de septiembre; "
                 "no hay ventas de octubre para recalcularla) y se cargan en el turno A (cuota 1,0: el turno B cargó 0,6 trailers por día en octubre). "
                 "'CARGA = vuelta' sigue siendo [hipótesis, confirmar con Elías]. La vuelta 1 (turno C) se mantiene fija en config.PROCESOS_POR_TURNO['C']."),
        "almacenamiento": ProcesoEnTurno(
            True, "confirmado", f"{_LOGO}: 31% de las tareas de almacenamiento, {_VIG}",
            nota="Antes del 22-sep 20%; 22-sep a 1-oct 61%; octubre 31% (el almacenamiento se movió al turno B). Driver y cuota en capacity.py. "
                 "Tiempo estándar `ciclo_total` validado con el WMS."),
        "retorno_fletero": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 14,8 tareas por día (14% del proceso), {_VIG}",
            nota="Rol por confirmar con Víctor: se asigna a Operario (79% de las tareas de A las hacen cuentas genéricas de bodega, 7% cuentas de gruero)."),
        "merma": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 7,0 tareas por día (19% del proceso), {_VIG}",
            nota="Rol por confirmar con Víctor: se asigna a Operario."),
    },
    # TURNO B (16:00-24:00), contado proceso por proceso con los exports de octubre (2 al 9-oct completos).
    "B": {
        "picking": ProcesoEnTurno(
            False, "confirmado", f"{_LOG}: 0,4 líneas por día en ambos regímenes de septiembre (los exports de octubre no traen picking)",
            nota="No ocurre en B: 4 de 22 días con alguna línea, máximo 4. Cuota ≈0."),
        "reabasto_pallet": ProcesoEnTurno(
            True, "pendiente", f"{_LOGO}: 1,3 palés por día (2% de los palés; 22-sep a 1-oct 6,9 por día, 7%; antes del 22-sep 33,6, 37%)",
            nota="INTERMITENTE: 4 palés el 5-oct y 3 el 6-oct, el resto de los días 0 a 1. Se deja True (hay actividad real) pero 'pendiente': confirmar con Víctor "
                 "si B repone por demanda. Aporte a la dotación de B: marginal. Tiempo: sin medición propia de B (sustituto_dia)."),
        "reabasto_caja": ProcesoEnTurno(
            True, "pendiente", f"{_LOGO}: 0,1 retiros de caja por día (1 retiro en 7 días; 22-sep a 1-oct 1,4 por día; antes del 22-sep 9,6)",
            nota="CASI DESAPARECIÓ de B. Se deja True/'pendiente' por no cambiar la estructura de golpe; aporte marginal."),
        "recepcion_directo_stage": ProcesoEnTurno(
            True, "confirmado", f"{_LOGO}: 5,0 camiones estimados por día (53%; 22-sep a 1-oct 3,5, 35%; antes del 22-sep 4,1, 47%), activa 7 de 7 días",
            nota="La recepción pasó de A a B: ahora B recibe más camiones que A. Tiempo estándar: la planilla casi no midió B (insuficiente). "
                 "El log no distingue 'directo a stage' del resto de la recepción."),
        "carga_camion": ProcesoEnTurno(
            False, "pendiente",
            f"{_LOGO}: 0,6 trailers por día en B (2 el 2-oct y 1 el 8 y el 9-oct); {_LOG}: 2 sueltos desde el 17-sep",
            nota="Antes del 17-sep hubo carga en B en 11 días (hasta 25 trailers el 15-sep, peak de Fiestas Patrias). "
                 "Queda 'pendiente' hasta que Víctor lo confirme: el patrón cambió dentro del mes y octubre sigue casi sin carga en B."),
        "almacenamiento": ProcesoEnTurno(
            True, "confirmado", f"{_LOGO}: 129 tareas por día (61%; 22-sep a 1-oct 82, 35%; antes del 22-sep 230, 77%), activa 7 de 7 días",
            nota="En octubre B vuelve a ser el turno PRINCIPAL del almacenamiento. Driver y cuota en capacity.py. Tiempo estándar `ciclo_total` validado en B (WMS)."),
        "retorno_fletero": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 90,4 tareas por día (85% del proceso), {_VIG}",
            nota="Es el turno del retorno de fletero. MUY irregular: de 43 a 180 tareas por día (26 a 238 desde el 22-sep). Rol por confirmar con Víctor: se asigna a Operario "
                 "(43% cuentas genéricas de bodega, 47% personales, 10% cuentas de gruero)."),
        "merma": ProcesoEnTurno(
            True, "confirmado", f"{_LOGM}: 4,0 tareas por día (11% del proceso), {_VIG}",
            nota="Rol por confirmar con Víctor: se asigna a Operario (32% de las tareas de B las hacen cuentas de gruero)."),
    },
}

# Régimen ANTERIOR (4 a 21-sep-2026): incluye la semana peak de Fiestas Patrias (14-17-sep).
# HISTÓRICO/ATÍPICO: no describe el estado actual del CD. Se conserva para comparar y para
# poder explicar por qué cambiaron las cuotas.
PROCESOS_POR_TURNO_HISTORICO: dict[str, dict[str, ProcesoEnTurno]] = {
    "C": {
        "picking": ProcesoEnTurno(True, "historico", f"{_LOG}: 97% de las líneas, 4-21-sep (14 días operados)"),
        "reabasto_pallet": ProcesoEnTurno(True, "historico", f"{_LOG}: 15% de los palés"),
        "reabasto_caja": ProcesoEnTurno(True, "historico", f"{_LOG}: 0% de los retiros de caja"),
        "carga_camion": ProcesoEnTurno(True, "historico", f"{_LOG}: 21,0 trailers por noche (conteo directo, media de 14 noches 4-21 sep, rango 17-22)",
                                       volumen_fijo=21.0, unidad="camiones", n_volumen=14),
        "recepcion_directo_stage": ProcesoEnTurno(False, "historico", f"{_LOG}: 0,3 pallets por noche"),
        "almacenamiento": ProcesoEnTurno(False, "historico", f"{_LOG}: 4% de las tareas"),
        "retorno_fletero": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 0,2 tareas por noche"),
        "merma": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 13,8 tareas por noche"),
    },
    "A": {
        "picking": ProcesoEnTurno(True, "historico", f"{_LOG}: 3% de las líneas (derrame del peak 14-16-sep)"),
        "reabasto_pallet": ProcesoEnTurno(True, "historico", f"{_LOG}: 48% de los palés"),
        "reabasto_caja": ProcesoEnTurno(True, "historico", f"{_LOG}: 81% de los retiros de caja"),
        "recepcion_directo_stage": ProcesoEnTurno(True, "historico", f"{_LOG}: 48% de los pallets"),
        "carga_camion": ProcesoEnTurno(True, "historico", f"{_LOG}: carga en A"),
        "almacenamiento": ProcesoEnTurno(True, "historico", f"{_LOG}: 20% de las tareas"),
        "retorno_fletero": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 3,2 tareas por día"),
        "merma": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 18,2 tareas por día"),
    },
    "B": {
        "picking": ProcesoEnTurno(False, "historico", f"{_LOG}: 0,4 líneas por día"),
        "reabasto_pallet": ProcesoEnTurno(True, "historico", f"{_LOG}: 37% de los palés"),
        "reabasto_caja": ProcesoEnTurno(True, "historico", f"{_LOG}: 19% de los retiros de caja"),
        "recepcion_directo_stage": ProcesoEnTurno(True, "historico", f"{_LOG}: 52% de los pallets"),
        "carga_camion": ProcesoEnTurno(True, "historico", f"{_LOG}: carga en B en 11 días (hasta 25 trailers el 15-sep)"),
        "almacenamiento": ProcesoEnTurno(True, "historico", f"{_LOG}: 76% de las tareas"),
        "retorno_fletero": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 5,1 tareas por día"),
        "merma": ProcesoEnTurno(True, "historico", f"{_LOGM}: 10-21 sep, 0,9 tareas por día"),
    },
}


# La parte del GRUERO del reabasto de cajas (bajar el pallet para que el operario reponga) ocurre donde ocurre el reabasto de cajas y con su mismo volumen
# (una reposición servida = una reposición). Se deriva aquí para que no pueda desincronizarse de la fila del operario.
for _tabla in (PROCESOS_POR_TURNO, PROCESOS_POR_TURNO_HISTORICO):
    for _d in _tabla.values():
        _rc = _d.get("reabasto_caja")
        if _rc is not None:
            _d["reabasto_caja_grua"] = ProcesoEnTurno(
                _rc.ocurre, _rc.estado, "igual que reabasto_caja: " + _rc.fuente,
                nota="Parte del gruero del reabasto de cajas (bajar los pallets). Tiempo de la planilla oficial: 6 visitas, 4 días, 1 gruero (insuficiente).")


# Régimen INTERMEDIO (22-sep a 1-oct): el «vigente» de las versiones del modelo anteriores al 2026-10-13. Mismas reglas de ocurrencia que el vigente salvo que el almacenamiento
# nocturno era < 4% (False, confirmado por David) y la carga del turno C se cuenta con las noches de esa ventana. Los volúmenes y las cuotas están en `calibracion.perfil("intermedio")`.
# Sirve para evaluar (backtest) los días de septiembre posteriores al 22 con la práctica que de verdad tenían, no con la de octubre.
from dataclasses import replace as _replace

_cn_i, _n_i = _cal.carga_noche_c("2026-09-22", "2026-10-02") if _cal.disponible() else (22.4, 9)
PROCESOS_POR_TURNO_INTERMEDIO: dict[str, dict[str, ProcesoEnTurno]] = {
    t: {k: _replace(v, fuente="régimen intermedio (22-sep a 1-oct): ocurrencia como en el vigente; volúmenes y cuotas en calibracion.perfil('intermedio')", nota="")
        for k, v in d.items()} for t, d in PROCESOS_POR_TURNO.items()}
PROCESOS_POR_TURNO_INTERMEDIO["C"]["almacenamiento"] = ProcesoEnTurno(
    False, "confirmado", f"{_DAVID} (prácticamente no ocurre); {_LOG}: 4% de las tareas, 22-sep a 1-oct")
PROCESOS_POR_TURNO_INTERMEDIO["C"]["carga_camion"] = _replace(
    PROCESOS_POR_TURNO["C"]["carga_camion"], volumen_fijo=round(_cn_i, 2), n_volumen=_n_i,
    fuente=f"{_LOG}: {_cn_i:.1f} trailers por noche, media de {_n_i} noches (22-sep a 1-oct)", nota="")


# Régimen COMBINADO (2026-10-16): 4-sep a 10-oct, los tres regímenes juntos (pedido de la jefatura: no separar la práctica en regímenes). Un proceso "ocurre" en un turno si en esa ventana
# tiene ≥ 0,5 tareas por día en promedio; los volúmenes y cuotas salen de `calibracion.perfil('combinado')`. La carga nocturna (volumen fijo) es la del vigente.
PROCESOS_POR_TURNO_COMBINADO: dict[str, dict[str, ProcesoEnTurno]] = {
    t: {k: _replace(v, fuente="régimen combinado (4-sep a 10-oct): volúmenes y cuotas en calibracion.perfil('combinado')", nota="") for k, v in d.items()} for t, d in PROCESOS_POR_TURNO.items()}
if _cal.disponible():
    _pc = _cal.perfil("combinado")
    for _t in ("A", "B", "C"):
        for _k, _v in list(PROCESOS_POR_TURNO_COMBINADO[_t].items()):
            _p = _pc.get(_k)
            if _p and _v.volumen_fijo is None and _v.ocurre is not True and _p["por_turno"][_t] >= 0.5:
                PROCESOS_POR_TURNO_COMBINADO[_t][_k] = _replace(_v, ocurre=True, estado="pendiente", fuente=f"régimen combinado: {_p['por_turno'][_t]:.1f} tareas por día en promedio (4-sep a 10-oct)")


# CARGA (2026-10-21, decisión del usuario): vueltas de venta; 22 fijas en el turno C y el resto en el A; nada en el B. Igual en todos los regímenes.
for _tabla in (PROCESOS_POR_TURNO, PROCESOS_POR_TURNO_HISTORICO, PROCESOS_POR_TURNO_INTERMEDIO, PROCESOS_POR_TURNO_COMBINADO):
    _tabla["C"]["carga_camion"] = _replace(_tabla["C"]["carga_camion"], volumen_fijo=_pd.VUELTAS_TURNO_C, n_volumen=_N_CARGA_C, unidad="vueltas",   # el log cuenta 22,6 trailers por noche (17 noches)
                                           fuente="Excel de Diana (`CDC`!B15): 22 vueltas fijas en el turno C [decisión del usuario, 2026-10-21]")
    _tabla["A"]["carga_camion"] = _replace(_tabla["A"]["carga_camion"], ocurre=True, estado="confirmado")
    _tabla["B"]["carga_camion"] = _replace(_tabla["B"]["carga_camion"], ocurre=False, estado="confirmado", fuente="decisión del usuario (2026-10-21): las vueltas fuera de C se cargan en A")


def procesos_que_ocurren(turno: str) -> dict[str, ProcesoEnTurno]:
    """Procesos con ocurre=True en un turno."""
    return {k: v for k, v in PROCESOS_POR_TURNO[turno].items() if v.ocurre is True}


def procesos_inciertos(turno: str) -> dict[str, ProcesoEnTurno]:
    """Procesos de los que no se sabe si ocurren (ocurre=None)."""
    return {k: v for k, v in PROCESOS_POR_TURNO[turno].items() if v.ocurre is None}


@dataclass(frozen=True)
class Turno:
    codigo: str
    hora_inicio: int
    hora_fin: int
    dotacion: dict[str, int]        # rol -> personas, de la fuente "turnos reales"


@lru_cache
def dotacion_real_por_turno() -> dict[str, Turno]:
    """Dotación real (bloque 'Turnos reales 24-09 al 28-09' de la planilla; ver
    dotacion.py). Es la línea base de personal contra la que se compara cualquier
    dotación que proponga el modelo."""
    filas = dotacion.cargar_turnos_reales()
    out: dict[str, dict[str, int]] = {t: {} for t in TURNOS}
    for d in filas:
        out.setdefault(d.turno, {})[d.rol] = d.personas
    return {t: Turno(t, *HORARIO_TURNO[t], out[t]) for t in TURNOS}


@lru_cache
def equipos_disponibles() -> dict[str, int]:
    """Techo físico real de grúas y transpaletas (planilla, contado a las 15:30).
    Es más estrecho que el límite del WMS (9999 = "sin límite sistémico", ver
    wms_config.py) y aplica igual a todos los turnos salvo que se indique lo contrario
    (no hay evidencia de que el número de equipos cambie por turno)."""
    equipos = dotacion.cargar_equipos()
    out: dict[str, int] = {}
    for e in equipos:
        if e.tipo in ("Grúas", "Transpaletas"):
            out[e.tipo] = e.cantidad
    return out


@lru_cache
def config_wms() -> wms_config.ConfigWMS:
    return wms_config.cargar()


# =====================================================================================================
# FUNCIONES (2026-10-06, decisión del equipo): los ROLES siguen siendo solo Operario y Gruero. Una FUNCIÓN es lo que hace una persona con su
# rol (picking, reposición, recepción, almacenamiento, carga); sirve para DESGLOSAR N, no para crear roles. «El Charter habla de pickers y
# reponedores; son funciones de los mismos operarios»: el Operario hace picking y reposición de cajas; el Gruero hace reposición de pallets,
# recepción, almacenamiento y carga. FUERA DEL MODELO (declarado): los parrilleros (CMS, Clean Services) y consolidadores, que traen cargas y
# consolidan pallets; no tienen tiempo estándar ni driver y no entran en ningún N.
# =====================================================================================================
FUNCION_POR_PROCESO = {
    "picking": "picking",
    "reabasto_caja": "reposición",
    "reabasto_caja_grua": "reposición",
    "retorno_fletero": "retorno de fletero",
    "merma": "merma",
    "retorno_fletero_gruero": "retorno de fletero", "carga_cl": "carga", "recogida_pallets": "carga", "traslado_mixtos": "traslado de mixtos", "traslados_operario": "traslados", "traslados_gruero": "traslados", "otras_actividades_operario": "otras actividades", "otras_actividades_gruero": "otras actividades",
    "reabasto_pallet": "reposición",
    "recepcion_directo_stage": "recepción",
    "traslado_stage": "recepción",
    "almacenamiento": "almacenamiento",
    "carga_camion": "carga",
}
FUNCIONES = ("picking", "reposición", "recepción", "almacenamiento", "carga", "retorno de fletero", "merma", "traslados", "traslado de mixtos", "otras actividades")
NOTA_FUNCIONES = ("El Charter habla de pickers y reponedores; son funciones de los mismos operarios (los roles del modelo siguen siendo Operario y Gruero). "
                  "Parrilleros (CMS) y consolidadores quedan fuera del modelo: traen cargas y consolidan pallets, sin tiempo estándar ni driver.")
FUERA_DEL_MODELO = ("Parrillero (CMS)", "Consolidador")


# =====================================================================================================
# R COMO FUNCIÓN DE LA FECHA (2026-10-06). La dotación real YA NO es una constante: sale de
#   - TURNO C: el roster programado por semana (`dotacion_historica`, Excel de turnos; solo CONTEOS, sin nombres:
#     `salidas/limpio/dotacion_dia_turno_rol.csv`), para la semana pedida; por defecto la ÚLTIMA semana disponible;
#   - TURNOS A y B (Gruero y Operario): el ODS de septiembre (headcount por empresa, cargo y turno), porque el roster no
#     nombra a los contratistas de A y B. Mapeo [provisional]: Gruero = CCU «Gruero-Operario» + Recifast «Gruero»;
#     Operario = CCU «Auxiliar Operac. de Bodega»; los contratistas (Recifast/Papic) quedan fuera del rol Operario;
#   - cualquier otro rol (Planificador, Chequeador en A/B, Subestándar…): la planilla de terreno del 24-28 sep, como antes.
# INTERMEDIO y D CUENTAN EN A [provisional, pendiente de confirmar con Víctor]; el roster los trae con su `subturno` visible
# (`detalle_roster`). Si el CSV del roster no existe, todo cae a la planilla y la fuente lo dice.
# Verificación ODS vs roster (2026-10-06): el C del ODS (26 = Despachador 3 + Gruero-Operario 4 + Auxiliar 15 + Papic parrillero 4)
# calza con el roster de las semanas del 21-sep al 3-oct (15 operarios + 4 grueros + 4 parrilleros + 3 de supervisión = 26).
# HISTÓRICO: Operario C = 12 en agosto (roster 17-ago a 29-ago; 11 del 3 al 15-ago) y 12 en la planilla de terreno del 24 al 28-sep.
# =====================================================================================================
ROSTER_CSV = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "dotacion_dia_turno_rol.csv"
ROLES_ODS = ("Gruero", "Operario")
HISTORICO_OPERARIO_C = (
    {"desde": "2026-08-03", "hasta": "2026-08-15", "operarios_c": 11, "fuente": "roster programado (hojas 3-8 y 10-15 ago)"},
    {"desde": "2026-08-17", "hasta": "2026-08-29", "operarios_c": 12, "fuente": "roster programado (hojas 17-22 y 24-29 ago)"},
    {"desde": "2026-09-24", "hasta": "2026-09-28", "operarios_c": 12, "fuente": "planilla de terreno «Personal (Turnos)», bloque «Turnos reales 24-09 al 28-09»"},
)


def _lunes(f) -> "dt.date":
    import datetime as dt
    f = dt.date.fromisoformat(f) if isinstance(f, str) else f
    return f - dt.timedelta(days=f.weekday())


@lru_cache
def _roster() -> dict:
    """{lunes_iso: {turno: {rol: {'personas': n, 'subturnos': 'A+INT+D', 'hoja': ...}}}} desde el CSV de conteos (sin nombres)."""
    import csv
    if not ROSTER_CSV.exists():
        return {}
    out: dict = {}
    with open(ROSTER_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sem = _lunes(r["fecha"]).isoformat()
            d = out.setdefault(sem, {}).setdefault(r["turno"], {}).setdefault(r["rol"], {"personas": 0, "subturnos": set(), "hoja": r["hoja"], "_fecha": r["fecha"]})
            if r["fecha"] == d["_fecha"]:                       # una sola fecha por semana (la misma dotación de lunes a sábado)
                d["personas"] += int(r["personas"])
                d["subturnos"] |= set(r["subturnos"].split("+"))
    for sem in out.values():
        for t in sem.values():
            for d in t.values():
                d["subturnos"] = "+".join(sorted(d["subturnos"]))
    return out


def semanas_dotacion() -> list[str]:
    """Lunes (ISO) de cada semana con roster, de la más antigua a la más reciente."""
    return sorted(_roster())


def semana_dotacion_defecto() -> str | None:
    """Última semana disponible del roster (por defecto del estimador)."""
    s = semanas_dotacion()
    return s[-1] if s else None


def etiqueta_semana(semana: str) -> str:
    import datetime as dt
    d = dt.date.fromisoformat(semana)
    return f"{d:%d-%m} al {d + dt.timedelta(days=5):%d-%m-%Y}"


@lru_cache
def dotacion_ods() -> dict[str, dict[str, int]]:
    """Headcount del ODS de septiembre por turno para los roles del modelo (mapeo provisional, ver arriba)."""
    filas = dotacion.cargar_dotacion_septiembre()
    out = {t: {"Gruero": 0, "Operario": 0} for t in TURNOS}
    for f in filas:
        vals = {"A": f.turno_a, "B": f.turno_b, "C": f.turno_c}
        if (f.empresa == "CCU" and f.cargo == "Gruero-Operario") or (f.empresa == "Recifast" and f.cargo == "Gruero"):
            for t, v in vals.items():
                out[t]["Gruero"] += v
        elif f.empresa == "CCU" and f.cargo.startswith("Auxiliar Operac"):
            for t, v in vals.items():
                out[t]["Operario"] += v
    return out


def dotacion_c_ods() -> int:
    """Headcount total del turno C en el ODS (26), para contrastar con el roster."""
    return sum(f.turno_c for f in dotacion.cargar_dotacion_septiembre())


# Operarios reales de los turnos A y B (dato del usuario, 2026-10-21): hay 4 por turno. El ODS solo cuenta al Auxiliar de bodega CCU (1 / 0) y deja fuera a los operarios de los
# contratistas (Recifast 4 / 2, Papic 2 / 2), por eso subestimaba. Pendiente de confirmar con Víctor quiénes hacen reabasto de cajas, retorno de fletero y merma.
OPERARIOS_AB = {"A": 4, "B": 4}
# DOTACIÓN REAL de grueros y operarios (dato del usuario, 2026-10-23; demanda normal y baja): grueros 4 / 3 / 4 y operarios 4 / 4 / 12 en A / B / C.
# Manda sobre el roster y el ODS para estos dos roles.
DOTACION_REAL = {("A", "Gruero"): 4, ("B", "Gruero"): 3, ("C", "Gruero"): 4, ("A", "Operario"): 4, ("B", "Operario"): 4, ("C", "Operario"): 14}
# [2026-10-24] CONFIRMADO por el usuario: Operario C = 14 (antes 12) y 6 grúas por turno. POR CONFIRMAR: el resto de la tabla (grueros A/B/C y operarios A/B); se dejan como estaban.
DOTACION_REAL_POR_CONFIRMAR = {("A", "Gruero"), ("B", "Gruero"), ("C", "Gruero"), ("A", "Operario"), ("B", "Operario")}


# =====================================================================================================
# PARÁMETROS DE LA CAPA DE OPTIMIZACIÓN (modelo/optimizacion.py, 2026-10-23; formulación en docs_modelo/formulacion_modelo_final.tex)
# Se derivan de lo que ya usa el estimador: no agregan valores nuevos salvo el tope R.
# =====================================================================================================
ROLES_OPTIMIZACION = ("Gruero", "Operario")
# G_s: grúas disponibles por turno. CONFIRMADO por el usuario el 2026-10-24: 6 por turno (la planilla de terreno contó 4 a las 15:30, dato anterior).
GRUAS_POR_TURNO = {"A": 6, "B": 6, "C": 6}
# M: NÚMERO DE RÉPLICAS de la simulación del día, ÚNICO para todo el proyecto (decisión del usuario, 2026-11-20). Fuente: modelo/analisis_replicas.py
# (piloto de 2.000 réplicas × 3 dotaciones × 2 días de septiembre; salidas/replicas/). Es el menor M de la grilla {50, ..., 2.000} que cumple:
# (1) semiamplitud de Wilson evaluada en p = 0,95 ≤ 0,015 (con 1.000: 0,0136; con 750: 0,0157);
# (2) la decisión viable/no viable coincide con la de 2.000 réplicas en ≥ 95% de 1.000 submuestras bootstrap, en las combinaciones con p fuera de [0,93; 0,97] (100%);
# (3) una decisión completa (3 dotaciones, 4 procesos, día de 42.151 cajas) tarda ≤ 30 min (8,1 min).
# 2026-11-23: el umbral pasa a p_min = 0,90 y se MANTIENEN 1.000 réplicas (decisión del usuario): en p = 0,90 la semiamplitud de Wilson es 0,0186 (±1,9 puntos),
# sobre el 0,015 del criterio (1); para llegar a 0,015 harían falta ≈ 1.550. El criterio (1) se declara ≤ 0,02 con el umbral nuevo.
N_REPLICAS = 1000
# R_{r,s}: TOPE de dotación (sin contratación). [DECISIÓN 2026-10-23] = la dotación real (DOTACION_REAL). El prompt externo proponía gruero 5/4/5 y operario 5/5/16
# «con contratistas, por confirmar», sin fuente en el repositorio: se deja como alternativa declarada, NO por defecto.
TOPE_DOTACION = dict(DOTACION_REAL)
TOPE_DOTACION_ALTERNATIVO_POR_CONFIRMAR = {("A", "Gruero"): 5, ("B", "Gruero"): 4, ("C", "Gruero"): 5, ("A", "Operario"): 5, ("B", "Operario"): 5, ("C", "Operario"): 16}


def elegibilidad() -> dict[tuple[str, str], int]:
    """E_{k,r}: 1 si el rol r ejecuta la tarea k (AS-IS: un solo rol por tarea, desde capacity.ROL_POR_PROCESO). Import diferido: capacity importa config."""
    from modelo import capacity
    return {(k, r): int(rol == r) for k, rol in capacity.ROL_POR_PROCESO.items() for r in ROLES_OPTIMIZACION}


def uso_grua() -> dict[str, float]:
    """g_k: fracción del tiempo de la tarea k que ocupa una grúa, desde capacity.USO_DE_EQUIPO (1,0 en las tareas de gruero con grúa [supuesto]; 0 en el resto,
    incluido el retorno de fletero del gruero, que no usa grúa)."""
    from modelo import capacity
    return {k: float((capacity.USO_DE_EQUIPO.get(k) or {}).get("Grúas") or 0.0) for k in capacity.ROL_POR_PROCESO}


def personas_turno(turno: str, rol: str, semana: str | None = None) -> int:
    """Dotación real de un rol en un turno para una semana (lunes ISO o fecha; por defecto, la última del roster).
    C: roster programado. A y B (Gruero, Operario): ODS. Otros roles: planilla de terreno (0 si el rol no existe en ese turno)."""
    if (turno, rol) in DOTACION_REAL:
        return DOTACION_REAL[(turno, rol)]
    ro = _roster()
    sem = _lunes(semana).isoformat() if semana else semana_dotacion_defecto()
    if turno == "C" and sem in ro and rol in ro[sem].get("C", {}):
        return ro[sem]["C"][rol]["personas"]
    if turno == "C" and sem in ro and rol in ("Operario", "Gruero", "Parrillero", "Planificador", "Chequeador"):
        return 0
    if turno in ("A", "B") and rol == "Operario":
        return OPERARIOS_AB[turno]
    if turno in ("A", "B") and rol in ROLES_ODS:
        return dotacion_ods()[turno][rol]
    return dotacion_real_por_turno()[turno].dotacion.get(rol, 0)


def fuente_dotacion(semana: str | None = None) -> dict[str, str]:
    """De dónde sale R por turno, para mostrarlo junto al resultado."""
    sem = _lunes(semana).isoformat() if semana else semana_dotacion_defecto()
    c = (f"roster programado, semana {etiqueta_semana(sem)} (Excel de turnos, solo conteos)" if sem in _roster() else "planilla de terreno 24-28 sep (no hay roster)")
    ab = "Gruero: ODS de septiembre (CCU + Recifast); Operario: 4 por turno (dato del usuario, 2026-10-21; el ODS solo contaba al auxiliar de bodega CCU)"
    return {"A": ab + "; Intermedio y D cuentan en A [provisional, pendiente de Víctor]", "B": ab, "C": c}


def detalle_roster(semana: str | None = None) -> dict:
    """Roster de la semana por turno y rol con su `subturno` (A, INT, D…): para ver qué hay detrás de cada cifra. {} si no hay roster."""
    sem = _lunes(semana).isoformat() if semana else semana_dotacion_defecto()
    return {t: {r: (d["personas"], d["subturnos"]) for r, d in roles.items()} for t, roles in _roster().get(sem, {}).items()}


def total_personas(turno: str | None = None) -> int:
    """Total de la PLANILLA de terreno (24-28 sep), no del roster."""
    turnos = TURNOS if turno is None else (turno,)
    return sum(sum(dotacion_real_por_turno()[t].dotacion.values()) for t in turnos)


if __name__ == "__main__":
    print(f"Umbral de criticidad de recurso: {UMBRAL_UTILIZACION_CRITICA} (ver docstring: pendiente de validar)")
    for t in TURNOS:
        tu = dotacion_real_por_turno()[t]
        print(f"Turno {t} ({tu.hora_inicio:02d}-{tu.hora_fin:02d}h): {sum(tu.dotacion.values())} personas -> {tu.dotacion}")
    print("Equipos (techo físico, todos los turnos):", equipos_disponibles())
    print("Equivalencia de roles (inferida, por confirmar):", EQUIVALENCIA_ROLES)
