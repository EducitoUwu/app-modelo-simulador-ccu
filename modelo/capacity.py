"""Capa analítica: dada una demanda, cuánta gente hace falta por turno y rol.

Es el entregable central del Charter ("dada una demanda X, entregar la dotación correcta
por día, turno y rol"). No simula: calcula. La simulación (`simulation.py`, pendiente)
después refina este punto de partida bajo variabilidad y contención.

FÓRMULA
    horas_de_trabajo      = volumen_del_proceso x tiempo_estándar
    horas_persona_necesarias = horas_de_trabajo / eficiencia
    personas_necesarias   = horas_persona_necesarias / duración_del_turno

  DECISIÓN DEL EQUIPO (2026-10-13): la duración del turno que entra a la fórmula es de 7 HORAS EFECTIVAS (de las 8 se descuentan colación, charlas y tiempos muertos) y la
  «eficiencia» es la UTILIZACIÓN OBJETIVO de 0,85: N = horas de trabajo / (7 x 0,85). Antes: 8 x 0,848. Todas las cifras de N son ≈14% mayores (8 x 0,848 / 7 x 0,85 = 1,14).

  Nota sobre la fórmula acordada: en `CLAUDE.md` está escrita como
  `Volumen x T_estándar / (Tiempo_disponible x Eficiencia)` y luego
  `Personas = Horas / Duración_turno`. Aplicada literalmente divide dos veces por el
  tiempo del turno y da personas/hora, no personas. Acá se implementa la versión
  dimensionalmente correcta, que es la misma idea: el tiempo disponible entra **una vez**,
  como divisor final.

VALIDACIÓN CONTRA LA REALIDAD (lo que hace creíble a este módulo)
  Con los datos de picking del 20-ago al 24-sep:
    152.225 líneas x 50,3 s = 2.128 h de trabajo
    2.128 h / 0,848 de eficiencia = 2.509 horas-persona calculadas
    contra 2.481 horas-persona realmente observadas en el log  ->  error de 1,1%
  O sea: la fórmula reproduce el turno noche real con datos independientes. Es la
  prueba `test_capacity.py::test_reproduce_las_horas_persona_reales_de_picking`.

AGREGACIÓN POR (ROL, TURNO) — corrección del 2026-09-30
  El cálculo por proceso (`requerimiento`) es un **diagnóstico**, no una dotación: las
  mismas personas cubren varios procesos dentro de un turno, así que comparar cada
  proceso contra la dotación completa del rol cuenta a la misma gente varias veces (en
  turno C, picking daba "−2,0 operarios" y reabasto de cajas "−12,0", como si los 12
  operarios fueran 24). La cifra que se reporta como dotación es `dotacion_por_turno`:
      horas_trabajo(rol, turno) = suma de horas_trabajo de TODOS los procesos del rol
      personas(rol, turno)      = horas_trabajo(rol, turno) / (duración_turno x eficiencia)
  dividiendo **una sola vez al final**. Lo mismo para equipos compartidos (grúas). La
  confianza de la cifra agregada es la **más baja** de los procesos que la componen.

LO QUE ESTE MÓDULO NO HACE, A PROPÓSITO
  - **No optimiza.** No existe meta de Fill Rate confirmada (brecha #7, David con
    Ernesto/Milton), así que `Requerimiento.meta_fill_rate` queda en `None` y el módulo
    entrega la **brecha contra la dotación real**, no un óptimo. Inventar una meta sería
    inventar el criterio de éxito del proyecto.
  - **No mezcla confianzas.** Cada resultado hereda el peor estado de los parámetros y
    del driver que usó. Un número de `reabasto_caja` (N=5, una jornada) no se presenta
    igual que uno de picking (N=131.428, 28 días).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

from modelo import calibracion, config, fit_params, listas_picking, parametros_diana, recepcion_pallets

# ---------------------------------------------------------------- confianza

# De mejor a peor. El resultado hereda el peor de sus insumos.
ORDEN_CONFIANZA = ("validado", "confiable", "preliminar", "insuficiente",
                   "sustituto_dia", "sin_datos")


def peor_confianza(*estados: str) -> str:
    return max((e for e in estados if e), key=ORDEN_CONFIANZA.index, default="sin_datos")


# ---------------------------------------------------------------- drivers

@dataclass(frozen=True)
class Driver:
    """Cuántas unidades de trabajo genera una caja vendida, y de dónde sale esa cifra.

    `volumen_dia`: si no es None, el proceso NO escala con la demanda (en septiembre no hubo relación
    detectable entre su actividad diaria y las cajas facturadas) y su volumen es ese nivel medio diario,
    se venda lo que se venda. `unidades_por_caja` queda como tasa media de referencia (tareas por caja
    facturada al volumen medio del mes), pero NO se multiplica por la demanda."""
    unidades_por_caja: float
    unidad: str
    fuente: str
    estado: str
    volumen_dia: float | None = None
    relacion: str = ""          # confianza de la relación actividad ~ facturado (demanda_actividad.py)
    intercepto: float = 0.0     # volumen = max(0, intercepto + unidades_por_caja x demanda): carga de camión
    sigma: float | None = None  # desvío residual de la regresión, en la unidad del driver


# =====================================================================================================
# DRIVERS (2026-10-04): FACTURADO REAL de septiembre-2026, ya no "recepción como proxy de venta".
# Salen de `modelo/demanda_actividad.py` (drivers_facturado.csv): actividad diaria del WMS (todos los turnos)
# contra las cajas FACTURADAS del mismo día (desfase 0, elegido con evidencia: ver el informe
# salidas/demanda_vs_actividad.md), 22 días operados del 4 al 30-sep, con los dos polos del mes incluidos
# (calibración 2: volumen de demanda = septiembre completo). Se recalculan en tests/test_capacity.py.
#
# Solo PICKING escala con la demanda (líneas ~ facturado: R²=0,86, p<0,001, `validado`). Reabasto de pallets
# (p=0,54), reabasto de cajas (p=0,94), almacenamiento (p=0,10) y recepción en camiones estimados (p=0,16)
# NO muestran relación con las cajas facturadas del día: dimensionarlos multiplicando la demanda por una
# tasa haría crecer su trabajo en los días peak cuando el dato dice que no crece. Para esos procesos el
# volumen es su NIVEL MEDIO DIARIO (`volumen_dia`), con estado `preliminar` (el nivel está medido en 22 días;
# lo que no está probado es cómo escala). Las cifras se interpretan como "a demanda de septiembre".
# Carga de camión no usa driver: su volumen por turno está en config.PROCESOS_POR_TURNO (conteo directo).
# PICKING SE MIDE POR LISTAS, NO POR LÍNEAS (decisión de la jefatura, 2026-10-16). Una lista = el volante de un operario (varias líneas, casi siempre de un pasillo, que arman un pallet);
# en el log = las líneas hacia un mismo LPN de destino (`modelo/listas_picking.py`, hipótesis a confirmar). El driver y el tiempo se convierten SIN cambiar las horas de trabajo:
# listas por 1.000 cajas (22 días de septiembre con venta) y líneas por lista = líneas ÷ listas de esos mismos días; tiempo por lista = tiempo por línea × líneas por lista.
def _datos_listas() -> tuple[float, float, float]:
    r = listas_picking.cargar_resumen()
    d = (r or {}).get("driver")
    if not d:
        return 23.0, 204.18 / 23.0, 464.61 / (204.18 / 23.0)          # respaldo si falta listas_picking_resumen.json (correr modelo.listas_picking)
    return d["listas_por_1000_cajas"], d["lineas_por_1000_cajas"] / d["listas_por_1000_cajas"], d["sigma_listas"]


LISTAS_POR_1000, LINEAS_POR_LISTA, SIGMA_LISTAS = _datos_listas()
_FV = "ventas septiembre-2026 (FACTURADO) vs log WMS, 22 días operados (4 al 30-sep), desfase 0"

# OJO: LOS VOLÚMENES DE REABASTO DE ESTOS DRIVERS SOLO CUENTAN LO QUE DISPARA EL WMS (2026-10-02). Hay una vía de disparo
# adicional que el log nunca ve: la herramienta de los planificadores indica a mano las reposiciones de posiciones que
# no alcanzan el umbral del WMS. El 2-oct listó 46 pallets (hoja de las 14:46) y 191 reposiciones de cajas (2.157 cajas, hoja de las
# 14:44) de las que, por la regla confirmada por el equipo (> 5 cajas), se ejecutarían ≈84 (1.883 cajas), contra ≈90 pallets y
# ≈36 retiros de caja por día en el log del régimen vigente. Es UNA jornada,
# confianza `insuficiente`, no se reparte por turno (una sola hora de emisión por hoja) y aporta frecuencia, nunca tiempo,
# así que NO entra en estos cálculos; solo queda declarado que el volumen real de reabasto está subestimado.
# Detalle y advertencias en `reabasto_planificadores.py`.
DRIVERS: dict[str, Driver] = {
    # 5.452,6 líneas/día contra 26.704,6 cajas facturadas/día = 204,18 líneas por 1.000 cajas
    # (p10-p90 de la razón diaria 189-226). R² 0,86, p<0,001, n=22: escala con la demanda.
    # sigma = desvío residual de la regresión lineal líneas ~ facturado del total del día (464,6 líneas); solo se usa
    # para dar un RANGO (±1 sigma), el valor central sigue siendo la tasa media.
    "picking": Driver(LISTAS_POR_1000 / 1000, "listas de picking", _FV + "; lista = líneas hacia un mismo LPN de destino (listas_picking.py)", "validado", None, "validado", sigma=SIGMA_LISTAS),
    # 91,5 tareas/día (3,43 por 1.000 cajas al volumen medio). Sin relación con el facturado (R² 0,02, p=0,54).
    "reabasto_pallet": Driver(91.50 / 26_704.59, "tareas de reabasto", _FV, "preliminar", 91.50, "sin_relacion"),
    # 45,2 retiros/día (1,69 por 1.000). Sin relación (R² 0,00, p=0,94). Solo lo que dispara el WMS (ver arriba).
    "reabasto_caja": Driver(45.18 / 26_704.59, "tareas de reabasto", _FV, "preliminar", 45.18, "sin_relacion"),
    # OJO CON LA UNIDAD: los tiempos de `recepcion_directo_stage` que hay en la planilla son por CAMIÓN
    # completo, no por pallet (con el driver en pallets el cálculo se va ~26x). Camiones ESTIMADOS por huecos
    # de 30 min entre pallets (método de recepcion_log): 9,23 por día; sin relación con el facturado (p=0,16).
    # Los pallets recepcionados sí escalan algo (R² 0,44), pero la estimación de camiones no lo capta.
    "recepcion_directo_stage": Driver(9.23 / 26_704.59, "camiones", _FV, "preliminar", 9.23, "sin_relacion"),
    # Tareas STREC -> ubicación final (las mismas que miden su tiempo): 262,9 por día (9,84 por 1.000). Sin
    # relación con el facturado del día (R² 0,13, p=0,10); antes se calculaba contra la recepción (r=0,74).
    "almacenamiento": Driver(262.86 / 26_704.59, "tareas de almacenamiento (pallets)", _FV, "preliminar", 262.86,
                             "sin_relacion"),
    # MOVIMIENTOS DE INVENTARIO PARCIAL (2026-10-15): retorno de fletero a bodega (RETFLET → posición de picking) y merma (→ MERMA01). Nivel diario del log de octubre (modelo/calibracion.py);
    # sin relación con el facturado de septiembre (17 días: r = −0,36 y 0,43, p > 0,08) => nivel fijo [hipótesis, n pequeño]. El reabasto de cajas por movimiento parcial NO es un proceso aparte:
    # suma al volumen de `reabasto_caja` (ver volumen_reabasto_caja).
    "retorno_fletero": Driver(105.9 / 26_704.59, "retornos de fletero (tareas)", _FV, "preliminar", 105.9, "sin_relacion"),
    "merma": Driver(35.9 / 26_704.59, "movimientos a merma (tareas)", _FV, "preliminar", 35.9, "sin_relacion"),
    # CARGA DE CAMIÓN, vueltas >= 2 (2026-10-04). La vuelta 1 se carga de noche y es casi fija (≈22, en
    # config.PROCESOS_POR_TURNO['C']); las vueltas >= 2 crecen con la venta y se cargan de día (turno A en el
    # régimen vigente). Regresión sobre los 25 días con venta de septiembre: camiones con CARGA >= 2 =
    # -14,77 + 1,2949 por 1.000 cajas facturadas; R² 0,95, p < 0,001, sigma 1,77 camiones. Coincide con los
    # trailers de A + B del log (r = 0,95). Unidad: camiones (= trailers). [hipótesis: CARGA = vuelta; confirmar con Elías]
    "carga_camion": Driver(1.2949 / 1000, "camiones", "ventas septiembre-2026: camiones con CARGA >= 2 ~ FACTURADO, n=25 días",
                           "validado", None, "validado", intercepto=-14.768, sigma=1.774),
}

# =====================================================================================================
# NIVEL DIARIO DE LOS PROCESOS QUE NO ESCALAN, POR RÉGIMEN (2026-10-13). Antes era UN nivel (el promedio de septiembre completo) para los dos regímenes; ahora cada
# régimen trae el suyo, calculado con el log del WMS (`calibracion.perfil`): vigente = octubre (2 al 10-oct), anterior = 4 al 21-sep. El respaldo (septiembre,
# 22 días) solo se usa si falta `actividad_turnos.csv`. Que estos procesos NO escalan con la venta sigue siendo la conclusión de septiembre (único mes con ventas):
# con las ventas de octubre habría que volver a probarlo.
# =====================================================================================================
_NIVELES_RESPALDO = {"reabasto_pallet": 91.50, "reabasto_caja": 45.18, "recepcion_directo_stage": 9.23, "almacenamiento": 262.86, "retorno_fletero": 105.9, "merma": 35.9}
_PROCESOS_CON_NIVEL = tuple(_NIVELES_RESPALDO)


def _niveles(regimen: str) -> dict[str, float]:
    p = calibracion.perfil(regimen)
    return {k: (p[k]["total"] if p else v) for k, v in _NIVELES_RESPALDO.items()}


NIVELES_REGIMEN = {"vigente": _niveles("vigente"), "intermedio": _niveles("intermedio"), "anterior": _niveles("anterior"), "combinado": _niveles("combinado")}
_FL = ("log WMS de octubre-2026 (2 al 10-oct: 7 días completos + la noche del 10), nivel medio diario de todos los turnos; que NO escala con la venta se probó en "
       "septiembre (sin relación con el facturado) — octubre no tiene ventas para repetir la prueba")
_FLP = ("movimientos de inventario parcial (log WMS 10-sep a 14-oct; ventana de octubre, 2 al 10-oct), nivel medio diario de todos los turnos; sin relación con el facturado de septiembre "
        "(17 días, p > 0,08) [hipótesis]")
for _p in _PROCESOS_CON_NIVEL:
    DRIVERS[_p] = replace(DRIVERS[_p], volumen_dia=NIVELES_REGIMEN["vigente"][_p], unidades_por_caja=NIVELES_REGIMEN["vigente"][_p] / 26_704.59,
                          fuente=_FLP if _p in ("retorno_fletero", "merma") else _FL)


# RETORNO DE FLETERO DEL OPERARIO ESCALA CON LA DEMANDA, VÍA EL RECHAZO (2026-10-26, decisión del usuario; `modelo/rechazo_familias.py`). Supuesto: TODO el rechazo de las ventas vuelve por retorno de fletero
# y de ahí a posiciones de picking, menos un 10% que es merma. Tarea = un SKU rechazado en el día. Recta sobre 811 días con venta (2-ene-2024 a 30-sep-2026, sin las lluvias de julio y sin la familia
# OTROS, que rechaza el 91% de lo que factura y no parece producto): tareas = a + b por 1.000 cajas facturadas; el modelo usa el 90% (retorno a picking). El tiempo (152 s) NO cambia. La merma queda con su
# nivel del log (35,0 por día): el 10% de merma del rechazo no se suma a ella. `Ajustes.retorno_fijo=True` recupera el nivel fijo anterior (68,6 por día).
DRIVER_RETORNO_FIJO = DRIVERS["retorno_fletero"]
try:
    import json as _json
    from pathlib import Path
    _RR = _json.loads((Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "retorno_rechazo.json").read_text(encoding="utf-8"))
except Exception:                                   # respaldo: valores de la corrida del 2026-10-26
    _RR = {"tareas_intercepto": 145.568, "tareas_por_1000_cajas": 2.04506, "sigma_tareas": 38.51, "r_tareas": 0.3435, "n_dias": 811, "pct_a_picking": 0.9}
DRIVERS["retorno_fletero"] = Driver(_RR["pct_a_picking"] * _RR["tareas_por_1000_cajas"] / 1000, "retornos de fletero (tareas)",
                                    f"rechazo de las ventas ({_RR['n_dias']} días, 2024 a sep-2026, sin la familia OTROS): 1 tarea por SKU rechazado en el día, {100 * _RR['pct_a_picking']:.0f}% a picking y 10% merma; "
                                    f"recta {_RR['pct_a_picking'] * _RR['tareas_intercepto']:.1f} + {1000 * _RR['pct_a_picking'] * _RR['tareas_por_1000_cajas'] / 1000:.3f} por 1.000 cajas, r {_RR['r_tareas']:.2f}",
                                    "preliminar", None, "rechazo de ventas (supuesto: todo el rechazo vuelve por retorno de fletero)", _RR["pct_a_picking"] * _RR["tareas_intercepto"], _RR["pct_a_picking"] * _RR["sigma_tareas"])


# RECEPCIÓN Y ALMACENAMIENTO ESCALAN CON LA DEMANDA, EN PALLETS (2026-10-18, decisión del equipo; `modelo/recepcion_pallets.py`).
# La recepción se medía en «camiones estimados» (huecos de 30 min), que NO sube con la venta (r = −0,30): los camiones seguidos se funden en uno. Los PALLETS recibidos sí
# escalan (r = 0,66 diario, 0,98 semanal; 0,99 cajas recibidas por caja vendida). Recepción = pallets recibidos por 1.000 cajas facturadas × demanda, con tiempo POR PALLET
# (tiempo del camión ÷ sus pallets, planilla de terreno). Almacenamiento = 1 pallet almacenado por pallet recibido (regla del equipo: no quedan pallets de un día a otro;
# el log empareja 0,69 tareas por pallet, referencia). El reparto por turno sigue saliendo del log de cada proceso (cuotas del régimen).
_RP = recepcion_pallets.cargar()
PALLETS_POR_1000 = _RP["driver"]["pallets_por_1000_cajas"] if _RP else 12.46
N_DIAS_PALLETS = _RP["driver"]["n_dias"] if _RP else 25
S_RECEPCION_PALLET = _RP["tiempo"]["s_por_pallet"] if _RP else 60.4
N_RECEPCION_PALLET = _RP["tiempo"]["n_camiones"] if _RP else 26
ALMACENAMIENTO_POR_PALLET = 1.0          # [regla del equipo, 2026-10-18]; el log da 0,69 (salidas/limpio/recepcion_pallets.json)
ALMACENAMIENTO_LOG_POR_PALLET = _RP["driver"]["almacenamiento_log_por_pallet"] if _RP else 0.69
_FRP = f"pallets recibidos por 1.000 cajas facturadas: {PALLETS_POR_1000:.2f} ({N_DIAS_PALLETS} días de septiembre con recepción y ventas; recepcion_pallets.py)"
DRIVERS["recepcion_directo_stage"] = Driver(PALLETS_POR_1000 / 1000, "pallets recibidos", _FRP, "preliminar", None, "escala (r 0,66 diario, 0,98 semanal)")
DRIVERS["almacenamiento"] = Driver(ALMACENAMIENTO_POR_PALLET * PALLETS_POR_1000 / 1000, "pallets almacenados (1 por pallet recibido)",
                                   _FRP + f"; 1 pallet almacenado por pallet recibido [regla del equipo; el log empareja {ALMACENAMIENTO_LOG_POR_PALLET:.2f}]", "preliminar", None, "regla del equipo")


# =====================================================================================================
# PARÁMETROS FINALES ACORDADOS (2026-10-21; `modelo/parametros_diana.py`, Excel «Carga de trabajo final.xlsx» de Diana + modelo, por decisión del usuario).
# Reemplazan los drivers y tiempos anteriores de recepción, almacenamiento, reabasto de pallets, carga, parte gruero del reabasto de cajas y retorno de fletero del gruero,
# y agregan la CARGA PARA EL CL. El reabasto de cajas del operario, el picking, el retorno de fletero y la merma del operario siguen siendo del modelo.
# =====================================================================================================
_PF = parametros_diana.valores()
_RV = parametros_diana.regresion_vueltas()
_FD = "Excel de Diana «Carga de trabajo final.xlsx» (decisión del usuario, 2026-10-21)"
# RECEPCIÓN EN PALLETS (corregido el 2026-10-22, decisión del usuario): los PALLETS del log están bien (12,46 por 1.000 cajas facturadas, 25 días de septiembre, r 0,66 diario y 0,98
# semanal; 0,99 cajas recibidas por caja vendida) y los CAMIONES estaban mal: el log los estima por huecos de 30 min (≈ 8,8 por día, el mismo número que usó Diana) y funde camiones
# seguidos. Camiones = pallets ÷ 17,5 pallets por camión (mezcla de la planilla: 17 camiones de 10, 10 de 26 y 4 de 28); p. ej. 296 pallets ≈ 7 de 28 + 10 de 10.
# Tiempo: un camión demora 230 s fijos + 46,3 s por pallet (recta sobre 26 camiones cronometrados, r 0,78); con la mezcla, 13 s fijos por pallet + 46 s = ≈ 60 s por pallet.
# Almacenamiento = los mismos pallets recibidos (1 por pallet).
_TR = _RP["tiempo"] if _RP and "s_por_pallet_mezcla" in _RP["tiempo"] else {"s_fijo_camion": 230.0, "s_por_pallet_marginal": 46.3, "pallets_por_camion_mezcla": 17.48, "n_camiones": 26,
                                                                              "s_por_pallet_mezcla": 230.0 / 17.48 + 46.3, "s_camion_mezcla": 230.0 + 46.3 * 17.48, "r_tiempo_pallets": 0.78}
PALLETS_POR_CAMION_CL = _TR["pallets_por_camion_mezcla"]
DRIVERS["recepcion_directo_stage"] = Driver(PALLETS_POR_1000 / 1000, "pallets recibidos", _FRP + f"; ≈ {PALLETS_POR_1000 / PALLETS_POR_CAMION_CL:.2f} camiones por 1.000 cajas "
                                            f"({PALLETS_POR_CAMION_CL:.1f} pallets por camión)", "preliminar", None, "escala (r 0,66 diario, 0,98 semanal)")
DRIVERS["almacenamiento"] = Driver(PALLETS_POR_1000 / 1000, "pallets almacenados (= pallets recibidos)", _FRP + "; 1 pallet almacenado por pallet recibido", "preliminar", None, "regla del equipo")
DRIVERS["reabasto_pallet"] = Driver(_PF["rdp_por_caja"], "pallets repuestos", _FD + f": pallets repuestos por caja vendida ({1000 * _PF['rdp_por_caja']:.3f} por 1.000; escala con la venta)", "preliminar", None, "factor fijo (Diana)")
DRIVERS["carga_camion"] = Driver(_RV["b"], "vueltas", f"vueltas de venta por día = {_RV['a']:.2f} + {1000 * _RV['b']:.4f} por 1.000 cajas facturadas (ventas ene-2025 a sep-2026, n = {_RV['n']}, r = {_RV['r']:.2f}); "
                                 f"{parametros_diana.VUELTAS_TURNO_C:g} fijas en el turno C, el resto en el A", "validado", None, "validado", intercepto=_RV["a"] - parametros_diana.VUELTAS_TURNO_C, sigma=_RV["sigma"])
REPARTO_REABASTO_PALLET = parametros_diana.REPARTO_REABASTO_PALLET
FRACCION_CARGA_CL = parametros_diana.FRACCION_CARGA_CL      # 1/3 de los pallets recibidos (decisión del usuario, 2026-10-22)
DEMANDA_NORMAL_RDC = _PF["demanda_normal_vendida"]           # el reabasto de cajas del log (≈ 87,3 por día) corresponde a esta demanda y escala proporcional a la venta
# Tiempos fijados (s por unidad): de Diana. El Excel no declara N, así que quedan `preliminar`; el retorno (N = 4) queda `insuficiente`.
from modelo import recogida_pallets as _recogida
_RG = _recogida.cargar()
# RECOGIDA DE PALLETS COMPLETOS para pedidos (2026-10-22): el gruero lleva un pallet completo de almacenamiento al andén. Casi solo en el turno C y no escala con la venta
# (r 0,29, p 0,35, 12 noches): nivel fijo por noche, editable.
RECOGIDA_PALLETS_NOCHE = _RG["pallets_noche_c"]               # referencia del log (nivel medio por noche, sep-oct)
RECOGIDA_POR_CAJA = _PF["recogida_por_caja"]                   # 2026-10-22: escala con la venta con la razón de Diana (pallets completos por caja vendida)
# RECEPCIÓN POR TIPO DE CAMIÓN (2026-10-26; `modelo/recepcion_camiones.py`): el tiempo de recepción es el del GRUERO HACIENDO TODO SOLO (descarga + escaneo; sin la espera del camión):
# 431 s el camión chico (~10 pallets) y 1.413 s el grande (~28), planilla de terreno. Segundos por pallet = Σ camiones × segundos ÷ Σ pallets de los 29 días del log. El TO-BE (planificador escaneando)
# usa 267 s y 910 s de gruero. Un solo tiempo por pallet en los tres turnos, como el resto de los procesos.
try:
    RECEPCION_CAMIONES = json.loads((Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "recepcion_camiones.json").read_text(encoding="utf-8"))
except Exception:
    RECEPCION_CAMIONES = None          # sin el JSON se usa el tiempo anterior (correr modelo.recepcion_camiones)


def seg_recepcion_por_pallet(pct_planificador: float | None = 0.0) -> float | None:
    """Segundos de gruero por pallet recibido con la mezcla de camiones del log. pct_planificador: fracción de camiones que escanea el planificador (0 = gruero solo; None = práctica del log)."""
    if not RECEPCION_CAMIONES:
        return None
    seg = pal = 0.0
    for x in RECEPCION_CAMIONES["tipos"].values():
        p = x["pct_escaneo_planificador_log"] if pct_planificador is None else pct_planificador
        n = x["camiones_dia_media_todos"]
        seg += n * (p * x["gruero_s_con_planificador"] + (1 - p) * x["gruero_s_sin_planificador"])
        pal += n * x["pallets_por_camion"]
    return seg / pal


TIEMPOS_FINALES = {
    "recepcion_directo_stage": (_TR["s_por_pallet_mezcla"], "preliminar", f"planilla de terreno: camión = {_TR['s_fijo_camion']:.0f} s + {_TR['s_por_pallet_marginal']:.1f} s por pallet "
                                f"({_TR['n_camiones']} camiones, r {_TR['r_tiempo_pallets']:.2f}); con {PALLETS_POR_CAMION_CL:.1f} pallets por camión = {_TR['s_camion_mezcla'] / 60:.1f} min por camión "
                                f"= {_TR['s_por_pallet_mezcla']:.1f} s por pallet (Diana: 23,5 min, que es el de un camión grande)"),
    "picking": (60 * _PF["min_lista_picking_pond"], "validado", f"{_FD}: {_PF['min_lista_picking_pond']:.2f} min por lista, promedio ponderado por listas de {_PF['n_pickers_picking']} pickers "
                f"(el simple daba {_PF['min_lista_picking']:.2f}); utilización {_PF['util_picking_pond']:.3f} sobre la presencia; validado contra 22 noches de septiembre"),
    "almacenamiento": (60 * _PF["min_almacenamiento_pallet"], "preliminar", f"{_FD}: {_PF['min_almacenamiento_pallet']:.2f} min por pallet"),
    "reabasto_pallet": (_PF["seg_rdp"], "preliminar", f"{_FD}: 2 min 9 s por pallet, sin desplazamiento (`RDP`!E9; decisión del usuario 2026-10-22)"),
    "reabasto_caja_grua": (_PF["seg_rdc_gruero"], "preliminar", f"{_FD}: 2 min 13 s por tarea"),
    "carga_camion": (_PF["seg_carga_vuelta"], "preliminar", f"{_FD}: 18 min 15 s por vuelta"),
    "retorno_fletero_gruero": (_PF["seg_retorno"], "insuficiente", f"{_FD}: 33 s por retorno (N = 4)"),
    "retorno_fletero": (152.0, "validado", "log de movimientos parciales, ciclo del turno A; el mismo en los tres turnos (decisión del usuario, 2026-10-23)"),
    "merma": (226.0, "validado", "log de movimientos parciales, ciclo del turno B; el mismo en los tres turnos (decisión del usuario, 2026-10-23)"),
    **({"recepcion_directo_stage": (seg_recepcion_por_pallet(0.0), "preliminar",
        "planilla de terreno (31 camiones, hoja «Recepción_»): el gruero hace descarga + escaneo solo = 431 s el camión chico y 1.413 s el grande; "
        f"{seg_recepcion_por_pallet(0.0):.1f} s por pallet con la mezcla del log; sin la espera del camión (recepcion_camiones.py)")} if RECEPCION_CAMIONES else {}),
    "carga_cl": (_PF["seg_carga_cl_pallet"], "preliminar", f"{_FD}: {_PF['seg_carga_cl_pallet']:.0f} s por pallet cargado para el CL (`extra`!B3)"),
    "traslado_mixtos": (62.0, "insuficiente", "observación en terreno de la noche (dato del equipo, 2026-11-08): 62 s por traslado de mixto; sin N declarado"),
    "traslado_stage": (58.0, "preliminar", "dato del equipo (2026-11-19): traslado del pallet recibido del patio a STREC, medido entre 45 y 65 s; fijo en 58 s"),
    "recogida_pallets": (_RG["s_ciclo"], _RG["estado_tiempo"], f"log WMS «Recogida de palés»: ciclo entre depósitos del mismo gruero sin otra tarea en medio, ≤ 900 s "
                         f"(N {_RG['n_ciclos']}, {_RG['dias_ciclo']} noches, {_RG['usuarios_ciclo']} grueros; recogida_pallets.py)"),
}

# La parte del gruero del reabasto de cajas tiene EXACTAMENTE el mismo volumen que la del operario (una reposición servida = una reposición): mismo driver.
DRIVERS["reabasto_caja_grua"] = replace(DRIVERS["reabasto_caja"], unidad="reposiciones de cajas servidas (pallet bajado)")


def nivel_modelo(proceso: str, regimen: str = "vigente") -> float | None:
    """Nivel diario que el modelo usa para un proceso que NO escala, en el régimen pedido."""
    return NIVELES_REGIMEN.get(regimen, {}).get(proceso, DRIVERS[proceso].volumen_dia if proceso in DRIVERS else None)


# Drivers ANTERIORES (hasta el 2026-10-03), con la recepción como proxy de la venta. Se conservan solo para
# mostrar el antes/después; no los usa ningún cálculo.
CAJAS_DIA_REFERENCIA = 24_747.0
_F = "log WMS 20-ago a 24-sep-2026 / recepción como proxy de venta [inferido]"
DRIVERS_PROXY_RECEPCION: dict[str, Driver] = {
    "picking": Driver(5436.6 / CAJAS_DIA_REFERENCIA, "líneas de picking", _F, "validado"),
    "reabasto_pallet": Driver(104.2 / CAJAS_DIA_REFERENCIA, "tareas de reabasto", _F, "validado"),
    "reabasto_caja": Driver(51.8 / CAJAS_DIA_REFERENCIA, "tareas de reabasto", _F, "validado"),
    "recepcion_directo_stage": Driver(8.8 / CAJAS_DIA_REFERENCIA, "camiones", _F, "validado"),
    "almacenamiento": Driver(262.9 / 26_516.0, "tareas de almacenamiento (pallets)",
                             "log WMS septiembre-2026 (4 a 30-sep, 22 días) / recepción como proxy de venta [inferido]",
                             "confiable"),
}


def drivers_desde_ventas() -> dict | None:
    """Recalcula los drivers desde `salidas/limpio/drivers_facturado.csv` (celdas Total), para que las constantes
    de arriba no puedan quedar desactualizadas. None si falta el archivo (correr modelo.demanda_actividad)."""
    from modelo import demanda_actividad as da
    d = da.leer_drivers()
    if not d:
        return None
    mapa = {"picking": "picking", "reabasto_pallet": "reabasto_pallet", "reabasto_caja": "reabasto_caja",
            "recepcion_directo_stage": "recepcion_directo_stage", "almacenamiento": "almacenamiento"}
    out = {k: {"por_1000": float(d[(v, "Total")]["por_1000_cajas"]), "media_dia": float(d[(v, "Total")]["media_dia"]),
                "sigma": float(d[(v, "Total")]["desvio_residual"]),
                "confianza": d[(v, "Total")]["confianza"], "n": int(d[(v, "Total")]["n"]),
                "r2": float(d[(v, "Total")]["r2"]), "p": float(d[(v, "Total")]["p"]),
                "facturado_medio": float(d[(v, "Total")]["facturado_medio"])}
           for k, v in mapa.items()}
    pc = d.get(("picking_cajas", "Total"))
    if pc:
        out["picking_cajas"] = {"por_1000": float(pc["por_1000_cajas"]), "sigma": float(pc["desvio_residual"])}
    return out


def driver_carga_desde_ventas() -> dict | None:
    """Recalcula la regresión de las vueltas >= 2 contra el facturado desde demanda_diaria.csv (septiembre)."""
    import csv
    from pathlib import Path
    from scipy import stats
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "demanda_diaria.csv"
    if not ruta.exists():
        return None
    with open(ruta, newline="", encoding="utf-8") as fh:
        f = [r for r in csv.DictReader(fh) if r.get("fuente") == "septiembre"]
    if not f:
        return None
    x = [float(r["facturado"]) for r in f]
    y = [int(r["camiones_carga2"]) + int(r["camiones_carga3mas"]) for r in f]
    lr = stats.linregress(x, y)
    res = [yy - (lr.intercept + lr.slope * xx) for xx, yy in zip(x, y)]
    sigma = (sum(e * e for e in res) / (len(y) - 2)) ** 0.5
    return {"n": len(y), "pendiente_por_1000": 1000 * lr.slope, "intercepto": lr.intercept, "r2": lr.rvalue ** 2,
            "p": lr.pvalue, "sigma": sigma}


def escenarios_demanda() -> dict | None:
    """Demanda de dimensionamiento con TODAS las ventas disponibles (2026-10-16): los 821 días con venta de `demanda_diaria.csv` (2-ene-2024 a 30-sep-2026: históricos 1 y 2 más septiembre).
    Antes eran solo los 25 días de septiembre-2026. Devuelve p25, p50, p90, p95, p99, el mínimo y máximo históricos, los dos polos (media de los días >= p90 y de los días <= p25) y, aparte,
    el RANGO DE CALIBRACIÓN de los drivers (`cal_min`, `cal_max`: los días de septiembre, único mes donde hay ventas y log de picking a la vez): fuera de él el modelo extrapola."""
    import csv
    import statistics
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "demanda_diaria.csv"
    if not ruta.exists():
        return None
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = [(r.get("fuente"), float(r["facturado"])) for r in csv.DictReader(fh) if float(r["facturado"]) > 0]
    f = sorted(x for _, x in filas)
    sep = sorted(x for fu, x in filas if fu == "septiembre")
    if not f:
        return None
    q = statistics.quantiles(f, n=100, method="inclusive")
    p25, p50, p90 = q[24], q[49], q[89]
    sat = [x for x in f if x >= p90]
    baj = [x for x in f if x <= p25]
    return {"dias": len(f), "p25": p25, "p50": p50, "p90": p90, "p95": q[94], "p99": q[98], "min": f[0], "max": f[-1],
            "saturado": statistics.mean(sat), "n_saturado": len(sat), "bajo": statistics.mean(baj), "n_bajo": len(baj),
            "cal_min": sep[0] if sep else f[0], "cal_max": sep[-1] if sep else f[-1], "cal_dias": len(sep)}


# Procesos cuyo tiempo estándar es un PISO (cota inferior) porque no cubre todo el trabajo del proceso.
# `reabasto_caja` (planilla oficial, 2026-10-13): el ciclo MEDIDO (hueco entre reposiciones consecutivas de una visita, 235 s, N=30) incluye reponer y el traslado entre
# posiciones, pero NO incluye anotar las tareas en papel ni esperar a que el planificador las elimine o a que llegue la grúa (hoja «Reabastecimiento_cajas», solo 2 jornadas:
# 730 y 2.780 s para anotar 6 y 10 tareas). Antes (hasta el 2026-10-12) el tiempo eran esas esperas, que NO son un ciclo; ahora hay un ciclo, pero parcial.
SIN_CICLO_MEDIDO = {
    "reabasto_caja": "el ciclo medido (235 s por reposición: reponer + traslado dentro de la visita, N=30) NO incluye anotar las tareas en papel "
                     "ni las esperas por eliminación de tarea o por la grúa (N=2 jornadas): es un piso del tiempo real",
}

# Qué rol ejecuta cada proceso. [inferido de docs/procesos_as_is.md + la columna Usuario
# del log; PENDIENTE de confirmar junto con config.EQUIVALENCIA_ROLES]
ROL_POR_PROCESO = {
    "picking": "Operario",
    "reabasto_pallet": "Gruero",
    "reabasto_caja": "Operario",
    "reabasto_caja_grua": "Gruero",          # bajar el pallet para que el operario reponga (hoja nueva de la planilla oficial)
    # [2026-10-15] movimientos de inventario parcial (modelo/movimientos_parciales.py): «la otra actividad de los operarios». Rol por confirmar con Víctor.
    "retorno_fletero": "Operario",
    "merma": "Operario",
    "recepcion_directo_stage": "Gruero",
    "carga_camion": "Gruero",
    "almacenamiento": "Gruero",
    # [2026-10-15] procesos nuevos pedidos por el equipo (todos editables en el estimador):
    "carga_cl": "Gruero",                    # carga de camiones para el CL (Excel de Diana, 2026-10-21)
    "traslado_stage": "Gruero",              # traslado del pallet recibido del patio a STREC (2026-11-20)
    "traslado_mixtos": "Gruero",             # traslado de pallets mixtos, solo turno C (observación de terreno, 2026-11-08; opción Ajustes.traslado_mixtos)
    "recogida_pallets": "Gruero",            # pallets completos para pedidos, turno C (log WMS, 2026-10-22)
    "retorno_fletero_gruero": "Gruero",      # el gruero atiende el retorno del fletero cuando llega una 2ª vuelta de camión de carga
    "otras_actividades_operario": "Operario",
    "otras_actividades_gruero": "Gruero",
    "traslados_operario": "Operario",
    "traslados_gruero": "Gruero",
}

# Tiempo estándar que un proceso toma prestado de otro mientras no tenga medición propia (se marca `insuficiente`).
ALIAS_TIEMPO = {"retorno_fletero_gruero": "retorno_fletero"}

# Cómo se combinan los pasos de un proceso para dar el tiempo de UNA unidad de trabajo.
#   "suma"   : los pasos son etapas consecutivas de la misma tarea -> se suman.
#   "mezcla" : los pasos son tipos alternativos de la misma unidad -> media ponderada
#              por N. Es el caso de picking: una línea es de UN tipo (mismo pasillo, o
#              cambio de pasillo, o cambio de onda), nunca de todos a la vez. Sumarlos
#              daría un tiempo por línea 4 veces mayor que el real.
AGREGACION = {
    "picking": "mezcla",
    "reabasto_pallet": "suma",
    "reabasto_caja": "mezcla",
    "reabasto_caja_grua": "suma",
    "retorno_fletero": "suma",
    "merma": "suma",
    "recepcion_directo_stage": "suma",
    "carga_camion": "suma",
    "almacenamiento": "suma",
    "retorno_fletero_gruero": "suma",
    "carga_cl": "suma",
    "recogida_pallets": "suma",
    "traslado_mixtos": "suma",
    "traslado_stage": "suma",
}

# Fracción del tiempo presente que es trabajo medido. Para picking está MEDIDA: la suma
# de los gaps (2.105 h) contra las horas-persona de presencia (2.481 h) da 0,848 — el
# 15,2% restante son pausas, esperas de onda y el arranque/cierre de la jornada.
# Para el resto **no hay medición**: se usa la misma cifra como supuesto declarado, y por
# eso `Requerimiento.supuestos` lo registra en cada resultado.
# [2026-10-13] Desde esta fecha la «eficiencia» es la UTILIZACIÓN OBJETIVO del equipo (0,85) aplicada a las 7 horas efectivas del turno; el 0,848 que se había medido en
# picking (horas de gap / horas de presencia del log) queda como referencia. Se usa el mismo valor en todos los procesos.
EFICIENCIA_MEDIDA = {"picking": config.UTILIZACION_PICKING}
EFICIENCIA_SUPUESTA = config.UTILIZACION_OBJETIVO
EFICIENCIA_PICKING_MEDIDA_ANTES = 0.848


# ---------------------------------------------------------------- resultado

@dataclass
class Requerimiento:
    proceso: str
    turno: str
    rol: str
    demanda_cajas_dia: float
    volumen: float                  # unidades de trabajo del proceso
    unidad_volumen: str
    tiempo_estandar_s: float
    horas_trabajo: float
    eficiencia: float
    horas_persona: float
    duracion_turno_h: float
    personas_necesarias: float
    personas_reales: int
    brecha: float                   # necesarias - reales (positivo = faltan)
    estado_confianza: str
    meta_fill_rate: None = None     # brecha #7: sin confirmar. NO inventar un valor.
    fuentes: list[str] = field(default_factory=list)
    supuestos: list[str] = field(default_factory=list)
    sin_ciclo_medido: bool = False  # su tiempo estándar NO es un ciclo medido (SIN_CICLO_MEDIDO): el rol queda en cota inferior

    @property
    def alcanza(self) -> bool | None:
        """None cuando la confianza no permite afirmarlo."""
        if self.estado_confianza in ("insuficiente", "sustituto_dia", "sin_datos"):
            return None
        return self.brecha <= 0


# ---------------------------------------------------------------- tiempos

# REABASTO DE PALLET POR ORIGEN (2026-10-02). Los eventos salen de dos orígenes con tiempos muy distintos:
# ALMAC (rack, con grúa: media ≈120-185 s) y ALMPIC (media ≈37 s, con cola larga). Promediarlos en un solo
# parámetro no describe a ninguno, así que `fit_params` los ajusta por separado (pasos `ciclo_total_almac` y
# `ciclo_total_almpic`, cada uno con su N, estado y fuente) y el tiempo estándar del proceso es la media
# ponderada por la proporción de eventos de cada origen.
#
# Proporción de eventos por origen (pares retiro->depósito, todos los turnos). ES ESTABLE ENTRE REGÍMENES
# (3,1 puntos; chi² p≈0,05: 0,053 con corrección de continuidad y 0,045 sin ella, en el límite; el cambio es mucho menor que el de otros procesos):
# régimen vigente 22 a 30-sep: ALMAC 75 de 720 pares (10,4%); anterior 4-21 sep: 173 de 1.282 (13,5%).
# Práctica vigente = calibración 1. Se recalcula de wms_sep_ciclos.csv en tests/test_capacity.py.
ORIGEN_REABASTO_PALLET = {"ALMAC": 0.104, "ALMPIC": 0.896}            # respaldo (22 a 30-sep); se reemplaza abajo con el log de octubre
ORIGEN_REABASTO_PALLET_HISTORICO = {"ALMAC": 0.135, "ALMPIC": 0.865}  # respaldo (4 a 21-sep)


def origen_reabasto_pallet_desde_log() -> dict | None:
    """Proporción de reabasto de pallet por origen en cada régimen, recalculada de `actividad_obs_reabasto_pallet.csv` (pares retiro->depósito dentro del corte de
    900 s, septiembre + octubre: la MISMA población que estima su tiempo). vigente = octubre, anterior = 4 a 21-sep. None si falta el archivo."""
    import csv
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "actividad_obs_reabasto_pallet.csv"
    if not ruta.exists():
        return None
    c = {k: {"ALMAC": 0, "ALMPIC": 0} for k in ("vigente", "anterior")}
    with open(ruta, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for reg in c:
                d, h = calibracion.REGIMENES[reg]
                if d <= r["fecha"] < h and r["paso"] in ("ciclo_total_almac", "ciclo_total_almpic"):
                    c[reg]["ALMAC" if r["paso"].endswith("almac") else "ALMPIC"] += 1
    if not all(sum(v.values()) for v in c.values()):
        return None
    return {k: {"pares": sum(v.values()), "almac": v["ALMAC"], "prop_almac": v["ALMAC"] / sum(v.values())}
            for k, v in c.items()}


_o = origen_reabasto_pallet_desde_log()
if _o:
    ORIGEN_REABASTO_PALLET = {"ALMAC": _o["vigente"]["prop_almac"], "ALMPIC": 1 - _o["vigente"]["prop_almac"]}
    ORIGEN_REABASTO_PALLET_HISTORICO = {"ALMAC": _o["anterior"]["prop_almac"], "ALMPIC": 1 - _o["anterior"]["prop_almac"]}


def _tiempo_por_origen(ps, proceso: str = "reabasto_pallet"):
    """Media ponderada por origen, o None si falta alguno de los dos pasos. La confianza es la más baja
    de los dos orígenes (conservador): cada origen conserva su estado en `fuentes`."""
    por_paso = {p.paso: p for p in ps}
    a, b = por_paso.get("ciclo_total_almac"), por_paso.get("ciclo_total_almpic")
    if a is None or b is None:
        return None
    w = ORIGEN_REABASTO_PALLET
    t = w["ALMAC"] * a.media_s + w["ALMPIC"] * b.media_s
    estado = peor_confianza(a.estado, b.estado)
    fuentes = [f"origen ALMAC ({w['ALMAC']:.1%} de los eventos): {a.media_s:.0f} s, N={a.n}, {a.estado}",
               f"origen ALMPIC ({w['ALMPIC']:.1%} de los eventos): {b.media_s:.0f} s, N={b.n}, {b.estado}",
               a.fuente]
    return t, estado, fuentes


def tiempo_estandar(proceso: str, turno: str, parametros=None) -> tuple[float, str, list[str]]:
    """Segundos por unidad de trabajo. PICKING: por LISTA (tiempo por línea × líneas por lista); el resto como `_tiempo_estandar_base`."""
    if proceso in TIEMPOS_FINALES:                    # 2026-10-23: un solo tiempo por proceso en los tres turnos (decisión del usuario)
        seg, est_, fu = TIEMPOS_FINALES[proceso]
        return seg, est_, [fu]
    t, e, f = _tiempo_estandar_base(proceso, turno, parametros)
    if proceso == "picking" and t:
        return t * LINEAS_POR_LISTA, e, f + [f"tiempo por lista = tiempo por línea ({t:.1f} s) × {LINEAS_POR_LISTA:.2f} líneas por lista"]
    return t, e, f


def _tiempo_estandar_base(proceso: str, turno: str,
                          parametros=None) -> tuple[float, str, list[str]]:
    """Segundos por unidad de trabajo (picking: por LÍNEA), con su estado de confianza y sus fuentes.

    Si el proceso tiene un paso `ciclo_total` medido, se usa ese (es el ciclo completo
    cronometrado punta a punta, más fiable que sumar pasos que pueden solaparse).
    Si no, se aplica la regla de `AGREGACION`."""
    if proceso in ALIAS_TIEMPO:
        t_, e_, f_ = tiempo_estandar(ALIAS_TIEMPO[proceso], turno, parametros)
        return t_, peor_confianza(e_, "insuficiente"), f_ + [f"tiempo PRESTADO de '{ALIAS_TIEMPO[proceso]}' (sin medición propia de '{proceso}'): editable en el estimador"]
    ps = [p for p in (parametros if parametros is not None else fit_params.cargar())
          if p.proceso == proceso and p.turno == turno]
    if not ps:
        return 0.0, "sin_datos", []
    fuentes = sorted({p.fuente for p in ps})
    if proceso == "reabasto_pallet":
        por_origen = _tiempo_por_origen(ps)
        if por_origen is not None:
            return por_origen
    total = next((p for p in ps if p.paso == "ciclo_total"), None)
    if total is not None:
        return total.media_s, total.estado, [total.fuente]
    pasos = [p for p in ps if p.paso != "ciclo_total"]
    estado = peor_confianza(*(p.estado for p in pasos))
    if AGREGACION.get(proceso, "suma") == "mezcla":
        n = sum(p.n for p in pasos)
        if n == 0 or proceso in SIN_CICLO_MEDIDO:
            # Media simple, igual en TODOS los turnos. Sin muestra propia (n=0) no hay pesos que usar. Y en un proceso
            # sin ciclo medido (reabasto de cajas) los pasos son esperas y acciones distintas, no tipos alternativos de
            # una misma tarea: pesarlos por N (2, 2, 6, 2, 2) sería pesar por cuántas veces los vio el cronómetro, no por
            # cuánto pesan en una tarea. [Unificado el 2026-10-05: antes A usaba pesos (864 s) y B/C media simple (1.109 s).]
            t = sum(p.media_s for p in pasos) / len(pasos)
        else:
            t = sum(p.media_s * p.n for p in pasos) / n
    else:
        t = sum(p.media_s for p in pasos)
    return t, estado, fuentes


# ---------------------------------------------------------------- cálculo

@dataclass(frozen=True)
class Ajustes:
    """Parámetros editables del estimador (pestaña «Capacidad necesaria»). Los valores por defecto son los del
    modelo; cambiarlos es un análisis "qué pasaría si", no una calibración.
      regimen: "vigente" (octubre, 2 al 10-oct: config.PROCESOS_POR_TURNO y CUOTAS_TURNO), "intermedio" (22-sep a 1-oct: PROCESOS_POR_TURNO_INTERMEDIO y
               CUOTAS_TURNO_INTERMEDIO) o "anterior" (4 al 21-sep: PROCESOS_POR_TURNO_HISTORICO y CUOTAS_TURNO_HISTORICAS).
      eficiencia_picking / eficiencia_resto: utilización objetivo del equipo (0,85, decisión 2026-10-13), aplicada a las 7 horas efectivas del turno.
      niveles_base: volumen diario de los procesos que NO escalan con la venta (reemplaza Driver.volumen_dia).
      k_sigma: desplaza el volumen de los procesos con sigma de regresión (picking, carga) en k desvíos: -1/+1 = rango."""
    regimen: str = "combinado"                    # 2026-10-21: repartos por turno del log combinado (decisión del usuario)
    eficiencia_picking: float = config.UTILIZACION_PICKING
    eficiencia_resto: float = config.UTILIZACION_OBJETIVO
    niveles_base: tuple = ()          # tuplas (proceso, volumen_dia); tuple para que sea hashable
    k_sigma: float = 0.0
    tiempos_override: tuple = ()      # tuplas (proceso, segundos por unidad): SOLO análisis de sensibilidad, no es una medición
    semana_dotacion: str | None = None  # lunes ISO de la semana cuyo roster da R (turno C); None = la última semana disponible (config)
    hojas: tuple | None = ()          # hojas de planificadores a sumar al reabasto de cajas: None = todas; () = ninguna (por defecto desde 2026-10-15); fechas ISO = esas
    con_parciales: bool = True        # suma al reabasto de cajas los movimientos de inventario parcial (log) — la ejecución real de lo que indican las hojas
    reabasto_caja_turno: tuple = ()   # tuplas (turno, tareas por día) que reemplazan el volumen de reabasto de cajas de ese turno (editable)
    volumenes_turno: tuple = ()       # tuplas (proceso, turno, volumen por día): reemplaza el volumen de CUALQUIER proceso en ese turno (editable; 'qué pasaría si')
    # [2026-10-15] tiempo del turno y trabajo fuera de los procesos (todo editable; los defectos reproducen el modelo anterior: 8 h − 30 − 30 min = 7 h, sin traslados ni otras actividades)
    charla_min: float = parametros_diana.CHARLA_MIN     # charla de inicio de turno (min), 20
    colacion_min: float = parametros_diana.COLACION_MIN  # colación (min), 30
    traslados_min: tuple = ()         # tuplas (rol, turno, minutos por persona por turno) dedicados a traslados (recorridos entre áreas que no están dentro del tiempo de una tarea)
    otras_h: tuple = ()               # tuplas (rol, turno, horas de trabajo por turno) de otras actividades no modeladas (el equipo las mide o estima)
    # [2026-10-21] PICKING: su utilización se midió sobre la presencia del log, que incluye la colación. Su tiempo disponible es 8 h − charla de picking (10 min observados),
    # sin restar la colación otra vez. False + 20 min reproduce el criterio anterior (7 h 10 también para picking).
    colacion_en_utilizacion_picking: bool = True
    charla_picking_min: float = parametros_diana.CHARLA_PICKING_MIN
    # [2026-10-24, reunión con Milton y Felipe] OPCIÓN DESACTIVADA POR DEFECTO. True: el picking se hace en el turno C hasta lo que alcanza su dotación real de operarios
    # (R del turno C, descontadas sus otras tareas) y solo el EXCEDENTE pasa al turno A; reemplaza el reparto fijo del log (≈98% C / 2% A). Ver `reparto_picking_excedente`.
    picking_excedente_a: bool = False
    # [2026-10-24, definición de Milton] POR DEFECTO el 100% de las listas de picking va al turno C (cuota C = 1, A = B = 0). Cuando el C no alcanza, el excedente NO se mueve: se informa en listas y
    # en vueltas (`excedente_picking_c`). True recupera el reparto observado en el log (≈98% C / 2% A, el derrame real) para comparar con el modelo anterior.
    picking_reparto_log: bool = False
    # [2026-10-26] El retorno de fletero del operario sigue al rechazo de las ventas (cambia con la demanda). True = el nivel fijo anterior del log (68,6 por día).
    retorno_fijo: bool = False
    # [2026-10-26] Dotación actual EDITABLE: tuplas (turno, rol, personas) que reemplazan a config.personas_turno (vacío = la de config.DOTACION_REAL). Cambia la columna «real», las brechas y el tope del MILP.
    dotacion_real: tuple = ()
    # [2026-10-26] POR DEFECTO ACTIVADA (decisión del usuario): recepción por TIPO DE CAMIÓN (chicos ~10 pallets, fijos por día; grandes ~28 pallets, escalan con la venta), con el tiempo del GRUERO por camión
    # según haya o no planificador escaneando (`modelo/recepcion_camiones.py`). Cambia los pallets recibidos (y con ellos el almacenamiento y la carga para el CL) y el tiempo por pallet de recepción.
    recepcion_por_camion: bool = True
    # TRASLADO DE MIXTOS (2026-11-08, terreno nocturno; SOLO turno C): una lista de picking es también un pallet mixto, y el gruero traslada el 40% de ellos (62 s cada uno; decisión del usuario, 2026-11-08, antes 1/3). Opción
    # DESACTIVADA por defecto (False reproduce el modelo de siempre). Volumen = `mixtos_fraccion` × listas del turno C. Usa grúa (confirmado por el usuario, 2026-11-08).
    traslado_mixtos: bool = True
    mixtos_fraccion: float = 0.40
    # Quién escanea en recepción con esa opción: fracción de camiones que escanea el planificador. 0,0 (por defecto) = el gruero hace TODO solo (431 s el camión chico, 1.413 s el grande: el tiempo del
    # modelo); 1,0 = siempre el planificador (TO-BE con ayuda: gruero 267 / 910 s); None = la práctica del log (67% chicos, 73% grandes).
    recepcion_pct_planificador: float | None = 0.0

    def personas(self, turno: str, rol: str) -> int:
        """Dotación actual de un rol en un turno: la editada si existe, si no la de config."""
        ov = {(t, r): n for t, r, n in self.dotacion_real}
        return int(ov[(turno, rol)]) if (turno, rol) in ov else config.personas_turno(turno, rol, self.semana_dotacion)

    def horas_picking(self) -> float:
        """Horas de presencia de un picker por turno sobre las que se aplica su utilización."""
        return max(0.0, config.HORAS_TURNO_NOMINAL - (self.charla_picking_min + (0.0 if self.colacion_en_utilizacion_picking else self.colacion_min)) / 60.0)

    def eficiencia_picking_efectiva(self) -> float:
        """Utilización de picking expresada sobre las horas efectivas del turno (las mismas que el resto): η × horas de picking ÷ horas efectivas. Así N = horas productivas ÷ (η × horas
        de picking) se calcula con el mismo divisor que los demás procesos del rol, y la simulación usa una sola ventana por turno."""
        h = self.horas_efectivas()
        return self.eficiencia_picking * self.horas_picking() / h if h else self.eficiencia_picking

    def nivel(self, proceso: str) -> float | None:
        return dict(self.niveles_base).get(proceso)

    def tiempo(self, proceso: str) -> float | None:
        return dict(self.tiempos_override).get(proceso)

    def traslado_min(self, rol: str, turno: str) -> float:
        return float({(r, t): m for r, t, m in self.traslados_min}.get((rol, turno), 0.0))

    def otras_horas(self, rol: str, turno: str) -> float:
        return float({(r, t): h for r, t, h in self.otras_h}.get((rol, turno), 0.0))

    def horas_efectivas(self) -> float:
        """Horas efectivas por persona y turno: 8 − charla − colación."""
        return max(0.0, config.HORAS_TURNO_NOMINAL - (self.charla_min + self.colacion_min) / 60.0)

    def volumen_editado(self, proceso: str, turno: str) -> float | None:
        """Volumen que el usuario fijó para (proceso, turno), o None. La parte gruero del reabasto de cajas sigue al volumen de reabasto de cajas."""
        clave = "reabasto_caja" if proceso == "reabasto_caja_grua" else proceso
        return {(p, t): v for p, t, v in self.volumenes_turno}.get((clave, turno))


def requerimiento(proceso: str, turno: str, demanda_cajas_dia: float,
                  parametros=None, cuota_turno: float = 1.0,
                  volumen_fijo: float | None = None,
                  unidad_fija: str | None = None, ajustes: Ajustes | None = None,
                  n_volumen: int | None = None, volumen_turno: float | None = None, fuente_volumen: str = "",
                  volumen_editado: bool = False) -> Requerimiento:
    """Personas necesarias para un proceso en un turno, dada una demanda diaria.

    `cuota_turno` es qué fracción del volumen diario del proceso cae en ese turno (1.0 =
    todo: pensado para diagnosticar un proceso aislado, NO para dimensionar un turno;
    `dotacion_por_turno` nunca lo usa, siempre pasa la cuota real o el volumen fijo).
    `volumen_fijo` reemplaza al driver cuando el volumen por turno está declarado
    directamente (p. ej. ~5 camiones por noche, config.PROCESOS_POR_TURNO); `n_volumen` es de cuántas noches/días
    contados sale (su estado se etiqueta con `fit_params.estado_por_n`; None = no se sabe = insuficiente).
    """
    drv = DRIVERS.get(proceso)
    if proceso == "retorno_fletero" and ajustes is not None and ajustes.retorno_fijo:
        drv = DRIVER_RETORNO_FIJO
    t_std, est_t, fuentes = tiempo_estandar(proceso, turno, parametros)
    rol = ROL_POR_PROCESO.get(proceso, "desconocido")
    h_turno = config.HORARIO_TURNO[turno]
    dur = _duracion_turno(turno, ajustes)

    supuestos = []
    if volumen_fijo is not None and volumen_editado:
        # el usuario fijó el volumen de este proceso en este turno: «qué pasaría si». Conserva el estado del driver (no degrada la confianza de los tiempos).
        volumen, unidad = volumen_fijo, unidad_fija or "unidades"
        est_d = drv.estado if drv is not None else "preliminar"
        fuentes = fuentes + [f"volumen EDITADO por el usuario ({volumen_fijo:g} {unidad} por día en el turno {turno})"]
        supuestos.append(f"volumen de '{proceso}' en el turno {turno} EDITADO por el usuario: {volumen_fijo:g} {unidad} por día (no es un dato del modelo)")
    elif volumen_fijo is not None:
        volumen, unidad = volumen_fijo, unidad_fija or "unidades"
        # El estado del volumen declarado sale de su N con las MISMAS reglas que los parámetros (fit_params.estado_por_n):
        # 10 noches contadas = preliminar. Sin N conocido (o un valor puesto a mano en una sensibilidad) = insuficiente.
        est_d = fit_params.estado_por_n(n_volumen) if n_volumen else "insuficiente"
        fuentes = fuentes + [f"volumen declarado en config.PROCESOS_POR_TURNO ({volumen_fijo:g} "
                             f"{unidad} por turno; N={n_volumen if n_volumen else 'desconocido'})"]
        supuestos.append(f"volumen de '{proceso}' en turno {turno} declarado por el equipo "
                         f"({volumen_fijo:g} {unidad}), no derivado de la demanda: no escala "
                         "con ella [dato exacto pendiente]")
    elif drv is not None and drv.volumen_dia is not None:
        # proceso que NO escala con la demanda (sin relación con el facturado en septiembre)
        nivel = ajustes.nivel(proceso) if ajustes else None
        base = nivel_modelo(proceso, ajustes.regimen if ajustes else "vigente")
        if nivel is not None and nivel != base:
            supuestos.append(f"nivel base de '{proceso}' editado: {nivel:g} {drv.unidad} por día (modelo: {base:g})")
        if volumen_turno is not None and nivel is None:
            # volumen propio del turno (log + hojas de planificadores): ya no es nivel diario x cuota
            volumen = volumen_turno
            unidad, est_d = drv.unidad, drv.estado
            fuentes = fuentes + [fuente_volumen or drv.fuente]
            supuestos.append(f"'{proceso}' no escala con la demanda; su volumen en el turno {turno} es {volumen_turno:.1f} {drv.unidad} por día: {fuente_volumen} [preliminar]")
        else:
            volumen = (base if nivel is None else nivel) * cuota_turno
            unidad, est_d = drv.unidad, drv.estado
            fuentes = fuentes + [drv.fuente]
            supuestos.append(f"'{proceso}' no escala con la demanda: en septiembre su actividad diaria no tuvo relación "
                             f"con las cajas facturadas, así que se usa su nivel medio del régimen ({base:g} {drv.unidad} "
                             "por día) a cualquier demanda [preliminar]")
    elif drv is None:
        volumen, unidad, est_d = 0.0, "sin driver", "sin_datos"
        supuestos.append(f"no hay driver de volumen para '{proceso}': "
                         "no se puede traducir demanda a unidades de trabajo")
    else:
        k = ajustes.k_sigma if ajustes else 0.0
        desvio = k * drv.sigma if (k and drv.sigma) else 0.0
        volumen = max(0.0, drv.intercepto + demanda_cajas_dia * drv.unidades_por_caja + desvio) * cuota_turno
        unidad, est_d = drv.unidad, drv.estado
        fuentes = fuentes + [drv.fuente]
        if proceso == "retorno_fletero":
            supuestos.append(f"retorno de fletero = {drv.intercepto:.1f} + {1000 * drv.unidades_por_caja:.3f} por cada 1.000 cajas facturadas (tareas por día, 90% del rechazo en SKU; el 10% se asume merma) [supuesto: todo el rechazo vuelve por retorno de fletero]")
        elif drv.intercepto:
            supuestos.append(f"volumen de '{proceso}' = {drv.intercepto:g} + {1000 * drv.unidades_por_caja:g} por cada 1.000 "
                             f"cajas facturadas (sigma {drv.sigma:g} {drv.unidad}); vueltas >= 2, [hipótesis: CARGA = vuelta]")
        supuestos.append("el volumen escala linealmente con las cajas facturadas (septiembre-2026: R² 0,86 en picking); "
                         "en los días saturados el turno C quedó ~9% bajo la recta y el derrame lo absorbió el turno A")

    ef = EFICIENCIA_MEDIDA.get(proceso)
    if ef is None:
        ef = EFICIENCIA_SUPUESTA if ajustes is None else ajustes.eficiencia_resto
        supuestos.append(f"eficiencia {ef} tomada de picking, NO medida para "
                         f"'{proceso}' [supuesto]")
    else:
        ef = (ajustes or Ajustes()).eficiencia_picking_efectiva()
    t_ov = ajustes.tiempo(proceso) if ajustes else None
    if t_ov is not None:
        supuestos.append(f"tiempo por unidad de '{proceso}' FIJADO a {t_ov:g} s para una sensibilidad (valor del modelo: {t_std:.0f} s); "
                         "no es una medición")
        t_std = t_ov
    horas_trabajo = volumen * t_std / 3600
    horas_persona = horas_trabajo / ef if ef else 0.0
    personas = horas_persona / dur if dur else 0.0
    reales = ajustes.personas(turno, rol) if ajustes else config.personas_turno(turno, rol, None)

    if proceso in SIN_CICLO_MEDIDO:
        supuestos.append(SIN_CICLO_MEDIDO[proceso])
    if est_t == "sustituto_dia":
        supuestos.append(f"el tiempo estándar de '{proceso}' en turno {turno} es la "
                         "distribución del turno A copiada: NO hay medición propia")

    return Requerimiento(
        proceso=proceso, turno=turno, rol=rol, demanda_cajas_dia=demanda_cajas_dia,
        volumen=volumen, unidad_volumen=unidad, tiempo_estandar_s=t_std,
        horas_trabajo=horas_trabajo, eficiencia=ef, horas_persona=horas_persona,
        duracion_turno_h=dur, personas_necesarias=personas, personas_reales=reales,
        brecha=personas - reales,
        estado_confianza=peor_confianza(est_t, est_d),
        fuentes=sorted(set(fuentes)), supuestos=supuestos, sin_ciclo_medido=proceso in SIN_CICLO_MEDIDO)


def _regimen(ajustes: Ajustes | None) -> str:
    return ajustes.regimen if ajustes is not None else "vigente"


def _tabla_procesos(regimen: str) -> dict:
    return {"anterior": config.PROCESOS_POR_TURNO_HISTORICO, "intermedio": config.PROCESOS_POR_TURNO_INTERMEDIO, "combinado": config.PROCESOS_POR_TURNO_COMBINADO}.get(regimen, config.PROCESOS_POR_TURNO)


def _procesos(turno: str, ajustes: Ajustes | None):
    """(procesos que ocurren, procesos inciertos) del turno según el régimen."""
    if _regimen(ajustes) != "vigente":
        tabla = _tabla_procesos(_regimen(ajustes))[turno]
        return ({k: v for k, v in tabla.items() if v.ocurre is True}, {k: v for k, v in tabla.items() if v.ocurre is None})
    return config.procesos_que_ocurren(turno), config.procesos_inciertos(turno)


def _cuotas(turno: str, ajustes: Ajustes | None) -> dict[str, float]:
    c = {"anterior": CUOTAS_TURNO_HISTORICAS, "intermedio": CUOTAS_TURNO_INTERMEDIO, "combinado": CUOTAS_TURNO_COMBINADO}.get(_regimen(ajustes), CUOTAS_TURNO)[turno]
    if "picking" in c and not (ajustes is not None and ajustes.picking_reparto_log):
        c = {**c, "picking": 1.0 if turno == "C" else 0.0}           # 2026-10-24: todo el picking en el turno C (definición de Milton)
    return c


def _volumen_turno(proceso: str, turno: str, ajustes: "Ajustes | None") -> tuple[float | None, str]:
    """Volumen propio de un turno para el reabasto de cajas (log + hojas), o (None, '') si el proceso usa nivel x cuota o el usuario editó el nivel diario."""
    if proceso not in ("reabasto_caja", "reabasto_caja_grua") or (ajustes is not None and ajustes.nivel("reabasto_caja") is not None):
        return None, ""
    reg = ajustes.regimen if ajustes else "vigente"
    ov = dict(ajustes.reabasto_caja_turno) if ajustes else {}
    v = volumen_reabasto_caja(reg, ajustes.hojas if ajustes else (), ajustes.con_parciales if ajustes else True)[turno]
    if turno in ov:
        return float(ov[turno]), f"valor EDITADO por el usuario (el modelo da {v['total']:.1f}: {v['fuente']})"
    return v["total"], v["fuente"]


def _resolver_turno(turno: str, demanda_cajas_dia: float,
                    cuotas: dict[str, float] | None = None, parametros=None,
                    volumenes: dict[str, float] | None = None, ajustes: Ajustes | None = None):
    """Qué procesos entran en un turno y cuáles no se pueden calcular.

    Solo entran los que `config.PROCESOS_POR_TURNO` declara que ocurren. Un proceso que
    ocurre pero no tiene volumen calculable (sin driver, sin cuota, o sin saber si ocurre)
    NO recibe un valor por defecto: se devuelve aparte, con el motivo, para que el
    resultado se marque incompleto en vez de subestimarse en silencio.

    `volumenes` permite sobrescribir un volumen fijo (análisis de sensibilidad).
    Devuelve (requerimientos, {proceso: motivo_sin_volumen})."""
    if parametros is None:
        parametros = fit_params.cargar()
    cuotas = _cuotas(turno, ajustes) if cuotas is None else cuotas
    if ajustes is not None and ajustes.picking_excedente_a and "picking" in cuotas:       # opción desactivada por defecto (reparto_picking_excedente)
        cuotas = {**cuotas, "picking": reparto_picking_excedente(demanda_cajas_dia, parametros, ajustes)["cuota"].get(turno, 0.0)}
    volumenes = volumenes or {}
    reqs: list[Requerimiento] = []
    sin_vol: dict[str, str] = {}
    ocurren, inciertos = _procesos(turno, ajustes)
    for proceso, pt in ocurren.items():
        editado = ajustes.volumen_editado(proceso, turno) if ajustes else None
        if editado is not None and proceso not in volumenes:
            unidad = pt.unidad or (DRIVERS[proceso].unidad if proceso in DRIVERS else "unidades")
            reqs.append(requerimiento(proceso, turno, demanda_cajas_dia, parametros, 1.0, editado, unidad, ajustes, volumen_editado=True))
            continue
        fijo = volumenes.get(proceso, pt.volumen_fijo)
        if fijo is not None:
            # el N del volumen solo vale para el valor declarado: un valor sobrescrito (sensibilidad) no tiene N
            reqs.append(requerimiento(proceso, turno, demanda_cajas_dia, parametros, 1.0,
                                      fijo, pt.unidad, ajustes, n_volumen=pt.n_volumen if proceso not in volumenes else None))
        elif proceso in DRIVERS and proceso in cuotas:
            vt, fv = _volumen_turno(proceso, turno, ajustes)
            if vt is not None and not fv.startswith("valor EDITADO"):     # 2026-10-22: el reabasto de cajas del log es el de la demanda normal y escala con la venta
                vt = vt * demanda_cajas_dia / DEMANDA_NORMAL_RDC
                fv = f"{fv}; × demanda ÷ {DEMANDA_NORMAL_RDC:,.0f} (el nivel del log corresponde a la demanda normal; escala con la venta como las tareas de Diana)"
            reqs.append(requerimiento(proceso, turno, demanda_cajas_dia, parametros,
                                      cuotas[proceso], ajustes=ajustes, volumen_turno=vt, fuente_volumen=fv))
        elif proceso not in DRIVERS:
            sin_vol[proceso] = "sin driver de volumen"
        else:
            sin_vol[proceso] = "sin cuota por turno"
    for proceso in inciertos:
        sin_vol[proceso] = "no se sabe si ocurre en este turno"
    if ajustes is not None and ajustes.recepcion_por_camion and RECEPCION_CAMIONES:          # opción desactivada por defecto (recepcion_camiones.py)
        reqs = [_recepcion_por_camion(r, turno, demanda_cajas_dia, ajustes) for r in reqs]
    # RETORNO DE FLETERO (gruero), 2026-10-15: se hace cuando llega una 2ª vuelta de camión de carga => un retorno por cada camión cargado fuera del turno C (vueltas ≥ 2).
    # Volumen = trailers de ese turno; el tiempo se toma prestado del retorno medido en el log (insuficiente) y es editable.
    if turno != "C":                          # 2026-11-25 (decisión del usuario): el retorno de fletero del gruero ocurre solo en A y B, no en C (antes, 2026-10-23, en los tres turnos; en C eran ≈ 10 min de gruero)
        carga = next((r for r in reqs if r.proceso == "carga_camion"), None)
        ed_rg = ajustes.volumen_editado("retorno_fletero_gruero", turno) if ajustes else None
        base_rg = carga.volumen if carga else 0.0
        if ed_rg is not None or base_rg > 0:
            rr = requerimiento("retorno_fletero_gruero", turno, demanda_cajas_dia, parametros, 1.0, ed_rg if ed_rg is not None else base_rg, "retornos de fletero (1 por camión cargado)",
                               ajustes, volumen_editado=ed_rg is not None)
            rr.supuestos.append("un retorno de fletero por cada camión cargado en el turno, solo en A y B (decisión del usuario, 2026-11-25); 33 s por retorno [Excel de Diana]")
            reqs.append(rr)
    # CARGA PARA EL CL (2026-10-21, Excel de Diana): el 60% de los camiones de recepción del turno se cargan para el CL. No en el turno C (decisión del usuario: de noche llega
    # a lo más un camión).
    rec = next((r for r in reqs if r.proceso == "recepcion_directo_stage"), None)
    ed_cl = ajustes.volumen_editado("carga_cl", turno) if ajustes else None
    base_cl = (FRACCION_CARGA_CL * rec.volumen) if (rec and turno != "C") else 0.0      # pallets: 1/3 de los recibidos en el turno (2026-10-22)
    if ed_cl is not None or base_cl > 0:
        rc = requerimiento("carga_cl", turno, demanda_cajas_dia, parametros, 1.0, ed_cl if ed_cl is not None else base_cl, "pallets cargados para el CL", ajustes, volumen_editado=ed_cl is not None)
        rc.supuestos.append(f"1/3 de los pallets recibidos en el turno se cargan para el CL (decisión del usuario, 2026-10-22), {_PF['seg_carga_cl_pallet']:.0f} s por pallet [Excel de Diana]")
        reqs.append(rc)
    # TRASLADO PATIO → STREC (2026-11-20): cada pallet recibido en el turno se lleva a STREC.
    ed_ts = ajustes.volumen_editado("traslado_stage", turno) if ajustes else None
    base_ts = rec.volumen if rec else 0.0
    if ed_ts is not None or base_ts > 0:
        ts = requerimiento("traslado_stage", turno, demanda_cajas_dia, parametros, 1.0, ed_ts if ed_ts is not None else base_ts, "pallets trasladados a STREC", ajustes, volumen_editado=ed_ts is not None)
        ts.supuestos.append("un traslado patio → STREC por pallet recibido, 58 s (dato del equipo, 45-65 s)")
        reqs.append(ts)
    # TRASLADO DE MIXTOS (2026-11-08): solo turno C; volumen = fracción de las listas de picking del turno. Incorporado al modelo por defecto (2026-11-10); Ajustes(traslado_mixtos=False) lo quita.
    ed_tm = ajustes.volumen_editado("traslado_mixtos", turno) if ajustes else None
    pk = next((r for r in reqs if r.proceso == "picking"), None)
    base_tm = (ajustes.mixtos_fraccion * pk.volumen) if (ajustes is not None and ajustes.traslado_mixtos and turno == "C" and pk) else 0.0
    if ed_tm is not None or base_tm > 0:
        tm = requerimiento("traslado_mixtos", turno, demanda_cajas_dia, parametros, 1.0, ed_tm if ed_tm is not None else base_tm, "mixtos trasladados", ajustes, volumen_editado=ed_tm is not None)
        tm.supuestos.append(f"{ajustes.mixtos_fraccion if ajustes else 0.40:.2f} de las listas de picking del turno C se trasladan como mixto (observación de terreno nocturna, 2026-11-08); 62 s por mixto; usa grúa durante todo el traslado (confirmado por el usuario)")
        reqs.append(tm)
    # RECOGIDA DE PALLETS COMPLETOS (2026-10-22): nivel fijo del log en el turno C (97% de la actividad); en A y B solo si se edita.
    ed_rp = ajustes.volumen_editado("recogida_pallets", turno) if ajustes else None
    base_rp = RECOGIDA_POR_CAJA * demanda_cajas_dia if turno == "C" else 0.0
    if ed_rp is not None or base_rp > 0:
        rp = requerimiento("recogida_pallets", turno, demanda_cajas_dia, parametros, 1.0, ed_rp if ed_rp is not None else base_rp, "pallets completos recogidos para pedidos", ajustes,
                           volumen_editado=ed_rp is not None, n_volumen=None if ed_rp is not None else _RG["n_noches"])
        rp.supuestos.append(f"{1000 * RECOGIDA_POR_CAJA:.3f} pallets completos por 1.000 cajas vendidas (razón de Diana; el log da {RECOGIDA_PALLETS_NOCHE:.1f} por noche con la venta media de sep-oct)")
        reqs.append(rp)
    return reqs, sin_vol


def reparto_picking_excedente(demanda_cajas_dia: float, parametros=None, ajustes: Ajustes | None = None) -> dict:
    """Reparto del picking entre turnos cuando `Ajustes.picking_excedente_a` está activo (opción desactivada por defecto, reunión 2026-10-24).

    Listas del día = driver de picking (igual que siempre). Capacidad del turno C en listas = (R·T − horas de presencia de las otras tareas del Operario en C − traslados)
    ÷ horas de presencia por lista, con R = dotación real de operarios del C (`config.personas_turno`) y T = horas efectivas del turno. El C hace min(listas, capacidad) y el
    turno A el resto; el B no pickea. No cambia tiempos, utilización ni driver: solo dónde cae el volumen."""
    if parametros is None:
        parametros = fit_params.cargar()
    aj0 = replace(ajustes or Ajustes(), picking_excedente_a=False)
    reqs_c, _ = _resolver_turno("C", demanda_cajas_dia, {**_cuotas("C", aj0), "picking": 0.0}, parametros, None, aj0)
    otras = sum(r.horas_persona for r in reqs_c if r.rol == "Operario" and r.proceso != "picking") + aj0.otras_horas("Operario", "C") / aj0.eficiencia_resto
    R = aj0.personas("C", "Operario")
    T = _duracion_turno("C", aj0)
    cap_h = max(0.0, R * T - otras - R * aj0.traslado_min("Operario", "C") / 60.0)
    sonda = requerimiento("picking", "C", demanda_cajas_dia, parametros, 1.0, ajustes=aj0)
    total = sonda.volumen
    hp_lista = sonda.horas_persona / total if total else 0.0
    cap = cap_h / hp_lista if hp_lista else float("inf")
    en_c = min(total, cap)
    exc = total - en_c
    return {"listas_dia": total, "capacidad_c_listas": cap, "listas_c": en_c, "listas_a": exc, "operarios_c": R, "horas_otras_tareas_c": otras,
            "cuota": {"C": en_c / total if total else 1.0, "A": exc / total if total else 0.0, "B": 0.0}}


LISTAS_POR_VUELTA_C = 31.0       # listas de picking por vuelta de camión del turno C (dato del usuario, 2026-10-24: «unas 31 listas por vuelta»)


def excedente_picking_c(demanda_cajas_dia: float, parametros=None, ajustes: Ajustes | None = None) -> dict:
    """Con el picking 100% en C: cuántas listas NO caben en la dotación real del C (`reparto_picking_excedente`), y a cuántas vueltas de camión equivalen (LISTAS_POR_VUELTA_C).
    Solo informa: no mueve nada al turno A."""
    r = reparto_picking_excedente(demanda_cajas_dia, parametros, ajustes)
    exc = max(0.0, r["listas_dia"] - r["capacidad_c_listas"])
    return {"listas_dia": r["listas_dia"], "capacidad_c_listas": r["capacidad_c_listas"], "excedente_listas": exc, "excedente_vueltas": exc / LISTAS_POR_VUELTA_C, "operarios_c": r["operarios_c"]}


def camiones_recepcion(demanda_cajas_dia: float) -> dict:
    """Camiones por día de cada tipo: grandes = razón por 1.000 cajas facturadas; chicos = nivel fijo del log (no escalan con la venta)."""
    out = {}
    for t, x in RECEPCION_CAMIONES["tipos"].items():
        out[t] = x["por_1000_cajas"] * demanda_cajas_dia / 1000 if x["escala_con_venta"] else x["camiones_dia_media_todos"]
    return out


def _rehacer(r: "Requerimiento", volumen: float, t_std: float, supuesto: str) -> "Requerimiento":
    ht = volumen * t_std / 3600
    hp = ht / r.eficiencia if r.eficiencia else 0.0
    n = hp / r.duracion_turno_h if r.duracion_turno_h else 0.0
    return replace(r, volumen=volumen, tiempo_estandar_s=t_std, horas_trabajo=ht, horas_persona=hp, personas_necesarias=n, brecha=n - r.personas_reales,
                   supuestos=r.supuestos + [supuesto], fuentes=r.fuentes + ["recepción por tipo de camión (modelo/recepcion_camiones.py)"])


def _recepcion_por_camion(r: "Requerimiento", turno: str, demanda: float, ajustes: "Ajustes") -> "Requerimiento":
    tipos = RECEPCION_CAMIONES["tipos"]
    n = camiones_recepcion(demanda)
    if r.proceso == "recepcion_directo_stage":
        pal = sum(n[t] * x["reparto_turno"][turno] * x["pallets_por_camion"] for t, x in tipos.items())
        pp = lambda x: x["pct_escaneo_planificador_log"] if ajustes.recepcion_pct_planificador is None else ajustes.recepcion_pct_planificador
        gs = lambda x: pp(x) * x["gruero_s_con_planificador"] + (1 - pp(x)) * x["gruero_s_sin_planificador"]
        seg = sum(n[t] * x["reparto_turno"][turno] * gs(x) for t, x in tipos.items())
        vol = r.volumen if ajustes.volumen_editado("recepcion_directo_stage", turno) is not None else pal
        t_to = seg_recepcion_por_pallet(ajustes.recepcion_pct_planificador)
        t_std = r.tiempo_estandar_s if (ajustes.tiempo("recepcion_directo_stage") is not None or t_to is None) else t_to      # un solo tiempo por pallet en los tres turnos
        detalle = ", ".join(f"{n[t] * x['reparto_turno'][turno]:.1f} {t}s" for t, x in tipos.items())
        # [2026-10-29] EDICIÓN POR TIPO DE CAMIÓN (parámetros editables del estimador): camiones por día y segundos por camión de cada tipo, claves «recepcion_chico» / «recepcion_grande» en
        # Ajustes.volumenes_turno y Ajustes.tiempos_override. Con alguna edición, los pallets son Σ camiones × pallets por camión y el tiempo por pallet del turno es Σ camiones × s ÷ Σ pallets
        # (las horas de gruero son exactamente Σ camiones × segundos por camión). Sin ediciones no cambia nada. Una edición en pallets de la recepción completa (clave anterior) tiene prioridad.
        ed_n = {t: ajustes.volumen_editado(f"recepcion_{t}", turno) for t in tipos}
        ed_s = {t: ajustes.tiempo(f"recepcion_{t}") for t in tipos}
        if (any(v is not None for v in ed_n.values()) or any(v is not None for v in ed_s.values())) \
                and ajustes.volumen_editado("recepcion_directo_stage", turno) is None and ajustes.tiempo("recepcion_directo_stage") is None:
            cam = {t: ed_n[t] if ed_n[t] is not None else n[t] * x["reparto_turno"][turno] for t, x in tipos.items()}
            sg = {t: ed_s[t] if ed_s[t] is not None else gs(x) for t, x in tipos.items()}
            vol = sum(cam[t] * x["pallets_por_camion"] for t, x in tipos.items())
            t_std = sum(cam[t] * sg[t] for t in tipos) / vol if vol > 0 else t_std
            detalle = ", ".join(f"{cam[t]:.1f} {t}s" + (" (editado)" if ed_n[t] is not None else "") for t in tipos)
            return _rehacer(r, vol, t_std, f"recepción por tipo de camión en el turno {turno} con edición del usuario: {detalle}; segundos de gruero por camión "
                                           + " y ".join(f"{sg[t]:.0f} s el {t}" + (" (editado)" if ed_s[t] is not None else "") for t in tipos) + f" = {t_std:.1f} s por pallet")
        return _rehacer(r, vol, t_std, f"recepción por tipo de camión en el turno {turno}: {detalle}; {t_std:.1f} s de gruero por pallet (sin la espera del camión; escanea el planificador en el "
                                       f"{100 * pp(tipos['chico']):.0f}% de los chicos y el {100 * pp(tipos['grande']):.0f}% de los grandes)")
    if r.proceso == "almacenamiento" and ajustes.volumen_editado("almacenamiento", turno) is None:
        nuevo = sum(n[t] * x["pallets_por_camion"] for t, x in tipos.items())
        viejo = PALLETS_POR_1000 * demanda / 1000
        f = nuevo / viejo if viejo else 1.0
        return _rehacer(r, r.volumen * f, r.tiempo_estandar_s, f"almacenamiento = pallets recibidos por tipo de camión ({nuevo:.0f} por día en vez de {viejo:.0f}); mismo reparto por turno y tiempo")
    return r


def por_turno(turno: str, demanda_cajas_dia: float, cuotas: dict[str, float] | None = None,
              parametros=None) -> list[Requerimiento]:
    """Detalle por proceso de los procesos que ocurren en el turno y tienen volumen."""
    return _resolver_turno(turno, demanda_cajas_dia, cuotas, parametros)[0]


# CALIBRACIÓN 1 DE 2 — PRÁCTICA DE TRABAJO (ver el bloque de config.PROCESOS_POR_TURNO).
# Reparto del volumen de cada proceso entre turnos en el RÉGIMEN VIGENTE (desde el 22-sep-2026:
# 8 días operados, 22 al 30-sep, de los exports WMS de septiembre). Es una decisión de
# organización del CD, y cambió el 22-sep, así que describe el estado ACTUAL y no se mezcla con
# el período anterior. Se recalcula de `wms_sep_turnos.csv` en tests/test_capacity.py.
# [ACTUALIZADO 2026-10-13: el régimen vigente es OCTUBRE, ver `calibracion.py`; ya no se escriben a mano.]
# Antes del 22-sep (CUOTAS_TURNO_HISTORICAS) palés era 48/37/15% y cajas 81/19/0%.
# Cuota = fracción del nivel diario de cada proceso que cae en cada turno, calculada con el log (`calibracion.perfil`) en la ventana de cada régimen. Picking NO tiene
# datos de octubre: conserva la cuota de septiembre (A 0 · B 0,0001 · C 0,9999 en el vigente de entonces; 0,03/0/0,97 en el anterior, que incluye el derrame del peak 14-16-sep).
# Carga de camión (vueltas >= 2): todas en A en el vigente (el B cargó 0,6 trailers por día en octubre); en el anterior se reparte A/B según el log.
_BASE_VIGENTE = {"A": {"picking": 0.0, "carga_camion": 1.0}, "B": {"picking": 0.0001, "carga_camion": 0.0}, "C": {"picking": 0.9999}}
_BASE_ANTERIOR = {"A": {"picking": 0.03, "carga_camion": 0.717}, "B": {"picking": 0.0, "carga_camion": 0.283}, "C": {"picking": 0.97}}
_RESPALDO_SEP22 = {"A": {"reabasto_pallet": 0.022, "reabasto_caja": 0.403, "recepcion_directo_stage": 0.706, "almacenamiento": 0.587, "retorno_fletero": 0.14, "merma": 0.20},
                   "B": {"reabasto_pallet": 0.076, "reabasto_caja": 0.038, "recepcion_directo_stage": 0.208, "almacenamiento": 0.368, "retorno_fletero": 0.85, "merma": 0.11},
                   "C": {"reabasto_pallet": 0.902, "reabasto_caja": 0.559, "recepcion_directo_stage": 0.086, "almacenamiento": 0.045, "retorno_fletero": 0.01, "merma": 0.69}}
_RESPALDO_ANT = {"A": {"reabasto_pallet": 0.48, "reabasto_caja": 0.81, "recepcion_directo_stage": 0.48, "almacenamiento": 0.197, "retorno_fletero": 0.38, "merma": 0.55},
                 "B": {"reabasto_pallet": 0.37, "reabasto_caja": 0.19, "recepcion_directo_stage": 0.52, "almacenamiento": 0.769, "retorno_fletero": 0.60, "merma": 0.03},
                 "C": {"reabasto_pallet": 0.15, "reabasto_caja": 0.0, "recepcion_directo_stage": 0.0, "almacenamiento": 0.033, "retorno_fletero": 0.03, "merma": 0.42}}


def _cuotas_regimen(regimen: str) -> dict:
    base = _BASE_ANTERIOR if regimen == "anterior" else _BASE_VIGENTE
    resp = _RESPALDO_ANT if regimen == "anterior" else _RESPALDO_SEP22
    p = calibracion.perfil(regimen)
    out = {t: dict(base[t]) for t in config.TURNOS}
    for t in config.TURNOS:
        for proc in _PROCESOS_CON_NIVEL:
            out[t][proc] = p[proc]["cuota"][t] if p else resp[t][proc]
    if regimen in ("anterior", "combinado") and p:
        a, b = p["carga_camion"]["por_turno"]["A"], p["carga_camion"]["por_turno"]["B"]
        out["A"]["carga_camion"], out["B"]["carga_camion"] = a / (a + b), b / (a + b)
    for t in config.TURNOS:          # 2026-10-21 (decisión del usuario): reabasto de pallets 20/2/78 y carga fuera de C toda en A, en todos los regímenes
        out[t]["reabasto_pallet"] = REPARTO_REABASTO_PALLET[t]
    out["A"]["carga_camion"], out["B"]["carga_camion"] = 1.0, 0.0
    if regimen == "combinado":      # picking: mezcla de las cuotas de septiembre por días de cada ventana (el log de picking solo llega al 30-sep)
        w = 18 / 28
        for t in config.TURNOS:
            out[t]["picking"] = w * _BASE_ANTERIOR[t]["picking"] + (1 - w) * _BASE_VIGENTE[t]["picking"]
    for t in config.TURNOS:
        out[t]["reabasto_caja_grua"] = out[t]["reabasto_caja"]
    return out


CUOTAS_TURNO = _cuotas_regimen("vigente")
CUOTAS_TURNO_INTERMEDIO = _cuotas_regimen("intermedio")          # 22-sep a 1-oct (el «vigente» de las versiones anteriores del modelo)
CUOTAS_TURNO_HISTORICAS = _cuotas_regimen("anterior")
CUOTAS_TURNO_COMBINADO = _cuotas_regimen("combinado")              # 4-sep a 10-oct: los tres regímenes juntos


def demanda_septiembre() -> dict | None:
    """CALIBRACIÓN 2 DE 2 — VOLUMEN DE DEMANDA: septiembre COMPLETO, sin recortar.

    A diferencia de las cuotas por turno (práctica vigente, solo desde el 22-sep), la demanda
    usa los 22 días operados del 4 al 30-sep: la semana peak de Fiestas Patrias (hasta 37.077
    cajas en la jornada del 15-sep) MÁS el período normal posterior (17-22k). Esa variedad es lo
    que permite correlacionar el volumen con las ventas; recortar al régimen vigente dejaría 8
    días casi constantes. La demanda es del mercado, no una decisión de organización del CD.

    Unidad: cajas de PICKING por jornada (log WMS), proxy de las cajas vendidas: las ventas de
    septiembre no están (terminan el 17-ago) y falta el factor de conversión. [supuesto]
    Devuelve None si no existe `wms_sep_turnos.csv` (correr `modelo.wms_septiembre`)."""
    import csv
    import statistics
    ruta = config_salidas_turnos()
    if not ruta.exists():
        return None
    por_dia: dict[str, int] = {}
    with open(ruta, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["fecha"] < "2026-10-01":
                por_dia[r["fecha"]] = por_dia.get(r["fecha"], 0) + int(r["picking_cajas"])
    v = sorted(por_dia.values())
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"dias": len(v), "media": statistics.mean(v), "p50": q(0.5), "p90": q(0.9),
            "min": v[0], "max": v[-1]}


def driver_almacenamiento_desde_log() -> dict | None:
    """Recalcula el driver de almacenamiento y sus cuotas desde las salidas de `wms_septiembre`
    (`wms_sep_ciclos.csv` y `wms_sep_turnos.csv`), para que no puedan quedar desactualizados.
    Driver (calibración 2: septiembre completo, 4 a 30-sep) y cuotas por turno (calibración 1:
    régimen vigente, 22 a 30-sep). Devuelve None si faltan los archivos."""
    import csv
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent / "salidas" / "limpio"
    f_ci, f_tu = base / "wms_sep_ciclos.csv", base / "wms_sep_turnos.csv"
    if not (f_ci.exists() and f_tu.exists()):
        return None
    finales = {"ALMAC", "ALMPIC", "SURTID"}
    por_dia: dict[str, int] = {}
    por_turno: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    with open(f_ci, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["proceso"] == "almacenamiento" and r["area_origen"] == "PATCA" and r["area_destino"] in finales:
                por_dia[r["fecha"]] = por_dia.get(r["fecha"], 0) + 1
                if "2026-09-22" <= r["fecha"] < "2026-10-01":
                    por_turno[r["turno"]] += 1
    cajas: dict[str, int] = {}
    with open(f_tu, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["fecha"] < "2026-10-01":
                cajas[r["fecha"]] = cajas.get(r["fecha"], 0) + int(r["recepcion_cajas"])
    dias = sorted(cajas)
    tareas = sum(por_dia.get(d, 0) for d in dias)
    total_t = sum(por_turno.values())
    return {"dias": len(dias), "tareas_dia": tareas / len(dias), "cajas_dia": sum(cajas.values()) / len(dias),
            "tareas_por_caja": tareas / sum(cajas.values()),
            "cuotas": {t: v / total_t for t, v in por_turno.items()}}


def config_salidas_turnos():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "wms_sep_turnos.csv"


# =====================================================================================================
# VOLUMEN DE REABASTO DE CAJAS POR TURNO (decisión del equipo, 2026-10-06). Antes: un nivel diario (45,18 retiros) x la cuota de cada turno.
#   - TURNO C: el log del WMS (tareas dirigidas tras el picking), contado directo: 162 retiros en 8 noches = 20,25 por noche (régimen vigente).
#   - TURNOS A y B: las hojas de los planificadores (reposiciones fuera de umbral, que el WMS no ve; ejecutado estimado = filas de más de 5 cajas)
#     MÁS el residual dirigido que muestra el log (A 14,6 y B 1,4 retiros por día, régimen vigente). Hoy hay UNA sola hoja (2-oct, emitida en el
#     turno A): suma 84 al turno A; el B no tiene hoja y queda solo con el log (subestimado). Estado `preliminar` por decisión del equipo (por N=1
#     día sería `insuficiente`); la fuente va a la vista.
#   - La hoja es un PARÁMETRO POR FECHA: `reabasto_planificadores` descubre los libros de la carpeta de hojas; `Ajustes.hojas` elige cuáles sumar.
# RECONCILIACIÓN 45,18 vs 20,2 [verificado]: 45,18 es el total diario A+B+C promediado en los 22 días operados de septiembre (régimen anterior ≈ 55 retiros
#   por día, casi todos de día; vigente ≈ 36,2). 45,18 x 0,559 (cuota C vigente) = 25,3 mezcla dos calibraciones (el nivel del mes completo, inflado
#   por el régimen anterior, con la cuota del vigente); la medición directa de C en el vigente es 20,25 por noche (-20%). Con los volúmenes por turno
#   desaparece esa mezcla.
# ¿SE PUEDEN SUMAR LOG Y HOJAS? Misma unidad [verificado]: un retiro del log es una reposición de un SKU (origen -> TRF) y una fila de la hoja es una
#   reposición de un SKU (origen -> destino, con cantidad). Cajas por unidad: 14,3 por retiro en el log vigente (18,6 si se miran solo los de más de 5
#   cajas) contra 22,4 por fila ejecutada de la hoja: mismo orden. Son DISJUNTAS por construcción (la hoja trae lo que no alcanza el umbral del WMS y por
#   eso no genera tarea) [dato del equipo]. Límites declarados: el umbral de ejecución difiere (la hoja excluye las de 5 cajas o menos; el log trae 27% de
#   retiros de 5 cajas o menos), la hoja es una foto de UN día y no se reparte por hora. Aun así se muestran siempre las dos cifras por separado.
# =====================================================================================================
VENTANA_VIGENTE = calibracion.REGIMENES["vigente"]       # [desde, hasta): octubre, 2 al 10-oct
VENTANA_ANTERIOR = calibracion.REGIMENES["anterior"]     # 4 al 21-sep (incluye Fiestas Patrias)
SALIDA_HOJAS = None                                      # ruta del CSV de hojas (se resuelve perezosamente)


def hojas_disponibles() -> list[dict]:
    """Hojas de ruta de CAJAS descubiertas (CSV de `reabasto_planificadores`): una fila por (fecha, turno de emisión) con las reposiciones listadas y
    las ejecutadas estimadas (> 5 cajas). Vacío si no hay hojas."""
    import csv
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "reabasto_planificadores.csv"
    if not ruta.exists():
        return []
    g: dict[tuple, dict] = {}
    with open(ruta, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["tipo"] != "caja":
                continue
            d = g.setdefault((r["fecha"], r["turno_emision"]), {"fecha": r["fecha"], "turno": r["turno_emision"], "hora": r["hora_emision"], "listadas": 0, "ejecutadas_estimadas": 0})
            d["listadas"] += 1
            d["ejecutadas_estimadas"] += int(r["ejecutable_estimado"])
    return sorted(g.values(), key=lambda d: d["fecha"])


def volumen_reabasto_caja(regimen: str = "vigente", hojas: tuple | None = (), con_parciales: bool = True) -> dict[str, dict]:
    """Reposiciones de cajas por día y turno = log (retiros dirigidos del WMS) + movimientos de inventario PARCIAL (log; origen una posición y destino una posición de picking) + hojas de los
    planificadores (ejecutado estimado; OPCIONAL). CAMBIO 2026-10-15: los movimientos parciales son la ejecución real, medida, de lo que indican las hojas; por eso las hojas ya NO se suman por
    defecto (`hojas=()`): sumarlas junto con los parciales cuenta dos veces la misma reposición. `hojas`: None = todas las descubiertas; () = ninguna; tupla de fechas ISO = esas (solo en el régimen
    vigente y desde su fecha). Devuelve {turno: {log, parcial, hojas, total, n_dias_log, hojas_usadas, fuente, estado}}."""
    desde = calibracion.REGIMENES[regimen][0]
    perf = calibracion.perfil(regimen).get("reabasto_caja")
    par = calibracion.perfil(regimen).get("reabasto_caja_parcial")
    hs = [h for h in hojas_disponibles() if regimen == "vigente" and h["fecha"] >= desde and (hojas is None or h["fecha"] in hojas)]
    out = {}
    for t in config.TURNOS:
        log_t = perf["por_turno"][t] if perf else 0.0
        n = perf["dias"][t] if perf else 0
        par_t = par["por_turno"][t] if (par and con_parciales) else 0.0
        ht = [h for h in hs if h["turno"] == t]
        h_val = sum(h["ejecutadas_estimadas"] for h in ht) / len(ht) if ht else 0.0
        fuente = (f"log WMS dirigido ({regimen}: {perf['suma'][t] if perf else 0} retiros en {n} días = {log_t:.2f} por día)" +
                  (f" + movimientos de inventario parcial ({par['suma'][t]} en {par['dias'][t]} turnos = {par_t:.2f} por día)" if (par and con_parciales) else "") +
                  (f" + {len(ht)} hoja(s) de planificadores ({', '.join(h['fecha'] for h in ht)}; ejecutado estimado, promedio {h_val:.0f} por hoja) [preliminar, 1 día por hoja; puede duplicar los parciales]" if ht else ""))
        out[t] = {"log": log_t, "parcial": par_t, "hojas": h_val, "total": log_t + par_t + h_val, "n_dias_log": n, "hojas_usadas": [h["fecha"] for h in ht],
                  "fuente": fuente, "estado": "preliminar"}
    return out


# ---------------------------------------------------------------- agregación

# Uso de equipos compartidos, DECLARADO POR PROCESO e independiente de los roles.
#
# POR QUÉ NO SE DERIVA DEL ROL (corregido el 2026-09-30): antes las grúas se calculaban con
# las horas de "los procesos del rol Gruero", o sea exactamente la misma variable que la
# dotación de Gruero. Daban el mismo número (0,23 de 4 en turno C) por construcción, no
# porque fueran magnitudes equivalentes: coincidían además porque hay 4 grúas y 4 grueros.
# Es un acoplamiento oculto — cambiar el rol de un proceso habría movido las grúas sin
# avisar, y un proceso que usa grúa sin que su rol sea Gruero (reabasto de cajas) no
# habría contado nunca. Ahora cada proceso declara el equipo que ocupa.
#
# Valor = fracción del tiempo de trabajo del proceso en que el equipo está ocupado.
#   None = el proceso SÍ usa el equipo pero sus horas de equipo no están medidas: no se
#          inventa una fracción, el resultado se marca incompleto.
USO_DE_EQUIPO: dict[str, dict[str, float | None]] = {
    # [confirmado, wms_config.py] todo reabasto exige la grúa GR_AB_ALMA. La fracción 1.0
    # para reabasto_pallet es [supuesto]: que la grúa esté ocupada todo el ciclo.
    "reabasto_pallet": {"Grúas": 1.0},
    # [confirmado, wms_config.py] REAPROCAJA exige GR_AB_ALMA. Desde la planilla oficial (2026-10-13) el tiempo de la grúa SÍ está medido: es la parte del
    # gruero (`reabasto_caja_grua`, bajar los pallets: 6 visitas, 98,8 s por reposición servida, `insuficiente`). Se asume la grúa ocupada todo ese tiempo [supuesto].
    "reabasto_caja_grua": {"Grúas": 1.0},
    # [supuesto, pendiente de confirmar con Víctor] La planilla registra grúas y
    # transpaletas usadas por muestra (hojas Carga, Recepción y Almacenamiento); esa
    # columna todavía no se carga y podría reemplazar estas fracciones por datos.
    "recepcion_directo_stage": {"Grúas": 1.0},
    "carga_camion": {"Grúas": 1.0},
    "carga_cl": {"Grúas": 1.0},                     # [supuesto] igual que la carga de camión
    "traslado_stage": {"Grúas": 1.0},
    "traslado_mixtos": {"Grúas": 1.0},              # el gruero toma el mixto (la lista armada) y lo lleva a su posición con grúa (confirmado por el usuario, 2026-11-08)
    "recogida_pallets": {"Grúas": 1.0},             # saca el pallet de altura: grúa todo el ciclo [supuesto]
    "almacenamiento": {"Grúas": 1.0},
}

# Procesos en los que un rol PARTICIPA pero cuyas horas en ese rol no están medidas.
# reabasto_caja: el AS-IS (docs/procesos_as_is.md, pasos 10-11b) pone al gruero a bajar el
# pallet y volver a subirlo, pero el proceso está medido desde el lado del operario.
# [Desde 2026-10-13 el gruero del reabasto de cajas SÍ está medido (`reabasto_caja_grua`); queda vacío, pero se conserva el mecanismo.]
PARTICIPACION_SIN_MEDIR: dict[str, tuple[str, ...]] = {}


@dataclass
class Dotacion:
    """Cifra agregada para un recurso (rol de personal o tipo de equipo) en un turno."""
    recurso: str
    tipo: str                       # "rol" | "equipo"
    turno: str
    horas_trabajo: float            # suma de los procesos que lo componen
    eficiencia: float               # efectiva: horas_trabajo / horas_persona
    duracion_turno_h: float
    necesarios: float               # horas_trabajo / (duracion x eficiencia)
    reales: int
    brecha: float                   # necesarios - reales (positivo = faltan)
    estado_confianza: str           # el MÁS BAJO de los procesos que lo componen
    limitado_por: str               # proceso que fija esa confianza
    pct_horas_validadas: float      # qué parte de las horas viene de parámetros validados
    procesos: list[Requerimiento] = field(default_factory=list)
    # Lo que falta para que la cifra sea completa. Si alguna de las dos no está vacía, la
    # cifra es una COTA INFERIOR: el recurso hace falta para más cosas de las que suma.
    sin_volumen: dict[str, str] = field(default_factory=dict)
    sin_horas_medidas: list[str] = field(default_factory=list)
    meta_fill_rate: None = None     # brecha #7: sin confirmar
    # Procesos que SÍ suman horas pero cuyo tiempo no es un ciclo medido (SIN_CICLO_MEDIDO: reabasto de cajas). Mientras
    # existan, la cifra es una cota inferior: ese tiempo podría ser menor o mayor y no hay medición que lo acote.
    sin_ciclo_medido: list[str] = field(default_factory=list)
    # N = n_respaldado + n_no_respaldado (personas). Respaldado: horas de procesos con tiempo Y driver `validado`.
    # No respaldado: todo el resto. A diferencia de HV (% de horas) no se diluye cuando crece el trabajo respaldado:
    # la parte no respaldada se ve en personas absolutas.
    n_respaldado: float = 0.0
    n_no_respaldado: float = 0.0
    # N sin los procesos de tiempo sin ciclo medido: el piso que sí se puede defender como cota inferior de N real.
    n_cota: float | None = None

    @property
    def completo(self) -> bool:
        return not self.sin_volumen and not self.sin_horas_medidas and not self.sin_ciclo_medido

    @property
    def alcanza(self) -> bool | None:
        """True / False solo cuando se puede afirmar. Una cota inferior puede probar que
        NO alcanza (si ya con lo que se suma falta gente), pero nunca que sí.

        Se revisa la cota ANTES que la confianza (2026-10-05): si lo calculado ya supera a los reales, 'no alcanza' vale
        aunque la confianza sea baja, porque lo que falta por sumar solo puede aumentar N. Para esa prueba se usa
        `n_cota` (N sin los procesos de tiempo sin ciclo medido), no N completo: un tiempo no medido podría ser menor."""
        if not self.completo:
            piso = self.necesarios if self.n_cota is None else self.n_cota
            if piso > self.reales:
                return False
        if self.estado_confianza in ("insuficiente", "sustituto_dia", "sin_datos"):
            return None
        if not self.completo:
            return None
        return self.brecha <= 0


def _duracion_turno(turno: str, ajustes: "Ajustes | None" = None) -> float:
    """Horas EFECTIVAS del turno: 8 − charla − colación (defecto 7, decisión del equipo 2026-10-13: `config.HORAS_EFECTIVAS_POR_TURNO`). Editables en `Ajustes`."""
    return config.HORAS_EFECTIVAS_POR_TURNO if ajustes is None else ajustes.horas_efectivas()


def _agregar(recurso: str, tipo: str, turno: str, rs: list[Requerimiento], reales: int,
             fraccion: dict[str, float | None] | None = None,
             sin_volumen: dict[str, str] | None = None,
             sin_horas_medidas: list[str] | None = None, ajustes: "Ajustes | None" = None, traslado_min: float = 0.0) -> Dotacion:
    """Suma las horas de los procesos `rs` y divide UNA sola vez al final.

    `fraccion` (solo equipos): qué parte del tiempo de cada proceso ocupa el equipo."""
    dur = _duracion_turno(turno, ajustes)
    sin_volumen = dict(sin_volumen or {})
    sin_horas = list(sin_horas_medidas or [])
    horas = horas_persona = 0.0
    hp_resp = hp_sin_ciclo = 0.0
    aportan: list[Requerimiento] = []
    for r in rs:
        f = 1.0 if fraccion is None else fraccion.get(r.proceso)
        if f is None:                       # usa el equipo, horas sin medir
            sin_horas.append(r.proceso)
            continue
        if r.horas_trabajo > 0:             # un proceso sin volumen no degrada la confianza
            aportan.append(r)
            horas += r.horas_trabajo * f
            horas_persona += r.horas_persona * f
            if r.estado_confianza == "validado":
                hp_resp += r.horas_persona * f
            if r.sin_ciclo_medido:
                hp_sin_ciclo += r.horas_persona * f
    ef = horas / horas_persona if horas_persona else EFICIENCIA_SUPUESTA
    if traslado_min > 0 and tipo == "rol" and horas > 0 and dur * ef > traslado_min / 60:
        # TRASLADOS: cada persona dedica `traslado_min` min por turno a recorridos que no están dentro del tiempo de una tarea. N = H / (dur·ef − m/60) (cerrada: el traslado crece con N).
        n_tr = horas / (dur * ef - traslado_min / 60)
        h_tr = n_tr * traslado_min / 60
        nombre = "traslados_" + recurso.lower()
        rt = Requerimiento(proceso=nombre, turno=turno, rol=recurso, demanda_cajas_dia=0.0, volumen=traslado_min, unidad_volumen="min por persona por turno", tiempo_estandar_s=60.0,
                           horas_trabajo=h_tr, eficiencia=ef, horas_persona=h_tr / ef, duracion_turno_h=dur, personas_necesarias=h_tr / ef / dur, personas_reales=0, brecha=0.0,
                           estado_confianza="insuficiente", fuentes=["traslados EDITADOS por el usuario"],
                           supuestos=[f"{traslado_min:g} min por persona y turno de traslados del rol {recurso}, fijados por el usuario (no hay medición)"])
        aportan.append(rt)
        horas += h_tr
        horas_persona += h_tr / ef
    necesarios = horas / (dur * ef) if dur and ef else 0.0
    if aportan:
        estado = peor_confianza(*(r.estado_confianza for r in aportan))
        limita = next(r.proceso for r in aportan if r.estado_confianza == estado)
        validadas = sum(r.horas_trabajo for r in aportan if r.estado_confianza == "validado")
        pct = validadas / sum(r.horas_trabajo for r in aportan)
    else:
        estado, limita, pct = "sin_datos", "—", 0.0
    n_resp = hp_resp / dur if dur else 0.0
    return Dotacion(recurso, tipo, turno, horas, ef, dur, necesarios, reales,
                    necesarios - reales, estado, limita, pct, aportan,
                    sin_volumen, sorted(set(sin_horas)),
                    sin_ciclo_medido=sorted({r.proceso for r in aportan if r.sin_ciclo_medido}),
                    n_respaldado=n_resp, n_no_respaldado=necesarios - n_resp,
                    n_cota=necesarios - (hp_sin_ciclo / dur if dur else 0.0))


def dotacion_por_turno(turno: str, demanda_cajas_dia: float,
                       cuotas: dict[str, float] | None = None, parametros=None,
                       volumenes: dict[str, float] | None = None,
                       ajustes: Ajustes | None = None) -> list[Dotacion]:
    """Dotación necesaria por rol y por equipo compartido en un turno.

    Es el **resultado principal** del módulo. Suma SOLO las horas de los procesos que
    `config.PROCESOS_POR_TURNO` declara para ese turno, agrupadas por rol (personal) y por
    `USO_DE_EQUIPO` (equipos, calculados por separado de los roles), y divide una sola vez
    al final. El detalle por proceso queda en `Dotacion.procesos`.

    `volumenes` sobrescribe volúmenes fijos declarados (análisis de sensibilidad)."""
    if parametros is None:
        parametros = fit_params.cargar()
    reqs, sin_vol = _resolver_turno(turno, demanda_cajas_dia, cuotas, parametros, volumenes, ajustes)
    ocurren = set(_procesos(turno, ajustes)[0])
    out: list[Dotacion] = []
    if ajustes is not None:                     # OTRAS ACTIVIDADES (editables): horas de trabajo por rol y turno fuera de los procesos modelados
        ef_o = ajustes.eficiencia_resto
        for rol_o in ROLES_ESTIMADOR:
            h_o = ajustes.otras_horas(rol_o, turno)
            if h_o > 0:
                dur_o = _duracion_turno(turno, ajustes)
                reqs.append(Requerimiento(proceso="otras_actividades_" + rol_o.lower(), turno=turno, rol=rol_o, demanda_cajas_dia=demanda_cajas_dia, volumen=h_o, unidad_volumen="horas de trabajo por turno",
                                          tiempo_estandar_s=3600.0, horas_trabajo=h_o, eficiencia=ef_o, horas_persona=h_o / ef_o, duracion_turno_h=dur_o, personas_necesarias=h_o / ef_o / dur_o,
                                          personas_reales=0, brecha=0.0, estado_confianza="insuficiente", fuentes=["otras actividades EDITADAS por el usuario"],
                                          supuestos=[f"{h_o:g} h de trabajo por turno de otras actividades del rol {rol_o}, fijadas por el usuario (no hay medición)"]))

    roles = ({r.rol for r in reqs}
             | {ROL_POR_PROCESO[p] for p in sin_vol if p in ROL_POR_PROCESO}
             | {rol for p, rs_ in PARTICIPACION_SIN_MEDIR.items() if p in ocurren for rol in rs_})
    for rol in sorted(roles):
        out.append(_agregar(
            rol, "rol", turno, [r for r in reqs if r.rol == rol], (ajustes.personas(turno, rol) if ajustes else config.personas_turno(turno, rol, None)),
            sin_volumen={p: m for p, m in sin_vol.items() if ROL_POR_PROCESO.get(p) == rol},
            sin_horas_medidas=[p for p, rs_ in PARTICIPACION_SIN_MEDIR.items()
                               if p in ocurren and rol in rs_], ajustes=ajustes, traslado_min=ajustes.traslado_min(rol, turno) if ajustes else 0.0))

    equipos = config.equipos_disponibles()
    for equipo in sorted({e for usos in USO_DE_EQUIPO.values() for e in usos}):
        usan = {p: usos[equipo] for p, usos in USO_DE_EQUIPO.items() if equipo in usos}
        out.append(_agregar(
            equipo, "equipo", turno, [r for r in reqs if r.proceso in usan],
            equipos.get(equipo, 0), fraccion=usan,
            sin_volumen={p: m for p, m in sin_vol.items() if p in usan}, ajustes=ajustes))
    return out


def sensibilidad_carga_nocturna(demanda_cajas_dia: float = 22_859.0,
                                camiones=(5.0, 17.0, 21.6, 22.6, 25.0), parametros=None) -> list[dict]:
    """Gruero y Grúas en turno C según cuántos camiones se carguen de noche.

    5 = lo que declaró David (el log lo contradice); 17 / 21,6 / 24 = mínimo, media y máximo del
    CONTEO DIRECTO de trailers por noche del 4-sep al 2-oct (24 noches); 22,6 = media de las 17 noches desde el 22-sep (valor de
    config.PROCESOS_POR_TURNO); 25 = el máximo (10-oct)."""
    if parametros is None:
        parametros = fit_params.cargar()
    res = []
    for c in camiones:
        d = {x.recurso: x for x in dotacion_por_turno("C", demanda_cajas_dia, None,
                                                      parametros, {"carga_camion": c})}
        res.append({"camiones": c, "gruero": d["Gruero"].necesarios,
                    "gruas": d["Grúas"].necesarios, "horas_gruero": d["Gruero"].horas_trabajo})
    return res


ANOTAR_POR_TAREA_S = 200.0     # [hipótesis, N=2 jornadas] (2.780 s / 10 tareas + 730 s / 6 tareas) / 2 ≈ 200 s por tarea anotada en papel (hoja «Reabastecimiento_cajas»)
TIEMPO_PLANILLA_ANTERIOR_S = 1109.0   # media simple de las 5 esperas de la planilla del 25-sep: el valor que usó el modelo hasta el 2026-10-12


def sensibilidad_reabasto_caja(demanda_cajas_dia: float = 24_204.0, fracciones=None, turnos=("C", "A", "B"),
                               parametros=None, regimen: str = "vigente") -> list[dict]:
    """Cuánto se mueve Operario según el tiempo por tarea de reabasto de cajas y la fuente de su volumen.

    El tiempo del modelo es un ciclo MEDIDO pero PARCIAL (SIN_CICLO_MEDIDO: 235 s, no incluye anotar tareas ni esperas), así que no se elige un valor: se barren
      · 0 s (el piso: Operario sin reabasto de cajas),
      · la mitad del ciclo medido,
      · el ciclo medido (el modelo),
      · el ciclo medido + anotar las tareas en papel (ANOTAR_POR_TAREA_S, N=2 jornadas) [hipótesis],
      · el valor de la planilla anterior (1.109 s de esperas), que fue el del modelo hasta el 2026-10-12,
    y tres variantes de volumen POR TURNO: «solo log dirigido», «log dirigido + movimientos parciales» (el modelo) y «+ hoja de los planificadores del 2-oct» (puede duplicar los parciales)."""
    if parametros is None:
        parametros = fit_params.cargar()
    out = []
    for t in turnos:
        ref = tiempo_estandar("reabasto_caja", t, parametros)[0]
        escenarios = ([(f"{fr:g} x ciclo medido", ref * fr) for fr in fracciones] if fracciones is not None else
                      [("piso: sin reabasto de cajas", 0.0), ("mitad del ciclo medido", ref * 0.5), ("ciclo medido (modelo)", ref),
                       ("ciclo + anotar tareas (N=2)", ref + ANOTAR_POR_TAREA_S), ("planilla anterior (esperas, 1.109 s)", TIEMPO_PLANILLA_ANTERIOR_S)])
        for variante, hojas, parc in (("solo log dirigido", (), False), ("log dirigido + parciales (modelo)", (), True), ("+ hoja de planificadores del 2-oct", None, True)):
            vol = volumen_reabasto_caja(regimen, hojas, parc)[t]["total"]
            for nombre, seg in escenarios:
                aj = Ajustes(regimen=regimen, hojas=hojas, con_parciales=parc, tiempos_override=(("reabasto_caja", seg),))
                op = next(d for d in dotacion_por_turno(t, demanda_cajas_dia, None, parametros, ajustes=aj) if d.recurso == "Operario")
                rc = next((r for r in op.procesos if r.proceso == "reabasto_caja"), None)
                out.append({"turno": t, "variante": variante, "escenario": nombre, "fraccion_tiempo": seg / ref if ref else 0.0, "tiempo_s": seg, "tareas_dia": vol,
                            "operario": op.necesarios, "reales": op.reales, "horas_reabasto_caja": rc.horas_persona if rc else 0.0})
    ref_ = {r["turno"]: r["operario"] for r in out if r["variante"].startswith("log dirigido + parciales") and abs(r["fraccion_tiempo"] - 1.0) < 1e-9}
    for r in out:
        r["vs_modelo"] = r["operario"] - ref_[r["turno"]]
    return out


# =====================================================================================================
# ESTIMADOR (pestaña «Capacidad necesaria» del dashboard, 2026-10-04). Toda la lógica vive aquí; el dashboard
# solo llama a estas funciones. Estima CARGA DE TRABAJO PROMEDIO: no modela colas ni contención (eso es de la
# simulación, pendiente).
# =====================================================================================================

# Cajas pickeadas por caja facturada (total del día, septiembre-2026: 846,11 por 1.000, R² 0,96); sigma = desvío
# residual de la regresión cajas pickeadas ~ facturado (1.094,9 cajas). Ver demanda_actividad.py.
FACTOR_CAJAS_PICKEADAS = 846.11 / 1000
SIGMA_CAJAS_PICKEADAS = 1094.87
ROLES_ESTIMADOR = ("Gruero", "Operario")
_PROCESOS_NIVEL_BASE = ("reabasto_pallet", "recepcion_directo_stage", "almacenamiento")      # reabasto_caja ya no: su volumen es POR TURNO (volumen_reabasto_caja)
_NOMBRE = {"picking": "picking", "reabasto_pallet": "reabasto de pallets", "reabasto_caja": "reabasto de cajas", "reabasto_caja_grua": "reabasto de cajas (gruero: bajar pallets)", "retorno_fletero": "retorno de fletero a bodega", "merma": "merma",
           "recepcion_directo_stage": "recepción (camiones)", "almacenamiento": "almacenamiento", "carga_camion": "carga de camión",
           "retorno_fletero_gruero": "retorno de fletero (gruero)", "carga_cl": "carga para el CL", "recogida_pallets": "recogida de pallets completos", "traslado_mixtos": "traslado de mixtos", "traslado_stage": "traslado a STREC", "otras_actividades_operario": "otras actividades (operario)", "otras_actividades_gruero": "otras actividades (gruero)",
           "traslados_operario": "traslados (operario)", "traslados_gruero": "traslados (gruero)"}


def ajustes_por_defecto() -> list[dict]:
    """Parámetros editables con su valor por defecto y su fuente, para mostrarlos en la interfaz."""
    out = [{"clave": "eficiencia_picking", "nombre": "Eficiencia de picking", "valor": EFICIENCIA_MEDIDA["picking"],
            "estado": "medida", "fuente": "horas de gaps de picking / horas-persona de presencia (log WMS, 28 noches)"},
           {"clave": "eficiencia_resto", "nombre": "Eficiencia del resto de los procesos", "valor": EFICIENCIA_SUPUESTA,
            "estado": "supuesto", "fuente": "NO medida: se copia la de picking"}]
    for p in _PROCESOS_NIVEL_BASE:
        d = DRIVERS[p]
        out.append({"clave": p, "nombre": f"Nivel base diario: {_NOMBRE[p]}", "valor": nivel_modelo(p), "estado": d.estado,
                    "fuente": f"{d.unidad} por día, media del régimen vigente ({d.fuente}); no escala con la venta"})
    return out


# Orden y rotulado de los procesos en las tablas de tiempo estándar (dashboard).
_NOMBRE_CAMION = {"chico": lambda x: f"Recepción, camión chico (~{x['pallets_por_camion']:.0f} pallets)",
                  "grande": lambda x: f"Recepción, camión grande (~{x['pallets_por_camion']:.0f} pallets)"}
PROCESOS_TIEMPO = ("picking", "reabasto_pallet", "reabasto_caja", "reabasto_caja_grua", "recepcion_directo_stage", "almacenamiento", "carga_camion", "carga_cl", "recogida_pallets", "traslado_mixtos", "traslado_stage", "retorno_fletero", "merma", "retorno_fletero_gruero")


def referencia_log_parciales() -> dict:
    """Ciclo medio entre tareas del log de movimientos parciales por proceso y turno (s), para mostrar junto al tiempo del modelo. {} si falta el CSV."""
    import csv
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "movimientos_parciales_tiempos.csv"
    if not ruta.exists():
        return {}
    clave = {"Reabasto de cajas": "reabasto_caja", "Retorno de fletero": "retorno_fletero", "Merma": "merma"}
    out: dict = {}
    with open(ruta, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = next((v for pre, v in clave.items() if r["proceso"].startswith(pre)), None)
            if k and r["ciclo_entre_tareas_media_s"]:
                out.setdefault(k, {})[r["turno"]] = (float(r["ciclo_entre_tareas_media_s"]), float(r["ciclo_entre_tareas_mediana_s"]), int(r["n_ciclos"]))
    return out


def tiempos_modelo(parametros=None, ajustes: Ajustes | None = None) -> list[dict]:
    """Tiempo estándar (s por unidad) que usa el modelo en cada proceso y turno, con su confianza, el valor EDITADO si lo hay y la referencia del log de movimientos parciales. Es la tabla que se edita
    en el estimador: un tiempo editado vale para los tres turnos del proceso (Ajustes.tiempos_override)."""
    if parametros is None:
        parametros = fit_params.cargar()
    a = ajustes or Ajustes()
    ref = referencia_log_parciales()
    out = []
    for p in PROCESOS_TIEMPO:
        if p == "recepcion_directo_stage" and a.recepcion_por_camion and RECEPCION_CAMIONES:       # [2026-10-29] dos tareas: segundos de gruero por camión chico y por camión grande
            for tp, x in RECEPCION_CAMIONES["tipos"].items():
                pc = x["pct_escaneo_planificador_log"] if a.recepcion_pct_planificador is None else a.recepcion_pct_planificador
                seg = pc * x["gruero_s_con_planificador"] + (1 - pc) * x["gruero_s_sin_planificador"]
                fila = {"proceso": f"recepcion_{tp}", "nombre": _NOMBRE_CAMION[tp](x), "rol": "Gruero", "editado": a.tiempo(f"recepcion_{tp}"),
                        "referencia_log": "", **{f"t_{t}": seg for t in config.TURNOS}, **{f"estado_{t}": "preliminar" for t in config.TURNOS}}
                out.append(fila)
            continue
        fila = {"proceso": p, "nombre": _NOMBRE.get(p, p), "rol": ROL_POR_PROCESO.get(p, ""), "editado": a.tiempo(p)}
        for t in config.TURNOS:
            seg, est, _ = tiempo_estandar(p, t, parametros)
            fila[f"t_{t}"], fila[f"estado_{t}"] = seg, est
        r = ref.get(p)
        fila["referencia_log"] = ("; ".join(f"{t} {m:.0f} s (mediana {md:.0f}, N={n})" for t, (m, md, n) in sorted(r.items())) if r else "")
        out.append(fila)
    return out


def escenarios_estimador() -> dict[str, float]:
    """Demandas predefinidas (cajas facturadas) calculadas con TODAS las ventas (821 días, 2024 a sep-2026): polo bajo, p50, p90, polo saturado y p99 (como «máximo»; el máximo absoluto es 56.870 y se trata como estrés)."""
    e = escenarios_demanda()
    if e is None:
        return {}
    return {"polo bajo": e["bajo"], "p50": e["p50"], "p90": e["p90"], "polo saturado": e["saturado"], "máximo": e["p99"]}


def dias_historicos() -> list[dict]:
    """Días con venta de septiembre-2026: facturado, preventa y fill rate real."""
    import csv
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "demanda_diaria.csv"
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as fh:
        return [{"fecha": r["fecha"], "facturado": float(r["facturado"]), "preventa": float(r["preventa"]),
                 "fill_rate": float(r["facturado"]) / float(r["preventa"]) if float(r["preventa"]) else None}
                for r in csv.DictReader(fh) if r.get("fuente") == "septiembre"]


def resumen_backtest() -> dict | None:
    """Errores del backtest de picking (salidas/limpio/backtest_picking.csv, validacion_septiembre.py)."""
    import csv
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "backtest_picking.csv"
    if not ruta.exists():
        return None
    with open(ruta, newline="", encoding="utf-8") as fh:
        f = list(csv.DictReader(fh))
    for r in f:
        for k in ("facturado", "horas_persona_pred", "horas_persona_log", "err_horas", "err_lineas", "err_horas_loo"):
            r[k] = float(r[k])
    pk = sorted(f, key=lambda r: -r["facturado"])[:3]
    peores = sorted(f, key=lambda r: r["err_horas"])[:2]
    antes = [r for r in f if r["fecha"] < "2026-09-22"]
    desp = [r for r in f if r["fecha"] >= "2026-09-22"]
    m = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else float("nan")
    return {"n": len(f), "error_mes": sum(r["horas_persona_pred"] for r in f) / sum(r["horas_persona_log"] for r in f) - 1,
            "mape_horas": m([{"e": abs(r["err_horas"])} for r in f], "e"),
            "mape_horas_loo": m([{"e": abs(r["err_horas_loo"])} for r in f], "e"),
            # sesgo con la definición de la ec. (1.12) del capítulo: (Σ predicho − Σ observado) / Σ observado. Antes se usaba la
            # media de los errores diarios (−6,7% antes del 22-sep); se conserva aparte como `sesgo_diario_medio_*`.
            "sesgo_antes_22sep": sum(r["horas_persona_pred"] for r in antes) / sum(r["horas_persona_log"] for r in antes) - 1,
            "sesgo_desde_22sep": sum(r["horas_persona_pred"] for r in desp) / sum(r["horas_persona_log"] for r in desp) - 1,
            "sesgo_diario_medio_antes_22sep": m(antes, "err_horas"), "sesgo_diario_medio_desde_22sep": m(desp, "err_horas"),
            "dias_mayor_volumen": [(r["fecha"], r["facturado"], r["err_horas"]) for r in pk],
            "sesgo_3_peak": m(pk, "err_horas"),
            "mayores_subestimaciones": [(r["fecha"], r["facturado"], r["err_horas"]) for r in peores]}


# ---------------------------------------------------------------------------------------------------
# Referencias del análisis del picking nocturno (2026-10-05): `analisis_picking_noche.referencias_estimador()` las deja en un JSON y
# acá solo se leen (no se recalcula nada pesado). Son referencias, no parte del cálculo de N.
def _ref_picking() -> dict | None:
    import json
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / "salidas" / "limpio" / "picking_referencias_estimador.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def referencias_operario_c() -> list[dict]:
    """Dos cifras de referencia junto a Operario C. NO se elige una: miden cosas distintas (dotación contratada de la planilla vs
    pickers observados activos a la vez en el log) y no se reconcilian con los datos disponibles [hipótesis: refuerzos u otros roles]."""
    ref = _ref_picking()
    out = [{"rotulo": "12, planilla", "valor": float(config.dotacion_real_por_turno()["C"].dotacion["Operario"]),
            "fuente": "Operario C de la planilla de terreno 24-28 sep (histórico: igual al roster programado de agosto, 12 en las semanas del 17 al 29-ago)"}]
    if ref:
        out.append({"rotulo": "~15 pickers simultáneos, log", "valor": float(ref["pickers_simultaneos_mediana"]),
                    "fuente": f"mediana por hora de pickers distintos activos en el turno C, 22 noches de septiembre (máximo típico {ref['pickers_simultaneos_max']:.0f}); "
                              "analisis_picking_noche.py"})
    return out


def hora_termino_picking(demanda: float) -> dict | None:
    """Hora estimada en que se completa el 99% de las líneas de picking del turno C, con el intervalo de predicción al 95%, la demanda donde
    la recta cruza las 08:00 y el error dejando una noche fuera. Regresión de 22 noches (4 al 30-sep); extrapola fuera de 18.050 a 42.151."""
    ref = _ref_picking()
    if not ref:
        return None
    import math
    a = ref["regresion_t99"]
    x = demanda / 1000
    m = a["intercepto"] + a["pendiente_h_por_1000"] * x
    se = a["sigma_h"] * math.sqrt(1 + 1 / a["n"] + (x - a["x_medio"]) ** 2 / a["sxx"])
    lo, hi = m - a["tcrit"] * se, m + a["tcrit"] * se
    from modelo import demanda_historica as dh
    hist = dh.superan(dh.cargar(), tuple(float(round(ref["cruce_8h"][k], -2)) for k in ("p5", "punto", "p95")))   # 31.600 / 33.200 / 36.400
    return {"demanda": demanda, "hora": m, "hora_min": lo, "hora_max": hi, "sigma_min": a["sigma_h"] * 60, "n": a["n"],
            "pendiente_h_por_1000": a["pendiente_h_por_1000"], "ic95_pendiente": tuple(a["ic95_pend"]), "r2": a["r2"],
            "cruce_8h": ref["cruce_8h"]["punto"], "cruce_p5": ref["cruce_8h"]["p5"], "cruce_p95": ref["cruce_8h"]["p95"],
            "pasa_de_8": m >= 8, "extrapola": not (ref["facturado_min"] <= demanda <= ref["facturado_max"]),
            "rango_observado": (ref["facturado_min"], ref["facturado_max"]), "loo": ref["loo"], "noches_pasan_de_8": ref["noches_pasan_de_8"],
            "efecto_regimen_min": ref["efecto_regimen_h"] * 60, "historia": hist}


# =====================================================================================================
# CAPACIDAD MÁXIMA SOSTENIBLE (2026-10-06): D*(R, U) = demanda diaria a la que N de Operario C iguala R x U. SOLO Operario C.
# En el modelo N de Operario C es LINEAL en la demanda: N(D) = a·D + b, con
#     a = λ · q_pick,C · τ_pick,C / (3600 · η_pick · T_C)        (picking: proporcional a las cajas facturadas; driver sin intercepto)
#     b = N(0) = horas de los demás procesos del rol en C (hoy, reabasto de cajas) / T_C        (constante: no escala con la venta)
# así que  D* = (R·U − b) / a  (forma cerrada) y la banda según el sigma de la regresión del picking (σ = 464,6 líneas por día) es
#     D*(k) = D* − k·σ/λ,  k = ±1  →  D* ± σ/λ  (±2.276 cajas). Se verifica con una búsqueda numérica (brentq) sobre `dotacion_por_turno`.
# «SOSTENIBLE» = U menor o igual a U*: una demanda D es sostenible bajo el criterio U* cuando N(D)/R ≤ U*; D*(R, U*) es la mayor para la que se cumple.
# U NO es una utilización medida (el modelo no tiene colas): es la razón N/R, con cuatro referencias: 1,00 (N = R), 0,89 (N/R en que la recta de la hora de
# término cruza las 08:00, análisis C exploratorio), 0,85 (ρ* propuesto, reservado a la simulación) y 0,72 (N/R medio de 36 semanas, análisis A).
# LÍMITES [declarados]: depende de R de la semana elegida; el piso b es el reabasto de cajas, cuyo tiempo no es un ciclo medido (sin respaldo); el modelo
# subestima las horas de picking ~20% en el peak (régimen anterior) y las sobrestima +10,8% en el régimen vigente; solo Operario C.
# =====================================================================================================
U_REFERENCIAS = (
    (1.00, "N = R"),
    (0.89, "N/R en que la recta de la hora de término cruza las 08:00 (análisis C, exploratorio)"),
    (0.85, "ρ* propuesto (reservado a la simulación; aquí solo una razón N/R)"),
    (0.72, "N/R medio de 36 semanas (análisis A)"),
)


def _operario_c(demanda: float, ajustes: Ajustes | None, parametros) -> Dotacion:
    return next(d for d in dotacion_por_turno("C", demanda, None, parametros, ajustes=ajustes) if d.recurso == "Operario")


def coeficientes_operario_c(ajustes: Ajustes | None = None, parametros=None) -> dict:
    """a y b de N(D) = a·D + b para Operario C, en forma cerrada (a, desde λ, cuota, τ, η y T) y b = N(0) del modelo."""
    if parametros is None:
        parametros = fit_params.cargar()
    aj = ajustes or Ajustes()
    drv = DRIVERS["picking"]
    tau = tiempo_estandar("picking", "C", parametros)[0]
    t_ov = aj.tiempo("picking")
    tau = t_ov if t_ov is not None else tau
    q = _cuotas("C", aj)["picking"]
    eta = aj.eficiencia_picking_efectiva()
    dur = _duracion_turno("C", aj)
    a = drv.unidades_por_caja * q * tau / (3600 * eta * dur)
    op0 = _operario_c(0.0, aj, parametros)
    return {"a": a, "b": op0.necesarios, "b_sin_respaldo": op0.n_no_respaldado, "lambda": drv.unidades_por_caja, "sigma_lineas": drv.sigma,
            "tau_s": tau, "eta": eta, "duracion_h": dur, "cuota_picking_c": q}


def capacidad_maxima(R: float | None = None, U: float = 1.0, ajustes: Ajustes | None = None, parametros=None) -> dict:
    """D*(R, U): demanda diaria (cajas facturadas) a la que N de Operario C = R·U, con su banda (±σ del picking). Forma cerrada. Si el piso b ya supera R·U,
    no existe (D* ≤ 0)."""
    aj = ajustes or Ajustes()
    c = coeficientes_operario_c(aj, parametros)
    if R is None:
        R = float(aj.personas("C", "Operario"))
    objetivo = R * U
    d_star = (objetivo - c["b"]) / c["a"]
    delta = c["sigma_lineas"] / c["lambda"]
    ref = _ref_picking()
    rango = (ref["facturado_min"], ref["facturado_max"]) if ref else (18_050.0, 42_151.0)
    return {"R": R, "U": U, "objetivo_N": objetivo, "d_estrella": d_star, "d_min": d_star - delta, "d_max": d_star + delta, "banda_cajas": delta,
            "a": c["a"], "b": c["b"], "piso_sin_respaldo": c["b_sin_respaldo"], "existe": d_star > 0,
            "extrapola": not (rango[0] <= d_star <= rango[1]), "rango_observado": rango}


def capacidad_maxima_numerica(R: float, U: float, ajustes: Ajustes | None = None, parametros=None, hasta: float = 300_000.0) -> float:
    """Verificación: la demanda donde N(D) = R·U por búsqueda numérica (brentq) llamando al modelo completo, sin usar la forma cerrada."""
    from scipy.optimize import brentq
    if parametros is None:
        parametros = fit_params.cargar()
    f = lambda D: _operario_c(D, ajustes, parametros).necesarios - R * U
    if f(0.0) >= 0:
        return 0.0
    return float(brentq(f, 0.0, hasta, xtol=1e-6))


def tabla_capacidad_maxima(ajustes: Ajustes | None = None, parametros=None, R: float | None = None) -> dict:
    """D*(R, U) para las cuatro referencias de U, el % de los días históricos (821 con venta desde 2024) que supera cada D*, y la comparación con el cruce de las
    08:00 del análisis de hora de término. Es solo Operario C."""
    if parametros is None:
        parametros = fit_params.cargar()
    aj = ajustes or Ajustes()
    from modelo import demanda_historica as dh
    hist = dh.cargar()
    filas = []
    for U, desc in U_REFERENCIAS:
        d = capacidad_maxima(R, U, aj, parametros)
        sup = dh.superan(hist, (max(d["d_estrella"], 0.0),))[0] if d["existe"] else None
        filas.append({**d, "descripcion": desc, "pct_dias_supera": sup["pct"] if sup else None, "dias_supera": sup["dias"] if sup else None,
                      "dias_total": sup["total"] if sup else None, "dias_fin_de_mes": sup["dias_fin_de_mes"] if sup else None})
    R_ = filas[0]["R"]
    ht = hora_termino_picking(24_204.0)
    comp = None
    if ht:
        n_cruce = _operario_c(ht["cruce_8h"], aj, parametros).necesarios
        comp = {"cruce_8h": ht["cruce_8h"], "cruce_p5": ht["cruce_p5"], "cruce_p95": ht["cruce_p95"], "N_en_el_cruce": n_cruce, "N_sobre_R_en_el_cruce": n_cruce / R_,
                "d_estrella_089": filas[1]["d_estrella"]}
    c = coeficientes_operario_c(aj, parametros)
    return {"filas": filas, "R": R_, "a": c["a"], "b": c["b"], "piso_sin_respaldo": c["b_sin_respaldo"], "comparacion_hora_termino": comp,
            "sesgos": {"peak_subestima": -0.20, "vigente_sobrestima": 0.108},
            "avisos": ["Usa la dotación real del turno C: R = %d operarios." % R_,
                       "En el peak (15 y 16-sep) el modelo predice menos listas de picking que las reales: D* queda sobreestimada para días de peak.",
                       "Solo Operario C: no incluye grueros ni los turnos A y B.",
                       "«Sostenible» significa U menor o igual a U*: una demanda es sostenible bajo el criterio U* cuando N/R ≤ U*; D* es la mayor para la que se cumple. U es la razón N/R, no una utilización medida."]}


def hhmm(h: float) -> str:
    """Hora decimal a texto; pasado de las 24 h se indica como día siguiente."""
    m = int(round(h * 60))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}" + (" (día siguiente)" if m >= 24 * 60 else "")


def opciones_tiempo_linea(parametros=None) -> list[dict]:
    """Tiempo estándar por línea de picking seleccionable en Ajustes. El primero es el VALOR POR DEFECTO del modelo (no cambia); los demás salen del
    análisis por ventana de noches (`analisis_picking_noche.tiempo_linea_ventanas`) y son sensibilidades, no calibraciones."""
    if parametros is None:
        parametros = fit_params.cargar()
    modelo_s = tiempo_estandar("picking", "C", parametros)[0]
    out = [{"clave": "modelo", "etiqueta": f"valor del modelo ({modelo_s:.0f} s por lista = {modelo_s / LINEAS_POR_LISTA:.1f} s por línea × {LINEAS_POR_LISTA:.1f} líneas; 28 noches 20-ago a 24-sep)", "segundos": modelo_s, "n_gaps": None}]
    ref = _ref_picking()
    for x in (ref["tiempo_linea"] if ref else []):
        out.append({"clave": x["clave"], "etiqueta": f"{x['etiqueta']}: {x['segundos'] * LINEAS_POR_LISTA:.0f} s por lista ({x['segundos']:.1f} s por línea)", "segundos": x["segundos"] * LINEAS_POR_LISTA, "n_gaps": x["n_gaps"]})
    return out


def banda_tiempo_linea(demanda: float, ajustes: Ajustes | None = None, parametros=None) -> dict:
    """Operario C con cada tiempo por línea (el resto de los ajustes se respeta). La banda es el mínimo y el máximo de las opciones: NO es un
    intervalo estadístico. Régimen, volumen y calendario no se pueden separar con estos datos (ver `regimen_vs_volumen` del JSON)."""
    if parametros is None:
        parametros = fit_params.cargar()
    a = ajustes or Ajustes()
    base = tuple(x for x in a.tiempos_override if x[0] != "picking")
    filas = []
    for o in opciones_tiempo_linea(parametros):
        aj = Ajustes(**{**a.__dict__, "tiempos_override": base + (("picking", o["segundos"]),)})
        op = next(d for d in dotacion_por_turno("C", demanda, None, parametros, ajustes=aj) if d.recurso == "Operario")
        filas.append({**o, "operario_c": op.necesarios, "reales": op.reales})
    v = [f["operario_c"] for f in filas]
    ref = _ref_picking()
    return {"filas": filas, "min": min(v), "max": max(v), "regimen_vs_volumen": ref["regimen_vs_volumen"] if ref else None}


def _por_funcion(d: "Dotacion") -> dict[str, float]:
    """N desglosado por FUNCIÓN (picking, reposición, …): horas-persona de cada proceso ÷ duración del turno. Suma exactamente N (misma fórmula, solo se reparte)."""
    out = {f: 0.0 for f in config.FUNCIONES}
    for r in d.procesos:
        out[config.FUNCION_POR_PROCESO.get(r.proceso, "otra")] = out.get(config.FUNCION_POR_PROCESO.get(r.proceso, "otra"), 0.0) + r.horas_persona / d.duracion_turno_h
    return {f: v for f, v in out.items() if v > 0 or f in config.FUNCIONES}


def _tiene_sigma(proceso: str) -> bool:
    d = DRIVERS.get(proceso)
    return bool(d and d.sigma and d.volumen_dia is None)


def estimar(demanda: float, ajustes: Ajustes | None = None, propuesta: dict | None = None, parametros=None) -> dict:
    """Dotación necesaria por turno y rol para una demanda (cajas facturadas), con rango, propuesta, actividad
    esperada, advertencias y supuestos. `propuesta`: {(turno, rol): personas}; por defecto la dotación real."""
    if parametros is None:
        parametros = fit_params.cargar()
    a = ajustes or Ajustes()
    lo, hi = Ajustes(**{**a.__dict__, "k_sigma": -1.0}), Ajustes(**{**a.__dict__, "k_sigma": 1.0})
    filas, detalle = [], []
    for t in config.TURNOS:
        base = {d.recurso: d for d in dotacion_por_turno(t, demanda, None, parametros, ajustes=a)}
        dlo = {d.recurso: d for d in dotacion_por_turno(t, demanda, None, parametros, ajustes=lo)}
        dhi = {d.recurso: d for d in dotacion_por_turno(t, demanda, None, parametros, ajustes=hi)}
        for rec, d in base.items():
            con_sigma = abs(dhi[rec].necesarios - dlo[rec].necesarios) > 1e-9      # algún proceso con sigma aporta
            prop = (propuesta or {}).get((t, rec), d.reales) if d.tipo == "rol" else d.reales
            avisos = []
            for r in d.procesos:
                if r.estado_confianza in ("sustituto_dia", "insuficiente", "sin_datos"):
                    avisos.append(f"{_NOMBRE.get(r.proceso, r.proceso)}: tiempo/volumen `{r.estado_confianza}`")
                if r.proceso == "reabasto_caja":
                    avisos.append("reabasto de cajas: ciclo medido PARCIAL (no incluye anotar tareas en papel ni esperas): es un piso")
                if r.proceso == "reabasto_caja_grua":
                    avisos.append("reabasto de cajas, parte del gruero: 6 visitas medidas en un solo gruero (insuficiente)")
                if r.proceso == "carga_camion":
                    avisos.append("carga: tiempo de la planilla de turno A; el WMS solo cuenta trailers (timestamps de lote)")
            for p_ in d.sin_horas_medidas:
                avisos.append(f"{_NOMBRE.get(p_, p_)}: participa pero sus horas no están medidas")
            for p_ in d.sin_ciclo_medido:
                avisos.append(f"{_NOMBRE.get(p_, p_)}: su tiempo es un ciclo medido PARCIAL (cota inferior)")
            for p_, mot in d.sin_volumen.items():
                avisos.append(f"{_NOMBRE.get(p_, p_)}: {mot}")
            brecha_prop = d.necesarios - prop
            lectura = (f"La brecha calculada contra la dotación {'propuesta' if prop != d.reales else 'real'} es "
                       f"{brecha_prop:+.2f}; confianza: {d.estado_confianza}.")
            if not d.completo:
                lectura += " Cota inferior: solo puede mostrar que NO alcanza respecto de la dotación de la planilla (no de la efectiva), nunca que sí alcanza."
            filas.append({"turno": t, "recurso": rec, "tipo": d.tipo, "necesarios": d.necesarios,
                          "referencias": referencias_operario_c() if (t == "C" and rec == "Operario") else [],
                          "por_funcion": _por_funcion(d) if d.tipo == "rol" else {},
                          "rango_min": dlo[rec].necesarios if con_sigma else None,
                          "rango_max": dhi[rec].necesarios if con_sigma else None,
                          "reales": d.reales, "propuesta": prop, "brecha_real": d.necesarios - d.reales,
                          "brecha_propuesta": brecha_prop, "confianza": d.estado_confianza,
                          "pct_horas_validadas": d.pct_horas_validadas, "cota_inferior": not d.completo,
                          "n_respaldado": d.n_respaldado, "n_no_respaldado": d.n_no_respaldado, "alcanza": d.alcanza,
                          "horas_trabajo": d.horas_trabajo, "avisos": sorted(set(avisos)), "lectura": lectura})
            if d.tipo == "rol":
                for r in d.procesos:
                    detalle.append({"turno": t, "rol": rec, "proceso": r.proceso, "volumen": r.volumen,
                                    "unidad": r.unidad_volumen, "tiempo_estandar_s": r.tiempo_estandar_s,
                                    "horas_persona": r.horas_persona, "eficiencia": r.eficiencia,
                                    "confianza": r.estado_confianza, "fuentes": "; ".join(r.fuentes)})
    # actividad esperada
    pk, cg = DRIVERS["picking"], DRIVERS["carga_camion"]
    v1 = _tabla_procesos(a.regimen)["C"]["carga_camion"].volumen_fijo
    v2 = max(0.0, cg.intercepto + demanda * cg.unidades_por_caja)
    actividad = [
        {"actividad": "listas de picking", "valor": demanda * pk.unidades_por_caja,
         "min": max(0.0, demanda * pk.unidades_por_caja - pk.sigma), "max": demanda * pk.unidades_por_caja + pk.sigma,
         "fuente": f"{1000 * pk.unidades_por_caja:.1f} listas por 1.000 cajas facturadas; rango ±1 sigma de la regresión ({pk.sigma:.0f} listas)"},
        {"actividad": "líneas de picking (equivalentes)", "valor": demanda * pk.unidades_por_caja * LINEAS_POR_LISTA,
         "min": max(0.0, (demanda * pk.unidades_por_caja - pk.sigma) * LINEAS_POR_LISTA), "max": (demanda * pk.unidades_por_caja + pk.sigma) * LINEAS_POR_LISTA,
         "fuente": f"listas × {LINEAS_POR_LISTA:.2f} líneas por lista (22 días de septiembre)"},
        {"actividad": "cajas pickeadas", "valor": demanda * FACTOR_CAJAS_PICKEADAS,
         "min": max(0.0, demanda * FACTOR_CAJAS_PICKEADAS - SIGMA_CAJAS_PICKEADAS),
         "max": demanda * FACTOR_CAJAS_PICKEADAS + SIGMA_CAJAS_PICKEADAS,
         "fuente": "0,846 cajas pickeadas por caja facturada; ±1 sigma (1.094,9 cajas)"},
        {"actividad": "pallets recibidos (= pallets a almacenar)", "valor": demanda * PALLETS_POR_1000 / 1000, "min": None, "max": None,
         "fuente": _FRP},
        {"actividad": "trailers vuelta 1 (turno C, fija)", "valor": v1, "min": None, "max": None,
         "fuente": "config.PROCESOS_POR_TURNO (conteo directo del log); no escala con la venta"},
        {"actividad": "trailers vueltas ≥ 2", "valor": v2, "min": max(0.0, v2 - cg.sigma), "max": v2 + cg.sigma,
         "fuente": "regresión vueltas ≥ 2 ~ facturado (R² 0,95, sigma 1,8, n = 25); [hipótesis: CARGA = vuelta]"},
    ]
    # advertencias dinámicas
    e = escenarios_demanda() or {}
    adv = []
    if e and not (e["cal_min"] <= demanda <= e["cal_max"]):
        adv.append(f"Demanda fuera del rango de calibración de los drivers (septiembre-2026: {e['cal_min']:,.0f} a {e['cal_max']:,.0f} cajas): "
                   "EXTRAPOLACIÓN.".replace(",", "."))
    if e and demanda >= e["p90"]:
        bt = resumen_backtest()
        if bt:
            pk3 = "; ".join(f"{f[8:]}-{f[5:7]} {err:+.0%}" for f, _, err in bt["dias_mayor_volumen"])
            sub = " y ".join(f"{f[8:]}-{f[5:7]} ({err:+.0%})" for f, _, err in bt["mayores_subestimaciones"])
            det = (f"En los 3 días de mayor volumen el backtest subestimó las horas de picking en {bt['sesgo_3_peak']:+.0%} en promedio "
                   f"({pk3}); las mayores subestimaciones fueron {sub}, en la semana del peak.")
        else:
            det = "El backtest subestimó las horas de picking en los días de mayor volumen."
        p90 = f"{e['p90']:,.0f}".replace(",", ".")
        adv.append(f"Zona saturada (≥ p90 = {p90} cajas). {det.replace('%', ' %').replace(' %', '%')} La cifra de picking es un PISO.")
    if any(f["cota_inferior"] for f in filas):
        roles = ", ".join(sorted({f"{f['recurso']} {f['turno']}" for f in filas if f["cota_inferior"]}))
        adv.append(f"Cota inferior en {roles}: esto solo puede mostrar que NO alcanza respecto de la dotación de la planilla (no de la efectiva), nunca que sí alcanza.")
    if a.eficiencia_resto != EFICIENCIA_SUPUESTA or a.eficiencia_picking != EFICIENCIA_MEDIDA["picking"] or a.niveles_base or a.tiempos_override:
        adv.append("Hay parámetros editados: el resultado es un 'qué pasaría si', no la estimación del modelo.")
    if a.regimen == "anterior":
        adv.append("Régimen anterior (4 al 21-sep, incluye Fiestas Patrias): práctica histórica, no la vigente.")
    if a.regimen == "combinado":
        adv.append("Régimen combinado (4-sep a 10-oct): promedia las tres prácticas del CD (anterior, intermedio y vigente) por turno; no describe ninguna práctica concreta, sino la proporción media de tareas.")
    if a.regimen == "intermedio":
        adv.append("Régimen intermedio (22-sep a 1-oct): era el «vigente» de las versiones anteriores del modelo; la práctica más reciente es la de octubre.")
    sem = a.semana_dotacion or config.semana_dotacion_defecto()
    fuentes_dot = config.fuente_dotacion(sem)
    supuestos = {"demanda_cajas_facturadas": demanda, "regimen": a.regimen,
                 "dotacion_semana": config.etiqueta_semana(config._lunes(sem).isoformat()) if sem else "planilla 24-28 sep",
                 "dotacion_fuente_C": fuentes_dot["C"], "dotacion_fuente_AB": "ODS sep-2026 (Gruero, Operario); Intermedio y D en A [provisional]", "eficiencia_picking": a.eficiencia_picking,
                 "eficiencia_resto (supuesto)": a.eficiencia_resto,
                 **{f"nivel_base_{p}": (a.nivel(p) if a.nivel(p) is not None else nivel_modelo(p, a.regimen)) for p in _PROCESOS_NIVEL_BASE},
                 **{f"reabasto_caja_{t}_{k}": round(v[k], 2) for t, v in volumen_reabasto_caja(a.regimen, a.hojas, a.con_parciales).items() for k in ("log", "parcial", "hojas", "total")},
                 "reabasto_caja_hojas_usadas": ", ".join(sorted({f for v in volumen_reabasto_caja(a.regimen, a.hojas, a.con_parciales).values() for f in v["hojas_usadas"]})) or "ninguna",
                 "drivers": "picking y carga: facturado real de septiembre-2026; volumen de los demás procesos: log WMS de octubre-2026 (régimen vigente)",
                 "tiempos_estandar": "parametros.json (planilla oficial .ods + logs WMS de septiembre y octubre)",
                 **{f"volumen_editado_{p_}_{t_}": v_ for p_, t_, v_ in a.volumenes_turno},
                 "meta_fill_rate": "no definida (None)"}
    dotacion = {"semana": config._lunes(sem).isoformat() if sem else None, "etiqueta": config.etiqueta_semana(config._lunes(sem).isoformat()) if sem else "planilla 24-28 sep",
                "fuentes": fuentes_dot, "detalle_roster": config.detalle_roster(sem), "historico_operario_c": config.HISTORICO_OPERARIO_C,
                "c_ods": config.dotacion_c_ods()}
    if a.volumenes_turno or a.reabasto_caja_turno:
        adv.append("Hay volúmenes editados por el usuario: el resultado es un 'qué pasaría si', no la estimación del modelo.")
    if a.charla_min != parametros_diana.CHARLA_MIN or a.colacion_min != parametros_diana.COLACION_MIN or a.traslados_min or a.otras_h:
        adv.append("Hay tiempo de turno, traslados u otras actividades editados: el resultado es un 'qué pasaría si', no la estimación del modelo.")
    supuestos.update({"horas_efectivas_por_turno": a.horas_efectivas(), "charla_min": a.charla_min, "colacion_min": a.colacion_min,
                      **{f"traslados_min_{r_}_{t_}": m_ for r_, t_, m_ in a.traslados_min}, **{f"otras_horas_{r_}_{t_}": h_ for r_, t_, h_ in a.otras_h}})
    est = {"demanda": demanda, "duracion_h": a.horas_efectivas(), "dotacion": dotacion, "funciones": {"nota": config.NOTA_FUNCIONES, "fuera_del_modelo": list(config.FUERA_DEL_MODELO)}, "filas": filas, "detalle": detalle, "actividad": actividad, "advertencias": adv,
            "supuestos": supuestos, "meta_fill_rate": None,
            "aviso_permanente": ("Estima CARGA DE TRABAJO PROMEDIO con tiempos estándar medios: no modela colas, esperas ni "
                                 "contención por grúas y gruero. Eso lo hará la simulación (pendiente)."),
            "aviso_meta": "No hay meta de fill rate definida (meta_fill_rate = None): el estimador entrega brechas, no un óptimo."}
    est["carga_por_tarea"] = carga_por_tarea(est)
    est["picking_excedente_c"] = excedente_picking_c(demanda, parametros, a) if not (a.picking_excedente_a or a.picking_reparto_log) else None
    return est


def volumenes_modelo(demanda: float, ajustes: Ajustes | None = None, parametros=None) -> list[dict]:
    """Volumen por (proceso, turno) que usa el modelo con la demanda dada, SIN las ediciones del usuario, junto al valor editado si lo hay.
    Es la tabla que se edita en el estimador: «el tamaño de las variables». Una fila por proceso que ocurre en el turno (la parte gruero del reabasto de cajas no
    aparece: sigue al volumen de reabasto de cajas). `modelo` = lo que da el modelo; `editado` = lo que fijó el usuario (None si no); `usado` = lo que entra al cálculo."""
    if parametros is None:
        parametros = fit_params.cargar()
    a = ajustes or Ajustes()
    limpio = replace(a, volumenes_turno=(), reabasto_caja_turno=())
    ed = {(p, t): v for p, t, v in a.volumenes_turno}
    ed.update({("reabasto_caja", t): v for t, v in a.reabasto_caja_turno if ("reabasto_caja", t) not in ed})
    filas = []
    for t in config.TURNOS:
        reqs, _ = _resolver_turno(t, demanda, None, parametros, None, limpio)
        for r in reqs:
            if r.proceso == "reabasto_caja_grua":
                continue
            if r.proceso == "recepcion_directo_stage" and a.recepcion_por_camion and RECEPCION_CAMIONES:       # [2026-10-29] la recepción se edita en DOS tareas: camión chico y camión grande
                n_ = camiones_recepcion(demanda)
                for tp, x in RECEPCION_CAMIONES["tipos"].items():
                    e = ed.get((f"recepcion_{tp}", t))
                    m_ = n_[tp] * x["reparto_turno"][t]
                    filas.append({"proceso": f"recepcion_{tp}", "nombre": _NOMBRE_CAMION[tp](x), "turno": t, "rol": r.rol, "modelo": m_, "unidad": "camiones",
                                  "editado": e, "usado": m_ if e is None else e, "nota": f"{x['pallets_por_camion']:.1f} pallets por camión"})
                continue
            e = ed.get((r.proceso, t))
            filas.append({"proceso": r.proceso, "nombre": _NOMBRE.get(r.proceso, r.proceso), "turno": t, "rol": r.rol, "modelo": r.volumen, "unidad": r.unidad_volumen,
                          "editado": e, "usado": r.volumen if e is None else e, "nota": r.supuestos[0] if r.supuestos else ""})
    return filas


def carga_por_tarea(est: dict) -> list[dict]:
    """PORCENTAJE DE LA CARGA POR TAREA. Una fila por (turno, rol, proceso) con sus horas-persona, el N parcial que representa y su porcentaje dentro de la carga del
    rol en ese turno (`pct_rol`) y dentro de la carga de TODO el turno (`pct_turno`, suma de roles). Sale del detalle por proceso de `estimar`: no calcula nada
    nuevo, solo reparte. La suma de `n_parcial` de un (turno, rol) es el N de la tabla principal."""
    det = est["detalle"]
    dur = {t: est.get("duracion_h", _duracion_turno(t)) for t in config.TURNOS}
    tot_rol, tot_turno = {}, {}
    for d in det:
        tot_rol[(d["turno"], d["rol"])] = tot_rol.get((d["turno"], d["rol"]), 0.0) + d["horas_persona"]
        tot_turno[d["turno"]] = tot_turno.get(d["turno"], 0.0) + d["horas_persona"]
    out = []
    for d in det:
        h = d["horas_persona"]
        out.append({"turno": d["turno"], "rol": d["rol"], "proceso": d["proceso"], "tarea": _NOMBRE.get(d["proceso"], d["proceso"]),
                    "funcion": config.FUNCION_POR_PROCESO.get(d["proceso"], ""), "volumen": d["volumen"], "unidad": d["unidad"],
                    "tiempo_estandar_s": d["tiempo_estandar_s"], "horas_persona": h, "n_parcial": h / dur[d["turno"]],
                    "pct_rol": h / tot_rol[(d["turno"], d["rol"])] if tot_rol[(d["turno"], d["rol"])] else 0.0,
                    "pct_turno": h / tot_turno[d["turno"]] if tot_turno[d["turno"]] else 0.0, "confianza": d["confianza"]})
    return out


def estimacion_a_filas(est: dict) -> list[dict]:
    """Una fila por (turno, recurso) con los supuestos en columnas, para descargar en CSV."""
    out = []
    for f in est["filas"]:
        out.append({"turno": f["turno"], "recurso": f["recurso"], "tipo": f["tipo"],
                    "necesarios": round(f["necesarios"], 3),
                    "rango_min": None if f["rango_min"] is None else round(f["rango_min"], 3),
                    "rango_max": None if f["rango_max"] is None else round(f["rango_max"], 3),
                    "reales": f["reales"], "propuesta": f["propuesta"], "brecha_propuesta": round(f["brecha_propuesta"], 3),
                    "confianza": f["confianza"], "pct_horas_validadas": round(f["pct_horas_validadas"], 3),
                    "n_respaldado": round(f["n_respaldado"], 3), "n_no_respaldado": round(f["n_no_respaldado"], 3),
                    **{f"N_{config.FUNCIONES[i].replace('ó', 'o').replace(' ', '_')}": round(f["por_funcion"].get(fn, 0.0), 3) if f["por_funcion"] else None for i, fn in enumerate(config.FUNCIONES)},
                    "cota_inferior": f["cota_inferior"], "avisos": " | ".join(f["avisos"]), **est["supuestos"]})
    return out


def _fmt_dotacion(d: Dotacion) -> str:
    alc = {True: "alcanza", False: "NO alcanza", None: "?"}[d.alcanza]
    cota = "" if d.completo else "  [COTA INFERIOR: falta " + ", ".join(
        [f"{p} ({m})" for p, m in d.sin_volumen.items()] +
        [f"horas de {p}" for p in d.sin_horas_medidas]) + "]"
    return (f"  {d.recurso:<10}{d.tipo:<7}{d.horas_trabajo:>7.1f}h{d.necesarios:>8.2f}"
            f"{d.reales:>6}{d.brecha:>+8.2f}  {d.estado_confianza:<14}"
            f"{d.pct_horas_validadas:>6.1%}  {alc:<11}{cota}")


def _fmt(r: Requerimiento) -> str:
    alc = {True: "alcanza", False: "NO alcanza", None: "?"}[r.alcanza]
    return (f"  {r.proceso:<26}{r.rol:<11}{r.volumen:>9,.0f} {r.unidad_volumen:<18}"
            f"{r.tiempo_estandar_s:>7.1f}s{r.horas_persona:>9.1f}h"
            f"{r.personas_necesarias:>8.1f}{r.personas_reales:>7}"
            f"{r.brecha:>+8.1f}  {r.estado_confianza:<14}{alc}").replace(",", ".")


if __name__ == "__main__":
    params = fit_params.cargar()
    demanda = 22_859.0   # p50 de 2026 sin lluvias (modelo/analisis.py)
    print(f"Demanda de entrada: {demanda:,.0f} cajas/día (p50 2026 sin lluvias)\n"
          .replace(",", "."))
    for turno in ("C", "A", "B"):
        rs = por_turno(turno, demanda, CUOTAS_TURNO[turno], params)
        print(f"TURNO {turno} ({config.HORARIO_TURNO[turno][0]:02d}-"
              f"{config.HORARIO_TURNO[turno][1]:02d}h)")
        print("  DOTACIÓN (agregada por rol/equipo — cifra principal)")
        print(f"  {'recurso':<10}{'tipo':<7}{'trabajo':>8}{'nec.':>8}{'real':>6}{'brecha':>8}"
              f"  {'confianza':<14}{'%val.':>6}  {'¿alcanza?':<11}")
        for d in dotacion_por_turno(turno, demanda, CUOTAS_TURNO[turno], params):
            print(_fmt_dotacion(d))
        print("  Detalle por proceso (diagnóstico; NO comparar cada fila con la dotación)")
        print(f"  {'proceso':<26}{'rol':<11}{'volumen':>9} {'unidad':<18}"
              f"{'t.std':>8}{'h-pers':>9}{'nec.':>8}{'real':>7}{'brecha':>8}  "
              f"{'confianza':<14}¿alcanza?")
        for r in sorted(rs, key=lambda x: -x.personas_necesarias):
            print(_fmt(r))
        print()
    print("SENSIBILIDAD turno C — camiones cargados por noche (5 = David; 14 = 22-09 parcial; "
          "21 = 23-09 completa)")
    for x in sensibilidad_carga_nocturna(demanda, parametros=params):
        print(f"  {x['camiones']:>4.0f} camiones -> Gruero {x['gruero']:.2f} de "
              f"{config.personas_turno('C', 'Gruero')} | Grúas {x['gruas']:.2f} de "
              f"{config.equipos_disponibles()['Grúas']}")
    print()
    print("meta_fill_rate = None en todos los resultados: no hay criterio de éxito "
          "confirmado (brecha #7).")
