"""Ajuste de distribuciones por (proceso, paso, turno) a partir de las observaciones
limpias (salidas/limpio/observaciones.csv, generado por modelo.data_loader).

Regla de rigor del proyecto: ningún tiempo entra a config.py/dists.py sin declarar su
fuente. Por eso esto no devuelve solo parámetros: devuelve, para cada uno, N, fechas,
días distintos, operarios distintos y un `estado` según el tamaño de muestra. config.py
lee `parametros.json`, nunca escribe un número a mano.

Umbrales de estado (propuestos por el modelo, PENDIENTES de confirmar con el tutor,
ver CLAUDE.md > Brechas de datos):
  N < 10                                  -> "insuficiente"
  10 <= N < 30                            -> "preliminar"
  N >= 30                                 -> "confiable"
  N >= 30 y >=3 días y >=3 operarios      -> "validado"

Turno noche (C) y buena parte de B no tienen observaciones propias (ver CLAUDE.md):
para esos pasos se reutiliza la distribución de turno A y se marca
`turno_real="A"` / `estado="sustituto_dia"`, nunca se inventa un parámetro nuevo.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from scipy import stats

RAIZ = Path(__file__).resolve().parent.parent
OBSERVACIONES = RAIZ / "salidas" / "limpio" / "observaciones.csv"
SALIDA = RAIZ / "salidas" / "limpio" / "parametros.json"

# Candidatas para tiempos de proceso: siempre positivos, cola a la derecha (decisión
# del usuario, 2026-09-22). Todas con soporte en scipy.stats y loc fijo en 0 (un tiempo
# de proceso no debería tener un desplazamiento negativo).
CANDIDATAS = {
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
}


@dataclass
class Parametro:
    proceso: str
    paso: str
    turno: str            # turno para el que vale (puede ser sustituto)
    turno_real: str        # turno del que realmente vienen las observaciones
    distribucion: str      # nombre de la distribución elegida, o "empirica"
    parametros: dict       # kwargs para scipy.stats.<distribucion>(**parametros)
    media_s: float
    desvio_s: float
    n: int
    dias: int
    operarios: int
    fechas: list[str]
    estado: str
    ks_pvalor: float | None
    fuente: str


# Fuentes de observaciones. Cada una declara de dónde salió el dato, porque no son
# equivalentes: la planilla es cronómetro en terreno (desglose fino por paso, N chico,
# casi todo turno A) y el log de picking es el hueco entre picks consecutivos del WMS
# (N enorme, turno noche real, pero sin desglose interno del paso). Ver
# modelo/picking_log.py para por qué el segundo es tiempo de ciclo válido y el
# historial general (docs/i.csv) no lo es.
# CORREGIDO 2026-09-30: la etiqueta de la planilla estaba escrita a mano y quedó
# desactualizada al migrar `data_loader` a la planilla del 25-sep — 69 de 81 parámetros
# declaraban una fuente falsa. Ahora se **deriva** de `data_loader.PLANILLA` y de
# `picking_log.LOGS`, así que no puede volver a desincronizarse: si alguien cambia el
# archivo de entrada, la etiqueta cambia sola. Es la regla de rigor del proyecto
# ("todo parámetro declara su fuente") aplicada al propio mecanismo de declaración.
def _etiqueta_planilla() -> str:
    from modelo.data_loader import PLANILLA
    return f"{PLANILLA.name} (levantamiento en terreno, cronómetro)"


def _etiqueta_picking() -> str:
    from modelo.picking_log import LOGS
    nombres = " + ".join(l.name for l in LOGS if l.exists()) or LOGS[0].name
    return f"{nombres} (log WMS, gap entre picks consecutivos)"


def _etiqueta_wms_almacenamiento() -> str:
    from modelo.wms_septiembre import NUEVOS
    return (f"{NUEVOS['almacenamiento'].name} + Almacenamiento octubre.csv (log WMS septiembre y octubre-2026, tiempo real retiro->deposito "
            "uniendo los dos tramos por LPN, sin filtrar por el rótulo del depósito)")


def _etiqueta_wms_pallet() -> str:
    from modelo.wms_septiembre import NUEVOS
    return (f"{NUEVOS['reabasto_pallet'].name} + Reabastecimientos de cajas y palletts octubre.csv (log WMS septiembre y octubre-2026, tiempo real "
            "retiro->deposito uniendo los dos tramos por LPN, separado por ORIGEN: ALMAC o ALMPIC)")


# Septiembre + octubre, generados por `modelo.wms_octubre` (códigos de usuario anónimos). Antes: wms_sep_obs_*.csv (solo septiembre, con el usuario del WMS).
def _etiqueta_movimientos() -> str:
    return ("movimientos de inventario parcial sep-oct.csv (log WMS, hueco ≤ 900 s entre tareas consecutivas del mismo operario y proceso; tareas = origen→TRF y TRF→destino unidos por LPN)")


OBS_MOVIMIENTOS = RAIZ / "salidas" / "limpio" / "movimientos_parciales_obs.csv"     # modelo.movimientos_parciales
OBS_WMS_ALMACENAMIENTO = RAIZ / "salidas" / "limpio" / "actividad_obs_almacenamiento.csv"
OBS_WMS_PALLET = RAIZ / "salidas" / "limpio" / "actividad_obs_reabasto_pallet.csv"

FUENTES = (
    (OBSERVACIONES, _etiqueta_planilla()),
    (RAIZ / "salidas" / "limpio" / "picking_observaciones.csv", _etiqueta_picking()),
    (OBS_WMS_ALMACENAMIENTO, _etiqueta_wms_almacenamiento()),
    # Reabasto de pallet por origen: pasos NUEVOS (`ciclo_total_almac`, `ciclo_total_almpic`), así que no
    # reemplaza nada; el `ciclo_total` de la planilla (A, N=17) queda como referencia cruzada de ALMAC.
    (OBS_WMS_PALLET, _etiqueta_wms_pallet()),
    # Retorno de fletero, merma y (como referencia) el reabasto de cajas por movimiento parcial: ciclo medido en el log, ver modelo/movimientos_parciales.py
    (OBS_MOVIMIENTOS, _etiqueta_movimientos()),
)

# Claves (proceso, paso) en las que el WMS REEMPLAZA a la planilla en vez de mezclarse con ella:
# son dos mediciones de la misma cosa con N muy distinto (63 vs ~6.000) y mezclarlas daría una
# distribución dominada por una sola fuente sin declararlo. Almacenamiento `ciclo_total`: el WMS
# da mediana 85-87 s en turno A contra 85 s del cronómetro (validación cruzada, 2026-10-02) y trae
# además B y C, que la planilla no tiene. Los pasos finos de la planilla (recoger_de_stage,
# ir_y_dejar_pallet, escanear_qr) se conservan tal cual. NO aplica a reabasto de cajas ni a
# carga: su log solo sirve para frecuencia (bypass / timestamps de lote).
REEMPLAZADAS_POR_WMS = {OBS_WMS_ALMACENAMIENTO: {("almacenamiento", "ciclo_total")}}


def _leer(rutas=FUENTES) -> dict[tuple[str, str, str], list[dict]]:
    """Agrupa por (proceso, paso, turno) desde todas las fuentes disponibles.
    Una fuente que todavía no se generó se salta en silencio: el módulo tiene que
    poder correr con solo la planilla."""
    if isinstance(rutas, (str, Path)):
        rutas = ((Path(rutas), FUENTES[0][1]),)
    grupos: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for ruta, fuente in rutas:
        if not Path(ruta).exists():
            continue
        reemplaza = REEMPLAZADAS_POR_WMS.get(Path(ruta), set())
        for k in [k for k in grupos if k[:2] in reemplaza]:
            del grupos[k]                      # esta fuente sustituye a las anteriores en esas claves
        with open(ruta, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r["duracion_s"]:
                    continue
                r["_fuente"] = fuente
                grupos[(r["proceso"], r["paso"], r["turno"])].append(r)
    return grupos


def estado_por_n(n: int) -> str:
    """Estado según el tamaño de muestra SOLO (sin días ni operarios): N<10 insuficiente, N<30 preliminar, si no confiable.
    Es la regla de `_estado` sin el escalón `validado`; `capacity` la usa para el volumen declarado de un proceso
    (p. ej. 10 noches contadas de carga = preliminar), de modo que un dato y un tiempo se etiqueten con el mismo criterio."""
    if n < 10:
        return "insuficiente"
    if n < 30:
        return "preliminar"
    return "confiable"


def _estado(n: int, dias: int, operarios: int) -> str:
    e = estado_por_n(n)
    return "validado" if e == "confiable" and dias >= 3 and operarios >= 3 else e


def _ajustar(valores: list[float]) -> tuple[str, dict, float | None]:
    """Elige la distribución de menor AIC entre las candidatas; si el ajuste falla
    (muestra muy chica o degenerada) cae a los momentos empíricos."""
    mejor = None
    for nombre, dist in CANDIDATAS.items():
        try:
            params = dist.fit(valores, floc=0)
        except Exception:
            continue
        k = len(params)
        ll = sum(dist.logpdf(v, *params) for v in valores)
        if not (ll == ll):  # NaN
            continue
        aic = 2 * k - 2 * ll
        ks_p = stats.kstest(valores, dist.name, args=params).pvalue
        cand = (aic, nombre, params, ks_p)
        if mejor is None or cand[0] < mejor[0]:
            mejor = cand
    if mejor is None:
        return "empirica", {}, None
    _, nombre, params, ks_p = mejor
    nombres_arg = ["s", "loc", "scale"] if nombre in ("lognorm", "weibull_min") else ["a", "loc", "scale"]
    return nombre, dict(zip(nombres_arg, params)), ks_p


def ajustar_todo(ruta=FUENTES) -> list[Parametro]:
    grupos = _leer(ruta)
    out: list[Parametro] = []
    for (proceso, paso, turno), filas in grupos.items():
        valores = [float(r["duracion_s"]) for r in filas]
        n = len(valores)
        dias = len({r["fecha"] for r in filas if r["fecha"]})
        operarios = len({r["operario"] for r in filas if r["operario"]})
        media = sum(valores) / n
        desvio = (sum((v - media) ** 2 for v in valores) / n) ** 0.5 if n > 1 else 0.0
        nombre, params, ks_p = (_ajustar(valores) if n >= 5 else ("empirica", {}, None))
        out.append(Parametro(
            proceso, paso, turno, turno, nombre, params, media, desvio, n, dias, operarios,
            sorted({r["fecha"] for r in filas if r["fecha"]}), _estado(n, dias, operarios),
            ks_p, filas[0].get("_fuente", FUENTES[0][1]),
        ))
    _agregar_sustitutos_turno_noche(out)
    return sorted(out, key=lambda p: (p.proceso, p.paso, p.turno))


def _agregar_sustitutos_turno_noche(parametros: list[Parametro]) -> None:
    """Para cada (proceso, paso) sin observaciones en B o C, copia el de A marcado
    como sustituto. No inventa un parámetro nuevo: reusa el de día con su estado propio."""
    por_paso: dict[tuple[str, str], dict[str, Parametro]] = defaultdict(dict)
    for p in parametros:
        por_paso[(p.proceso, p.paso)][p.turno] = p
    for (proceso, paso), por_turno in por_paso.items():
        base = por_turno.get("A")
        if base is None:
            continue
        for turno in ("B", "C"):
            if turno in por_turno:
                continue
            # n/dias/operarios/fechas quedan en 0: son los de ESTE turno (no tiene
            # observaciones propias), no los del turno A que se está reutilizando.
            sust = Parametro(**{**asdict(base), "turno": turno, "n": 0, "dias": 0,
                                "operarios": 0, "fechas": [], "estado": "sustituto_dia",
                                "fuente": base.fuente + " (turno A usado como sustituto; sin muestras de "
                                          f"turno {turno}, ver CLAUDE.md > Brechas de datos)"})
            parametros.append(sust)


def guardar(parametros: list[Parametro], ruta: Path = SALIDA) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump([asdict(p) for p in parametros], fh, ensure_ascii=False, indent=2)


def cargar(ruta: Path = SALIDA) -> list[Parametro]:
    with open(ruta, encoding="utf-8") as fh:
        return [Parametro(**d) for d in json.load(fh)]


if __name__ == "__main__":
    ps = ajustar_todo()
    guardar(ps)
    from collections import Counter
    print(f"{len(ps)} parámetros -> {SALIDA}")
    print("estados:", dict(Counter(p.estado for p in ps)))
    print("distribuciones elegidas:", dict(Counter(p.distribucion for p in ps if p.estado != "sustituto_dia")))
    for p in sorted(ps, key=lambda p: -p.n)[:5]:
        print(f"  {p.proceso}.{p.paso} [{p.turno}] n={p.n} {p.distribucion}{p.parametros} "
              f"media={p.media_s:.0f}s estado={p.estado}")
