"""Exports de actividad del WMS de septiembre-2026 (`docs/wms actividades septiembre/`):
tiempos de ciclo reales por proceso, uniendo los DOS tramos que el WMS escribe cuando una
tarea pasa por un punto de transferencia (TRF).

ESTRUCTURA (verificada 2026-10-01 contra los datos, no supuesta)
  Una tarea con paso por TRF deja dos registros:
    tramo 1 "retiro"   : ubicación de origen -> TRF_*   (el operario levantó la carga)
    tramo 2 "depósito" : TRF_* -> ubicación de destino  (el operario la dejó)
  El timestamp es el de la confirmación por RF, así que `ts2 - ts1` es el tiempo de
  **traslado con carga** (levantar -> dejar). NO incluye el viaje en vacío hasta el
  origen ni la espera previa. Para eso se calcula además `t_desde_tarea_anterior_s`:
  hueco entre el depósito de la tarea anterior del mismo usuario y éste.

  Qué trae cada export (contado, ver `auditoria_compatibilidad()`):
    Recepción        solo tramo 1 (EXPR000015 -> TRF). Se cuenta ese tramo.
    Carga de tráiler Desde el 2026-10-02 llegan 3 exports contiguos (3-sep 14:39 a 2-oct 10:45, ninguno
                     truncado: 99.473 / 92.360 / 60.286 registros) con AMBOS tramos: PATCA -> TRF y
                     TRF -> trailer (TRL). El número de trailer viene en TODAS las líneas despachadas
                     (tramo TRF -> TRL y líneas directas PATCA -> TRL), así que los trailers por noche
                     se CUENTAN, no se estiman. El export del mismo día anterior (truncado en 100.000,
                     sin tramo TRF -> TRL) quedó obsoleto: sus líneas están contenidas en estos tres.
    Almacenamiento   ambos (STREC -> TRF, TRF -> ubicación final).
    Reabasto pallet  ambos. Reabasto caja: ambos (pero timestamps sospechosos, ver abajo).
    Picking          ambos, pero de distinta naturaleza: tramo 1 = una línea pickeada
                     (SURTID -> TRF), tramo 2 = el pallet armado se deja en el stage
                     (TRF -> PATCA, un registro por línea, todos con el mismo timestamp).
                     El par se hace a nivel de **pallet** (LPN destino).
    Trabajo suspendido: un registro por tarea suspendida, sin tramos.

CONFIABILIDAD DE TIMESTAMPS
  Reabasto de cajas: el equipo reporta que se marcan varias tareas como terminadas de una
  vez. Se marca `confiabilidad = reducida_bypass` en TODAS sus filas y se cuantifica el
  patrón (`rafagas`) contra almacenamiento y reabasto de palés como control.

CORTE DE OUTLIERS: `CORTE_S = 900` s, el mismo de `picking_log.py` (un solo umbral).
Los pares más largos se conservan en el CSV con `dentro_corte = 0`. En picking (pallet armado)
`dentro_corte` usa 3 h, porque armar un pallet no es una línea.

COMBINACIÓN CON LOS LOGS VIEJOS: ver `auditoria_compatibilidad()`. Resultado: en el traslape
los registros del tramo 1 coinciden uno a uno, así que se combina deduplicando; los logs
viejos solo aportan lo que el export nuevo no cubre (picking 3 a 20-sep, tramo 1).

Salidas (`salidas/limpio/`):
  wms_sep_eventos.csv      cada registro, trazable a archivo y fila de origen
  wms_sep_ciclos.csv       un par de tramos por fila (o un pallet armado, en picking)
  wms_sep_intervalos.csv   huecos entre eventos consecutivos del mismo usuario
  wms_sep_suspendidos.csv  suspensiones por fecha/turno/tipo/planificador
  wms_sep_turnos.csv       una fila por (fecha, turno), enlaza volumen y suspensiones
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_NUEVOS = RAIZ / "docs" / "wms actividades septiembre"
NUEVOS = {
    "recepcion": DIR_NUEVOS / "Recepcion Septiembre.csv",
    "almacenamiento": DIR_NUEVOS / "almacenamiento dirigido septiembre.csv",
    "reabasto_pallet": DIR_NUEVOS / "Reabastecimiento de pallets Septiembre.csv",
    "reabasto_caja": DIR_NUEVOS / "Reabastos de cajas Septiembre.csv",
    "picking": DIR_NUEVOS / "Picking septiembre 20 para adelante.csv",
    "suspendido": DIR_NUEVOS / "trabajo suspendido septiembre.csv",
}
DOCS = RAIZ / "docs"
# Carga de tráiler: tres exports contiguos que cubren 3-sep 14:39 a 2-oct 10:45 (2026-10-02).
CARGA_EXPORTS = (DIR_NUEVOS / "Carga de tráiler Septiembre 1.csv.csv",
                 DIR_NUEVOS / "Carga de tráiler Septiembre 2.csv",
                 DIR_NUEVOS / "Carga de tráiler Septiembre 3.csv")
TRUNCADO_EN = 100_000
VIEJOS = {
    "reabasto": DOCS / ("Reabastecimiento de cajas y pallets con reabasto cancelado y "
                        " trabajo suspendido.csv"),
    "picking_sep": DOCS / "recogida de lista.csv",
    "picking_ago": DOCS / "recogida de lista agosto.csv",
    "recepcion": DOCS / "recepcion.csv",
}
LIMPIO = RAIZ / "salidas" / "limpio"
SAL_EVENTOS = LIMPIO / "wms_sep_eventos.csv"
SAL_CICLOS = LIMPIO / "wms_sep_ciclos.csv"
SAL_INTERVALOS = LIMPIO / "wms_sep_intervalos.csv"
SAL_SUSP = LIMPIO / "wms_sep_suspendidos.csv"
SAL_TURNOS = LIMPIO / "wms_sep_turnos.csv"
# Observaciones de tiempo en el formato de `fit_params` (mismas columnas que observaciones.csv).
SAL_OBS_FIT = LIMPIO / "wms_sep_obs_almacenamiento.csv"
SAL_OBS_PALLET = LIMPIO / "wms_sep_obs_reabasto_pallet.csv"
# Reabasto de pallet se mide POR ORIGEN: ALMAC (rack, con grúa; ≈12% de los casos, ≈120-150 s) y ALMPIC
# (≈88%, mediana 7-22 s pero media ≈37 s por una cola larga) son recorridos distintos; un promedio
# único no describe a ninguno. SURTID (2 pares) se descarta.
ORIGENES_PALLET = ("ALMAC", "ALMPIC")
# Almacenamiento hacia su ubicación FINAL (lo que mide el `ciclo_total` de la planilla: de stage a
# la posición). Se excluyen los movimientos dentro de stage (PATCA->PATCA), ALMPIC->ALMPIC,
# EXPR->PATCA y SUBEST->ALMPIC: son otra tarea y bajan la mediana.
AREAS_DESTINO_FINAL = ("ALMAC", "ALMPIC", "SURTID")

csv.field_size_limit(10 ** 7)

CORTE_S = 900
CORTE_JORNADA_S = 3 * 3600
# Septiembre: el equipo reporta operación normal del 22-sep al 1-oct (Fiestas Patrias 18-19).
NORMAL_DESDE, NORMAL_HASTA = dt.date(2026, 9, 22), dt.date(2026, 10, 1)
# El WMS conserva ~1 mes: el 3-sep es el primer día de los exports nuevos y viene cortado.
PRIMER_DIA_COMPLETO = dt.date(2026, 9, 4)

# Qué tramo es "el útil" para contar actividad de cada proceso (instrucción del equipo).
CONFIABILIDAD = {
    "reabasto_caja": "reducida_bypass",   # el equipo reporta tareas marcadas "terminado" en bloque
    # Carga: las líneas de un mismo trailer se confirman en bloque (97% comparten usuario y
    # segundo): el timestamp es del cierre del trailer, no de cada pallet.
    "carga_camion": "reducida_lote",
}


def turno_de_hora(h: int) -> str:
    return "A" if 8 <= h < 16 else ("B" if 16 <= h < 24 else "C")


def _s2(celda: str) -> tuple[str, str]:
    """Celdas con dos valores apilados (origen arriba / destino abajo). Conserva la
    posición: un origen vacío no corre al destino."""
    p = [x.strip() for x in celda.split("\n")]
    while len(p) < 2:
        p.append("")
    return p[0].replace('="', "").replace('"', ""), p[1].replace('="', "").replace('"', "")


def leer_export(ruta: Path, archivo: str | None = None) -> pd.DataFrame:
    """Un export del WMS -> DataFrame con una fila por registro y columna `fila` (número de
    fila de datos en el archivo, 1 = primera) para trazabilidad."""
    rows = []
    with open(ruta, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for n, r in enumerate(rd, start=1):
            if len(r) < 10:
                continue
            f, a = _s2(r[0]), _s2(r[1])
            try:
                ts = dt.datetime.strptime(f[0], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue
            lo, ld = _s2(r[5])
            uo, ud = _s2(r[8])
            ao, ad = _s2(r[9])
            rows.append((archivo or ruta.name, n, ts, f[1], a[0], a[1], r[2].strip(),
                         int(r[3] or 0), r[4].strip(), lo, ld, uo, ud, ao, ad))
    df = pd.DataFrame(rows, columns=[
        "archivo", "fila", "ts", "usuario", "actividad", "operacion", "articulo", "cantidad",
        "um", "lpn_origen", "lpn_destino", "ubic_origen", "ubic_destino", "area_origen",
        "area_destino"])
    df["tramo1"] = df.ubic_destino.str.startswith("TRF")   # origen real -> TRF
    df["tramo2"] = df.ubic_origen.str.startswith("TRF")    # TRF -> destino real
    return df


def _agregar_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    df["fecha"] = df.ts.dt.date
    df["turno"] = df.ts.dt.hour.map(turno_de_hora)
    return df


# ---------------------------------------------------------------------------------------
# Auditoría de compatibilidad con los logs viejos
# ---------------------------------------------------------------------------------------
def _traslape(viejo: pd.DataFrame, nuevo: pd.DataFrame, cols: list[str]) -> dict:
    if viejo.empty or nuevo.empty:
        return {"desde": None, "hasta": None, "n_viejo": len(viejo), "n_nuevo": 0, "coinciden": 0}
    lo, hi = max(viejo.ts.min(), nuevo.ts.min()), min(viejo.ts.max(), nuevo.ts.max())
    if lo > hi:
        return {"desde": None, "hasta": None, "n_viejo": 0, "n_nuevo": 0, "coinciden": 0}
    v = viejo[(viejo.ts >= lo) & (viejo.ts <= hi)]
    n = nuevo[(nuevo.ts >= lo) & (nuevo.ts <= hi)]
    kv, kn = v.groupby(cols).size(), n.groupby(cols).size()
    j = pd.concat([kv.rename("v"), kn.rename("n")], axis=1).fillna(0)
    return {"desde": lo, "hasta": hi, "n_viejo": len(v), "n_nuevo": len(n),
            "coinciden": int(j[["v", "n"]].min(axis=1).sum())}


def auditoria_compatibilidad() -> list[dict]:
    """Compara, por proceso, el log viejo contra el export nuevo en el rango común.
    `coinciden` = registros idénticos (multiconjunto sobre ts, usuario, artículo, cantidad,
    ubicaciones). Si coinciden == n_viejo == n_nuevo el traslape es el mismo dato y se
    deduplica; si n_viejo < n_nuevo, el log viejo estaba incompleto en ese tramo."""
    base = ["ts", "usuario", "articulo", "cantidad", "ubic_origen", "ubic_destino"]
    out = []

    def fila(proc, tramo, v, n, cols=base):
        t = _traslape(v, n, cols)
        t.update({"proceso": proc, "tramo": tramo})
        out.append(t)

    rv = leer_export(VIEJOS["reabasto"], "viejo_reabasto")
    nuevo = {k: leer_export(p, k) for k, p in NUEVOS.items()}
    rec_v = leer_export(VIEJOS["recepcion"], "viejo_recepcion")
    pk_v = leer_export(VIEJOS["picking_sep"], "viejo_picking")
    fila("recepcion", 1, rec_v[rec_v.tramo1], nuevo["recepcion"][nuevo["recepcion"].tramo1])
    fila("recepcion", 2, rec_v[rec_v.tramo2], nuevo["recepcion"][nuevo["recepcion"].tramo2])
    fila("picking", 1, pk_v[pk_v.tramo1], nuevo["picking"][nuevo["picking"].tramo1])
    fila("picking", 2, pk_v[pk_v.tramo2], nuevo["picking"][nuevo["picking"].tramo2])
    for proc, act in (("reabasto_pallet", "Reabastecimiento de palés"),
                      ("reabasto_caja", "Reabastecimiento de cajas")):
        v = rv[rv.actividad == act]
        for t in (1, 2):
            c = f"tramo{t}"
            fila(proc, t, v[v[c]], nuevo[proc][nuevo[proc][c]])
    sv = rv[rv.actividad == "Trabajo suspendido"]
    fila("suspendido", 0, sv, nuevo["suspendido"], ["ts", "usuario", "operacion"])
    return out


# ---------------------------------------------------------------------------------------
# Pares de tramos
# ---------------------------------------------------------------------------------------
def _parear(df: pd.DataFrame, k1: list[str], k2: list[str], tolerancia_h: int = 12) -> pd.DataFrame:
    """Une cada tramo 1 con el tramo 2 siguiente de la misma clave (LPN, artículo,
    cantidad). Un tramo 2 se usa una sola vez: si dos tramos 1 caen sobre el mismo,
    se queda el más cercano en el tiempo."""
    a = df[df.tramo1].copy()
    b = df[df.tramo2].copy()
    a["_k"] = a[k1].astype(str).agg("|".join, axis=1)
    b["_k"] = b[k2].astype(str).agg("|".join, axis=1)
    a = a.sort_values("ts").reset_index()
    b = b.sort_values("ts").reset_index()
    b["_j"] = np.arange(len(b))
    b = b.rename(columns={"ts": "ts_2"})
    m = pd.merge_asof(a, b[["_k", "ts_2", "_j", "index", "usuario", "ubic_origen", "ubic_destino",
                            "area_origen", "area_destino", "fila"]],
                      left_on="ts", right_on="ts_2", by="_k", direction="forward",
                      tolerance=pd.Timedelta(hours=tolerancia_h), suffixes=("_1", "_2"),
                      allow_exact_matches=True)
    m = m.sort_values("ts").dropna(subset=["_j"])
    # tramo 2 reclamado por varios tramos 1: gana el último (el más cercano)
    m = m.drop_duplicates("_j", keep="last")
    m["t_traslado_s"] = (m.ts_2 - m.ts).dt.total_seconds()
    return m


def _ciclos_pares(proceso: str, df: pd.DataFrame, k1: list[str], k2: list[str],
                  filtro_op: str | None = None) -> pd.DataFrame:
    d = df if filtro_op is None else df[df.operacion == filtro_op]
    m = _parear(d, k1, k2)
    out = pd.DataFrame({
        "proceso": proceso,
        "id_par": m["archivo"] + "#" + m["fila_1"].astype(str) + "+" + m["fila_2"].astype(int).astype(str),
        "archivo": m["archivo"], "fila_tramo1": m["fila_1"], "fila_tramo2": m["fila_2"].astype(int),
        "ts_tramo1": m["ts"], "ts_tramo2": m["ts_2"],
        "fecha": m["ts"].dt.date, "turno": m["ts"].dt.hour.map(turno_de_hora),
        "usuario_tramo1": m["usuario_1"], "usuario_tramo2": m["usuario_2"],
        "articulo": m["articulo"], "cantidad": m["cantidad"],
        "ubic_origen_real": m["ubic_origen_1"], "ubic_trf": m["ubic_destino_1"],
        "ubic_destino_real": m["ubic_destino_2"],
        "area_origen": m["area_origen_1"], "area_destino": m["area_destino_2"],
        "t_traslado_s": m["t_traslado_s"],
    })
    out["flujo"] = out.area_origen + "->" + out.area_destino
    out["confiabilidad"] = CONFIABILIDAD.get(proceso, "normal")
    out["dentro_corte"] = ((out.t_traslado_s >= 0) & (out.t_traslado_s <= CORTE_S)).astype(int)
    return out.sort_values("ts_tramo1").reset_index(drop=True)


def _ciclos_picking(df: pd.DataFrame) -> pd.DataFrame:
    """Picking: pallet armado = líneas pickeadas (tramo 1) hasta su depósito en stage
    (tramo 2). Clave: LPN destino del tramo 1 = LPN origen del tramo 2."""
    p1 = df[df.tramo1].copy()
    p2 = df[df.tramo2].copy()
    g1 = p1.groupby("lpn_destino").agg(
        primer_pick=("ts", "min"), ultimo_pick=("ts", "max"), lineas=("ts", "size"),
        cajas=("cantidad", "sum"), usuarios=("usuario", "nunique"),
        usuario=("usuario", lambda s: s.mode().iat[0]), lista=("operacion", lambda s: s.mode().iat[0]),
        fila_primer_pick=("fila", "min"), archivo=("archivo", "first"))
    g2 = p2.groupby("lpn_origen").agg(
        deposito=("ts", "max"), lineas_deposito=("ts", "size"), stage=("ubic_destino", "first"),
        fila_deposito=("fila", "min"))
    g = g1.join(g2, how="left")
    g["t_armado_s"] = (g.deposito - g.primer_pick).dt.total_seconds()
    g["t_cierre_s"] = (g.deposito - g.ultimo_pick).dt.total_seconds()
    g = g.reset_index().rename(columns={"lpn_destino": "lpn"})
    g["proceso"] = "picking"
    g["id_par"] = g.archivo + "#" + g.fila_primer_pick.astype(str) + "+" + \
        g.fila_deposito.fillna(-1).astype(int).astype(str)
    g["fecha"] = g.primer_pick.dt.date
    g["turno"] = g.primer_pick.dt.hour.map(turno_de_hora)
    g["confiabilidad"] = "normal"
    return g


def intervalos_por_usuario(df: pd.DataFrame, proceso: str, tramo: str) -> pd.DataFrame:
    """Hueco entre eventos consecutivos del mismo usuario (misma jornada). Es tiempo de
    ciclo solo cuando una fila = una confirmación (picking, identificación de recepción)."""
    x = df[df[tramo]].sort_values(["usuario", "ts"]).copy()
    x["gap_s"] = x.groupby("usuario").ts.diff().dt.total_seconds()
    x = x[(x.gap_s.notna()) & (x.gap_s < CORTE_JORNADA_S)]
    x["proceso"] = proceso
    x["dentro_corte"] = (x.gap_s <= CORTE_S).astype(int)
    return x[["proceso", "archivo", "fila", "ts", "usuario", "gap_s", "dentro_corte"]].assign(
        fecha=x.ts.dt.date, turno=x.ts.dt.hour.map(turno_de_hora))


def rafagas(df: pd.DataFrame, tramo: str) -> float:
    """% de registros del tramo cuyo (usuario, ts al segundo) se repite en otro registro:
    la firma de varias tareas confirmadas de una vez."""
    x = df[df[tramo]]
    n = x.groupby(["usuario", "ts"]).ts.transform("size")
    return float((n >= 2).mean()) if len(x) else 0.0


# ---------------------------------------------------------------------------------------
# Construcción de las salidas
# ---------------------------------------------------------------------------------------
def leer_carga() -> pd.DataFrame:
    """Los tres exports de carga unidos (sus rangos no se solapan). `trailer` = número de TRL de la línea; las líneas del tramo 1 (PATCA->TRF) lo heredan
    de su tramo 2 emparejado por LPN, artículo y cantidad. Quedan sin trailer 137 líneas del tramo 1
    (137 el 29-sep 13:15, 14 pallets que pasaron a TRF y no figuran despachados a ningún trailer, más 3 sueltas)."""
    d = pd.concat([leer_export(r, r.name) for r in CARGA_EXPORTS if r.exists()], ignore_index=True)
    # NO se deduplica: los tres rangos no se solapan (verificado) y hay líneas gemelas legítimas
    # dentro de un mismo archivo (mismo pallet, artículo y cantidad dos veces en el mismo segundo).
    d = d.sort_values("ts", kind="stable").reset_index(drop=True)
    d["trailer"] = np.where(d.ubic_destino.str.startswith("TRL"), d.ubic_destino, "")
    m = _parear(d, ["lpn_origen", "articulo", "cantidad"], ["lpn_origen", "articulo", "cantidad"])
    d.loc[m["index_1"].values, "trailer"] = m["ubic_destino_2"].values
    return d


def cobertura_exports() -> list[dict]:
    """Alerta de truncamiento: el WMS corta cada export en 100.000 registros."""
    out = []
    for k, ruta in list(NUEVOS.items()) + [("carga_camion", r) for r in CARGA_EXPORTS]:
        d = leer_export(ruta)
        out.append({"export": k, "archivo": ruta.name, "registros": len(d), "desde": d.ts.min(),
                    "hasta": d.ts.max(), "truncado": len(d) >= TRUNCADO_EN})
    return out


def trailers_directos(eventos: pd.DataFrame) -> pd.DataFrame:
    """CONTEO DIRECTO de trailers: números de TRL distintos por (fecha, turno), asignando cada
    trailer al turno de su primera línea despachada. Un trailer se cuenta una sola vez aunque su
    carga cruce de turno (3 de 983 lo hacen)."""
    c = eventos[(eventos.proceso == "carga_camion") & (eventos.tramo != 1) & eventos.ubic_destino.str.startswith("TRL")]
    g = c.groupby("ubic_destino").agg(inicio=("ts", "min"), lineas=("ts", "size"), cajas=("cantidad", "sum")).reset_index()
    g["fecha"] = g.inicio.dt.date
    g["turno"] = g.inicio.dt.hour.map(turno_de_hora)
    return g


def lineas_por_trailer() -> dict:
    """Calibración del proxy ANTERIOR (líneas / líneas-por-trailer), que ya no se usa para el
    conteo pero se conserva para medir cuánto se equivocaba. Se calibró con 2 noches que tenían
    número de trailer: 23-sep (Inventario.csv, 21 trailers) y 1-oct (22 trailers)."""
    noches = {}
    iv = leer_export(DOCS / "Inventario.csv")
    iv = iv[(iv.actividad == "Carga de tráiler") & iv.ubic_destino.str.startswith("TRL")]
    c23 = iv[(iv.ts.dt.date == dt.date(2026, 9, 23)) & (iv.ts.dt.hour < 8)]
    noches["2026-09-23"] = (len(c23), c23.ubic_destino.nunique())
    # 1-oct: líneas del export anterior (tramo 1 de esa noche) y trailers del conteo directo
    c = leer_carga()
    c1 = c[(c.ts.dt.date == dt.date(2026, 10, 1)) & (c.ts.dt.hour < 8)]
    noches["2026-10-01"] = (int((~c1.tramo2).sum()), c1.trailer[c1.trailer != ""].nunique())
    lineas = sum(v[0] for v in noches.values())
    trailers = sum(v[1] for v in noches.values())
    return {"noches": noches, "lineas_por_trailer": lineas / trailers}


def trailers_estimados(eventos: pd.DataFrame) -> pd.DataFrame:
    """LEGADO (proxy del 2026-10-02 mañana): trailers = líneas de la noche (tramo 1 + directas) /
    líneas por trailer. Se conserva solo para compararlo contra el conteo directo."""
    lpt = lineas_por_trailer()["lineas_por_trailer"]
    c = eventos[(eventos.proceso == "carga_camion") & (eventos.tramo != 2)]
    g = c.groupby(["fecha", "turno"]).size().rename("lineas").reset_index()
    g["trailers_est"] = g.lineas / lpt
    return g


def tipo_suspension(op: str) -> str:
    if op == "Reabast. de cajas":
        return "reabasto_caja"
    if op.startswith("Lista de surtido"):
        return "picking_lista"
    if op.startswith("RECOGIDA ALMACENAMIENTO"):
        return "recogida_almacenamiento"
    if op == "Reabast de palés":
        return "reabasto_pallet"
    return "otro"


def construir() -> dict[str, pd.DataFrame]:
    nuevo = {k: _agregar_tiempo(leer_export(p, p.name)) for k, p in NUEVOS.items()}
    nuevo["carga_camion"] = _agregar_tiempo(leer_carga())

    # Picking 3-20 sep desde el log viejo (tramo 1 solamente), solo lo que el nuevo no cubre.
    pk_v = _agregar_tiempo(leer_export(VIEJOS["picking_sep"], VIEJOS["picking_sep"].name))
    pk_v = pk_v[pk_v.ts < nuevo["picking"].ts.min()]

    ev = []
    for proc, df in nuevo.items():
        d = df.copy()
        d["proceso"] = proc
        d["fuente"] = "nuevo"
        ev.append(d)
    pv = pk_v.copy()
    pv["proceso"] = "picking"
    pv["fuente"] = "viejo_solo_tramo1"
    ev.append(pv)
    eventos = pd.concat(ev, ignore_index=True)

    def util(r):
        return None
    eventos["tramo"] = np.where(eventos.tramo1, 1, np.where(eventos.tramo2, 2, 0))
    # tramo "útil" para contar actividad (instrucción del equipo)
    eventos["tramo_util"] = (
        ((eventos.proceso == "recepcion") & (eventos.tramo == 1))
        | ((eventos.proceso == "carga_camion") & (eventos.tramo != 1))   # línea despachada al trailer (TRF->TRL o directa)
        | (eventos.proceso.isin(["almacenamiento", "reabasto_pallet", "reabasto_caja"]) & (eventos.tramo == 2))
        | ((eventos.proceso == "picking") & (eventos.tramo == 1))
        | (eventos.proceso == "suspendido")).astype(int)
    eventos["confiabilidad"] = eventos.proceso.map(CONFIABILIDAD).fillna("normal")
    eventos["tipo_suspension"] = np.where(
        eventos.proceso == "suspendido", eventos.operacion.map(tipo_suspension), "")

    # --- ciclos
    alm = nuevo["almacenamiento"]
    ciclos = [
        _ciclos_pares("almacenamiento", alm, ["lpn_origen", "articulo", "cantidad"],
                      ["lpn_origen", "articulo", "cantidad"], "Transferencia de carga no dirigida"),
        _ciclos_pares("reabasto_pallet", nuevo["reabasto_pallet"], ["lpn_origen", "articulo"],
                      ["lpn_origen", "articulo"], "Reabast de palés"),
        # caja: el tramo 1 cambia de LPN (origen -> COQ...), el tramo 2 trae el COQ... de origen
        _ciclos_pares("reabasto_caja", nuevo["reabasto_caja"], ["lpn_destino", "articulo"],
                      ["lpn_origen", "articulo"], "Reabast. de cajas"),
    ]
    ciclos = pd.concat(ciclos, ignore_index=True)
    # tiempo desde el depósito de la tarea anterior del mismo usuario (viaje en vacío incluido)
    ciclos = ciclos.sort_values(["proceso", "usuario_tramo2", "ts_tramo2"])
    ciclos["t_desde_tarea_anterior_s"] = ciclos.groupby(
        ["proceso", "usuario_tramo2"]).ts_tramo2.diff().dt.total_seconds()
    ciclos.loc[ciclos.t_desde_tarea_anterior_s >= CORTE_JORNADA_S, "t_desde_tarea_anterior_s"] = np.nan
    ciclos = ciclos.sort_values("ts_tramo1").reset_index(drop=True)
    # REABASTO DE CAJA: el log sirve SOLO para frecuencia, nunca para duración (decisión del
    # usuario, 2026-10-02: los timestamps son poco confiables por el bypass, y el tiempo real
    # viene del cronometraje de terreno). El par se conserva para contar tareas, pero NO se
    # publica ningún tiempo. `diag_bypass_gap_s` es solo el diagnóstico de la sospecha (hueco
    # entre tramos) y no debe leerse como duración.
    caja = ciclos.proceso == "reabasto_caja"
    ciclos["diag_bypass_gap_s"] = np.where(caja, ciclos.t_traslado_s, np.nan)
    for c in ("t_traslado_s", "t_desde_tarea_anterior_s"):
        ciclos.loc[caja, c] = np.nan
    ciclos.loc[caja, "dentro_corte"] = 0
    ciclos["tiempo_valido"] = (~caja).astype(int)

    pick = _ciclos_picking(nuevo["picking"])

    # --- intervalos
    intervalos = pd.concat([
        intervalos_por_usuario(nuevo["recepcion"], "recepcion", "tramo1"),
        intervalos_por_usuario(pd.concat([pk_v, nuevo["picking"]]), "picking", "tramo1"),
    ], ignore_index=True)

    # --- suspendidos
    s = nuevo["suspendido"].copy()
    s["tipo"] = s.operacion.map(tipo_suspension)
    susp = (s.groupby(["fecha", "turno", "tipo", "usuario"]).size().rename("suspendidas")
            .reset_index())

    return {"eventos": eventos, "ciclos": ciclos, "picking_pallets": pick,
            "intervalos": intervalos, "suspendidos": susp, "suspendidos_raw": s,
            "nuevo": nuevo}


def dias_operados(eventos: pd.DataFrame) -> list[dt.date]:
    """Fechas del 4-sep al 1-oct con al menos un registro de picking, recepción o
    almacenamiento. Quedan fuera los días sin NINGÚN registro (domingos y 18, 19 y 20-sep):
    no son días de baja actividad sino días sin operación [inferido: el WMS no registra nada]."""
    e = eventos[eventos.proceso.isin(["picking", "recepcion", "almacenamiento"])]
    f = pd.to_datetime(e.fecha).dt.date
    return sorted(d for d in set(f) if PRIMER_DIA_COMPLETO <= d <= dt.date(2026, 10, 1))


def dias_turnos_completos(eventos: pd.DataFrame) -> set[tuple]:
    """(fecha, turno) analizables: días operados del 4-sep al 30-sep con sus tres turnos, y
    el 1-oct solo C (los exports se bajaron a media mañana del 1-oct)."""
    out = set()
    for d in dias_operados(eventos):
        for t in "ABC":
            if d < dt.date(2026, 10, 1) or t == "C":
                out.add((d, t))
    return out


def grupo(fecha: dt.date) -> str:
    return "normal_22sep_1oct" if NORMAL_DESDE <= fecha <= NORMAL_HASTA else "resto_sep"


def tabla_turnos(r: dict) -> pd.DataFrame:
    ev = r["eventos"].assign(fecha=lambda d: pd.to_datetime(d.fecha).dt.date)
    u = ev[ev.tramo_util == 1]
    filas = []
    piv = u.groupby(["fecha", "turno", "proceso"]).size().unstack(fill_value=0)
    cj = ev[(ev.proceso == "picking") & (ev.tramo == 1)].groupby(["fecha", "turno"]).cantidad.sum()
    pk_u = ev[(ev.proceso == "picking") & (ev.tramo == 1)].groupby(["fecha", "turno"]).usuario.nunique()
    rec_c = ev[(ev.proceso == "recepcion")].groupby(["fecha", "turno"]).cantidad.sum()
    ret_caj = ev[(ev.proceso == "reabasto_caja") & (ev.tramo == 1)].groupby(["fecha", "turno"]).size()
    sp = r["suspendidos_raw"].groupby(["fecha", "turno", "tipo"]).size().unstack(fill_value=0)
    comp = dias_turnos_completos(ev)
    idx = sorted(comp)
    t = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["fecha", "turno"]))
    t["completa"] = 1
    t["grupo"] = [grupo(f) for f, _ in idx]
    for c in ("recepcion", "almacenamiento", "reabasto_pallet", "reabasto_caja",
              "carga_camion", "picking"):
        t[f"{c}_lineas_tramo_util"] = piv[c].reindex(t.index).fillna(0).astype(int) if c in piv else 0
    t["picking_cajas"] = cj.reindex(t.index).fillna(0).astype(int)
    t["picking_usuarios"] = pk_u.reindex(t.index).fillna(0).astype(int)
    t["recepcion_cajas"] = rec_c.reindex(t.index).fillna(0).astype(int)
    t["reabasto_caja_retiros"] = ret_caj.reindex(t.index).fillna(0).astype(int)
    for c in ("reabasto_caja", "picking_lista", "recogida_almacenamiento", "reabasto_pallet", "otro"):
        t[f"susp_{c}"] = sp[c].reindex(t.index).fillna(0).astype(int) if c in sp else 0
    t["susp_total"] = t[[c for c in t.columns if c.startswith("susp_")]].sum(axis=1)
    # --- columnas agregadas el 2026-10-04 para cruzar con las ventas (demanda_actividad.py) ---
    # carga: trailers distintos (conteo directo), asignados al turno de su primera línea despachada
    tr = trailers_directos(ev)
    t["carga_trailers"] = tr.groupby(["fecha", "turno"]).size().reindex(t.index).fillna(0).astype(int)
    # almacenamiento a ubicación final: la misma definición que su tiempo y su driver en capacity.py
    ci = r["ciclos"].assign(fecha=lambda d: pd.to_datetime(d.fecha).dt.date)
    af = ci[(ci.proceso == "almacenamiento") & (ci.area_origen == "PATCA") & ci.area_destino.isin(AREAS_DESTINO_FINAL)]
    t["almacenamiento_destino_final"] = af.groupby(["fecha", "turno"]).size().reindex(t.index).fillna(0).astype(int)
    # picking en pallets armados (solo desde el 21-sep: antes no hay tramo de depósito)
    pp = r["picking_pallets"].assign(fecha=lambda d: pd.to_datetime(d.fecha).dt.date)
    t["picking_pallets"] = pp.groupby(["fecha", "turno"]).size().reindex(t.index)
    # recepción en camiones ESTIMADOS: mismo método que recepcion_log (un camión nuevo cuando pasan más de
    # GAP_CAMION_S entre dos pallets del día), asignando cada camión al turno de su primer pallet.
    from modelo.recepcion_log import GAP_CAMION_S
    rc = ev[(ev.proceso == "recepcion") & (ev.tramo == 1)].sort_values("ts")
    cam = []
    for f, g in rc.groupby("fecha"):
        ts = list(g.ts)
        ini = [ts[0]] + [b_ for a_, b_ in zip(ts, ts[1:]) if (b_ - a_).total_seconds() > GAP_CAMION_S]
        cam += [(f, turno_de_hora(x.hour)) for x in ini]
    cam = pd.Series(1, index=pd.MultiIndex.from_tuples(cam, names=["fecha", "turno"])).groupby(level=[0, 1]).sum() \
        if cam else pd.Series(dtype=int)
    t["recepcion_camiones_est"] = cam.reindex(t.index).fillna(0).astype(int)
    return t.reset_index()


def obs_almacenamiento(ciclos: pd.DataFrame) -> pd.DataFrame:
    """Tiempos reales de almacenamiento (retiro en stage -> depósito en la ubicación final) en el
    formato de observaciones que lee `fit_params`. Solo pares dentro del corte de 900 s."""
    a = ciclos[(ciclos.proceso == "almacenamiento") & (ciclos.dentro_corte == 1)
               & (ciclos.area_origen == "PATCA") & ciclos.area_destino.isin(AREAS_DESTINO_FINAL)]
    return pd.DataFrame({
        "proceso": "almacenamiento", "paso": "ciclo_total",
        "muestra": np.arange(1, len(a) + 1), "fecha": a.fecha.astype(str).values,
        "turno": a.turno.values, "operario": a.usuario_tramo2.values,
        "duracion_s": a.t_traslado_s.round().astype(int).values,
        "origen": "wms_retiro_a_deposito", "hoja": a.archivo.values, "fila": a.fila_tramo1.values})


def obs_reabasto_pallet(ciclos: pd.DataFrame) -> pd.DataFrame:
    """Tiempos retiro->depósito de reabasto de pallet, uno por par, con el origen en el nombre del paso
    (`ciclo_total_almac` / `ciclo_total_almpic`), en el formato de observaciones de `fit_params`.
    Solo pares dentro del corte de 900 s, de septiembre completo (el tiempo medio por origen es estable
    entre regímenes: ALMPIC 36,8 s antes del 22-sep y 36,9 s después)."""
    a = ciclos[(ciclos.proceso == "reabasto_pallet") & (ciclos.dentro_corte == 1)
               & ciclos.area_origen.isin(ORIGENES_PALLET)].sort_values("ts_tramo1")
    return pd.DataFrame({
        "proceso": "reabasto_pallet", "paso": "ciclo_total_" + a.area_origen.str.lower().values,
        "muestra": np.arange(1, len(a) + 1), "fecha": a.fecha.astype(str).values,
        "turno": a.turno.values, "operario": a.usuario_tramo2.values,
        "duracion_s": a.t_traslado_s.round().astype(int).values,
        "origen": "wms_retiro_a_deposito", "hoja": a.archivo.values, "fila": a.fila_tramo1.values})


def guardar() -> dict[str, pd.DataFrame]:
    r = construir()
    LIMPIO.mkdir(parents=True, exist_ok=True)
    ev = r["eventos"].drop(columns=["tramo1", "tramo2"])
    ev.to_csv(SAL_EVENTOS, index=False)
    ci = pd.concat([r["ciclos"], r["picking_pallets"].assign(
        ts_tramo1=r["picking_pallets"].primer_pick, ts_tramo2=r["picking_pallets"].deposito,
        usuario_tramo1=r["picking_pallets"].usuario, t_traslado_s=r["picking_pallets"].t_armado_s,
        dentro_corte=lambda d: ((d.t_armado_s >= 0) & (d.t_armado_s <= 3 * 3600)).astype(int),
        articulo="(varios)", cantidad=r["picking_pallets"].cajas)], ignore_index=True)
    ci.to_csv(SAL_CICLOS, index=False)
    r["intervalos"].to_csv(SAL_INTERVALOS, index=False)
    r["suspendidos"].to_csv(SAL_SUSP, index=False)
    t = tabla_turnos(r)
    t.to_csv(SAL_TURNOS, index=False)
    obs_almacenamiento(r["ciclos"]).to_csv(SAL_OBS_FIT, index=False)
    obs_reabasto_pallet(r["ciclos"]).to_csv(SAL_OBS_PALLET, index=False)
    r["turnos"] = t
    r["ciclos_csv"] = ci
    return r


if __name__ == "__main__":
    r = guardar()
    ev, ci = r["eventos"], r["ciclos"]
    print(f"{len(ev):,} eventos | {len(ci):,} pares | {len(r['picking_pallets']):,} pallets de picking")
    for p, g in ci.groupby("proceso"):
        ok = g[g.dentro_corte == 1].t_traslado_s
        print(f"  {p:<16} pares {len(g):>6,}  dentro de {CORTE_S}s {len(ok):>6,}  "
              f"mediana {ok.median():.0f}s  [{g.confiabilidad.iat[0]}]")
