"""Exports de actividad del WMS de octubre-2026 (`docs/wms actividades octubre hasta el 11/`): conteos por día y turno de los cuatro procesos que traen,
y la tabla UNIFICADA septiembre + octubre que calibra el volumen por turno en `capacity.py`.

QUÉ TRAE Y QUÉ NO (verificado el 2026-10-13 contra los archivos)
  Almacenamiento       1-oct 17:21 a 10-oct 20:28, 3.706 registros (los dos tramos STREC -> TRF -> ubicación final)
  Carga de tráiler     2-oct 01:34 a 10-oct 17:36, 81.548 registros (ambos tramos y número de trailer)
  Reabasto cajas+pallet 2-oct 00:26 a 10-oct 11:45, 1.654 registros (palés 1.238, cajas 416)
  Recepción            1-oct 18:34 a 10-oct 19:57, 3.912 registros
  NO TRAE: picking, ni "Trabajo suspendido" (reabasto cancelado/suspendido). Por eso el tiempo y el driver de picking, la tasa de suspensión de
  reabasto de cajas y las suspensiones de listas de picking siguen siendo los de septiembre: NADA de eso se actualiza con estos archivos.
  Tampoco hay ventas de octubre: los drivers contra el facturado (picking, carga de las vueltas >= 2) siguen siendo los de septiembre.

COBERTURA POR TURNO (qué (fecha, turno) cuenta como completo)
  Los exports de septiembre llegan hasta el 1-oct ~07:50 y estos arrancan entre el 1-oct 17:21 y el 2-oct 01:34, así que el 1-oct de día y de tarde no está
  en NINGÚN export (solo el turno C del 1-oct, que es de septiembre). Del 2 al 9-oct los tres turnos de los cuatro procesos están cubiertos. El 10-oct solo
  está completo el turno C: el export de reabasto termina a las 11:45 y el turno A llega hasta las 16:00.
  Los domingos (4-oct) no hay ningún registro: se tratan como días sin operación [inferido], igual que en septiembre.

COMBINACIÓN: carga se solapa con septiembre entre el 2-oct 01:34 y 10:45; se usa el export de septiembre hasta su último registro y el de octubre desde ahí.
Sin solape en los demás procesos.

Salidas (`salidas/limpio/`):
  wms_oct_turnos.csv        una fila por (fecha, turno) de octubre con los conteos
  actividad_turnos.csv      septiembre (de wms_sep_turnos.csv) + octubre, con `fuente` y `completa`; es lo que lee capacity.py
  actividad_obs_almacenamiento.csv / actividad_obs_reabasto_pallet.csv   tiempos retiro->depósito de septiembre + octubre en el formato de `fit_params`
PRIVACIDAD: estas salidas llevan solo conteos y tiempos; el usuario del WMS solo se usa en memoria para el pareo y se guarda como código anónimo.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from modelo import wms_septiembre as ws

RAIZ = Path(__file__).resolve().parent.parent
DIR = RAIZ / "docs" / "wms actividades octubre hasta el 11"
ARCHIVOS = {
    "almacenamiento": DIR / "Almacenamiento octubre.csv",
    "carga_camion": DIR / "carga octubre.csv",
    "reabasto": DIR / "Reabastecimientos de cajas y palletts octubre.csv",
    "recepcion": DIR / "Recepcion octubre.csv",
}
LIMPIO = RAIZ / "salidas" / "limpio"
SAL_TURNOS_OCT = LIMPIO / "wms_oct_turnos.csv"
SAL_ACTIVIDAD = LIMPIO / "actividad_turnos.csv"
SAL_OBS_ALM = LIMPIO / "actividad_obs_almacenamiento.csv"       # septiembre + octubre, para fit_params
SAL_OBS_PAL = LIMPIO / "actividad_obs_reabasto_pallet.csv"

PRIMER_DIA_OCT, ULTIMO_DIA_COMPLETO = dt.date(2026, 10, 2), dt.date(2026, 10, 9)
ULTIMO_DIA_PARCIAL = dt.date(2026, 10, 10)             # solo el turno C está completo
DIAS_SIN_PICKING = "desde 2026-10-02"                  # los exports de octubre no traen picking ni suspendidos
COLUMNAS = ["reabasto_pallet_lineas_tramo_util", "reabasto_caja_lineas_tramo_util", "reabasto_caja_retiros", "recepcion_lineas_tramo_util",
            "recepcion_cajas", "recepcion_camiones_est", "almacenamiento_lineas_tramo_util", "almacenamiento_destino_final", "carga_trailers"]


def _anon(df: pd.DataFrame) -> pd.DataFrame:
    """Código estable por usuario (solo en memoria y en las salidas de tiempos): ningún usuario del WMS sale a disco."""
    cod = {u: f"U{k + 1:03d}" for k, u in enumerate(sorted(df.usuario.unique()))}
    return df.assign(usuario=df.usuario.map(cod))


def leer() -> dict[str, pd.DataFrame]:
    """Los cuatro exports como DataFrames (tramos marcados, fecha y turno), con el reabasto separado en palés y cajas y la carga unida a septiembre."""
    out: dict[str, pd.DataFrame] = {}
    for k in ("almacenamiento", "recepcion"):
        out[k] = ws._agregar_tiempo(ws.leer_export(ARCHIVOS[k], ARCHIVOS[k].name))
    rb = ws._agregar_tiempo(ws.leer_export(ARCHIVOS["reabasto"], ARCHIVOS["reabasto"].name))
    out["reabasto_pallet"] = rb[rb.actividad == "Reabastecimiento de palés"].copy()
    out["reabasto_caja"] = rb[rb.actividad == "Reabastecimiento de cajas"].copy()
    # carga: septiembre hasta su último registro + octubre desde ahí (sin duplicar el solape)
    sep = ws.leer_carga()
    oct_ = ws._agregar_tiempo(ws.leer_export(ARCHIVOS["carga_camion"], ARCHIVOS["carga_camion"].name))
    oct_["trailer"] = np.where(oct_.ubic_destino.str.startswith("TRL"), oct_.ubic_destino, "")
    m = ws._parear(oct_, ["lpn_origen", "articulo", "cantidad"], ["lpn_origen", "articulo", "cantidad"])
    oct_.loc[m["index_1"].values, "trailer"] = m["ubic_destino_2"].values
    out["carga_camion"] = pd.concat([sep, oct_[oct_.ts > sep.ts.max()]], ignore_index=True)
    out["carga_camion_solape"] = oct_[oct_.ts <= sep.ts.max()]
    for k, d in out.items():
        d["tramo"] = np.where(d.tramo1, 1, np.where(d.tramo2, 2, 0))
    return out


def auditoria_solape(r: dict) -> dict:
    """Verifica que el solape de carga (2-oct 01:34 a 10:45) sea el mismo dato en los dos exports: líneas por trailer y total."""
    sep = ws.leer_carga()
    ov = r["carga_camion_solape"]
    lo, hi = ov.ts.min(), ov.ts.max()
    s = sep[(sep.ts >= lo) & (sep.ts <= hi)]
    return {"desde": lo, "hasta": hi, "septiembre": len(s), "octubre": len(ov),
            "trailers_sep": s[s.ubic_destino.str.startswith("TRL")].ubic_destino.nunique(),
            "trailers_oct": ov[ov.ubic_destino.str.startswith("TRL")].ubic_destino.nunique(),
            "septiembre_antes_de_oct": int((sep.ts < lo).sum() and (sep[(sep.ts >= pd.Timestamp("2026-10-02")) & (sep.ts < lo)].shape[0]))}


def pares_almacenamiento(df: pd.DataFrame) -> pd.DataFrame:
    """Pares retiro (STREC -> TRF) -> depósito (TRF -> ubicación) SIN filtrar por la operación del registro. CORRECCIÓN (2026-10-13): hasta septiembre solo se
    pareaban los depósitos rotulados «Transferencia de carga no dirigida» y se dejaban fuera los rotulados «Identificación no dirigida» (322 en septiembre,
    463 en octubre = 25% de los depósitos). Son la MISMA tarea —pallet de stage a su ubicación— con otro rótulo: mismo tiempo (media 104 s contra 106 s,
    mediana 86 contra 87 s) y mismos destinos. Dejarlos fuera subcontaba el volumen de almacenamiento, sobre todo el 7, 8 y 9-oct."""
    return ws._ciclos_pares("almacenamiento", df, ["lpn_origen", "articulo", "cantidad"], ["lpn_origen", "articulo", "cantidad"], None)


def _destino_final(df: pd.DataFrame) -> pd.DataFrame:
    p = pares_almacenamiento(df)
    return p[(p.area_origen == "PATCA") & p.area_destino.isin(ws.AREAS_DESTINO_FINAL)]


def completa(fecha: dt.date, turno: str) -> bool:
    if fecha < PRIMER_DIA_OCT or fecha > ULTIMO_DIA_PARCIAL:
        return False
    return fecha <= ULTIMO_DIA_COMPLETO or turno == "C"


def tabla_turnos(r: dict) -> pd.DataFrame:
    """Una fila por (fecha, turno) de octubre con los conteos, con las MISMAS definiciones que `wms_septiembre.tabla_turnos`."""
    def cnt(df, mask=None):
        d = df if mask is None else df[mask]
        return d.groupby(["fecha", "turno"]).size()

    pal, caj, rec, alm, car = r["reabasto_pallet"], r["reabasto_caja"], r["recepcion"], r["almacenamiento"], r["carga_camion"]
    af = _destino_final(alm)
    # recepción: camiones estimados con el mismo método (hueco de GAP_CAMION_S entre pallets del día)
    from modelo.recepcion_log import GAP_CAMION_S
    rc = rec[rec.tramo == 1].sort_values("ts")
    cam = []
    for f, g in rc.groupby("fecha"):
        ts = list(g.ts)
        ini = [ts[0]] + [b for a, b in zip(ts, ts[1:]) if (b - a).total_seconds() > GAP_CAMION_S]
        cam += [(f, ws.turno_de_hora(x.hour)) for x in ini]
    cam = pd.Series(1, index=pd.MultiIndex.from_tuples(cam, names=["fecha", "turno"])).groupby(level=[0, 1]).sum()
    # carga: trailers distintos, cada uno en el turno de su primera línea despachada
    c = car[(car.tramo != 1) & car.ubic_destino.str.startswith("TRL")]
    g = c.groupby("ubic_destino").agg(inicio=("ts", "min"))
    g["fecha"], g["turno"] = g.inicio.dt.date, g.inicio.dt.hour.map(ws.turno_de_hora)
    tr = g.groupby(["fecha", "turno"]).size()
    idx = sorted({(f, t) for f in set(pal.fecha) | set(caj.fecha) | set(rec.fecha) | set(alm.fecha) | set(c.ts.dt.date)
                  for t in "ABC" if completa(f, t)})
    t = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["fecha", "turno"]))
    t["reabasto_pallet_lineas_tramo_util"] = cnt(pal, pal.tramo == 2).reindex(t.index)
    t["reabasto_caja_lineas_tramo_util"] = cnt(caj, caj.tramo == 2).reindex(t.index)
    t["reabasto_caja_retiros"] = cnt(caj, caj.tramo == 1).reindex(t.index)
    t["recepcion_lineas_tramo_util"] = cnt(rec, rec.tramo == 1).reindex(t.index)
    t["recepcion_cajas"] = rec[rec.tramo == 1].groupby(["fecha", "turno"]).cantidad.sum().reindex(t.index)
    t["recepcion_camiones_est"] = cam.reindex(t.index)
    t["almacenamiento_lineas_tramo_util"] = cnt(alm, alm.tramo == 2).reindex(t.index)
    t["almacenamiento_destino_final"] = af.groupby(["fecha", "turno"]).size().reindex(t.index)
    t["carga_trailers"] = tr.reindex(t.index)
    t = t.fillna(0).astype(int)
    t["completa"] = 1
    t["fuente"] = "octubre"
    return t.reset_index()


def _codigos(*series: pd.Series) -> dict:
    return {u: f"U{k + 1:03d}" for k, u in enumerate(sorted(set().union(*[set(x) for x in series])))}


def obs_tiempos(r: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiempos retiro->depósito de OCTUBRE en el formato de `fit_params`: almacenamiento (STREC -> ubicación final, sin filtrar por rótulo) y reabasto de pallet
    por origen. Mismas reglas que `wms_septiembre.obs_*`: solo pares dentro del corte de 900 s. El usuario va como código anónimo."""
    alm = pares_almacenamiento(r["almacenamiento"])
    pal = ws._ciclos_pares("reabasto_pallet", r["reabasto_pallet"], ["lpn_origen", "articulo"], ["lpn_origen", "articulo"], "Reabast de palés")
    cod = _codigos(alm.usuario_tramo2, pal.usuario_tramo2)
    for d in (alm, pal):
        d["usuario_tramo2"] = d.usuario_tramo2.map(cod)
    return ws.obs_almacenamiento(alm), ws.obs_reabasto_pallet(pal)


def obs_combinadas(r: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Septiembre + octubre para `fit_params`. Almacenamiento: septiembre RE-PAREADO sin filtro de rótulo (ver `pares_almacenamiento`) + octubre. Reabasto de
    pallet: septiembre tal cual (`wms_septiembre`: no cambia, 2 de 2.216 depósitos tenían otro rótulo) + octubre. Los usuarios se reemplazan por códigos."""
    sep_alm = ws._agregar_tiempo(ws.leer_export(ws.NUEVOS["almacenamiento"], ws.NUEVOS["almacenamiento"].name))
    a_s, a_o = pares_almacenamiento(sep_alm), pares_almacenamiento(r["almacenamiento"])
    sep_pal = ws._agregar_tiempo(ws.leer_export(ws.NUEVOS["reabasto_pallet"], ws.NUEVOS["reabasto_pallet"].name))
    p_s = ws._ciclos_pares("reabasto_pallet", sep_pal, ["lpn_origen", "articulo"], ["lpn_origen", "articulo"], "Reabast de palés")
    p_o = ws._ciclos_pares("reabasto_pallet", r["reabasto_pallet"], ["lpn_origen", "articulo"], ["lpn_origen", "articulo"], "Reabast de palés")
    cod = _codigos(a_s.usuario_tramo2, a_o.usuario_tramo2, p_s.usuario_tramo2, p_o.usuario_tramo2)
    out = []
    for fn, partes in ((ws.obs_almacenamiento, (a_s, a_o)), (ws.obs_reabasto_pallet, (p_s, p_o))):
        d = pd.concat([x.assign(usuario_tramo2=x.usuario_tramo2.map(cod)) for x in partes], ignore_index=True)
        o = fn(d)
        o["muestra"] = np.arange(1, len(o) + 1)
        out.append(o)
    return out[0], out[1]


def unificar(t_oct: pd.DataFrame) -> pd.DataFrame:
    """Septiembre (de `wms_sep_turnos.csv`, sin picking ni suspensiones) + octubre. En septiembre, 4-sep a 30-sep los tres turnos y el 1-oct solo C."""
    sep = pd.read_csv(ws.SAL_TURNOS)
    sep["fecha"] = pd.to_datetime(sep.fecha).dt.date
    sep["fuente"] = "septiembre"
    # almacenamiento de septiembre RE-CONTADO sin filtrar por rótulo (misma definición que octubre)
    alm = ws._agregar_tiempo(ws.leer_export(ws.NUEVOS["almacenamiento"], ws.NUEVOS["almacenamiento"].name))
    af = _destino_final(alm).groupby(["fecha", "turno"]).size()
    clave = pd.MultiIndex.from_frame(sep[["fecha", "turno"]])
    sep["almacenamiento_destino_final"] = af.reindex(clave).fillna(0).astype(int).values
    cols = ["fecha", "turno", "completa", "fuente"] + COLUMNAS
    return pd.concat([sep[cols], t_oct[cols]], ignore_index=True).sort_values(["fecha", "turno"]).reset_index(drop=True)


def guardar() -> dict:
    r = leer()
    LIMPIO.mkdir(parents=True, exist_ok=True)
    t = tabla_turnos(r)
    t.to_csv(SAL_TURNOS_OCT, index=False)
    u = unificar(t)
    u.to_csv(SAL_ACTIVIDAD, index=False)
    a, p = obs_combinadas(r)
    a.to_csv(SAL_OBS_ALM, index=False)
    p.to_csv(SAL_OBS_PAL, index=False)
    return {"datos": r, "turnos": t, "unificada": u, "obs_almacenamiento": a, "obs_reabasto_pallet": p}


if __name__ == "__main__":
    res = guardar()
    r, t = res["datos"], res["turnos"]
    print("solape de carga:", auditoria_solape(r))
    pd.set_option("display.width", 220)
    print(t.drop(columns=["completa", "fuente"]).to_string(index=False))
    print(f"tiempos sep+oct: almacenamiento {len(res['obs_almacenamiento'])} pares, reabasto de pallet {len(res['obs_reabasto_pallet'])} pares")
