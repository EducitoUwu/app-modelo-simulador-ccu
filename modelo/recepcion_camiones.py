"""RECEPCIÓN POR TIPO DE CAMIÓN: chicos (~10 pallets) y grandes (~28 pallets) (2026-10-26, idea del usuario).

VOLUMEN (log del WMS, `docs/wms actividades septiembre/Recepcion Septiembre.csv` + `docs/wms actividades octubre hasta el 11/Recepcion octubre.csv`, 3-sep a 10-oct)
  Cada identificación de pallet trae en «ubicación de origen» el NÚMERO DE DOCUMENTO de recepción. Los pallets por documento tienen dos modas muy claras: 11 (camión chico) y 27-29 (camión
  grande); un documento de más de ~35 pallets son varios camiones grandes bajo un mismo documento. Regla: documento ≤ CORTE_CHICO pallets = 1 camión chico; si no, round(pallets / 28) camiones
  grandes (mínimo 1). Días completos: 4-sep a 9-oct sin el 1-oct (el 3-sep, el 1-oct y el 10-oct están cortados por los exports).
  Contra la venta (días con venta de septiembre): los GRANDES escalan (razón por 1.000 cajas facturadas) y los CHICOS no (nivel fijo por día).
TIEMPOS (planilla oficial de terreno, hoja «Recepción_», 31 camiones; columnas Pallets y «Planificador?»)
  descarga = «Empieza descarga» → «Termina descarga» (gruero con grúa); escaneo = «Empieza a ingresar pallets» → «Termina de ingresar pallets».
  Sin planificador el gruero hace las dos cosas seguidas: gruero = descarga + escaneo. Con planificador el escaneo lo hace el planificador (en los grandes, en paralelo con la descarga):
  gruero = descarga y planificador = escaneo. NO se cuenta como trabajo del gruero la espera del camión (llega → estaciona; hasta 3 h) ni el tramo estaciona → empieza descarga.
  % de camiones que escanea el planificador: log de recepción con usuarios (`docs/recepcion personas.csv`, 30-sep a 19-oct; lista local de planificadores), por tipo de camión.
  Tiempo esperado del gruero por camión = p · gruero_con + (1 − p) · gruero_sin. Las horas del planificador se informan aparte: no es un rol del modelo.
PRIVACIDAD: ninguna salida lleva nombres ni usuarios (la planilla trae «Gruero responsable» y el log el usuario: no se leen a disco).
USO: .venv/bin/python -m modelo.recepcion_camiones     Salidas: salidas/limpio/recepcion_camiones.json (lo lee capacity.py) y salidas/recepcion_camiones/*.csv
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent.parent
JSON = RAIZ / "salidas" / "limpio" / "recepcion_camiones.json"
SAL = RAIZ / "salidas" / "recepcion_camiones"
PLANILLA = RAIZ / "docs" / "planilla tiempos oficial.xlsx"
CORTE_CHICO = 19                     # pallets: ≤ 19 = camión chico (la moda es 11; entre 12 y 19 hay 28 documentos)
PALLETS_GRANDE_NOMINAL = 28
DIAS_CORTADOS = {dt.date(2026, 9, 3), dt.date(2026, 10, 1), dt.date(2026, 10, 10)}
TIPOS = ("chico", "grande")


def documentos() -> pd.DataFrame:
    from modelo import wms_octubre as wo, wms_septiembre as ws
    d = pd.concat([ws._agregar_tiempo(ws.leer_export(ws.NUEVOS["recepcion"])), ws._agregar_tiempo(ws.leer_export(wo.ARCHIVOS["recepcion"]))])
    d = d[d.tramo1].sort_values("ts").drop_duplicates("lpn_origen")
    g = d.groupby("ubic_origen").agg(pallets=("lpn_origen", "size"), fecha=("fecha", "first"), turno=("turno", "first")).reset_index(drop=True)
    g["tipo"] = np.where(g.pallets <= CORTE_CHICO, "chico", "grande")
    g["camiones"] = np.where(g.tipo == "chico", 1, np.maximum(1, np.round(g.pallets / PALLETS_GRANDE_NOMINAL))).astype(int)
    return g


def por_dia(g: pd.DataFrame) -> pd.DataFrame:
    g = g[~g.fecha.isin(DIAS_CORTADOS)]
    day = g.pivot_table(index="fecha", columns="tipo", values="camiones", aggfunc="sum", fill_value=0)
    for t in TIPOS:
        day[f"pallets_{t}"] = g[g.tipo == t].groupby("fecha").pallets.sum()
    day = day.fillna(0)
    day["pallets"] = day.pallets_chico + day.pallets_grande
    dd = pd.read_csv(RAIZ / "salidas" / "limpio" / "demanda_diaria.csv", parse_dates=["fecha"])
    dd["fecha"] = dd.fecha.dt.date
    return day.join(dd.set_index("fecha")[["facturado"]]).reset_index()


def _p_intercepto(x, y) -> float:
    n = len(x)
    r = stats.linregress(x, y)
    res = y - (r.intercept + r.slope * x)
    se = np.sqrt(((res ** 2).sum() / (n - 2)) * (1 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
    return float(2 * stats.t.sf(abs(r.intercept / se), n - 2))


def tiempos() -> pd.DataFrame:
    d = pd.read_excel(PLANILLA, sheet_name="Recepción_", header=2)
    d = d[pd.to_numeric(d["Pallets"], errors="coerce").notna()].copy()
    s = lambda v: v.hour * 3600 + v.minute * 60 + v.second if hasattr(v, "hour") else np.nan
    c = list(d.columns)
    col = {"llega": c[2], "estaciona": c[3], "ini_desc": c[4], "fin_desc": c[5], "ini_esc": next(x for x in c if x.startswith("Empieza a ingresar")),
           "fin_esc": next(x for x in c if x.startswith("Termina de ingresar"))}
    t = pd.DataFrame({k: d[v].map(s).values for k, v in col.items()})
    t["fecha"] = d["Fechas"].ffill().values
    t["pallets"] = d["Pallets"].astype(float).values
    t["planificador"] = d["Planificador?"].astype(str).str.strip().str.lower().eq("si").values
    t["tipo"] = np.where(t.pallets <= 15, "chico", "grande")
    t["descarga_s"] = t.fin_desc - t.ini_desc
    t["escaneo_s"] = t.fin_esc - t.ini_esc
    t["gruero_s"] = np.where(t.planificador, t.descarga_s, t.descarga_s + t.escaneo_s)
    t["planificador_s"] = np.where(t.planificador, t.escaneo_s, 0.0)
    t["en_paralelo"] = t.planificador & (t.ini_esc < t.fin_desc)
    t["espera_camion_s"] = t.estaciona - t.llega
    t["preparacion_s"] = t.ini_desc - t.estaciona
    return t.drop(columns=["llega", "estaciona", "ini_desc", "fin_desc", "ini_esc", "fin_esc"])


def pct_planificador() -> dict:
    """% de camiones (documentos) cuyo escaneo hizo un planificador, por tipo (un documento mixto cuenta como medio)."""
    from modelo import recepcion_roles as rr, wms_septiembre as ws
    plan = rr._planificadores()
    d = ws._agregar_tiempo(ws.leer_export(rr.ARCHIVO))
    d = d[d.tramo1].sort_values("ts").drop_duplicates("lpn_origen")
    rol = pd.Series([rr._rol(u, plan) == rr.CONTROL for u in d.usuario], index=d.index)
    g = pd.DataFrame({"doc": d.ubic_origen, "ctrl": rol}).groupby("doc").agg(pallets=("ctrl", "size"), frac=("ctrl", "mean"))
    g["tipo"] = np.where(g.pallets <= CORTE_CHICO, "chico", "grande")
    g["p"] = np.where(g.frac >= 0.999, 1.0, np.where(g.frac <= 0.001, 0.0, 0.5))
    return {t: {"p": float(g[g.tipo == t].p.mean()), "n_documentos": int((g.tipo == t).sum())} for t in TIPOS}


def calcular() -> dict:
    g = documentos()
    day = por_dia(g)
    v = day[day.facturado > 0]
    out = {"periodo_log": [str(day.fecha.min()), str(day.fecha.max())], "dias_completos": int(len(day)), "dias_con_venta": int(len(v)), "corte_chico_pallets": CORTE_CHICO, "tipos": {}}
    tot = g[~g.fecha.isin(DIAS_CORTADOS)]
    reparto = tot.groupby(["tipo", "turno"]).camiones.sum().unstack(fill_value=0)
    t = tiempos()
    pp = pct_planificador()
    for tipo in TIPOS:
        x = tot[tot.tipo == tipo]
        r = stats.linregress(v.facturado / 1000, v[tipo])
        tt = t[t.tipo == tipo]
        con, sin = tt[tt.planificador], tt[~tt.planificador]
        p = pp[tipo]["p"]
        g_con, g_sin, pl_con = float(con.gruero_s.mean()), float(sin.gruero_s.mean()), float(con.planificador_s.mean())
        out["tipos"][tipo] = {
            "pallets_por_camion": float(x.pallets.sum() / x.camiones.sum()),
            "camiones_dia_media_todos": float(day[tipo].mean()), "camiones_dia_media_con_venta": float(v[tipo].mean()), "desvio_dia": float(day[tipo].std()),
            "por_1000_cajas": float(1000 * v[tipo].sum() / v.facturado.sum()), "recta_intercepto": float(r.intercept), "recta_por_1000": float(r.slope), "r": float(r.rvalue), "p": float(r.pvalue),
            "p_intercepto": _p_intercepto(v.facturado / 1000, v[tipo]),
            "escala_con_venta": bool(r.pvalue < 0.05 and r.slope > 0),
            "reparto_turno": {s: float(reparto.loc[tipo, s] / reparto.loc[tipo].sum()) for s in ("A", "B", "C")},
            "n_camiones_cronometrados": int(len(tt)), "n_con_planificador": int(len(con)), "n_sin_planificador": int(len(sin)),
            "descarga_s_media": float(tt.descarga_s.mean()), "escaneo_s_con_planificador": float(con.escaneo_s.mean()), "escaneo_s_sin_planificador": float(sin.escaneo_s.mean()),
            "gruero_s_con_planificador": g_con, "gruero_s_sin_planificador": g_sin, "planificador_s": pl_con,
            "p_test_gruero_con_vs_sin": float(stats.mannwhitneyu(con.gruero_s, sin.gruero_s).pvalue),
            "pct_escaneo_planificador_log": p, "n_documentos_log_usuarios": pp[tipo]["n_documentos"],
            "gruero_s_por_camion": p * g_con + (1 - p) * g_sin, "planificador_s_por_camion": p * pl_con,
            "pct_escaneo_en_paralelo_con_planificador": float(con.en_paralelo.mean()) if len(con) else None,
            "espera_camion_s_mediana": float(tt.espera_camion_s.median()), "preparacion_s_mediana": float(tt.preparacion_s.median())}
    r = stats.linregress(v.facturado / 1000, v.pallets)
    out["pallets_por_1000_cajas"] = float(1000 * v.pallets.sum() / v.facturado.sum())
    out["pallets_r_venta"] = float(r.rvalue)
    SAL.mkdir(parents=True, exist_ok=True)
    day.to_csv(SAL / "camiones_por_dia.csv", index=False)
    t.to_csv(SAL / "tiempos_por_camion.csv", index=False)
    pd.crosstab(tot.pallets.clip(upper=60), tot.tipo).to_csv(SAL / "pallets_por_documento.csv")
    return out


def volumen(tipo: str, demanda: float, p: dict) -> float:
    """Camiones por día de un tipo: grandes = razón por 1.000 cajas × demanda; chicos = nivel fijo (no escalan con la venta)."""
    x = p["tipos"][tipo]
    return x["por_1000_cajas"] * demanda / 1000 if x["escala_con_venta"] else x["camiones_dia_media_todos"]


def comparar(p: dict, demandas=(15_913, 22_681, 30_000, 33_051)) -> pd.DataFrame:
    """Horas de gruero de recepción por día: modelo actual (pallets × 59,5 s) contra el modelo por camión."""
    from modelo import capacity
    t_actual = capacity.tiempo_estandar("recepcion_directo_stage", "A")[0]
    filas = []
    for D in demandas:
        n = {t: volumen(t, D, p) for t in TIPOS}
        pal = sum(n[t] * p["tipos"][t]["pallets_por_camion"] for t in TIPOS)
        filas.append({"demanda": D, "camiones_chicos": n["chico"], "camiones_grandes": n["grande"], "pallets_por_camion_nuevo": pal,
                      "pallets_modelo_actual": capacity.PALLETS_POR_1000 * D / 1000,
                      "horas_gruero_actual": capacity.PALLETS_POR_1000 * D / 1000 * t_actual / 3600,
                      "horas_gruero_nuevo": sum(n[t] * p["tipos"][t]["gruero_s_por_camion"] for t in TIPOS) / 3600,
                      "horas_planificador_nuevo": sum(n[t] * p["tipos"][t]["planificador_s_por_camion"] for t in TIPOS) / 3600})
    return pd.DataFrame(filas)


def main() -> int:
    p = calcular()
    JSON.write_text(json.dumps(p, indent=1, ensure_ascii=False), encoding="utf-8")
    c = comparar(p)
    c.to_csv(SAL / "comparacion_horas.csv", index=False)
    pd.set_option("display.width", 220)
    print(json.dumps(p, indent=1, ensure_ascii=False))
    print(c.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
