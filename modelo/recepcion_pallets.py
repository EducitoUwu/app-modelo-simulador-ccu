"""RECEPCIÓN Y ALMACENAMIENTO POR PALLET (2026-10-18, decisión del equipo).

Por qué: el modelo medía la recepción en «camiones estimados» (pallets agrupados por huecos de 30 min, porque el log no trae folio de camión). Esa medida NO sube con la
demanda (r = −0,30 contra la venta del día en septiembre) porque los días de mucha recepción los camiones llegan seguidos y se funden en uno. Los PALLETS recibidos sí escalan
(r = 0,66 diario; por semana r = 0,98 y entran 0,99 cajas por cada caja vendida). Por eso:
  - RECEPCIÓN se mide en pallets: volumen = pallets recibidos por cada 1.000 cajas facturadas × demanda; tiempo por pallet = tiempo del camión ÷ sus pallets (planilla de terreno).
  - ALMACENAMIENTO = un pallet almacenado por cada pallet recibido [regla del equipo: no quedan pallets de un día a otro]. El log empareja 0,69 tareas por pallet recibido
    (se muestra como referencia: parte de los pallets va directo a stage o queda sin par en el log).

Salida: `salidas/limpio/recepcion_pallets.json`.  USO: .venv/bin/python -m modelo.recepcion_pallets
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
L = RAIZ / "salidas" / "limpio"
SALIDA = L / "recepcion_pallets.json"
CORTE_S_POR_PALLET = 300.0     # un camión con más de 5 min por pallet trae una espera (p. ej. 3 h estacionado), no trabajo: se excluye del promedio


def tiempos_planilla() -> dict:
    """Tiempo por pallet de cada camión cronometrado (estacionar + descarga + ingreso al WMS) ÷ pallets del camión (columna «Pallets» de la hoja «Recepción_»)."""
    from modelo import data_loader as dl
    wb, _ = dl._abrir(dl.PLANILLA)
    hoja = next((w for w in wb.worksheets if dl._normalizar(w.title).rstrip("_ ") == "recepcion"), None)
    pal = {}
    for i, r in dl._filas(hoja):
        if isinstance(r[0], str):
            break
        if r[2] is not None and isinstance(r[13], (int, float)):
            pal[i] = float(r[13])
    o = pd.read_csv(L / "observaciones.csv")
    o = o[o.proceso == "recepcion_directo_stage"]
    g = o.groupby("fila").agg(t=("duracion_s", "sum"), pasos=("paso", "nunique"), fecha=("fecha", "first")).reset_index()
    g["pallets"] = g.fila.map(pal)
    g = g.dropna(subset=["pallets"])
    g = g[g.pasos == 3]
    g["s_pallet"] = g.t / g.pallets
    ok = g[g.s_pallet <= CORTE_S_POR_PALLET]
    # 2026-10-22: el tiempo de un camión = fijo (estacionar, papeles) + por pallet (descarga e ingreso al WMS). Recta sobre los camiones cronometrados; los pallets por camión
    # de la mezcla salen de TODOS los camiones con pallets anotados (17 de 10, 10 de 26 y 4 de 28 → 17,5).
    from scipy import stats
    sl = stats.linregress(ok.pallets, ok.t)
    ppc = float(sum(pal.values()) / len(pal))
    return {"s_fijo_camion": float(sl.intercept), "s_por_pallet_marginal": float(sl.slope), "r_tiempo_pallets": float(sl.rvalue), "p_tiempo_pallets": float(sl.pvalue),
            "pallets_por_camion_mezcla": ppc, "n_camiones_mezcla": len(pal), "s_camion_mezcla": float(sl.intercept + sl.slope * ppc),
            "s_por_pallet_mezcla": float(sl.intercept / ppc + sl.slope),"s_por_pallet": float(ok.t.sum() / ok.pallets.sum()), "mediana_s_por_pallet": float(ok.s_pallet.median()), "n_camiones": int(len(ok)), "n_dias": int(ok.fecha.nunique()),
            "excluidos": int(len(g) - len(ok)), "pallets_por_camion_media": float(ok.pallets.mean()), "corte_s_por_pallet": CORTE_S_POR_PALLET}


def driver_pallets() -> dict:
    """Pallets recibidos por cada 1.000 cajas facturadas en los días con recepción completa Y ventas, y la razón de almacenamiento del log."""
    r = pd.read_csv(L / "recepcion_dias.csv")
    r = r[r.cobertura == "completa"]
    v = pd.read_csv(L / "demanda_diaria.csv")
    v = v[v.facturado > 0]
    m = r.merge(v[["fecha", "facturado"]], on="fecha")
    a = pd.read_csv(L / "actividad_turnos.csv")
    al = a[a.completa == 1].groupby("fecha").almacenamiento_destino_final.sum().reset_index().merge(r[["fecha", "pallets"]], on="fecha")
    return {"pallets_por_1000_cajas": float(1000 * m.pallets.sum() / m.facturado.sum()), "n_dias": int(len(m)), "r_diario": float(m.pallets.corr(m.facturado)),
            "cajas_recibidas_por_caja_vendida": float(m.cajas.sum() / m.facturado.sum()), "desde": m.fecha.min(), "hasta": m.fecha.max(),
            "almacenamiento_log_por_pallet": float(al.almacenamiento_destino_final.sum() / al.pallets.sum()), "n_dias_almacenamiento": int(len(al))}


def guardar() -> dict:
    d = {"tiempo": tiempos_planilla(), "driver": driver_pallets()}
    SALIDA.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def cargar() -> dict | None:
    return json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else None


if __name__ == "__main__":
    d = guardar()
    t, dr = d["tiempo"], d["driver"]
    print(f"Recepción: {t['s_por_pallet']:.1f} s por pallet (mediana {t['mediana_s_por_pallet']:.1f}; {t['n_camiones']} camiones en {t['n_dias']} días; {t['excluidos']} excluido(s) por espera > {t['corte_s_por_pallet']:.0f} s/pallet); "
          f"{t['pallets_por_camion_media']:.1f} pallets por camión cronometrado.")
    print(f"Driver: {dr['pallets_por_1000_cajas']:.2f} pallets recibidos por 1.000 cajas facturadas ({dr['n_dias']} días, {dr['desde']} a {dr['hasta']}; r diario {dr['r_diario']:.2f}; "
          f"{dr['cajas_recibidas_por_caja_vendida']:.3f} cajas recibidas por caja vendida). Almacenamiento en el log: {dr['almacenamiento_log_por_pallet']:.2f} tareas por pallet recibido ({dr['n_dias_almacenamiento']} días).")
