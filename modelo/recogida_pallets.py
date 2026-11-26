"""RECOGIDA DE PALLETS COMPLETOS PARA PEDIDOS (2026-10-22). El gruero saca un pallet completo de almacenamiento (ALMAC o ALMPIC) y lo deja en el andén (PATCA) para
un pedido. No es reabasto (no va a una posición de picking) ni carga (no entra al trailer): es la parte de pallet completo del picking.

Fuente: `docs/recogida de pales picking.csv` (export del WMS, actividad «Recogida de palés», operaciones RECOGIDA ALMACENAMIENTO y RECOGIDA ALMACENAMIENTO PICKING),
15-sep 00:33 a 20-oct 00:46. Cada pallet deja dos registros: ubicación → TRF_OP en pallet (tramo 1) y TRF_OP → andén en cajas (tramo 2). Se cuenta el tramo 1.
  - VOLUMEN: pallets por noche del turno C (el 97% de la actividad); el 20-oct se excluye (el export termina a las 00:46). No escala con la venta (septiembre, noches con venta):
    entra como nivel fijo por noche, como la carga del C.
  - TIEMPO: ciclo = hueco entre dos depósitos consecutivos del mismo gruero en el turno C, ≤ 900 s, SIN otra tarea del mismo usuario en medio (carga, reabasto, recepción o
    almacenamiento del log de septiembre). Incluye volver, retirar y dejar el pallet. Solo las noches de septiembre (las que tienen el log de las otras tareas para descartar).
Sin nombres: los usuarios se usan solo para agrupar en memoria; la salida trae conteos.

Salida: `salidas/limpio/recogida_pallets.json`.  USO: .venv/bin/python -m modelo.recogida_pallets
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
LOG = RAIZ / "docs" / "recogida de pales picking.csv"
L = RAIZ / "salidas" / "limpio"
SALIDA = L / "recogida_pallets.json"
CORTE_S = 900.0
RESPALDO = {"pallets_noche_c": 37.9, "n_noches": 27, "s_ciclo": 234.0, "n_ciclos": 487, "dias_ciclo": 13, "usuarios_ciclo": 8, "estado_tiempo": "validado"}


def _leer() -> pd.DataFrame:
    from modelo import recepcion_log as rl, wms_septiembre as ws
    d = ws.leer_export(LOG)
    d = d[d.operacion.str.startswith("RECOGIDA")].copy()
    d["turno"] = d.ts.dt.hour.map(rl.turno_de_hora)
    d["fecha"] = d.ts.dt.date.astype(str)
    return d


def por_noche(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pallets recogidos (tramo 1) por fecha y turno; sin el último día (parcial)."""
    d = _leer() if d is None else d
    t1 = d[d.um == "PA"]
    ultimo = t1.fecha.max()
    return t1[t1.fecha < ultimo].groupby(["fecha", "turno"]).size().unstack(fill_value=0).reindex(columns=["A", "B", "C"], fill_value=0).reset_index()


def calcular() -> dict:
    from scipy import stats
    d = _leer()
    n = por_noche(d)
    t2 = d[(d.um == "CJ")].sort_values(["usuario", "ts"]).copy()
    t2["prev"] = t2.groupby("usuario").ts.shift()
    t2 = t2[(t2.turno == "C") & t2.prev.notna()]
    t2["gap"] = (t2.ts - t2.prev).dt.total_seconds()
    ev = pd.read_csv(L / "wms_sep_eventos.csv", usecols=["ts", "usuario", "proceso"], low_memory=False)
    ev = ev[ev.proceso.isin(["carga_camion", "reabasto_pallet", "almacenamiento", "recepcion", "reabasto_caja"])]
    ev["ts"] = pd.to_datetime(ev.ts)
    hasta = ev.ts.max()
    idx = {u: np.sort(g.ts.values) for u, g in ev.groupby("usuario")}
    t2 = t2[(t2.ts <= hasta) & (t2.gap > 0) & (t2.gap <= CORTE_S)]
    otros = [int(np.searchsorted(idx[u], np.datetime64(a)) - np.searchsorted(idx[u], np.datetime64(b), side="right")) if u in idx else 0
             for u, a, b in zip(t2.usuario, t2.ts, t2.prev)]
    lim = t2[np.array(otros) == 0]
    v = pd.read_csv(L / "demanda_diaria.csv")[["fecha", "facturado"]]
    m = n.merge(v, on="fecha")
    m = m[m.facturado > 0]
    r, p = stats.pearsonr(m.facturado, m.C) if len(m) > 2 else (float("nan"), float("nan"))
    from modelo import fit_params
    n_c, dias, usu = len(lim), lim.ts.dt.date.nunique(), lim.usuario.nunique()
    estado = "validado" if (n_c >= 30 and dias >= 3 and usu >= 3) else fit_params.estado_por_n(n_c)
    return {"pallets_noche_c": float(n.C.mean()), "n_noches": int(len(n)), "rango_noche_c": [int(n.C.min()), int(n.C.max())],
            "pallets_dia_a": float(n.A.mean()), "pct_turno_c": float(n.C.sum() / n[["A", "B", "C"]].sum().sum()),
            "desde": n.fecha.min(), "hasta": n.fecha.max(), "s_ciclo": float(lim.gap.mean()), "s_ciclo_mediana": float(lim.gap.median()),
            "n_ciclos": int(len(lim)), "dias_ciclo": int(lim.ts.dt.date.nunique()), "usuarios_ciclo": int(lim.usuario.nunique()), "descartados_otra_tarea": int(len(t2) - len(lim)),
            "estado_tiempo": estado, "r_venta": float(r), "p_venta": float(p), "n_noches_venta": int(len(m)),
            "cajas_por_pallet": float(d[d.um == "PA"].cantidad.mean())}


def guardar() -> dict:
    x = calcular()
    SALIDA.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding="utf-8")
    return x


def cargar() -> dict:
    """Valores guardados, o el respaldo (los del 2026-10-22) si no se ha corrido el módulo."""
    return {**RESPALDO, **json.loads(SALIDA.read_text(encoding="utf-8"))} if SALIDA.exists() else dict(RESPALDO)


if __name__ == "__main__":
    x = guardar()
    print(f"Recogida de pallets completos: {x['pallets_noche_c']:.1f} por noche en el turno C ({x['n_noches']} noches, {x['desde']} a {x['hasta']}, rango {x['rango_noche_c'][0]}-{x['rango_noche_c'][1]}; "
          f"{100 * x['pct_turno_c']:.0f}% del total en C). Ciclo {x['s_ciclo']:.0f} s (mediana {x['s_ciclo_mediana']:.0f}; N {x['n_ciclos']}, {x['dias_ciclo']} noches, {x['usuarios_ciclo']} grueros; "
          f"{x['estado_tiempo']}). Contra la venta: r {x['r_venta']:.2f} (p {x['p_venta']:.2f}, {x['n_noches_venta']} noches) → nivel fijo.")
