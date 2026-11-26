"""LISTAS DE PICKING: el picking se mide por LISTAS, no por líneas (decisión de la jefatura, 2026-10-16).

Una LISTA es el volante que recibe un operario: un conjunto de líneas de picking (SKU, cantidad, posición) que arma UN pallet, casi siempre en un solo pasillo
(p. ej. 5 líneas: 3 cajas de un SKU, 10 de otro, 2, 1 y 12). En el export de picking del WMS (`Recogida de lista`) cada fila es una línea confirmada y trae el LPN de
DESTINO, es decir, el pallet que se está armando. **Definición operativa [hipótesis, confirmar con Elías/Víctor]: una lista = todas las líneas confirmadas hacia un mismo LPN de destino.**
Evidencia a favor (181.697 líneas, 20-ago a 1-oct): 19.968 LPN de destino distintos; solo 2 tienen más de un operario y 2 más de un día; el 72% está en un solo pasillo; mediana de 7 líneas por lista.

Qué entrega (todo en conteos; el usuario del WMS se usa solo en memoria para ordenar, no se guarda):
  - `listas_picking.csv`        una fila por lista (fecha, turno, líneas, cajas, pasillos, duración y hueco hasta la lista siguiente del mismo operario)
  - `listas_picking_dias.csv`  listas, líneas y cajas por día junto al facturado (para el driver)
  - `listas_picking_resumen.json` líneas por lista, listas por 1.000 cajas facturadas, regresión, tiempo por lista y distribución de líneas por lista (la usa la simulación)

El modelo (capacity.py) convierte el driver y el tiempo de picking de líneas a listas SIN cambiar las horas de trabajo: listas por 1.000 cajas = líneas por 1.000 cajas ÷ líneas por lista, y
tiempo por lista = tiempo por línea × líneas por lista. El tiempo por lista medido por el equipo reemplaza ese valor desde el estimador (columna «Tu tiempo»).

USO:  .venv/bin/python -m modelo.listas_picking
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from modelo import picking_log as pl

RAIZ = Path(__file__).resolve().parent.parent
LIMPIO = RAIZ / "salidas" / "limpio"
SAL_LISTAS = LIMPIO / "listas_picking.csv"
SAL_DIAS = LIMPIO / "listas_picking_dias.csv"
SAL_RESUMEN = LIMPIO / "listas_picking_resumen.json"
CORTE_HUECO_S = 1800.0      # un hueco entre inicios de listas consecutivas mayor a 30 min no es tiempo de trabajo (colación, espera de onda, fin de turno)


def turno_de_hora(h: int) -> str:
    return "C" if h < 8 else ("A" if h < 16 else "B")


def leer_lineas() -> list[tuple]:
    """(ts, usuario, ubicacion, articulo, cantidad, lpn_destino) de todos los exports de picking, sin duplicados ni filas TRF."""
    vistos, out = set(), []
    for ruta in pl.LOGS_COMPLETO:
        if not Path(ruta).exists():
            continue
        with open(ruta, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd)
            for r in rd:
                if len(r) < 10:
                    continue
                f, l, ub = pl._partes(r[0]), pl._partes(r[5]), pl._partes(r[8])
                if not f or not ub or ub[0].startswith("TRF") or len(l) < 2:
                    continue
                try:
                    ts = dt.datetime.strptime(f[0], "%d/%m/%Y %H:%M:%S")
                except ValueError:
                    continue
                k = (ts, f[1] if len(f) > 1 else "", ub[0], r[2].strip(), r[3])
                if k in vistos:
                    continue
                vistos.add(k)
                out.append((ts, k[1], ub[0], k[3], int(r[3] or 0), l[1]))
    out.sort(key=lambda x: x[0])
    return out


def armar_listas(lineas: list[tuple] | None = None) -> list[dict]:
    """Agrupa las líneas por LPN de destino. Devuelve una fila por lista (sin usuario)."""
    lineas = leer_lineas() if lineas is None else lineas
    g: dict[str, list[tuple]] = defaultdict(list)
    for x in lineas:
        g[x[5]].append(x)
    listas = []
    for lpn, xs in g.items():
        xs.sort(key=lambda x: x[0])
        usuario = Counter(x[1] for x in xs).most_common(1)[0][0]
        pas = {x[2].split("-")[0] for x in xs}
        listas.append({"inicio": xs[0][0], "fin": xs[-1][0], "usuario": usuario, "lineas": len(xs), "cajas": sum(x[4] for x in xs), "pasillos": len(pas), "skus": len({x[3] for x in xs})})
    listas.sort(key=lambda d: d["inicio"])
    # hueco entre el inicio de una lista y el de la siguiente del MISMO operario (incluye traslado y armado)
    por_u: dict[str, list[dict]] = defaultdict(list)
    for d in listas:
        por_u[d["usuario"]].append(d)
    for ds in por_u.values():
        for a, b in zip(ds, ds[1:]):
            h = (b["inicio"] - a["inicio"]).total_seconds()
            a["hueco_s"] = h if h <= CORTE_HUECO_S else None
    for d in listas:
        d.setdefault("hueco_s", None)
        d["duracion_s"] = (d["fin"] - d["inicio"]).total_seconds()
        d["fecha"] = d["inicio"].date().isoformat()
        d["turno"] = turno_de_hora(d["inicio"].hour)
    return listas


def dias_completos() -> dict[str, float]:
    """{fecha: facturado} de los 22 días de septiembre con ventas y picking completos (los del backtest de picking)."""
    ruta = LIMPIO / "backtest_picking.csv"
    if not ruta.exists():
        return {}
    with open(ruta, newline="", encoding="utf-8") as fh:
        return {r["fecha"]: float(r["facturado"]) for r in csv.DictReader(fh)}


def resumen(listas: list[dict]) -> dict:
    lin = np.array([d["lineas"] for d in listas])
    cnt = Counter(lin.tolist())
    valores = sorted(cnt)
    por_dia: dict[str, dict] = defaultdict(lambda: {"listas": 0, "lineas": 0, "cajas": 0})
    for d in listas:
        x = por_dia[d["fecha"]]
        x["listas"] += 1
        x["lineas"] += d["lineas"]
        x["cajas"] += d["cajas"]
    fac = dias_completos()
    dias = [(f, por_dia[f], fac[f]) for f in sorted(fac) if f in por_dia]
    out = {"listas": len(listas), "lineas": int(lin.sum()), "lineas_por_lista": float(lin.mean()), "mediana_lineas": float(np.median(lin)), "p90_lineas": float(np.percentile(lin, 90)),
           "pct_un_pasillo": float(np.mean([d["pasillos"] == 1 for d in listas])), "cajas_por_lista": float(np.mean([d["cajas"] for d in listas])),
           "dist_lineas": {"valores": valores, "cuentas": [cnt[v] for v in valores]}}
    if dias:
        x = np.array([d[2] for d in dias])
        y = np.array([d[1]["listas"] for d in dias], float)
        l_ = np.array([d[1]["lineas"] for d in dias], float)
        k = float(y.sum() / x.sum())                         # listas por caja facturada (razón de sumas)
        A = np.vstack([x, np.ones(len(x))]).T
        (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ np.array([m, b])
        r2 = 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out["driver"] = {"n_dias": len(dias), "listas_por_1000_cajas": 1000 * k, "lineas_por_1000_cajas": 1000 * float(l_.sum() / x.sum()), "pendiente_por_1000": 1000 * float(m), "intercepto": float(b),
                         "r2": r2, "sigma_listas": float(np.sqrt((res ** 2).sum() / (len(x) - 2))), "listas_dia_medio": float(y.mean()), "facturado_medio": float(x.mean())}
    huecos = [d["hueco_s"] for d in listas if d["hueco_s"] is not None]
    out["tiempo_lista"] = {"hueco_inicio_a_inicio_media_s": float(np.mean(huecos)), "mediana_s": float(np.median(huecos)), "n": len(huecos), "corte_s": CORTE_HUECO_S}
    por_t = Counter(d["turno"] for d in listas)
    out["por_turno"] = {t: por_t.get(t, 0) / len(listas) for t in ("A", "B", "C")}
    return out


def guardar() -> dict:
    listas = armar_listas()
    LIMPIO.mkdir(parents=True, exist_ok=True)
    with open(SAL_LISTAS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fecha", "turno", "inicio", "lineas", "cajas", "skus", "pasillos", "duracion_s", "hueco_a_siguiente_s"])
        for d in listas:
            w.writerow([d["fecha"], d["turno"], d["inicio"].strftime("%H:%M:%S"), d["lineas"], d["cajas"], d["skus"], d["pasillos"], round(d["duracion_s"]), "" if d["hueco_s"] is None else round(d["hueco_s"])])
    res = resumen(listas)
    fac = dias_completos()
    por_dia: dict[str, dict] = defaultdict(lambda: {"listas": 0, "lineas": 0, "cajas": 0})
    for d in listas:
        x = por_dia[d["fecha"]]
        x["listas"] += 1
        x["lineas"] += d["lineas"]
        x["cajas"] += d["cajas"]
    with open(SAL_DIAS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fecha", "listas", "lineas", "cajas", "facturado", "listas_por_1000_cajas"])
        for f in sorted(por_dia):
            fv = fac.get(f)
            w.writerow([f, por_dia[f]["listas"], por_dia[f]["lineas"], por_dia[f]["cajas"], "" if fv is None else round(fv), "" if not fv else round(1000 * por_dia[f]["listas"] / fv, 2)])
    SAL_RESUMEN.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def cargar_resumen() -> dict | None:
    return json.loads(SAL_RESUMEN.read_text(encoding="utf-8")) if SAL_RESUMEN.exists() else None


if __name__ == "__main__":
    r = guardar()
    print(f"Listas: {r['listas']:,} ({r['lineas']:,} líneas, 20-ago a 1-oct). Líneas por lista: media {r['lineas_por_lista']:.2f}, mediana {r['mediana_lineas']:.0f}, p90 {r['p90_lineas']:.0f}; "
          f"{100 * r['pct_un_pasillo']:.0f}% en un solo pasillo; {r['cajas_por_lista']:.1f} cajas por lista.".replace(",", "."))
    d = r.get("driver")
    if d:
        print(f"Driver ({d['n_dias']} días de septiembre con venta): {d['listas_por_1000_cajas']:.2f} listas por 1.000 cajas facturadas (= {d['lineas_por_1000_cajas']:.1f} líneas por 1.000 ÷ {r['lineas_por_lista']:.2f}); "
              f"regresión {d['pendiente_por_1000']:.2f} por 1.000 {d['intercepto']:+.0f}, R² {d['r2']:.2f}, sigma {d['sigma_listas']:.1f} listas; {d['listas_dia_medio']:.0f} listas por día con {d['facturado_medio']:,.0f} cajas.".replace(",", "."))
    t = r["tiempo_lista"]
    print(f"Tiempo por lista (hueco de inicio a inicio del mismo operario, corte {t['corte_s']:.0f} s, N = {t['n']:,}): media {t['hueco_inicio_a_inicio_media_s']:.0f} s, mediana {t['mediana_s']:.0f} s.".replace(",", "."))
    print("Reparto por turno de inicio:", {k: f"{100 * v:.1f}%" for k, v in r["por_turno"].items()})
