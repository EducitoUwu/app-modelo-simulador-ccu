"""PARÁMETROS FINALES ACORDADOS (2026-10-21): mezcla de «Carga de trabajo final.xlsx» (Diana) y del modelo, por decisión del usuario.

Cada valor se LEE del Excel de Diana (celda indicada) o de las ventas; si el Excel no está, se usa el respaldo escrito aquí (el mismo valor leído el 2026-10-21).

TIEMPO DISPONIBLE: 8 h − 30 min de colación − 20 min de charla = 7 h 10 min por persona y turno, × utilización: picking 0,644 (la de Diana: promedio por picker de
  `Picking base`!K36:K62, celda `Carga de trabajo - Picking`!B12); el resto 0,7.
POR PROCESO (decisión del usuario):
  - Almacenamiento: volumen = pallets recibidos (1 por pallet); reparto por turno del log (régimen combinado); tiempo de Diana 1,7 min = 102 s por pallet (`Almacenamiento`!G9;
    la celda dice «pallets/minuto» pero la fórmula lo usa como minutos por pallet).
  - Recepción (corregido de nuevo el 2026-10-22): ya NO se usan los camiones de Diana (0,351 por 1.000 cajas = ≈ 8,8 por día, el conteo por huecos de 30 min del log, que funde
    camiones seguidos). La recepción va en PALLETS del log (12,46 por 1.000 cajas, `recepcion_pallets.py`) y los camiones = pallets ÷ 17,5 (mezcla de la planilla). El tiempo
    sale de la planilla: 230 s por camión + 46,3 s por pallet (Diana: 23,5 min por camión, que es el de un camión grande). Ver `capacity.py`, bloque «RECEPCIÓN EN PALLETS».
  - Carga: vueltas de venta (lo que usa Diana, sacado de las ventas) como función de la demanda diaria: regresión de las vueltas por día contra las cajas facturadas en la
    misma ventana de Diana (ene-2025 a sep-2026, días con venta). 22 vueltas fijas en el turno C (`CDC`!B15) y el resto en el A; tiempo 18 min 15 s por vuelta (`CDC`!C10).
  - Reabasto de pallets: ESCALA con la venta como en Diana (Σ pallets repuestos ÷ Σ cajas vendidas, `Pallets por mes` L y C); tiempo con desplazamiento 2 min 9 s + 2,88 min
    (`RDP`!E9 + B14) = 5,03 min; reparto 20 / 2 / 78% (decisión del usuario, igual en todos los niveles: corrige el reparto distinto de alta del Excel).
  - Reabasto de cajas: volumen del modelo (log dirigido + movimientos parciales, que Diana no cuenta). Operario: tiempo del modelo (234,5 s). Gruero: tiempo de Diana 2 min 13 s
    (`RDC`!B14).
  - Carga para el CL (NUEVO): el 60% de los camiones de recepción se cargan para el CL (`extra`!B5 = 0,6) × 24 min por camión (`extra`!B4); reparto como la recepción, SIN turno C
    (de noche llega a lo más un camión, decisión del usuario).
  - Picking (turno C): 6,92 min por lista (`Picking`!B15) con la utilización de Diana 0,644; volumen = listas del modelo (23,0 por 1.000 cajas).
  - Retorno de fletero (gruero): volumen del modelo = todas las vueltas menos la primera (= la carga fuera del turno C); tiempo de Diana 33 s (`extra`!B22, N = 4 en «Hoja2»).
  - Retorno de fletero y merma del operario: del modelo (log de movimientos parciales).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXCEL = RAIZ / "docs" / "Carga de trabajo (2).xlsx"                             # 2026-10-22: Excel actualizado de Diana (antes «Carga de trabajo final.xlsx»)
# DECISIONES FINALES DEL USUARIO (2026-10-22, tras comparar con «Carga de trabajo (2).xlsx» en los tres niveles; docs_modelo/ALINEACION_DIANA_2.md):
#   - Carga para el CL por PALLET: 1/3 de los pallets recibidos en el turno (p. ej. de 300 que llegan, 100 se cargan para el CL), solo A y B, × 62 s por pallet (`extra`!B3).
#   - Recogida de pallets completos: escala con la venta con la razón de Diana (pallets completos por caja vendida, `Pallets por mes` W ÷ C).
#   - Reabasto de cajas: el nivel del log (≈ 87,3 tareas por día) corresponde a la DEMANDA NORMAL de Diana en cajas vendidas y escala proporcional a la venta, igual que sus tareas.
#   - Reabasto de pallets: 2 min 9 s SIN desplazamiento (`RDP`!E9; antes se sumaban 2,88 min).
#   - Utilización de los grueros: 0,70 (hasta el 2026-11-24; ver UTILIZACION_RESTO).
FRACCION_CARGA_CL = 1 / 3

HORAS_NOMINALES = 8.0
COLACION_MIN = 30.0
CHARLA_MIN = 20.0
HORAS_EFECTIVAS = HORAS_NOMINALES - (COLACION_MIN + CHARLA_MIN) / 60.0       # 7 h 10 min
# Utilización de todo salvo picking: SUPUESTO, nunca medido. 0,70 hasta el 2026-11-24; desde entonces 0,80 en los tres turnos (decisión del usuario):
# observación de terreno (los grueros siempre están ocupados) y un día de ≈28.000 cajas que se sacó con 4/3/3 grueros y 1 chequeador; con 0,70 la simulación
# da 0% de días viables para esa dotación, con 0,80 el turno A cumple (98%) y el C 50%, con 0,85-0,90 cumple todo. Sensibilidad 0,70 / 0,80 / 0,85 en salidas/polivalencia/.
UTILIZACION_RESTO = 0.80
REPARTO_REABASTO_PALLET = {"A": 0.20, "B": 0.02, "C": 0.78}
# PICKING (corregido el 2026-10-21 tras validar la simulación contra las 22 noches de septiembre; `modelo/validacion_simulacion.py`):
#   - tiempo por lista y utilización = promedio PONDERADO por listas de la hoja «Picking base» (Σ minutos productivos ÷ Σ listas; Σ productivo ÷ Σ presencia). El 6,92 / 0,644 de la
#     hoja «Carga de trabajo - Picking» es un promedio simple de 27 pickers y da 10,73 min de presencia por lista contra 10,14 ponderado.
#   - la utilización se mide sobre la presencia del log (primera a última confirmación del picker), que YA incluye la colación: para picking no se descuenta la colación otra vez, y
#     la charla previa es la observada (primer pick de la noche entre 00:06 y 00:18 en las 22 noches).
CHARLA_PICKING_MIN = 10.0
VUELTAS_TURNO_C = 22.0
VENTANA_VUELTAS = ("2025-01-01", "2026-10-01")                                  # la de Diana (ene-2025 a sep-2026)

_RESPALDO = {"min_lista_picking_pond": 6.58, "util_picking_pond": 0.649, "n_pickers_picking": 27, "camiones_cl_por_caja": 0.3509 / 1000, "pallets_por_camion_planilla": 17.48, "min_lista_picking": 6.916265083543232, "util_picking": 0.6444057964598717, "pallets_cl_por_caja": 12.328 / 1000, "pallets_por_camion": 1 / 0.02847, "min_recepcion_camion": 23.5,
             "min_almacenamiento_pallet": 1.7, "rdp_por_caja": 3.249 / 1000, "seg_rdp": 2 * 60 + 9, "seg_carga_cl_pallet": 62.0, "recogida_por_caja": 1.3726 / 1000, "demanda_normal_vendida": 22431.1, "seg_rdc_gruero": 133.0,
             "seg_carga_vuelta": 18 * 60 + 15, "frac_carga_cl": 0.6, "min_carga_cl": 24.0, "seg_retorno": 33.0}


def _horas(v) -> float:
    if hasattr(v, "hour"):
        return v.hour + v.minute / 60 + v.second / 3600 + getattr(v, "microsecond", 0) / 3.6e9
    return float(v) * 24


@lru_cache(maxsize=1)
def valores() -> dict:
    """Valores leídos del Excel de Diana (o el respaldo) con su origen."""
    if not EXCEL.exists():
        return {**_RESPALDO, "fuente": "respaldo (no se encontró el Excel de Diana)"}
    import openpyxl
    wv = openpyxl.load_workbook(EXCEL, data_only=True)
    pm = wv["Pallets por mes"]
    filas = [r for r in pm.iter_rows(min_row=2, max_row=22, values_only=True) if r[0] and str(r[0]) != "2026-08"]   # sin agosto-2026 incompleto, como Diana
    # 2026-10-22: las columnas se buscan por su ENCABEZADO (Diana borra e inserta columnas entre versiones)
    enc = [str(c.value or "").strip().lower() for c in pm[1]]

    def col(*nombres):
        for n in nombres:
            if n.lower() in enc:
                return enc.index(n.lower())
        raise KeyError(f"«Pallets por mes» sin la columna {nombres[0]!r}")
    C_CAJ, C_LLEGAN, C_PAL, C_CAM = col("cajas vendidas"), col("cajas que llegan de CL"), col("pallets que llegan de CL"), col("Vueltas por camion mes (reabastecimiento)")
    C_RDP, C_COMPL = col("Pallet repuestos en el mes", "Promedio de pallet repuestos en el mes"), col("pallets completos por mes")
    cajas = sum(r[C_CAJ] for r in filas)
    rdp = wv["Carga de trabajo- RDP"]
    desplaz = rdp["B14"].value if isinstance(rdp["B14"].value, (int, float)) else 2.88     # Diana lo quitó en el Excel del 22-oct
    out = {
        "util_picking": float(wv["Carga de trabajo - Picking"]["B12"].value),
        "min_lista_picking": float(wv["Carga de trabajo - Picking"]["B15"].value),       # 6,92 min por lista (promedio simple; referencia)
        **_picking_ponderado(wv["Picking base"]),
        "camiones_cl_por_caja": sum(r[C_CAM] for r in filas) / cajas,                      # camiones de reabastecimiento del CL por caja vendida (columna I)
        "pallets_por_camion_planilla": pallets_por_camion_planilla(),
        "pallets_cl_por_caja": sum(r[C_PAL] for r in filas) / cajas,
        "pallets_por_camion": sum(r[C_PAL] for r in filas) / sum(r[C_CAM] for r in filas),
        "min_recepcion_camion": _horas(wv["Carga de trabajo-Recepcion"]["H10"].value) * 60,
        "min_almacenamiento_pallet": float(wv["Carga de trabajo-Almacenamiento"]["G9"].value),
        "rdp_por_caja": sum(r[C_RDP] for r in filas) / cajas,
        "seg_rdp": _horas(wv["Carga de trabajo- RDP"]["E9"].value) * 3600,                       # 2 min 9 s, sin desplazamiento (decisión 2026-10-22)
        "seg_rdp_con_desplazamiento": (_horas(rdp["E9"].value) * 60 + desplaz) * 60,
        "seg_carga_cl_pallet": _horas(wv["Carga de trabajo extra"]["B3"].value) * 3600,          # 62 s por pallet
        "recogida_por_caja": sum(r[C_COMPL] for r in filas) / cajas,                               # pallets completos por caja vendida (W ÷ C)
        # demanda normal en cajas VENDIDAS: la columna H de Diana son cajas recibidas (E ÷ B); se pasa a vendidas con Σ C ÷ Σ E
        "demanda_normal_vendida": float(pm["C34"].value) * cajas / sum(r[C_LLEGAN] for r in filas),
        "seg_rdc_gruero": _horas(wv["Carga de trabajo- RDC"]["B14"].value) * 3600,
        "seg_carga_vuelta": _horas((wv["Carga de trabajo- CDC"]["B13"].value or wv["Carga de trabajo- CDC"]["C10"].value)) * 3600,   # el Excel (2) lo movió de C10 a B13
        "frac_carga_cl": float(wv["Carga de trabajo extra"]["B5"].value),
        "min_carga_cl": _horas(wv["Carga de trabajo extra"]["B4"].value) * 60,
        "seg_retorno": _horas(wv["Carga de trabajo extra"]["B22"].value) * 3600,
        "n_meses": len(filas), "fuente": f"{EXCEL.name} (leído celda por celda)",
    }
    return out


def _picking_ponderado(ws) -> dict:
    """Promedios ponderados por listas de «Picking base» filas 36-62: E listas, F horas-persona (presencia), I minutos productivos por lista."""
    filas = [(ws[f"E{r}"].value, ws[f"F{r}"].value, ws[f"I{r}"].value) for r in range(36, 63)]
    filas = [(float(e), float(f), float(i)) for e, f, i in filas if e and f and i is not None]
    prod = sum(e * i for e, _, i in filas)
    return {"min_lista_picking_pond": prod / sum(e for e, _, _ in filas), "util_picking_pond": prod / (60 * sum(f for _, f, _ in filas)), "n_pickers_picking": len(filas)}


@lru_cache(maxsize=1)
def pallets_por_camion_planilla() -> float:
    """Pallets por camión de recepción = promedio de los camiones cronometrados en la planilla de terreno (hoja «Recepción_», columna «Pallets»): 31 camiones,
    17 de 10 pallets, 10 de 26 y 4 de 28 → 17,5 [decisión del usuario, 2026-10-21: reemplaza los 35,1 implícitos del Excel]."""
    try:
        from modelo import data_loader as dl
        wb, _ = dl._abrir(dl.PLANILLA)
        ws = next(w for w in wb.worksheets if dl._normalizar(w.title).rstrip("_ ") == "recepcion")
        p = [float(r[13]) for _, r in dl._filas(ws) if not isinstance(r[0], str) and r[2] is not None and isinstance(r[13], (int, float))]
        return sum(p) / len(p) if p else _RESPALDO["pallets_por_camion_planilla"]
    except Exception:
        return _RESPALDO["pallets_por_camion_planilla"]


@lru_cache(maxsize=1)
def regresion_vueltas() -> dict:
    """Vueltas de venta por día = a + b × cajas facturadas (días con venta de la ventana de Diana). Respaldo: 10,81 + 1,4987 por 1.000 cajas."""
    import csv
    ruta = RAIZ / "salidas" / "limpio" / "demanda_diaria.csv"
    if not ruta.exists():
        return {"a": 10.81, "b": 1.4987 / 1000, "sigma": 4.45, "n": 518, "r": 0.916}
    with open(ruta, newline="", encoding="utf-8") as fh:
        xs, ys = [], []
        for r in csv.DictReader(fh):
            if VENTANA_VUELTAS[0] <= r["fecha"] < VENTANA_VUELTAS[1] and float(r["facturado"]) > 0 and r.get("vueltas"):
                xs.append(float(r["facturado"]))
                ys.append(float(r["vueltas"]))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sigma = (sum(e * e for e in res) / (n - 2)) ** 0.5
    syy = sum((y - my) ** 2 for y in ys)
    return {"a": a, "b": b, "sigma": sigma, "n": n, "r": (b * sxx ** 0.5) / syy ** 0.5}


def resumen() -> str:
    v, rv = valores(), regresion_vueltas()
    return "\n".join([
        f"Tiempo disponible: {HORAS_EFECTIVAS:.3f} h (8 h − {COLACION_MIN:g} min colación − {CHARLA_MIN:g} min charla) × utilización picking {v['util_picking']:.3f} / resto {UTILIZACION_RESTO}",
        f"Recepción: {1000 * v['camiones_cl_por_caja']:.3f} camiones por 1.000 cajas × {v['min_recepcion_camion']:.2f} min; {v['pallets_por_camion_planilla']:.1f} pallets por camión (planilla) "
        f"→ {1000 * v['camiones_cl_por_caja'] * v['pallets_por_camion_planilla']:.2f} pallets recibidos y almacenados por 1.000 cajas (el Excel daba {1000 * v['pallets_cl_por_caja']:.2f})",
        f"Picking: {v['min_lista_picking_pond']:.2f} min por lista y utilización {v['util_picking_pond']:.3f} (ponderados por listas, {v['n_pickers_picking']} pickers; el promedio simple de la hoja daba "
        f"{v['min_lista_picking']:.2f} y {v['util_picking']:.3f}); presencia = 8 h − {CHARLA_PICKING_MIN:g} min de charla (la colación ya está dentro de la utilización)",
        f"Almacenamiento: 1 por pallet recibido; {v['min_almacenamiento_pallet']:.2f} min = {60 * v['min_almacenamiento_pallet']:.0f} s por pallet",
        f"Carga: vueltas = {rv['a']:.2f} + {1000 * rv['b']:.4f} por 1.000 cajas (n {rv['n']}, r {rv['r']:.3f}, sigma {rv['sigma']:.2f}); {VUELTAS_TURNO_C:g} en C; {v['seg_carga_vuelta']:.0f} s por vuelta",
        f"Reabasto de pallets: {1000 * v['rdp_por_caja']:.3f} por 1.000 cajas; {v['seg_rdp']:.1f} s; reparto {REPARTO_REABASTO_PALLET}",
        f"Reabasto de cajas gruero: {v['seg_rdc_gruero']:.0f} s · Carga para el CL: {100 * v['frac_carga_cl']:.0f}% de los camiones × {v['min_carga_cl']:.0f} min · Retorno gruero: {v['seg_retorno']:.0f} s",
        f"Fuente: {v['fuente']}"])


if __name__ == "__main__":
    print(resumen())
