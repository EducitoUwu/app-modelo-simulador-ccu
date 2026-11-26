"""Calibración del VOLUMEN POR TURNO desde el log del WMS (septiembre + octubre), para `capacity.py` y `config.py`.

Por qué un módulo aparte: el volumen por turno y las cuotas dejaron de ser números escritos a mano. Cada vez que llega un export nuevo se corre
`python -m modelo.wms_octubre` (que arma `salidas/limpio/actividad_turnos.csv`) y estas cifras se recalculan solas; los tests comparan lo que usa el modelo
contra este cálculo.

REGÍMENES (práctica de trabajo del CD, no la demanda)
  vigente  : 2-oct a 10-oct-2026 (la práctica MÁS RECIENTE; 7 días operados completos del 2 al 9-oct, más el turno C del 10-oct).
  anterior : 4 a 21-sep-2026 (incluye la semana peak de Fiestas Patrias).
  intermedio (22-sep a 1-oct): el «vigente» de las versiones anteriores del modelo. Ya no es el vigente: en octubre la práctica siguió cambiando
  (el reabasto de cajas pasó de 56% a 95% en el turno C, la recepción y el almacenamiento se movieron del turno A al B). Se conserva solo para comparar.
  Mezclar el intermedio con octubre promediaría dos prácticas distintas y no describiría a ninguna.

DEFINICIÓN DE CADA VOLUMEN (la misma en septiembre y en octubre)
  reabasto_pallet          depósitos de reabasto de palés (tramo 2)
  reabasto_caja            RETIROS de reabasto de cajas (tramo 1; el cierre casi no se registra, ver CLAUDE.md > trampa de conteo)
  recepcion_directo_stage  camiones ESTIMADOS por huecos de 30 min entre pallets (el log no trae folio de camión)
  almacenamiento           pares retiro -> depósito en su ubicación final, sin filtrar por el rótulo del depósito
  carga_camion             trailers distintos (conteo directo del número de TRL)
Cada promedio por turno se calcula sobre los (fecha, turno) COMPLETOS de la ventana: un turno incompleto no entra, y nunca se divide por días de otro turno.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ACTIVIDAD = RAIZ / "salidas" / "limpio" / "actividad_turnos.csv"
TURNOS = ("A", "B", "C")

# [desde, hasta)
REGIMENES = {
    "vigente": ("2026-10-02", "2026-10-11"),
    "intermedio": ("2026-09-22", "2026-10-02"),
    "anterior": ("2026-09-04", "2026-09-22"),
    # COMBINADO (2026-10-16, pedido de la jefatura): los tres regímenes juntos, 4-sep a 10-oct. Promedia las prácticas en vez de elegir una; los movimientos parciales empiezan el 10-sep.
    "combinado": ("2026-09-04", "2026-10-11"),
}
COLUMNA = {
    "reabasto_pallet": "reabasto_pallet_lineas_tramo_util",
    "reabasto_caja": "reabasto_caja_retiros",
    "recepcion_directo_stage": "recepcion_lineas_tramo_util",          # 2026-10-18: pallets (antes camiones estimados, que no escalan con la venta)
    "almacenamiento": "almacenamiento_destino_final",
    "carga_camion": "carga_trailers",
}


@lru_cache
def _filas() -> tuple:
    if not ACTIVIDAD.exists():
        return ()
    with open(ACTIVIDAD, newline="", encoding="utf-8") as fh:
        return tuple(r for r in csv.DictReader(fh) if r["completa"] == "1")


# Procesos que salen de «movimientos de inventario parcial» (modelo/movimientos_parciales.py): clave en este módulo -> proceso en el CSV de conteos.
PARCIALES = {"reabasto_caja_parcial": "reabasto_cajas_parcial", "retorno_fletero": "retorno_fletero", "merma": "merma"}
MOVIMIENTOS = RAIZ / "salidas" / "limpio" / "movimientos_parciales_turnos.csv"


@lru_cache
def _filas_parciales() -> tuple:
    if not MOVIMIENTOS.exists():
        return ()
    with open(MOVIMIENTOS, newline="", encoding="utf-8") as fh:
        return tuple(csv.DictReader(fh))


def disponible() -> bool:
    return bool(_filas())


def perfil(regimen: str = "vigente") -> dict:
    """{proceso: {'por_turno': {t: media por día}, 'total': suma, 'cuota': {t: fracción}, 'dias': {t: n}}} para la ventana del régimen. {} si falta el archivo."""
    if not disponible():
        return {}
    desde, hasta = REGIMENES[regimen]
    f = [r for r in _filas() if desde <= r["fecha"] < hasta]
    out = {}
    for proc, col in COLUMNA.items():
        suma = {t: 0 for t in TURNOS}
        n = {t: 0 for t in TURNOS}
        for r in f:
            suma[r["turno"]] += int(r[col])
            n[r["turno"]] += 1
        por_turno = {t: (suma[t] / n[t] if n[t] else 0.0) for t in TURNOS}
        total = sum(por_turno.values())
        out[proc] = {"por_turno": por_turno, "total": total, "dias": n, "suma": suma,
                     "cuota": {t: (por_turno[t] / total if total else 0.0) for t in TURNOS}}
    # movimientos de inventario parcial: mismas ventanas, sobre los (fecha, turno) completos de ese archivo (empieza el 10-sep: la ventana «anterior» solo cubre del 10 al 21-sep)
    fp = [r for r in _filas_parciales() if desde <= r["fecha"] < hasta]
    for clave, proc in PARCIALES.items():
        suma = {t: 0 for t in TURNOS}
        n = {t: 0 for t in TURNOS}
        for r in fp:
            if r["proceso"] == proc:
                suma[r["turno"]] += int(r["tareas"])
                n[r["turno"]] += 1
        por_turno = {t: (suma[t] / n[t] if n[t] else 0.0) for t in TURNOS}
        total = sum(por_turno.values())
        out[clave] = {"por_turno": por_turno, "total": total, "dias": n, "suma": suma, "cuota": {t: (por_turno[t] / total if total else 0.0) for t in TURNOS}}
    return out


def dias(regimen: str = "vigente") -> dict:
    """Días operados con los tres turnos completos y días con el turno C completo."""
    desde, hasta = REGIMENES[regimen]
    f = [r for r in _filas() if desde <= r["fecha"] < hasta]
    por_dia: dict[str, set] = {}
    for r in f:
        por_dia.setdefault(r["fecha"], set()).add(r["turno"])
    return {"completos": sorted(d for d, t in por_dia.items() if len(t) == 3), "noches_c": sorted(d for d, t in por_dia.items() if "C" in t)}


def carga_noche_c(desde: str = "2026-09-22", hasta: str = "2026-10-11") -> tuple[float, int]:
    """(trailers por noche, número de noches) del turno C. La carga NOCTURNA no depende del régimen (21 a 25 trailers por noche desde el 22-sep, 17 a 22 antes),
    así que se usa toda la ventana posterior al 22-sep para tener más noches contadas."""
    v = [int(r["carga_trailers"]) for r in _filas() if r["turno"] == "C" and desde <= r["fecha"] < hasta]
    return (sum(v) / len(v), len(v)) if v else (0.0, 0)


NOMBRE = {"reabasto_pallet": "Reabasto de pallets (tareas)", "reabasto_caja": "Reabasto de cajas (retiros)", "recepcion_directo_stage": "Recepción (pallets)",
          "almacenamiento": "Almacenamiento (tareas a ubicación final)", "carga_camion": "Carga de camión (trailers)",
          "reabasto_caja_parcial": "Reabasto de cajas, movimiento parcial (tareas)", "retorno_fletero": "Retorno de fletero a bodega (tareas)", "merma": "Merma (tareas)"}


def tabla_regimenes() -> list[dict]:
    """Una fila por (proceso, turno) con el volumen medio por día en cada régimen y la cuota del vigente. Es lo que alimenta la tabla del dashboard
    «Actividad por turno y régimen» y lo que usa capacity.py. Sin picking: octubre no lo trae."""
    ps = {r: perfil(r) for r in REGIMENES}
    if not all(ps.values()):
        return []
    out = []
    for proc, nombre in NOMBRE.items():
        for t in TURNOS:
            out.append({"proceso": proc, "nombre": nombre, "turno": t,
                        "vigente": ps["vigente"][proc]["por_turno"][t], "intermedio": ps["intermedio"][proc]["por_turno"][t],
                        "anterior": ps["anterior"][proc]["por_turno"][t], "combinado": ps["combinado"][proc]["por_turno"][t], "cuota_combinado": ps["combinado"][proc]["cuota"][t],
                        "cuota_vigente": ps["vigente"][proc]["cuota"][t],
                        "dias_vigente": ps["vigente"][proc]["dias"][t]})
    return out


def resumen_texto() -> str:
    out = []
    for reg in ("vigente", "intermedio", "anterior", "combinado"):
        p = perfil(reg)
        if not p:
            continue
        d = dias(reg)
        out.append(f"{reg} {REGIMENES[reg]}: {len(d['completos'])} días completos, {len(d['noches_c'])} noches C")
        for k, v in p.items():
            out.append(f"  {k:24s} total {v['total']:7.1f}/día  A {v['por_turno']['A']:6.1f}  B {v['por_turno']['B']:6.1f}  C {v['por_turno']['C']:6.1f}  "
                       f"cuotas {v['cuota']['A']:.3f}/{v['cuota']['B']:.3f}/{v['cuota']['C']:.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    print(resumen_texto())
    print("carga C (22-sep en adelante): %.2f trailers por noche, %d noches" % carga_noche_c())
