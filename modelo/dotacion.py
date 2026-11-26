"""Dotación real de personal y equipos: hoja 'Personal (Turnos)' de la planilla de
terreno y docs/Dotacion Sep 2026 x empresa.ods.

CORREGIDO 2026-09-30: este módulo apuntaba a `planilla tiempos2.xlsx` mientras
`data_loader.py` ya leía la planilla del 25-sep. No cambiaba ningún número (la hoja
"Personal (Turnos)" era idéntica salvo el formato 3.0 vs 3), pero una actualización de
dotación en la planilla nueva no habría llegado al modelo y nada lo habría avisado.
Ahora **reutiliza la misma constante que `data_loader`**, así que las dos no pueden
volver a divergir. Hay un test que falla si alguien las separa.

La hoja 'Personal (Turnos)' tiene tres bloques con la misma forma (roles x turno A/B/C):
  planificado          filas 4-12   dotación planificada de referencia
  turnos_reales        filas 22-32  "Turnos reales 24-09 al 28-09": la dotación que
                        REALMENTE estuvo en esas fechas (aclarado por el usuario,
                        2026-09-21: no es un plan, es observación de terreno)
Se usa `turnos_reales` como fuente de verdad. Los bloques "Turno Intermedio" (12-20 y
10-18) son turnos adicionales de refuerzo, no A/B/C puros; se guardan aparte.

El equipo (grúas, transpaletas) es compartido entre TODOS los procesos que lo piden
(recepción, carga, almacenamiento, reabasto de pallet y de caja: ver wms_config.py,
todo reabasto exige GR_AB_ALMA). Por eso su conteo es el techo físico real del sistema,
más estrecho que el límite del WMS (9999, no aplica).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import openpyxl

# La hoja «Personal (Turnos)» y el conteo de equipos NO están en la planilla oficial de tiempos (.ods): siguen saliendo de la del 25-sep (xlsx).
from modelo.data_loader import PLANILLA_PERSONAL as PLANILLA   # única fuente de verdad de qué planilla se lee para dotación
DOTACION_ODS = Path(__file__).resolve().parent.parent / "docs" / "Dotacion Sep 2026 x empresa.ods"

TURNOS = ("A", "B", "C")
HORARIO_TURNO = {"A": (8, 16), "B": (16, 24), "C": (0, 8)}

# Roles de la hoja de terreno -> filas de cada bloque (offset desde la fila del bloque).
ROLES_PRINCIPALES = ("Planificador", "Chequeador", "Gruero", "Operario", "Externos")


@dataclass(frozen=True)
class Dotacion:
    turno: str
    rol: str
    personas: int


@dataclass(frozen=True)
class Equipo:
    tipo: str            # "Grúas" | "Transpaletas"
    cantidad: int
    momento: str          # cuándo se contó (la planilla marca 15:30 junto a estos datos)


def _n(v) -> int:
    return int(v) if isinstance(v, (int, float)) else 0


def cargar_turnos_reales(ruta: Path = PLANILLA) -> list[Dotacion]:
    """Bloque 'Turnos reales 24-09 al 28-09': fuente de verdad de dotación por turno."""
    ws = openpyxl.load_workbook(ruta, data_only=True)["Personal (Turnos)"]
    filas = list(ws.iter_rows(values_only=True))
    out = []
    # fila 25 (índice 24) = Planificador; A en col 1(0-idx), B en col4, C en col7
    for i in range(24, 32):
        r = filas[i]
        rol = str(r[1]).strip() if r[1] else None
        if not rol:
            continue
        out.append(Dotacion("A", rol, _n(r[2])))
        if r[4]:
            out.append(Dotacion("B", str(r[4]).strip(), _n(r[5])))
        if r[7]:
            out.append(Dotacion("C", str(r[7]).strip(), _n(r[8])))
    return out


def cargar_equipos(ruta: Path = PLANILLA) -> list[Equipo]:
    """Filas 43-44: cantidad de transpaletas y grúas contadas a las 15:30 (fila 42)."""
    ws = openpyxl.load_workbook(ruta, data_only=True)["Personal (Turnos)"]
    filas = list(ws.iter_rows(values_only=True))
    momento = filas[41][1]
    momento = momento.strftime("%H:%M") if isinstance(momento, dt.time) else str(momento)
    out = [Equipo("Transpaletas", _n(filas[42][2]), momento)]
    if filas[42][3]:  # segunda cifra en la misma fila, sin etiqueta -> queda como aviso
        out.append(Equipo("Transpaletas (otra cifra sin etiquetar, columna D)", _n(filas[42][3]), momento))
    out.append(Equipo("Grúas", _n(filas[43][2]), momento))
    return out


def cargar_tareas_turno_a(ruta: Path = PLANILLA) -> list[str]:
    """Columna I, filas 35-47: lista de subtareas del turno A (día).

    CORREGIDO 2026-09-22: esta función se llamaba `cargar_tareas_turno_c` y decía
    documentar el turno noche, pero el encabezado real de esa columna (fila 34) es
    "Tareas (08:00-16:00)" — turno A, no C. Las primeras 5 filas del rango además
    calzan una a una con los roles de la dotación real de A (Planificador -> "Se
    encarga del wms y supervision", Chequeador -> "supervisión", Gruero -> "Descarga
    camión", Operario -> "Chequea pallets y los ingresa", Externos -> "Carga
    camiones"), lo que confirma que es la matriz rol->tarea del turno día. El turno
    noche sigue sin esta matriz (ver CLAUDE.md > Brechas de datos)."""
    ws = openpyxl.load_workbook(ruta, data_only=True)["Personal (Turnos)"]
    filas = list(ws.iter_rows(values_only=True))
    return [str(filas[i][8]).strip() for i in range(34, 47) if filas[i][8]]


def _leer_ods_hoja1(ruta: Path) -> list[list[str]]:
    """Parser mínimo de ODS (formato zip+XML) sin depender de odfpy."""
    import zipfile
    import xml.etree.ElementTree as ET

    T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    O = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
    X = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    root = ET.fromstring(zipfile.ZipFile(ruta).read("content.xml"))
    filas = []
    for tabla in root.iter(T + "table"):
        for row in tabla.iter(T + "table-row"):
            rep = int(row.get(T + "number-rows-repeated", "1"))
            if rep > 50:  # filas de relleno vacías repetidas miles de veces
                continue
            celdas = []
            for c in row:
                if c.tag not in (T + "table-cell", T + "covered-table-cell"):
                    continue
                r = min(int(c.get(T + "number-columns-repeated", "1")), 20)
                v = c.get(O + "value") or "".join("".join(p.itertext()) for p in c.iter(X + "p"))
                celdas += [v] * r
            while celdas and celdas[-1] == "":
                celdas.pop()
            if celdas:
                filas.append(celdas)
    return filas


@dataclass(frozen=True)
class DotacionEmpresa:
    empresa: str
    cargo: str
    turno_a: int
    turno_b: int
    turno_c: int


def cargar_dotacion_septiembre(ruta: Path = DOTACION_ODS) -> list[DotacionEmpresa]:
    """docs/Dotacion Sep 2026 x empresa.ods: headcount por empresa y cargo, turnos A/B/C.
    Incluye contratistas (Recifast, Papic) además de CCU. No calza uno a uno con los
    roles de la planilla de terreno: falta la tabla de equivalencia (CLAUDE.md > Brechas)."""
    filas = _leer_ods_hoja1(ruta)
    out = []
    for r in filas:
        if len(r) < 5 or r[0] not in ("CCU", "Recifast", "Papic"):
            continue
        try:
            out.append(DotacionEmpresa(r[0], r[1].strip(), int(float(r[2])), int(float(r[3])), int(float(r[4]))))
        except (ValueError, IndexError):
            continue
    return out


if __name__ == "__main__":
    reales = cargar_turnos_reales()
    print("Dotación real por turno (24 al 28-09-2026), hoja 'Personal (Turnos)':")
    for d in reales:
        print(f"  {d.turno} {d.rol}: {d.personas}")
    print("\nEquipos (planilla):")
    for e in cargar_equipos():
        print(f"  {e.tipo}: {e.cantidad} (contado {e.momento})")
    print("\nTareas de turno A (día; ver docstring de cargar_tareas_turno_a):")
    for t in cargar_tareas_turno_a():
        print(f"  - {t}")
    emp = cargar_dotacion_septiembre()
    print(f"\nDotación Sep-2026 por empresa: {len(emp)} filas, total "
          f"{sum(e.turno_a + e.turno_b + e.turno_c for e in emp)} personas")
