"""Configuración del WMS (docs/ccu wms configuracion.xlsx) que condiciona el modelo.

El WMS no se puede modificar (restricción del proyecto), así que esto entra como
dato fijo. Se leen solo las hojas necesarias, en modo lazy (el libro pesa 6,6 MB
comprimido y algunas hojas superan 20 MB de XML).

Hojas usadas:
  25-Asign Producto-Ubicación  slotting: min/max en cajas por SKU y posición de surtido
  16-Ubicaciones               posiciones físicas, área y capacidad
  35-Métodos de Reabasto       qué equipo exige cada tipo de reabasto
  37-Políticas Reabasto        cuándo se ejecuta cada política (columna "Horario")

Pendiente de confirmar con Felipe Mayo (admin WMS): el significado real del
"Horario ejecución reabastecimientos" (Lun-sáb / 12 am y 6 pm). Los datos de
ventas contradicen leerlo como tope de 2 reposiciones diarias por SKU (ver
tests/test_wms_config.py), así que no se usa como restricción dura.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warnings

import openpyxl

warnings.filterwarnings("ignore", message="Unknown extension", module="openpyxl")

LIBRO = Path(__file__).resolve().parent.parent / "docs" / "ccu wms configuracion.xlsx"


@dataclass(frozen=True)
class Slot:
    sku: str            # clave tal como viene en el WMS (puede llevar ceros a la izquierda)
    ubicacion: str
    minimo: float       # cajas: al bajar de aquí se genera reabasto
    maximo: float       # cajas: capacidad de la posición


@dataclass(frozen=True)
class Ubicacion:
    codigo: str
    area: str           # SURTID, ALMPIC, ALMAC, PATCA, ...
    tipo_capacidad: str  # P = pallet, E = cajas (unidad por confirmar, hoja 16)
    capacidad_max: float | None
    surtido: bool
    reabasto: bool


@dataclass(frozen=True)
class PoliticaReabasto:
    prioridad: float
    nombre: str
    zona_destino: str
    metodo: str          # REAPROPALLET | REAPROCAJA
    horario: str         # texto libre de la planilla
    tipo: str            # por nivel | Top off (Llenado completo)


@dataclass(frozen=True)
class MetodoReabasto:
    id: str
    descripcion: str
    equipo_requerido: str   # p. ej. GR_AB_ALMA = grúa de almacenamiento
    prioridad: float


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _filas(wb, hoja: str, desde: int, ancho: int = 26):
    """Filas rellenadas hasta `ancho` columnas: en modo lazy openpyxl recorta las celdas finales vacías."""
    for r in wb[hoja].iter_rows(min_row=desde, values_only=True):
        yield tuple(r) + (None,) * max(0, ancho - len(r))


def cargar_slotting(wb) -> list[Slot]:
    out = []
    for r in _filas(wb, "25-Asign Producto-Ubicación", 3):
        if r[1] is None or r[4] is None:
            continue
        mn, mx = _num(r[6]), _num(r[7])
        if mn is None or mx is None:
            continue
        out.append(Slot(str(r[1]).strip(), str(r[4]).strip(), mn, mx))
    return out


def cargar_ubicaciones(wb) -> list[Ubicacion]:
    out = []
    for r in _filas(wb, "16-Ubicaciones", 2):
        if not r[2]:
            continue
        out.append(Ubicacion(str(r[2]), str(r[3] or ""), str(r[4] or "").upper(), _num(r[5]),
                             r[22] == 1, r[23] == 1))
    return out


def cargar_politicas_reabasto(wb) -> list[PoliticaReabasto]:
    out = []
    for r in _filas(wb, "37-Políticas Reabasto", 3):
        if _num(r[0]) is None:
            continue
        out.append(PoliticaReabasto(_num(r[0]), str(r[1]), str(r[2]), str(r[7]), str(r[8] or ""), str(r[9] or "")))
    return out


def cargar_metodos_reabasto(wb) -> list[MetodoReabasto]:
    out = []
    for r in _filas(wb, "35-Métodos de Reabasto", 4):
        if not r[0] or _num(r[9]) is None:   # descarta títulos y encabezados de la hoja
            continue
        out.append(MetodoReabasto(str(r[0]), str(r[1]), str(r[8] or ""), _num(r[9])))
    return out


@dataclass
class ConfigWMS:
    slots: list[Slot]
    ubicaciones: list[Ubicacion]
    politicas_reabasto: list[PoliticaReabasto]
    metodos_reabasto: list[MetodoReabasto]

    def slot_por_sku(self) -> dict[str, Slot]:
        """Indexa por SKU con y sin ceros a la izquierda (las ventas usan ID_SKU_INV)."""
        idx: dict[str, Slot] = {}
        for s in self.slots:
            idx[s.sku] = s
            idx.setdefault(s.sku.lstrip("0"), s)
        return idx

    def equipo_para(self, metodo: str) -> str:
        return next(m.equipo_requerido for m in self.metodos_reabasto if m.id == metodo)


def cargar(ruta: Path = LIBRO) -> ConfigWMS:
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    try:
        return ConfigWMS(cargar_slotting(wb), cargar_ubicaciones(wb),
                         cargar_politicas_reabasto(wb), cargar_metodos_reabasto(wb))
    finally:
        wb.close()


if __name__ == "__main__":
    c = cargar()
    print(f"slots {len(c.slots)} | ubicaciones {len(c.ubicaciones)} | "
          f"políticas reabasto {len(c.politicas_reabasto)} | métodos {len(c.metodos_reabasto)}")
    for m in c.metodos_reabasto:
        print(f"  {m.id}: equipo {m.equipo_requerido}, prioridad {m.prioridad}")
