"""Paquete del modelo de capacidad y simulación (versión de la app).

Los archivos crudos de `docs/` (planillas de terreno, Excel de Diana, ODS de dotación, configuración del WMS) NO vienen en esta app.
Los pocos valores que el modelo lee de ellos están congelados en `datos/congelado.json`; aquí se reemplazan las funciones que los leían.
"""
import json
from pathlib import Path
from types import SimpleNamespace

_CONGELADO = Path(__file__).resolve().parent.parent / "datos" / "congelado.json"


def _instalar_congelado() -> None:
    from modelo import dotacion, parametros_diana, wms_config
    d = json.loads(_CONGELADO.read_text(encoding="utf-8"))
    parametros_diana.valores = lambda: dict(d["parametros_diana_valores"])
    parametros_diana.pallets_por_camion_planilla = lambda: d["pallets_por_camion_planilla"]
    dotacion.cargar_dotacion_septiembre = lambda ruta=None: [dotacion.DotacionEmpresa(**x) for x in d["dotacion_septiembre"]]
    dotacion.cargar_equipos = lambda ruta=None: [dotacion.Equipo(**x) for x in d["equipos"]]
    dotacion.cargar_turnos_reales = lambda ruta=None: [dotacion.Dotacion(**x) for x in d["turnos_reales"]]
    wms_config.cargar = lambda ruta=None: SimpleNamespace(metodos_reabasto=[SimpleNamespace(**m) for m in d["metodos_reabasto"]])


if _CONGELADO.exists():
    _instalar_congelado()
