"""Carga y limpieza de la planilla de tiempos de terreno.

Produce una tabla larga (una fila por paso medido) y un reporte de calidad con
cada fila descartada o corregida. No ajusta distribuciones (eso es fit_params.py).

Fuente: `docs/planilla tiempos actualizada 25 septiembre 9:27.xlsx`.

CAMBIO DE PLANILLA (2026-09-25). La versión anterior (`planilla tiempos2.xlsx`) se
mantiene en docs/ solo como referencia histórica; este cargador ya NO la lee, porque
la nueva reestructuró hojas y columnas:
  - "Carga de camion" -> "Carga de camión" (con tilde) y el encabezado bajó de la
    fila 3 a la fila 2: los datos empiezan en la 3, no en la 4.
  - "Recepción (excepcion)" desapareció; su contenido (Directo a Stage) es ahora la
    hoja "Recepción " (ojo: con espacio final), con dos columnas de duración nuevas
    intercaladas que corren todos los índices.
  - **Desapareció la hoja de Recepción dirigida.** Sus 7 observaciones ya no existen
    en la fuente; el proceso `recepcion_dirigida` deja de aparecer en la salida.
  - "Reabastecimiento de pallets" intercaló una columna de duración entre cada par de
    horas: los timestamps ya no son columnas consecutivas.
  - **Hojas nuevas**: "Reabastecimiento cajas" (con datos: es el proceso del problema
    central, medido por primera vez) y "Retorno fletero" (plantilla vacía).

Sigue siendo casi todo turno A (día). Todo lo que salga de aquí hereda esa limitación.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import openpyxl

_DOCS = Path(__file__).resolve().parent.parent / "docs"
# PLANILLA OFICIAL (2026-10-13): `planilla tiempos oficial.ods` reemplaza a la del 25-sep. Es la fuente de los TIEMPOS de terreno.
# La del 25-sep (xlsx) sigue en docs/ y solo se usa para lo que la oficial NO trae: la hoja «Personal (Turnos)» y el conteo de equipos
# (`dotacion.py` → PLANILLA_PERSONAL). La versión de tiempos2.xlsx es histórica.
PLANILLA = _DOCS / "planilla tiempos oficial.ods"
if not PLANILLA.exists() and (_DOCS / "planilla tiempos oficial.xlsx").exists():      # 2026-10-18: llegó con extensión .xlsx pero es un ODS por dentro
    PLANILLA = _DOCS / "planilla tiempos oficial.xlsx"
PLANILLA_PERSONAL = _DOCS / "planilla tiempos actualizada 25 septiembre 9:27.xlsx"
PLANILLA_25SEP = PLANILLA_PERSONAL
PLANILLA_ANTERIOR = _DOCS / "planilla tiempos2.xlsx"   # solo referencia histórica

# Turnos según hoja "Personal (Turnos)": A 08-16, B 16-24, C 00-08.
def turno_de_hora(h: int) -> str:
    if 8 <= h < 16:
        return "A"
    if 16 <= h < 24:
        return "B"
    return "C"


@dataclass
class Obs:
    """Una duración medida de un paso de un proceso."""
    proceso: str
    paso: str
    muestra: int
    fecha: str          # ISO, "" si desconocida
    turno: str          # A/B/C, "sin_hora" si la fila no trae hora del día
    operario: str
    duracion_s: float | None
    origen: str         # "timestamps" | "duracion_planilla"
    hoja: str
    fila: int


@dataclass
class Aviso:
    hoja: str
    fila: int
    muestra: str
    tipo: str           # descartada | corregida | dudosa | sin_medir
    detalle: str


@dataclass
class Espera:
    proceso: str
    muestra: int
    fecha: str
    donde: str
    duracion_s: float | None
    texto_duracion: str
    comentario: str
    tipo: str           # falla_sistemica | sin_clasificar (revisión manual)
    hoja: str
    fila: int


@dataclass
class Resultado:
    obs: list[Obs] = field(default_factory=list)
    avisos: list[Aviso] = field(default_factory=list)
    esperas: list[Espera] = field(default_factory=list)
    _agregados: dict = field(default_factory=dict)

    def aviso_hoja(self, hoja: str, tipo: str, detalle: str, fila: int):
        """Aviso que aplica a toda una columna: se emite una vez con el conteo."""
        k = (hoja, tipo, detalle)
        self._agregados.setdefault(k, []).append(fila)

    def cerrar(self):
        for (hoja, tipo, detalle), filas in self._agregados.items():
            self.avisos.append(Aviso(hoja, filas[0], f"{len(filas)} filas", tipo, detalle))
        self._agregados.clear()


# ---------------------------------------------------------------- utilidades
def _s(v) -> float | None:
    """datetime.time o timedelta -> segundos desde 00:00. Otros tipos -> None."""
    if isinstance(v, dt.time):
        return v.hour * 3600 + v.minute * 60 + v.second
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    return None


def _diff(a, b) -> float | None:
    """b - a en segundos entre dos horas del día; None si falta alguna."""
    sa, sb = _s(a), _s(b)
    if sa is None or sb is None:
        return None
    return sb - sa


def _txt_duracion(txt) -> float | None:
    """'00:01:12 ; 00:56:42' -> suma; '3 minutos aprox' -> 180; 'hh:mm:ss' -> s."""
    if txt is None:
        return None
    if isinstance(txt, dt.time):
        return _s(txt)
    t = str(txt)
    hms = re.findall(r"(\d+):(\d{2}):(\d{2})", t)
    if hms:
        return float(sum(int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in hms))
    m = re.search(r"(\d+)\s*min", t)
    return float(int(m.group(1)) * 60) if m else None


def _es_falla_sistemica(comentario: str) -> bool:
    c = comentario.lower()
    return "pallets en esa ubicaci" in c or "no encontraba la ubicaci" in c


def _col_responsable(ws, respaldo: int) -> int:
    """Columna del operario, buscada POR ENCABEZADO («Responsables», «Resonsable»…) en las primeras filas: la planilla oficial (.ods) movió esa
    columna respecto de la del 25-sep. Si no se encuentra, el índice de respaldo (el de la planilla anterior)."""
    for r in ws.iter_rows(min_row=1, max_row=4, values_only=True):
        for k, c in enumerate(r):
            if isinstance(c, str) and c.strip().lower().startswith(("responsable", "resonsable")):
                return k
    return respaldo


def _filas(ws, desde: int = 4):  # fila 3 es el encabezado de columnas
    for i, r in enumerate(ws.iter_rows(min_row=desde, values_only=True), start=desde):
        yield i, r


class _Fecha:
    """La fecha solo aparece en la primera fila de cada día: arrastrarla."""
    def __init__(self):
        self.actual = ""

    def de(self, v) -> str:
        if isinstance(v, dt.datetime) and v.year > 1900:
            self.actual = v.date().isoformat()
        return self.actual


def _n(v) -> int | None:
    return int(v) if isinstance(v, (int, float)) and v == v else None


def _turno(*horas) -> str:
    for h in horas:
        if isinstance(h, dt.time):
            return turno_de_hora(h.hour)
    return "sin_hora"


# ---------------------------------------------------------------- por hoja
def _cargar_carga(ws, res: Resultado):
    """Carga de camión (planilla 2026-09-25). Encabezado en la fila 2, datos desde la 3.
    0 N · 1 fecha · 2 llegada · 3 estacionar · 4/5 buscar pedidos (+6 duración) ·
    7/8 escaneo (+9 duración) · 10/11 carga (+12 duración) · 22 responsable · 23 pallets."""
    col_op = _col_responsable(ws, 22)
    fecha = _Fecha()
    for i, r in _filas(ws, 3):
        if isinstance(r[0], str):          # encabezado de la tabla de esperas
            break
        n = _n(r[0])
        f = fecha.de(r[1])
        if n is None or r[2] is None:   # fila de plantilla sin medir
            continue
        op = str(r[col_op] or "")
        tur = _turno(r[2])
        A, B, C, F, H, I, L, M = (r[2], r[3], r[4], r[5], r[7], r[8], r[10], r[11])

        def add(paso, dur, origen="timestamps"):
            res.obs.append(Obs("carga_camion", paso, n, f, tur, op, dur, origen, ws.title, i))

        # Estacionar: llegada -> estacionado
        d = _diff(A, B)
        if d is not None and d >= 0:
            add("estacionar", d)
        elif d is not None:
            res.avisos.append(Aviso(ws.title, i, str(n), "descartada",
                                    f"estacionar: estacionado ({B}) antes de llegada ({A}); hora mal anotada"))
        # Buscar pedidos / achicar
        d = _diff(C, F)
        if d is not None and d >= 0:
            add("buscar_pedidos", d)
        else:
            dp = _s(r[6])
            if dp is not None:
                add("buscar_pedidos", dp, "duracion_planilla")
                res.avisos.append(Aviso(ws.title, i, str(n), "corregida",
                                        "buscar_pedidos: timestamps inválidos, se usa la duración de la planilla"))
        # Escanear
        d = _diff(H, I)
        if d is not None and d >= 0:
            add("escanear", d)
        elif _s(r[9]) is not None:
            add("escanear", _s(r[9]), "duracion_planilla")
        # Cargar
        d = _diff(L, M)
        if d is not None and d > 0:
            add("cargar", d)
        else:
            dp = _s(r[12])
            if dp:
                add("cargar", dp, "duracion_planilla")
                res.avisos.append(Aviso(ws.title, i, str(n), "corregida",
                                        "cargar: inicio/fin en 00:00:00, se usa la duración de la planilla"))
        # Pallets: aceptar solo numérico
        if not isinstance(r[23], (int, float)):
            res.avisos.append(Aviso(ws.title, i, str(n), "dudosa",
                                    f"pallets no numérico ({r[23]!r}); no se usa como volumen"))

    # Tabla de esperas (debajo)
    _leer_esperas(ws, "carga_camion", res, col_donde=3, col_tiempo=4, col_com=None, col_desc=3)


def _leer_esperas(ws, proceso, res: Resultado, col_donde, col_tiempo, col_com, col_desc):
    """Tabla «Observaciones» (¿Hubo espera? / Donde / Tiempo / Comentarios) debajo de las mediciones. Las columnas se buscan POR ENCABEZADO: la planilla
    oficial (.ods) quitó «Donde» en Almacenamiento y la tabla de Carga, así que los índices fijos de la planilla del 25-sep ya no sirven.
    Los argumentos col_* son solo el respaldo si el encabezado no se reconoce."""
    fecha = _Fecha()
    en_tabla = False
    for i, r in _filas(ws):
        if isinstance(r[0], str) and "Muestras" in r[0] and isinstance(r[2], str) and "espera" in r[2].lower():
            en_tabla = True
            enc = [str(c or "").strip().lower() for c in r]
            col_donde = next((k for k, c in enumerate(enc) if c.startswith("donde")), None)
            col_tiempo = next((k for k, c in enumerate(enc) if c.startswith("tiempo")), col_tiempo)
            col_com = next((k for k, c in enumerate(enc) if c.startswith("comentario")), None)
            continue
        if not en_tabla:
            continue
        n = _n(r[0])
        if n is None or str(r[2] or "").strip().lower() != "si":
            continue
        f = fecha.de(r[1]) if isinstance(r[1], dt.datetime) else ""
        donde = str(r[col_donde] or "") if col_donde is not None else ""
        com = str(r[col_com] or "") if col_com is not None else donde
        tipo = "falla_sistemica" if _es_falla_sistemica(donde + " " + com) else "sin_clasificar"
        txt = r[col_tiempo] if col_tiempo is not None else None
        res.esperas.append(Espera(proceso, n, f, donde, _txt_duracion(txt),
                                  str(txt or ""), com, tipo, ws.title, i))


def _cargar_recepcion_directo_stage(ws, res: Resultado):
    """Hoja "Recepción " (Directo a Stage). Índices 2026-09-25:
    2 llega · 3 estaciona · 4/5 descarga (+6 duración) · 7/8 ingreso WMS (+9 duración) ·
    10 término · 13 pallets · 14 grúas · 15 personas. Ya NO trae las columnas de
    guardado que tenía la hoja "Recepción (excepcion)" de la planilla anterior."""
    fecha = _Fecha()
    for i, r in _filas(ws):
        if isinstance(r[0], str):
            break
        n = _n(r[0])
        f = fecha.de(r[1])
        if n is None or r[2] is None:
            continue
        tur = _turno(r[2])
        tramos = [("estacionar", r[2], r[3], None),
                  ("descarga", r[4], r[5], r[6]),
                  ("ingreso_wms", r[7], r[8], r[9])]
        for nom, a, b, dur_planilla in tramos:
            d = _diff(a, b)
            if d is None or d < 0:
                if d is not None and d < 0:
                    res.avisos.append(Aviso(ws.title, i, str(n), "descartada",
                                            f"{nom}: fin antes que inicio"))
                d2 = _s(dur_planilla)
                if d2 is not None:
                    res.obs.append(Obs("recepcion_directo_stage", nom, n, f, tur, "", d2,
                                       "duracion_planilla", ws.title, i))
                continue
            res.obs.append(Obs("recepcion_directo_stage", nom, n, f, tur, "", d,
                               "timestamps", ws.title, i))
        d = _diff(r[2], r[7])
        if d is not None and d < 0:
            res.avisos.append(Aviso(ws.title, i, str(n), "dudosa",
                                    "empieza a ingresar pallets antes de llegar el camión; "
                                    "revisar horas de ese camión"))
    # tabla «Observaciones» de la hoja (¿Hubo espera? / Donde / Tiempo / Comentarios), que la planilla oficial trae también aquí
    _leer_esperas(ws, "recepcion_directo_stage", res, col_donde=3, col_tiempo=4, col_com=5, col_desc=3)


def _cargar_almacenamiento(ws, res: Resultado):
    """Almacenamiento (putaway). C pinchar LPN (hora), D/E/F duraciones, G término (hora)."""
    col_op = _col_responsable(ws, 12)
    fecha = _Fecha()
    for i, r in _filas(ws):
        if isinstance(r[0], str):
            break
        n = _n(r[0])
        f = fecha.de(r[1])
        if n is None or r[2] is None:
            continue
        op = str(r[col_op] or "")
        tur = _turno(r[2])
        for nom, v in (("recoger_de_stage", r[3]), ("ir_y_dejar_pallet", r[4])):
            d = _s(v)
            if d is not None:
                res.obs.append(Obs("almacenamiento", nom, n, f, tur, op, d, "duracion_planilla", ws.title, i))
        qr = _s(r[5])
        if qr == 1:
            res.aviso_hoja(ws.title, "sin_medir",
                           "escanear_qr = 00:00:01 (valor por defecto, no medido); no se usa", i)
        elif qr is not None:
            res.obs.append(Obs("almacenamiento", "escanear_qr", n, f, tur, op, qr, "duracion_planilla", ws.title, i))
        total = _diff(r[2], r[6])
        tareas = sum(x for x in (_s(r[3]), _s(r[4]), qr if qr and qr != 1 else 0) if x)
        if total is not None and total >= 0:
            res.obs.append(Obs("almacenamiento", "ciclo_total", n, f, tur, op, total, "timestamps", ws.title, i))
            if tareas and total + 1 < tareas:
                res.avisos.append(Aviso(ws.title, i, str(n), "dudosa",
                                        f"ciclo total ({total:.0f}s) menor que suma de pasos ({tareas:.0f}s)"))
    _leer_esperas(ws, "almacenamiento", res, col_donde=3, col_tiempo=4, col_com=5, col_desc=3)


def _cargar_reabasto_pallets(ws, res: Resultado):
    """Reabastecimiento de pallets dirigido. En la planilla 2026-09-25 los timestamps
    ya NO son columnas consecutivas: entre cada par hay una columna de duración.
    Horas en 2,3,5,7,9,11; duraciones en 4,6,8,10,12; 15 ¿escaneó antes?; 19 responsable."""
    col_op = _col_responsable(ws, 19)
    fecha = _Fecha()
    nombres = ["ir_a_origen", "retirar_pallet", "escanear_lpn",
               "ir_a_destino_y_posicionar", "escanear_destino"]
    COL_TS = [2, 3, 5, 7, 9, 11]
    COL_DUR = [4, 6, 8, 10, 12]
    for i, r in _filas(ws):
        if isinstance(r[0], str):
            break
        n = _n(r[0])
        f = fecha.de(r[1])
        if n is None or r[2] is None:
            continue
        op = str(r[col_op] or "")
        tur = _turno(r[2])
        ts = [r[k] for k in COL_TS]
        for k, nom in enumerate(nombres):
            d = _diff(ts[k], ts[k + 1])
            if d is None or d < 0:
                if d is not None and d < 0:
                    res.avisos.append(Aviso(ws.title, i, str(n), "descartada",
                                            f"{nom}: eventos fuera de orden ({ts[k]} -> {ts[k+1]})"))
                d2 = _s(r[COL_DUR[k]])
                if d2 is not None:
                    res.obs.append(Obs("reabasto_pallet", nom, n, f, tur, op, d2,
                                       "duracion_planilla", ws.title, i))
                continue
            res.obs.append(Obs("reabasto_pallet", nom, n, f, tur, op, d, "timestamps", ws.title, i))
        total = _diff(ts[0], ts[-1])
        if total is not None and total >= 0:
            res.obs.append(Obs("reabasto_pallet", "ciclo_total", n, f, tur, op, total,
                               "timestamps", ws.title, i))


def _cargar_reabasto_cajas(ws, res: Resultado):
    """HOJA NUEVA (planilla 2026-09-25): el proceso del problema central, medido por
    primera vez. No mide un ciclo lineal sino **las esperas** que describe el AS-IS:
      2/3   anotar las tareas en papel (paso 5 del AS-IS)
      4/5   tareas anotadas vs tareas eliminadas por el planificador
      6/7   pide que la eliminen -> la eliminan  = espera del operario (C3 del AS-IS)
      8/9   llamar grúa -> llega la grúa        = espera de grúa   (C5 del AS-IS)
      9/10  llegada de grúa -> baja el pallet
      10/11 baja el pallet -> termina de reponer
    Varias filas comparten el N° de muestra: la primera trae el ciclo completo y las
    siguientes solo repiten el par 6/7 (cada tarea eliminada de esa misma jornada).
    Por eso el N° se arrastra hacia abajo en vez de descartar la fila."""
    col_op = _col_responsable(ws, 18)
    fecha = _Fecha()
    ultimo_n = None
    for i, r in _filas(ws):
        if isinstance(r[0], str):
            break
        n = _n(r[0])
        if n is not None:
            ultimo_n = n
        n = n if n is not None else ultimo_n
        if n is None:
            continue
        f = fecha.de(r[1])
        op = str(r[col_op] or "")
        hora_ref = r[2] or r[6] or r[8]
        tur = _turno(hora_ref)
        tramos = [("anotar_tareas", r[2], r[3]),
                  ("espera_eliminacion_tarea", r[6], r[7]),
                  ("espera_grua", r[8], r[9]),
                  ("bajar_pallet", r[9], r[10]),
                  ("reponer_cajas", r[10], r[11])]
        for nom, a, b in tramos:
            d = _diff(a, b)
            if d is None:
                continue
            if d < 0:
                res.avisos.append(Aviso(ws.title, i, str(n), "descartada",
                                        f"{nom}: fin antes que inicio ({a} -> {b})"))
                continue
            res.obs.append(Obs("reabasto_caja", nom, n, f, tur, op, d, "timestamps", ws.title, i))
        # Las tareas eliminadas son la variable de resultado de H5, no un tiempo:
        # se registran como aviso para que queden visibles y auditables.
        anotadas, eliminadas = _n(r[4]), _n(r[5])
        if anotadas is not None and eliminadas is not None:
            res.avisos.append(Aviso(ws.title, i, str(n), "dudosa",
                                    f"{eliminadas} de {anotadas} tareas anotadas fueron "
                                    f"eliminadas por el planificador"))


def _cargar_reabasto_cajas_ciclo(ws, res: Resultado):
    """HOJA NUEVA de la planilla oficial: «Reabastecimiento_de_cajas». Ahora SÍ mide el ciclo de reposición de cajas (37 reposiciones, 6 visitas de grúa):
      2/3 llamar al gruero -> llega la grúa            (espera de grúa)
      5/6 comienzo -> término de bajar pallets          (trabajo del GRUERO, 8 = pallets bajados)
      9/10 comenzar -> terminar de reponer              (trabajo del OPERARIO por reposición, 12 = cajas)
    Una «visita» empieza en una fila con «Llamar a gruero» y cubre las filas siguientes hasta la próxima llamada.
    Observaciones que genera:
      reabasto_caja / reponer_tarea            duración de reponer una posición (N = reposiciones)
      reabasto_caja / ciclo_total              hueco entre el inicio de una reposición y el de la siguiente DENTRO de la visita, tope 900 s (el mismo corte
                                               de picking): incluye el traslado entre posiciones. Es el tiempo ocupado por reposición.
      reabasto_caja / espera_grua              llamar -> llega la grúa
      reabasto_caja_grua / ciclo_total         tiempo de bajar pallets por reposición servida (bajar pallets de la visita / reposiciones de la visita)
    El operario NO incluye aquí anotar las tareas en papel ni esperar a que el planificador elimine tareas (hoja «Reabastecimiento_cajas», N=2 jornadas)."""
    col_op = _col_responsable(ws, 14)
    fecha = _Fecha()
    visitas: list[dict] = []
    filas = []
    for i, r in _filas(ws):
        if isinstance(r[0], str):
            break
        n = _n(r[0])
        if n is None:
            continue
        f = fecha.de(r[1])
        op = str(r[col_op] or "")
        if r[2] is not None:
            visitas.append({"fila": i, "n": n, "fecha": f, "op": op, "bajar": _s(r[7]), "pallets": _n(r[8]),
                            "espera": _diff(r[2], r[3]), "tareas": 0, "hora": r[2]})
        filas.append((i, n, f, op, r))
        if visitas:
            visitas[-1]["tareas"] += 1
    ult_visita = None
    prev_ini = None
    for i, n, f, op, r in filas:
        v = next((x for x in visitas if x["fila"] == i), None)
        if v is not None:
            ult_visita, prev_ini = v, None
        ini, fin = r[9], r[10]
        tur = _turno(ini, r[2])
        d = _s(r[11])
        if d is None:
            d = _diff(ini, fin)
        if d is not None and d >= 0:
            res.obs.append(Obs("reabasto_caja", "reponer_tarea", n, f, tur, op, d, "duracion_planilla", ws.title, i))
        if isinstance(ini, dt.time):
            if prev_ini is not None and ult_visita is not None:
                g = _diff(prev_ini, ini)
                if g is not None and 0 < g <= 900:
                    res.obs.append(Obs("reabasto_caja", "ciclo_total", n, f, tur, op, g, "timestamps", ws.title, i))
                elif g is not None:
                    res.avisos.append(Aviso(ws.title, i, str(n), "descartada",
                                            f"ciclo_total: hueco de {g:.0f} s entre reposiciones de la misma visita fuera del corte (0-900 s)"))
            prev_ini = ini
    for v in visitas:
        if v["espera"] is not None and v["espera"] >= 0:
            res.obs.append(Obs("reabasto_caja", "espera_grua", v["n"], v["fecha"], _turno(v["hora"]), v["op"], v["espera"], "timestamps", ws.title, v["fila"]))
        if v["bajar"] is not None and v["tareas"]:
            res.obs.append(Obs("reabasto_caja_grua", "ciclo_total", v["n"], v["fecha"], _turno(v["hora"]), "gruero", v["bajar"] / v["tareas"],
                               "duracion_planilla", ws.title, v["fila"]))


def _cargar_recepcion_por_pallet(ws, res: Resultado):
    """Descarga de pallets 26-ago: una fila por pallet; F = duración; C = gruero (1/2)."""
    for i, r in _filas(ws, 4):
        p = _n(r[1])
        if p is None or i > 30:
            continue
        d = _s(r[5])
        if d is None:
            continue
        res.obs.append(Obs("recepcion_pallet_individual", "descarga_por_pallet", p, "2026-08-26",
                           _turno(r[3]), f"gruero_{_n(r[2])}", d, "duracion_planilla", ws.title, i))


# Los nombres de hoja cambian entre versiones de la planilla (tildes, espacios al
# final). Se buscan normalizados para que un renombre cosmético no rompa el cargador.
HOJAS = {
    "carga de camion": _cargar_carga,
    "recepcion": _cargar_recepcion_directo_stage,
    "almacenamiento": _cargar_almacenamiento,
    "reabastecimiento de pallets": _cargar_reabasto_pallets,
    "reabastecimiento cajas": _cargar_reabasto_cajas,
    "reabastecimiento de cajas": _cargar_reabasto_cajas_ciclo,
    "recepcion por pallets": _cargar_recepcion_por_pallet,
}
# Hojas que existen como plantilla pero todavía no tienen mediciones.
HOJAS_VACIAS = ("picking", "retorno fletero")


def _normalizar(nombre: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", nombre.strip().lower().replace("_", " "))
    return " ".join("".join(c for c in n if not unicodedata.combining(c)).split())


def _es_simulada(nombre: str) -> bool:
    """La planilla oficial trae hojas «…-simulado» (datos generados, no medidos): NUNCA entran como observaciones."""
    return "simulad" in nombre.lower()


def _libro_desde_ods(ruta: Path):
    """Convierte las hojas REALES de un .ods en un libro de openpyxl en memoria, conservando la posición de cada celda (fila y columna), para que los
    cargadores por hoja funcionen igual que con el .xlsx. Se omiten las hojas simuladas y los vínculos externos. Devuelve (libro, [hojas_simuladas])."""
    import pandas as pd
    xl = pd.ExcelFile(ruta, engine="odf")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    simuladas = []
    for nombre in xl.sheet_names:
        if nombre.startswith("'"):                  # vínculos a libros externos (vacíos)
            continue
        if _es_simulada(nombre):
            simuladas.append(nombre)
            continue
        df = xl.parse(nombre, header=None)
        ws = wb.create_sheet(nombre[:31])
        for fila in df.itertuples(index=False):
            vals = []
            for v in fila:
                if v is None or (isinstance(v, float) and v != v) or (hasattr(v, "__class__") and v.__class__.__name__ == "NaTType"):
                    vals.append(None)
                elif hasattr(v, "to_pydatetime"):
                    vals.append(v.to_pydatetime())
                elif hasattr(v, "item"):
                    vals.append(v.item())
                else:
                    vals.append(v)
            ws.append(vals)
    return wb, simuladas


def _es_ods(ruta: Path) -> bool:
    """Un .xlsx que en realidad es un OpenDocument (zip con `mimetype` = application/vnd.oasis.opendocument.spreadsheet)."""
    import zipfile
    try:
        with zipfile.ZipFile(ruta) as z:
            return "mimetype" in z.namelist() and b"opendocument" in z.read("mimetype")
    except Exception:
        return False


def _abrir(ruta: Path):
    """(libro, hojas simuladas omitidas)."""
    if str(ruta).lower().endswith(".ods") or _es_ods(ruta):
        return _libro_desde_ods(ruta)
    wb = openpyxl.load_workbook(ruta, data_only=True)
    return wb, [t for t in wb.sheetnames if _es_simulada(t)]


def _anonimizar(res: Resultado) -> None:
    """PRIVACIDAD (regla del proyecto): ningún nombre de operario sale de aquí. Cada persona pasa a un código estable `Operario_NN` (por orden de aparición),
    de modo que los conteos de operarios distintos —lo único que usa `fit_params`— no cambian."""
    import unicodedata
    vistos: dict[str, str] = {}

    def cod(n: str) -> str:
        if not n or not n.strip():
            return ""
        k = "".join(c for c in unicodedata.normalize("NFKD", n.strip().lower()) if not unicodedata.combining(c))
        if k.startswith("gruero_"):
            return k
        return vistos.setdefault(k, f"Operario_{len(vistos) + 1:02d}")
    for o in res.obs:
        o.operario = cod(o.operario)


def cargar(ruta: Path = PLANILLA) -> Resultado:
    wb, simuladas = _abrir(ruta)
    por_nombre = {_normalizar(t): t for t in wb.sheetnames}
    res = Resultado()
    for h in simuladas:
        res.avisos.append(Aviso(h, 0, "-", "descartada", "hoja SIMULADA (datos generados, no medidos): no se usa"))
    for clave, fn in HOJAS.items():
        real = por_nombre.get(clave)
        if real is None:
            res.avisos.append(Aviso(clave, 0, "-", "sin_medir",
                                    "la hoja no existe en esta versión de la planilla"))
            continue
        fn(wb[real], res)
    res.cerrar()
    _anonimizar(res)
    for clave in HOJAS_VACIAS:
        real = por_nombre.get(clave)
        if real is None:
            continue
        ws = wb[real]
        if not any(_n(r[0]) is not None and any(c not in (None, "") for c in r[2:])
                   for r in ws.iter_rows(min_row=3, values_only=True)):
            res.avisos.append(Aviso(real, 0, "-", "sin_medir",
                                    "hoja sin mediciones: proceso sin datos"))
    return res


def guardar(res: Resultado, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, filas in (("observaciones", res.obs), ("avisos", res.avisos), ("esperas", res.esperas)):
        if not filas:
            continue
        with open(salida / f"{nombre}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(filas[0]).keys()))
            w.writeheader()
            w.writerows(asdict(x) for x in filas)


def resumen(res: Resultado) -> str:
    from collections import Counter, defaultdict
    muestras = defaultdict(set)
    fechas = defaultdict(set)
    turnos = defaultdict(Counter)
    for o in res.obs:
        muestras[o.proceso].add(o.muestra)
        if o.fecha:
            fechas[o.proceso].add(o.fecha)
        turnos[o.proceso][o.turno] += 1
    out = ["proceso | muestras | días | observaciones por turno"]
    for p in sorted(muestras):
        out.append(f"{p} | {len(muestras[p])} | {len(fechas[p])} | {dict(turnos[p])}")
    c = Counter(a.tipo for a in res.avisos)
    out.append(f"avisos: {dict(c)}")
    por_paso = Counter((o.proceso, o.paso) for o in res.obs if o.duracion_s is not None)
    out.append("N por paso:")
    out.extend(f"  {p}.{q}: {n}" for (p, q), n in sorted(por_paso.items()))
    out.append(f"esperas registradas: {len(res.esperas)}")
    return "\n".join(out)


if __name__ == "__main__":
    r = cargar()
    guardar(r, Path(__file__).resolve().parent.parent / "salidas" / "limpio")
    print(resumen(r))
