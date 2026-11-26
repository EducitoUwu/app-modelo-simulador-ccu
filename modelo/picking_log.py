"""Picking desde el log del WMS (`docs/recogida de lista.csv`): el primer tiempo de
ciclo real de turno noche del proyecto, y con N grande.

POR QUÉ ESTE ARCHIVO SÍ SIRVE PARA TIEMPOS Y `docs/i.csv` NO
  En `i.csv` cada tarea deja dos registros escritos al confirmar (0-1 s aparte), así
  que la resta entre ellos no mide nada. Acá el export viene filtrado a `Recogida de
  lista` con origen real (SURTID) — es decir, **una fila = una confirmación de picking
  en una posición**. El tiempo aprovechable es el **hueco entre picks consecutivos del
  mismo operario**, que incluye desplazamiento + toma + confirmación por RF. Es
  exactamente el tiempo de ciclo que necesita `simulation.py`.

QUÉ MIDE Y QUÉ NO
  Mide: tiempo por línea de picking, de punta a punta, en condiciones reales.
  NO mide: el desglose interno del paso 8 del AS-IS (escanear ubicación / confirmar
  cantidad / confirmar UM) — para eso hace falta cronómetro. Tampoco mide las esperas
  de la ruta de excepción (llamar al gruero a gritos), porque esas terminan en una
  cancelación que no deja fila acá.

SEGMENTACIÓN (decisión de diseño, no cosmética)
  El gap no es una sola población. Se parte en cuatro según qué cambió respecto del
  pick anterior del mismo operario:
    - `pick_misma_ubicacion`: misma posición. Sale MÁS lento que cambiar de posición
      dentro del pasillo [hipótesis: es el cierre/cambio de pallet, no el pick].
    - `pick_mismo_pasillo`:   cambia posición, mismo pasillo (primer campo del código).
    - `pick_cambia_pasillo`:  cambia de pasillo → incluye el desplazamiento largo.
    - `cambio_de_onda`:       además cambia la lista de surtido (nueva onda).
  Un solo número promedio escondería que cambiar de pasillo cuesta ~4x lo que quedarse.

CORTE DE OUTLIERS (declarado, no "a ojo")
  Se descartan los gaps > `CORTE_S` porque no son tiempo de trabajo sino colación,
  espera de onda o fin de turno. El corte y cuántas observaciones deja fuera quedan
  registrados en la salida para poder auditarlo.

LIMITACIÓN: el export volvió a truncarse en 100.000 registros exactos (03-09 a 24-09).
El día más antiguo está incompleto y se marca como tal.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# El WMS solo conserva ~1 mes de historial y trunca cada export en 100.000 registros,
# así que el picking llega en varios archivos. Se leen todos los que existan y se
# deduplica por (timestamp, usuario, ubicación, artículo): los rangos se solapan.
LOGS = (RAIZ / "docs" / "recogida de lista.csv",
        RAIZ / "docs" / "recogida de lista agosto.csv")
LOG = LOGS[0]
# Export nuevo (21-sep a 1-oct, 2026-10-01): trae los DOS tramos; el tramo 1 (origen SURTID = una línea pickeada) coincide registro a
# registro con los logs viejos en el traslape (verificar_traslape), así que se combina deduplicando; el tramo 2 (TRF -> PATCA) no es
# un pick y se descarta. `LOGS` (los viejos) queda INTACTO para `fit_params` (tiempos estándar congelados, 2026-10-05): ampliar las
# observaciones de tiempo a las noches 25-sep a 1-oct movería parametros.json y es una decisión aparte.
NUEVO = RAIZ / "docs" / "wms actividades septiembre" / "Picking septiembre 20 para adelante.csv"
LOGS_COMPLETO = LOGS + (NUEVO,)
SALIDA_OBS = RAIZ / "salidas" / "limpio" / "picking_observaciones.csv"
SALIDA_NOCHES = RAIZ / "salidas" / "limpio" / "picking_noches.csv"

csv.field_size_limit(10 ** 7)

# Gaps por sobre esto no son trabajo: colación, espera entre ondas, fin de turno.
# 900 s deja dentro el 99,5% de los gaps y excluye las pausas largas.
CORTE_S = 900
# Hueco que separa dos jornadas distintas del mismo operario (para no encadenar la
# noche del día D con la del día D+1).
CORTE_JORNADA_S = 3 * 3600


def turno_de_hora(h: int) -> str:
    if 8 <= h < 16:
        return "A"
    if 16 <= h < 24:
        return "B"
    return "C"


def percentil(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


@dataclass
class Pick:
    ts: dt.datetime
    usuario: str
    articulo: str
    cantidad: int
    ubicacion: str
    lista: str
    fila: int

    @property
    def pasillo(self) -> str:
        return self.ubicacion.split("-")[0]


def _partes(celda: str) -> list[str]:
    return [p.strip() for p in celda.split("\n") if p.strip() != ""]


def leer(rutas=None) -> list[Pick]:
    """Lee uno o varios exports de picking y devuelve los picks únicos, ordenados."""
    if rutas is None:
        rutas = LOGS
    elif isinstance(rutas, (str, Path)):
        rutas = (rutas,)
    vistos, out = set(), []
    for r in rutas:
        if Path(r).exists():
            for p in _leer_uno(Path(r)):
                k = (p.ts, p.usuario, p.ubicacion, p.articulo, p.cantidad)
                if k not in vistos:
                    vistos.add(k)
                    out.append(p)
    out.sort(key=lambda p: p.ts)
    return out


def leer_completo() -> list[Pick]:
    """Picks (tramo 1, origen real) de TODOS los exports: viejos + el nuevo del 21-sep al 1-oct, deduplicados. Cubre 20-ago a 1-oct."""
    return [p for p in leer(LOGS_COMPLETO) if not p.ubicacion.startswith("TRF")]


def verificar_traslape() -> dict:
    """En la ventana donde coinciden los exports viejos y el nuevo (21 al 24-sep), ¿son los mismos picks?"""
    viejos = leer(LOGS)
    nuevos = [p for p in leer((NUEVO,)) if not p.ubicacion.startswith("TRF")]
    lo, hi = max(viejos[0].ts, nuevos[0].ts), min(viejos[-1].ts, nuevos[-1].ts)
    k = lambda p: (p.ts, p.usuario, p.ubicacion, p.articulo, p.cantidad)
    a = {k(p) for p in viejos if lo <= p.ts <= hi}
    b = {k(p) for p in nuevos if lo <= p.ts <= hi}
    return {"desde": lo, "hasta": hi, "viejos": len(a), "nuevos": len(b), "comunes": len(a & b)}


def _leer_uno(ruta: Path) -> list[Pick]:
    out = []
    with open(ruta, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for n, r in enumerate(rd, start=1):
            if len(r) < 10:
                continue
            f, a, ub = _partes(r[0]), _partes(r[1]), _partes(r[8])
            if not f:
                continue
            try:
                ts = dt.datetime.strptime(f[0], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue
            out.append(Pick(ts=ts, usuario=f[1] if len(f) > 1 else "",
                            articulo=r[2].strip(), cantidad=int(r[3] or 0),
                            ubicacion=ub[0] if ub else "",
                            lista=a[1] if len(a) > 1 else "", fila=n))
    return out


# ---------------------------------------------------------------- tiempos

def observaciones(picks: list[Pick]) -> tuple[list[dict], dict]:
    """Gap entre picks consecutivos del mismo operario, segmentado y limpio.

    Devuelve (filas con el mismo esquema que observaciones.csv, diagnóstico)."""
    por_user: dict[str, list[Pick]] = defaultdict(list)
    for p in picks:
        por_user[p.usuario].append(p)

    filas, diag = [], {"gap_total": 0, "descartado_corte": 0, "descartado_jornada": 0}
    for usuario, ps in por_user.items():
        ps.sort(key=lambda p: p.ts)
        for ant, act in zip(ps, ps[1:]):
            g = (act.ts - ant.ts).total_seconds()
            if g <= 0:
                continue
            diag["gap_total"] += 1
            if g > CORTE_JORNADA_S:
                diag["descartado_jornada"] += 1
                continue
            if g > CORTE_S:
                diag["descartado_corte"] += 1
                continue
            if act.lista != ant.lista:
                paso = "cambio_de_onda"
            elif act.ubicacion == ant.ubicacion:
                paso = "pick_misma_ubicacion"
            elif act.pasillo == ant.pasillo:
                paso = "pick_mismo_pasillo"
            else:
                paso = "pick_cambia_pasillo"
            filas.append({
                "proceso": "picking", "paso": paso, "muestra": act.fila,
                "fecha": act.ts.date().isoformat(), "turno": turno_de_hora(act.ts.hour),
                "operario": usuario, "duracion_s": int(g), "origen": "wms_log_gap",
                "hoja": "recogida de lista.csv", "fila": act.fila,
            })
    filas.sort(key=lambda r: (r["fecha"], r["fila"]))
    return filas, diag


# ---------------------------------------------------------------- jornadas

@dataclass
class Noche:
    fecha: str
    dia_semana: str
    picks: int
    cajas: int
    pickers: int
    primer_pick: str
    fin_nocturno: str        # último pick antes de un hueco > 2 h
    ultimo_pick: str
    picks_turno_a: int       # picking que se pasó del turno C al A
    picks_a_de_gente_de_la_noche: int
    horas_persona: float
    picks_por_hora_persona: float
    cobertura: str


def noches(picks: list[Pick]) -> list[Noche]:
    por_dia: dict[dt.date, list[Pick]] = defaultdict(list)
    for p in picks:
        por_dia[p.ts.date()].append(p)
    primer_dia = min(por_dia)
    out = []
    for d in sorted(por_dia):
        ps = sorted(por_dia[d], key=lambda p: p.ts)
        noche_users = {p.usuario for p in ps if p.ts.hour < 8}
        en_a = [p for p in ps if 8 <= p.ts.hour < 16]
        # fin de la jornada nocturna: se corta al primer hueco > 2 h
        fin = ps[0].ts
        for p in ps:
            if (p.ts - fin).total_seconds() > 7200:
                break
            fin = p.ts
        # horas-persona: por operario, tiempo entre su primer y último pick de la jornada
        hp = 0.0
        pu: dict[str, list[dt.datetime]] = defaultdict(list)
        for p in ps:
            if p.ts <= fin:
                pu[p.usuario].append(p.ts)
        for ts in pu.values():
            hp += (max(ts) - min(ts)).total_seconds() / 3600
        npicks = sum(1 for p in ps if p.ts <= fin)
        out.append(Noche(
            fecha=d.isoformat(), dia_semana=d.strftime("%a"),
            picks=len(ps), cajas=sum(p.cantidad for p in ps),
            pickers=len({p.usuario for p in ps}),
            primer_pick=ps[0].ts.time().isoformat(),
            fin_nocturno=fin.time().isoformat(),
            ultimo_pick=ps[-1].ts.time().isoformat(),
            picks_turno_a=len(en_a),
            picks_a_de_gente_de_la_noche=sum(1 for p in en_a if p.usuario in noche_users),
            horas_persona=round(hp, 1),
            picks_por_hora_persona=round(npicks / hp, 1) if hp else 0.0,
            cobertura="parcial" if d == primer_dia else "completa",
        ))
    return out


# ---------------------------------------------------------------- resumen

def resumen_tiempos(filas: list[dict]) -> list[dict]:
    por_paso: dict[str, list[float]] = defaultdict(list)
    for r in filas:
        por_paso[r["paso"]].append(float(r["duracion_s"]))
    out = []
    for paso, v in sorted(por_paso.items(), key=lambda kv: -len(kv[1])):
        out.append({"paso": paso, "n": len(v), "p25": round(percentil(v, .25)),
                    "p50": round(percentil(v, .5)), "p75": round(percentil(v, .75)),
                    "p90": round(percentil(v, .9)), "media": round(st.fmean(v)),
                    "dias": len({r["fecha"] for r in filas if r["paso"] == paso}),
                    "operarios": len({r["operario"] for r in filas if r["paso"] == paso})})
    return out


def guardar(ruta=None):
    picks = leer(ruta)
    filas, diag = observaciones(picks)
    SALIDA_OBS.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_OBS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    # Las noches (conteos por jornada) salen de TODOS los exports; las observaciones de tiempo (arriba) de los viejos, congeladas.
    ns = noches(leer_completo() if ruta is None else picks)
    with open(SALIDA_NOCHES, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(ns[0])))
        w.writeheader()
        w.writerows(asdict(n) for n in ns)
    return picks, filas, ns, diag


if __name__ == "__main__":
    picks, filas, ns, diag = guardar()
    print(f"{len(picks):,} picks leídos | {len(filas):,} observaciones de tiempo "
          f"-> {SALIDA_OBS.name}")
    print(f"descartados: {diag['descartado_corte']:,} por gap > {CORTE_S}s, "
          f"{diag['descartado_jornada']:,} por cambio de jornada "
          f"(de {diag['gap_total']:,} gaps)\n")
    print(f"{'paso':<24}{'N':>7}{'p25':>6}{'p50':>6}{'p75':>6}{'p90':>7}{'media':>7}"
          f"{'días':>6}{'oper.':>7}")
    for r in resumen_tiempos(filas):
        print(f"{r['paso']:<24}{r['n']:>7,}{r['p25']:>6}{r['p50']:>6}{r['p75']:>6}"
              f"{r['p90']:>7}{r['media']:>7}{r['dias']:>6}{r['operarios']:>7}")
    print(f"\n{'fecha':<12}{'día':<5}{'picks':>7}{'cajas':>8}{'pk':>4}{'inicio':>10}"
          f"{'fin noche':>11}{'h-pers':>8}{'picks/h':>9}  turno A")
    for n in ns:
        print(f"{n.fecha:<12}{n.dia_semana:<5}{n.picks:>7,}{n.cajas:>8,}{n.pickers:>4}"
              f"{n.primer_pick:>10}{n.fin_nocturno:>11}{n.horas_persona:>8.0f}"
              f"{n.picks_por_hora_persona:>9.1f}  {n.picks_turno_a:>5} "
              f"({n.picks_a_de_gente_de_la_noche} de la noche)")
