"""LLEGADAS DE CAMIONES EN EL TIEMPO (recepción y carga, turnos A y B), medidas en el log del WMS (2026-11-09).

PREGUNTA: la simulación hoy hace aparecer todo el trabajo de un turno en su hora de inicio. ¿Se pueden sacar del log del WMS (a) cuántos camiones llegan por día según la demanda y (b) a qué hora
llegan, con una distribución que se pueda simular? Solo turnos A y B: el C queda fijo (22 trailers por noche) y su recepción es el 3%.

QUÉ ES CADA «LLEGADA» (esto decide cuánto sirve el resultado)
  Recepción: un DOCUMENTO de recepción = un camión (≤ 19 pallets: chico; si no, grande; un documento con más de ~35 pallets son varios camiones grandes juntos). Su hora es la de la PRIMERA
             identificación de un pallet en el WMS: es el inicio del ingreso, no la entrada del camión a portería (la planilla de terreno registra esperas de hasta 3 h entre la llegada y la descarga).
  Carga:     un TRAILER (número TRL). Las líneas de un trailer se confirman juntas (mediana ≈ 20 s entre la primera y la última): la hora es la del CIERRE/despacho del trailer, no la del inicio de la
             carga. Para el trabajo del gruero sirve como hora de término de una tarea de ~18 min (Diana), no como llegada.

MÉTODO
  Cantidad por día: Poisson contra binomial negativa (AIC) y dependencia de la venta (días con venta: septiembre).
  Hora: perfil por hora (proceso de Poisson no homogéneo, uniforme dentro de la hora), contra uniforme y Beta (AIC), y validación con la mitad de los días (se ajusta con los días pares y se evalúa con
        los impares y al revés: logverosimilitud por llegada). Agrupamiento: fracción de huecos < 10 min contra llegadas independientes con el mismo perfil (Monte Carlo).
  Generador: `generar_dia(demanda, rng)` y su validación contra lo observado.

USO:  .venv/bin/python -m modelo.llegadas_camiones      Salidas: salidas/llegadas_camiones/*.csv, salidas/limpio/llegadas_camiones.json, salidas/llegadas_camiones.md
PRIVACIDAD: solo conteos y horas; ningún usuario ni nombre.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modelo import recepcion_camiones as rc
from modelo import wms_octubre as wo
from modelo import wms_septiembre as ws

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "salidas" / "llegadas_camiones"
JSON = RAIZ / "salidas" / "limpio" / "llegadas_camiones.json"
INFORME = RAIZ / "salidas" / "llegadas_camiones.md"
VENTANA = (8.0, 24.0)                  # turnos A y B
VENTANA_A = (8.0, 16.0)
GRANDE_NOMINAL = rc.PALLETS_GRANDE_NOMINAL
SEMILLA = 20261109


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Tablas de llegadas
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _horas(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    return (ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600).to_numpy()


def documentos_recepcion() -> pd.DataFrame:
    """Un documento de recepción por fila: pallets, hora de la primera y de la última identificación, tipo y camiones. Días completos (4-sep a 9-oct sin los días cortados)."""
    d = pd.concat([ws._agregar_tiempo(ws.leer_export(ws.NUEVOS["recepcion"])), ws._agregar_tiempo(ws.leer_export(wo.ARCHIVOS["recepcion"]))])
    d = d[d.tramo1].sort_values("ts").drop_duplicates("lpn_origen")
    g = d.groupby("ubic_origen").agg(pallets=("lpn_origen", "size"), ini=("ts", "min"), fin=("ts", "max")).reset_index(drop=True)
    g["fecha"] = pd.to_datetime(g.ini).dt.date
    g = g[~g.fecha.isin(rc.DIAS_CORTADOS)].copy()
    g["tipo"] = np.where(g.pallets <= rc.CORTE_CHICO, "chico", "grande")
    g["camiones"] = np.where(g.tipo == "chico", 1, np.maximum(1, np.round(g.pallets / GRANDE_NOMINAL))).astype(int)
    g["hora"] = _horas(g.ini)
    g["dur_ident_min"] = (pd.to_datetime(g.fin) - pd.to_datetime(g.ini)).dt.total_seconds() / 60
    return g.reset_index(drop=True)


def trailers_carga() -> pd.DataFrame:
    """Un trailer por fila: hora de la primera y de la última línea despachada (el cierre), líneas y cajas."""
    c = wo.leer()["carga_camion"]
    c = c[(c.tramo == 2) & c.trailer.ne("")]
    g = c.groupby("trailer").agg(ini=("ts", "min"), fin=("ts", "max"), lineas=("ts", "size"), cajas=("cantidad", "sum")).reset_index(drop=True)
    g["fecha"] = pd.to_datetime(g.ini).dt.date
    g["hora"] = _horas(g.ini)
    g["span_min"] = (pd.to_datetime(g.fin) - pd.to_datetime(g.ini)).dt.total_seconds() / 60
    g["turno"] = np.where(g.hora < 8, "C", np.where(g.hora < 16, "A", "B"))
    # días completos para A y B: el export de carga empieza el 3-sep 14:39 y termina el 10-oct 17:36
    return g[(g.fecha >= dt.date(2026, 9, 4)) & (g.fecha <= dt.date(2026, 10, 9))].reset_index(drop=True)


def facturado_dia() -> pd.Series:
    d = pd.read_csv(RAIZ / "salidas" / "limpio" / "demanda_diaria.csv", parse_dates=["fecha"])
    d["fecha"] = d.fecha.dt.date
    return d.set_index("fecha")["facturado"]


def dias_operados(doc: pd.DataFrame, tr: pd.DataFrame) -> list:
    """Días con alguna actividad de recepción o de carga (los domingos y los días sin operación quedan fuera: un día sin camiones NO es un cero)."""
    return sorted(set(doc.fecha) | set(tr.fecha))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cantidad por día
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _nb_loglik(x: np.ndarray, mu: np.ndarray, k: float) -> float:
    """Binomial negativa con media mu y tamaño k (varianza = mu + mu²/k)."""
    p = k / (k + mu)
    return float(stats.nbinom.logpmf(x, k, p).sum())


def ajustar_conteo(n: np.ndarray, mu_dia: np.ndarray | None = None) -> dict:
    """Poisson y binomial negativa para la cantidad por día. `mu_dia` = media diaria esperada dada la venta (None: media constante)."""
    n = np.asarray(n, float)
    mu = np.full(len(n), n.mean()) if mu_dia is None else np.asarray(mu_dia, float)
    ll_p = float(stats.poisson.logpmf(n, mu).sum())
    ks = np.geomspace(0.5, 500, 200)
    lls = np.array([_nb_loglik(n, mu, k) for k in ks])
    k_nb, ll_nb = float(ks[lls.argmax()]), float(lls.max())
    res = (n - mu)
    return {"n_dias": len(n), "media": float(n.mean()), "varianza": float(n.var(ddof=1)), "indice_dispersion": float(n.var(ddof=1) / n.mean()) if n.mean() else float("nan"),
            "aic_poisson": -2 * ll_p, "aic_nbinom": 2 * 1 - 2 * ll_nb, "k_nbinom": k_nb, "dispersion_residual": float((res ** 2).sum() / max(1, len(n) - 1) / mu.mean()) if mu.mean() else float("nan"),
            "mejor": "nbinom" if (2 - 2 * ll_nb) < (-2 * ll_p) - 2 else "poisson"}


def conteos_por_dia(doc: pd.DataFrame, tr: pd.DataFrame, dias: list, fact: pd.Series) -> pd.DataFrame:
    """Camiones por día y por tipo / turno dentro de la ventana A y B (08:00 a 24:00), con la venta del día cuando existe."""
    d = pd.DataFrame({"fecha": dias})
    dv = doc[doc.hora >= 8]
    for t in ("chico", "grande"):
        d[f"rec_{t}_docs"] = d.fecha.map(dv[dv.tipo == t].groupby("fecha").size()).fillna(0).astype(int)
        d[f"rec_{t}_pallets"] = d.fecha.map(dv[dv.tipo == t].groupby("fecha").pallets.sum()).fillna(0).astype(int)
    d["rec_grande_camiones"] = d.fecha.map(dv[dv.tipo == "grande"].groupby("fecha").camiones.sum()).fillna(0).astype(int)
    for t in ("A", "B", "C"):
        d[f"carga_{t}"] = d.fecha.map(tr[tr.turno == t].groupby("fecha").size()).fillna(0).astype(int)
    d["facturado"] = d.fecha.map(fact)
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hora de llegada
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def perfil_horario(horas: np.ndarray, ventana: tuple[float, float], suave: float = 0.5) -> np.ndarray:
    """Probabilidad de llegada por hora entera dentro de la ventana (con suavizado de Laplace `suave`)."""
    lo, hi = int(ventana[0]), int(ventana[1])
    h = np.asarray(horas)
    c = np.array([((h >= k) & (h < k + 1)).sum() for k in range(lo, hi)], float)
    return (c + suave) / (c + suave).sum()


def _logdens_perfil(h: np.ndarray, p: np.ndarray, ventana: tuple[float, float]) -> np.ndarray:
    k = np.clip(np.floor(h - ventana[0]).astype(int), 0, len(p) - 1)
    return np.log(p[k])                                   # densidad por hora (cada hora mide 1)


def _ll_beta(h: np.ndarray, par: tuple, ventana: tuple[float, float]) -> np.ndarray:
    u = np.clip((h - ventana[0]) / (ventana[1] - ventana[0]), 1e-6, 1 - 1e-6)
    return stats.beta.logpdf(u, par[0], par[1]) - np.log(ventana[1] - ventana[0])


def evaluar_hora(horas_por_dia: dict, ventana: tuple[float, float]) -> dict:
    """Compara uniforme, Beta y perfil horario. AIC en la muestra completa y logverosimilitud por llegada con validación en dos mitades (días alternados)."""
    fechas = sorted(horas_por_dia)
    todas = np.concatenate([horas_por_dia[f] for f in fechas])
    todas = todas[(todas >= ventana[0]) & (todas < ventana[1])]
    L = ventana[1] - ventana[0]
    nb = int(L)
    ll_u = -len(todas) * np.log(L)
    par = stats.beta.fit(np.clip((todas - ventana[0]) / L, 1e-6, 1 - 1e-6), floc=0, fscale=1)[:2]
    ll_b = float(_ll_beta(todas, par, ventana).sum())
    p = perfil_horario(todas, ventana)
    ll_h = float(_logdens_perfil(todas, p, ventana).sum())
    chi = stats.chisquare(np.array([((todas >= ventana[0] + k) & (todas < ventana[0] + k + 1)).sum() for k in range(nb)]))
    cv = {}
    mitades = [fechas[0::2], fechas[1::2]]
    for nombre in ("uniforme", "beta", "perfil"):
        v = []
        for a, b in ((0, 1), (1, 0)):
            tr_ = np.concatenate([horas_por_dia[f] for f in mitades[a]]); te = np.concatenate([horas_por_dia[f] for f in mitades[b]])
            tr_ = tr_[(tr_ >= ventana[0]) & (tr_ < ventana[1])]; te = te[(te >= ventana[0]) & (te < ventana[1])]
            if nombre == "uniforme":
                v.append(np.full(len(te), -np.log(L)).mean())
            elif nombre == "beta":
                pr = stats.beta.fit(np.clip((tr_ - ventana[0]) / L, 1e-6, 1 - 1e-6), floc=0, fscale=1)[:2]
                v.append(_ll_beta(te, pr, ventana).mean())
            else:
                v.append(_logdens_perfil(te, perfil_horario(tr_, ventana), ventana).mean())
        cv[nombre] = float(np.mean(v))
    return {"n": int(len(todas)), "dias": len(fechas), "uniforme_chi2_p": float(chi.pvalue), "aic": {"uniforme": float(-2 * ll_u), "beta": float(2 * 2 - 2 * ll_b), "perfil": float(2 * (nb - 1) - 2 * ll_h)},
            "beta": [float(par[0]), float(par[1])], "ll_validacion_por_llegada": cv, "perfil_hora": [float(x) for x in p],
            "mediana_h": float(np.median(todas)), "p10_h": float(np.percentile(todas, 10)), "p90_h": float(np.percentile(todas, 90))}


def agrupamiento(horas_por_dia: dict, p: np.ndarray, ventana: tuple[float, float], umbral_min: float = 10.0, reps: int = 2000) -> dict:
    """¿Llegan en grupos? Fracción de huecos entre llegadas consecutivas del mismo día menores que `umbral_min`, contra llegadas independientes con el mismo perfil y la misma cantidad por día."""
    rng = np.random.default_rng(SEMILLA)
    def frac(listas):
        g = np.concatenate([np.diff(np.sort(x)) for x in listas if len(x) > 1])
        return float((g < umbral_min / 60).mean()), float(np.median(g) * 60)
    obs = [np.sort(v[(v >= ventana[0]) & (v < ventana[1])]) for v in horas_por_dia.values()]
    obs = [x for x in obs if len(x) > 1]
    f_obs, med_obs = frac(obs)
    cum = np.concatenate([[0], np.cumsum(p)])
    sim_f, sim_m = [], []
    for _ in range(reps):
        lst = []
        for x in obs:
            u = rng.random(len(x)); k = np.searchsorted(cum, u, side="right") - 1
            lst.append(ventana[0] + k + rng.random(len(x)))
        a, b = frac(lst); sim_f.append(a); sim_m.append(b)
    sim_f = np.array(sim_f)
    return {"frac_huecos_menores": f_obs, "umbral_min": umbral_min, "mediana_hueco_min": med_obs, "sim_frac_media": float(sim_f.mean()), "sim_frac_p5_p95": [float(np.percentile(sim_f, 5)), float(np.percentile(sim_f, 95))],
            "p_valor_mas_grupos": float((sim_f >= f_obs).mean()), "sim_mediana_hueco_min": float(np.mean(sim_m))}


def hora_vs_volumen(horas_por_dia: dict, conteo: pd.Series, ventana: tuple[float, float]) -> dict:
    """¿Los días con más camiones llegan más tarde? Hora media del día contra su cantidad (Spearman)."""
    f = [x for x in sorted(horas_por_dia) if len(horas_por_dia[x][(horas_por_dia[x] >= ventana[0]) & (horas_por_dia[x] < ventana[1])]) >= 2]
    m = np.array([horas_por_dia[x][(horas_por_dia[x] >= ventana[0]) & (horas_por_dia[x] < ventana[1])].mean() for x in f])
    n = np.array([conteo.get(x, 0) for x in f], float)
    r = stats.spearmanr(n, m)
    return {"n_dias": len(f), "spearman_cantidad_vs_hora_media": float(r.statistic), "p": float(r.pvalue)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generador de llegadas y su validación
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def _muestra_conteo(mu: float, k: float | None, rng) -> int:
    if mu <= 0:
        return 0
    if k is None or not np.isfinite(k) or k > 1e5:
        return int(rng.poisson(mu))
    return int(rng.negative_binomial(k, k / (k + mu)))


def _horas_perfil(n: int, p: np.ndarray, ventana: tuple[float, float], rng) -> np.ndarray:
    k = rng.choice(len(p), size=n, p=p)
    return np.sort(ventana[0] + k + rng.random(n))


def generar_dia(demanda: float, rng, par: dict | None = None) -> dict:
    """Llegadas de UN día para los turnos A y B: horas (reloj, 08:00 a 24:00) de camiones chicos y grandes de recepción y de cierres de trailers de carga. El turno C no se genera (fijo)."""
    par = par or cargar()
    out = {}
    for clave, ventana in (("rec_chico", VENTANA), ("rec_grande", VENTANA), ("carga_A", VENTANA_A), ("carga_B", (16.0, 24.0))):
        c = par["conteo"][clave]
        mu = c["a"] * demanda + c["b"]
        n = _muestra_conteo(mu, c["k"], rng)
        out[clave] = _horas_perfil(n, np.array(par["hora"][clave]["perfil_hora"]), ventana, rng)
    return out


def cargar() -> dict:
    return json.loads(JSON.read_text(encoding="utf-8"))


def validar_generador(par: dict, conteos: pd.DataFrame, horas: dict, reps: int = 400) -> dict:
    """Simula cada día observado con su venta y compara cantidad por día, horas y máximo de llegadas en una hora con lo observado."""
    rng = np.random.default_rng(SEMILLA + 1)
    claves = {"rec_chico": ("rec_chico_docs", VENTANA), "rec_grande": ("rec_grande_docs", VENTANA), "carga_A": ("carga_A", VENTANA_A)}
    out = {}
    for clave, (col, ventana) in claves.items():
        d = conteos[conteos.facturado.notna()]
        if clave == "rec_chico":
            d = conteos
        obs_n = d[col].to_numpy(float)
        sim_n, sim_max = [], []
        for _ in range(reps):
            for _, f in d.iterrows():
                v = generar_dia(float(f.facturado) if pd.notna(f.facturado) else float(np.nanmean(conteos.facturado)), rng, par)[clave]
                sim_n.append(len(v))
                sim_max.append(np.bincount(np.floor(v).astype(int) - int(ventana[0]), minlength=int(ventana[1] - ventana[0])).max() if len(v) else 0)
        obs_max = []
        for f in d.fecha:
            v = horas[clave].get(f, np.array([]))
            v = v[(v >= ventana[0]) & (v < ventana[1])]
            obs_max.append(np.bincount(np.floor(v).astype(int) - int(ventana[0]), minlength=int(ventana[1] - ventana[0])).max() if len(v) else 0)
        out[clave] = {"dias": len(d), "obs_media": float(obs_n.mean()), "obs_var": float(obs_n.var(ddof=1)), "sim_media": float(np.mean(sim_n)), "sim_var": float(np.var(sim_n, ddof=1)),
                      "obs_max_por_hora": float(np.mean(obs_max)), "sim_max_por_hora": float(np.mean(sim_max))}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Llegadas dentro de la simulación del día (opción desactivada por defecto: `simulacion_dia.simular_dia(..., llegadas=None)` reproduce la simulación de siempre)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
VARIANTES = {"rec_carga": "recepción y carga A llegan según el perfil", "con_derivados": "además, almacenamiento y carga para el CL siguen a los camiones de recepción"}
TURNO_INICIO = {"A": 8.0, "B": 16.0}


class GeneradorLlegadas:
    """Hora de aparición de las entidades de recepción y carga de los turnos A y B en la simulación. MISMO VOLUMEN que el modelo: solo cambia CUÁNDO llega cada entidad.
      - Recepción: se generan los camiones del día (cantidad por la venta, hora Beta) y los pallets del turno (los del modelo) se reparten entre los camiones del turno en proporción a su tamaño
        nominal (chico 10,3 y grande 29,5 pallets): los pallets de un camión llegan juntos. Si el turno no tiene camiones, el trabajo aparece al inicio (como en la base).
      - Carga A: cada trailer termina a una hora Beta (cierre del log) y su carga empieza `tiempo_carga_h` antes (1.095 s de Diana). Carga B y todo el turno C: sin cambio.
      - variante `con_derivados`: el almacenamiento y la carga para el CL de A y B se reparten entre los mismos camiones (llegan cuando llegan los pallets, sin retraso).
    Usa su PROPIO generador aleatorio por réplica: no consume los números aleatorios de la simulación (las duraciones salen idénticas a la base)."""

    def __init__(self, demanda: float, variante: str = "rec_carga", par: dict | None = None, charla_min: float = 20.0, colacion_min: float = 30.0, horas: float = 7.0 + 10 / 60,
                 tiempo_carga_h: float = 1095.0 / 3600.0):
        assert variante in VARIANTES
        self.demanda, self.variante, self.par = demanda, variante, par or cargar()
        self.charla, self.colacion, self.horas, self.tiempo_carga = charla_min / 60.0, colacion_min / 60.0, horas, tiempo_carga_h
        self.peso = {t: self.par["recepcion"]["pallets_por_documento"][t]["media"] for t in ("chico", "grande")}

    def a_reloj_sim(self, t: str, h_real: np.ndarray) -> np.ndarray:
        """Inversa de `simulacion_dia.hora_real`: hora de reloj → hora del reloj efectivo (descuenta la charla de inicio y la colación; lo que cae en la colación espera al fin de ella)."""
        x = np.asarray(h_real, float) - TURNO_INICIO[t] - self.charla
        mitad = self.horas / 2
        rel = np.where(x > mitad + self.colacion, x - self.colacion, np.where(x > mitad, mitad, x))
        return TURNO_INICIO[t] + np.clip(rel, 0.0, self.horas)

    def replica(self, rng: np.random.Generator) -> "_Replica":
        return _Replica(self, rng)


class _Replica:
    def __init__(self, g: GeneradorLlegadas, rng):
        self.g, self.rng = g, rng
        par = g.par
        self.camiones = {"A": ([], []), "B": ([], [])}                       # por turno: (horas reloj, pesos)
        for clave, tipo in (("rec_chico", "chico"), ("rec_grande", "grande")):
            c = par["conteo"][clave]
            n = _muestra_conteo(c["a"] * g.demanda + c["b"], c["k"], rng)
            a, b = par["hora"][clave]["beta"]
            h = VENTANA[0] + rng.beta(a, b, size=n) * (VENTANA[1] - VENTANA[0])
            for x in h:
                t = "A" if x < 16.0 else "B"
                self.camiones[t][0].append(float(x)); self.camiones[t][1].append(g.peso[tipo])
        self.carga_a = None

    def _carga_a(self, n: int) -> np.ndarray:
        a, b = self.g.par["hora"]["carga_A"]["beta"]
        cierre = VENTANA_A[0] + self.rng.beta(a, b, size=n) * (VENTANA_A[1] - VENTANA_A[0])
        return self.g.a_reloj_sim("A", np.maximum(cierre - self.g.tiempo_carga, TURNO_INICIO["A"]))

    def _pallets_a_camiones(self, t: str, n: int) -> np.ndarray | None:
        h, w = self.camiones[t]
        if not h:
            return None
        k = self.rng.choice(len(h), size=n, p=np.array(w) / np.sum(w))
        return self.g.a_reloj_sim(t, np.array(h)[k])

    def tiempos(self, t: str, nombre: str, n: int) -> np.ndarray | None:
        """Hora de aparición (reloj efectivo de la simulación) de las `n` entidades de (turno, proceso); None = al inicio del turno, como siempre."""
        if t not in ("A", "B") or n <= 0:
            return None
        if nombre == "recepcion_directo_stage":
            return self._pallets_a_camiones(t, n)
        if nombre in ("almacenamiento", "carga_cl") and self.g.variante == "con_derivados":
            return self._pallets_a_camiones(t, n)
        if nombre == "carga_camion" and t == "A":
            return self._carga_a(n)
        return None


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Corrida completa
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def analizar() -> dict:
    doc, tr, fact = documentos_recepcion(), trailers_carga(), facturado_dia()
    dias = dias_operados(doc, tr)
    cnt = conteos_por_dia(doc, tr, dias, fact)
    res: dict = {"dias_operados": len(dias), "desde": str(dias[0]), "hasta": str(dias[-1]), "dias_con_venta": int(cnt.facturado.notna().sum())}
    horas = {"rec_chico": {}, "rec_grande": {}, "carga_A": {}, "carga_B": {}}
    for f in dias:
        for t in ("chico", "grande"):
            horas[f"rec_{t}"][f] = doc[(doc.fecha == f) & (doc.tipo == t)].hora.to_numpy()
        for t in ("A", "B"):
            horas[f"carga_{t}"][f] = tr[(tr.fecha == f) & (tr.turno == t)].hora.to_numpy()
    res["conteo"], res["hora"], res["agrupamiento"], res["hora_vs_volumen"] = {}, {}, {}, {}
    # --- cantidad por día
    cv = cnt[cnt.facturado.notna()]
    series = {"rec_chico": ("rec_chico_docs", cnt, False), "rec_grande": ("rec_grande_docs", cnt, True), "carga_A": ("carga_A", cnt, True), "carga_B": ("carga_B", cnt, False)}
    for clave, (col, dat, usa_venta) in series.items():
        d = cv if usa_venta else dat
        x = d.facturado.to_numpy(float) / 1000 if usa_venta else None
        y = d[col].to_numpy(float)
        info = {"n_dias": int(len(d)), "media": float(y.mean()), "usa_venta": bool(usa_venta)}
        if usa_venta:
            r = stats.linregress(x, y)
            info.update({"a": float(r.slope / 1000), "b": float(r.intercept), "r": float(r.rvalue), "p": float(r.pvalue), "sigma_residual": float(np.std(y - (r.intercept + r.slope * x), ddof=2))})
            mu = np.clip(r.intercept + r.slope * x, 0.05, None)
            # la media diaria de un día cualquiera es a·D + b (en cajas, no en miles)
            info["ajuste"] = ajustar_conteo(y, mu)
        else:
            info.update({"a": 0.0, "b": float(y.mean())})
            info["ajuste"] = ajustar_conteo(y)
        info["k"] = info["ajuste"]["k_nbinom"] if info["ajuste"]["mejor"] == "nbinom" else None
        res["conteo"][clave] = info
    # --- hora
    ventanas = {"rec_chico": VENTANA, "rec_grande": VENTANA, "carga_A": VENTANA_A, "carga_B": (16.0, 24.0)}
    cuentas = {"rec_chico": cnt.set_index("fecha").rec_chico_docs, "rec_grande": cnt.set_index("fecha").rec_grande_docs, "carga_A": cnt.set_index("fecha").carga_A, "carga_B": cnt.set_index("fecha").carga_B}
    for clave, vent in ventanas.items():
        e = evaluar_hora(horas[clave], vent)
        res["hora"][clave] = e
        res["agrupamiento"][clave] = agrupamiento(horas[clave], np.array(e["perfil_hora"]), vent)
        res["hora_vs_volumen"][clave] = hora_vs_volumen(horas[clave], cuentas[clave], vent)
    # --- recepción: pallets por documento, antes de las 08:00 y duración de la identificación
    dv = doc[doc.hora >= 8]
    res["recepcion"] = {"docs": int(len(doc)), "antes_de_las_08_pct": float(100 * (doc.hora < 8).mean()),
                        "pallets_por_documento": {t: {"media": float(doc[doc.tipo == t].pallets.mean()), "sd": float(doc[doc.tipo == t].pallets.std()), "n": int((doc.tipo == t).sum())} for t in ("chico", "grande")},
                        "docs_varios_camiones": int((doc.camiones > 1).sum()), "dur_ident_min_mediana": {t: float(doc[doc.tipo == t].dur_ident_min.median()) for t in ("chico", "grande")}}
    res["carga"] = {"trailers": int(len(tr)), "por_turno": {t: int((tr.turno == t).sum()) for t in ("A", "B", "C")}, "span_min_mediana": float(tr.span_min.median()), "span_min_p95": float(tr.span_min.quantile(0.95)),
                    "pct_span_menor_5min": float(100 * (tr.span_min < 5).mean()), "lineas_mediana": float(tr.lineas.median())}
    SAL.mkdir(parents=True, exist_ok=True)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    cnt.to_csv(SAL / "conteos_por_dia.csv", index=False)
    pd.DataFrame({"hora": range(24), **{f"{k}": [res["hora"][k]["perfil_hora"][int(h - ventanas[k][0])] if ventanas[k][0] <= h < ventanas[k][1] else 0.0 for h in range(24)] for k in ventanas}}).to_csv(SAL / "perfil_horario.csv", index=False)
    res["validacion_generador"] = validar_generador(res, cnt, horas)
    JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main() -> int:
    r = analizar()
    print(json.dumps({k: r[k] for k in ("dias_operados", "desde", "hasta", "dias_con_venta", "recepcion", "carga")}, ensure_ascii=False, indent=1))
    for k, c in r["conteo"].items():
        a = c["ajuste"]
        print(f"\n{k}: {c['n_dias']} días, media {c['media']:.2f}, a={c['a']:.3g} b={c['b']:.3g} r={c.get('r', float('nan')):.2f}; índice de dispersión {a['indice_dispersion']:.2f}; mejor {a['mejor']} (k {a['k_nbinom']:.1f}); ΔAIC {a['aic_poisson'] - a['aic_nbinom']:.1f}")
    for k, h in r["hora"].items():
        print(f"\n{k}: N {h['n']}, uniforme chi2 p={h['uniforme_chi2_p']:.2g}, AIC {h['aic']}, validación {h['ll_validacion_por_llegada']}, agrupamiento {r['agrupamiento'][k]}, hora~volumen {r['hora_vs_volumen'][k]}")
    print("\nvalidación del generador:", json.dumps(r["validacion_generador"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
