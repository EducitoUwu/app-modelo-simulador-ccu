"""App de capacidad y simulación del CD Coquimbo (Transportes CCU, Capstone EIC 2026-2 UCN).

Es la pestaña «Capacidad y simulación» del dashboard del proyecto, sola: demanda → dotación por turno y rol (modelo de capacidad),
dotación mínima entera (MILP) y simulación de eventos discretos del día completo. Generada por `modelo/exportar_app.py`.

Correr:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from modelo import calibracion, capacity, config, fit_params, optimizacion  # noqa: E402
from modelo.fit_params import cargar as cargar_parametros  # noqa: E402
from modelo import simulacion_dia as sd, simulacion_turno_c as sc_  # noqa: E402


def _agregar_personas(r: dict, vigente: bool, sd):
    """Si el día NO es viable, indica qué rol de qué turno falla y ofrece agregar 1 persona ahí y volver a simular (mismas réplicas y semilla). Solo mira los roles que deciden
    la viabilidad del turno (`roles_criterio`). Cada prueba queda en la tabla de comparación para ver si agregar gente sirve (con las grúas fijas, más grueros puede no alcanzar)."""
    if not r["viable"] and vigente:
        fallan = []
        for t in sd.TURNOS:
            x = r["turnos"][t]
            if r.get("traspaso_tol") is not None:           # con traspaso falla el TURNO (suelta más de lo tolerado): se ofrece el rol que más suelta
                if x["pct_en_ventana"] < r["umbral"] - 1e-9:
                    rol_max = max(x["roles_criterio"], key=lambda ro: x["traspaso_h"][ro])
                    fallan.append((x["pct_en_ventana"], t, rol_max.capitalize()))
                continue
            for rol in x["roles_criterio"]:
                pct = x["pct_rol"][rol]
                if pct < r["umbral"] - 1e-9:
                    fallan.append((pct, t, rol.capitalize()))
        if not fallan:                                                       # el día falla sin que un solo rol esté bajo el umbral: se ofrece el más débil
            fallan = sorted((x["pct_rol"][rol], t, rol.capitalize()) for t in sd.TURNOS for x in [r["turnos"][t]] for rol in x["roles_criterio"])[:1]
        fallan.sort()
        st.error("**El día NO es viable.** Falla: " + "; ".join(f"**{rol} del turno {t}** (termina a tiempo en {100 * pct:.0f}% de las réplicas, se pide ≥ {100 * r['umbral']:.0f}%)"
                                                               for pct, t, rol in fallan))
        st.caption("Puedes agregar 1 persona al rol que falla y volver a simular con las mismas réplicas y semilla. Es una prueba «qué pasaría si»: la dotación total sube, y la regla del proyecto es "
                   "que todo +1 salga con su −1 en otro turno o rol, con la justificación del modelo.")
        cols = st.columns(min(len(fallan), 3))
        for i_, (pct, t, rol) in enumerate(fallan[:3]):
            ya = int(st.session_state.get("sim_extra", {}).get((t, rol), 0))
            if rol == "Gruero" and r["dot"][t][rol] >= _gruas_cd():
                cols[i_].warning(f"El turno {t} ya tiene {r['dot'][t][rol]} grueros, uno por grúa: agregar otro no sirve (hay {_gruas_cd()} grúas).")
                continue
            if cols[i_].button(f"➕ Agregar 1 {rol} al turno {t} y volver a simular", key=f"sim_mas_{t}_{rol}", type="primary" if i_ == 0 else "secondary",
                               help=f"Hoy se simulan {r['dot'][t][rol]} {rol.lower()}s en el turno {t}" + (f" (ya agregaste {ya})" if ya else "")):
                ex = dict(st.session_state.get("sim_extra", {}))
                ex[(t, rol)] = ex.get((t, rol), 0) + 1
                st.session_state["sim_extra"] = ex
                st.session_state["sim_rerun"] = True
                st.rerun()
    elif r["viable"] and st.session_state.get("sim_extra"):
        st.success("Con las personas agregadas el día **sí es viable**. Recuerda que la dotación total sube: para que sea neutra hay que quitar esas personas de otro turno o rol.")
    if st.session_state.get("sim_extra"):                                     # las pruebas quedan en la tabla de comparación de abajo
        if st.button("Quitar las personas agregadas", key="sim_quitar"):
            st.session_state["sim_extra"] = {}
            st.session_state["sim_rerun"] = True
            st.rerun()


_MODOS_APOYO = {"N": "No ayuda en el patio", "A50": "Apoyo la mitad del turno (escanea y traslada)", "A100": "Apoyo todo el turno (escanea y traslada)",
                "P50": "Polivalente la mitad del turno (además descarga, carga y retorno)", "P100": "Polivalente todo el turno (además descarga, carga y retorno)"}


# Escenarios de trabajo en el patio (2026-11-23, decisión del usuario): misma configuración en A, B y C. combo = personas de apoyo de `polivalencia_grilla` (None = modelo base).
_ESCENARIOS_PATIO = {
    "Gruero hace todo el patio": (None, "El gruero descarga, escanea, traslada a STREC, carga camiones y hace el retorno de fletero. Es el modelo base, validado contra las noches de septiembre."),
    "Apoyo operacional por turno": ("N+A100", "Una persona de apoyo por turno, todo el turno: escanea (14 s por pallet) y traslada a STREC (58 s, con equipo móvil, sin grúa). "
                                              "Descarga, carga, retorno, almacenamiento y reabastos siguen con el gruero. Con 1 o 2 personas de apoyo el resultado es el mismo "
                                              "(el trabajo que pueden tomar alcanza para una), por eso se deja una."),
    "1 polivalente por turno": ("N+P100", "Una persona por turno, todo el turno: escanea y traslada como el apoyo, y además descarga (con grúa), carga camiones y hace el retorno de fletero. "
                                          "Almacenamiento y reabastos siguen con el gruero."),
    "2 polivalentes por turno": ("P100+P100", "Dos personas por turno, cada una como el polivalente de arriba. Almacenamiento y reabastos siguen con el gruero."),
    "Personalizado": ("custom", "Eliges qué hace cada una de las 2 personas de apoyo de cada turno (la misma configuración en A, B y C): no ayuda, apoyo (escanea y traslada) "
                                "o polivalente (además descarga, carga y retorno), la mitad o todo el turno. Si las dos no ayudan, es lo mismo que «Gruero hace todo el patio»."),
}


@st.dialog("¿Qué hace cada quien en el patio?", width="large")
def _dialogo_escenarios_patio():
    for nom_, (_, txt_) in _ESCENARIOS_PATIO.items():
        st.markdown(f"**{nom_}.** {txt_}")
    st.markdown("En todos los escenarios con apoyo, si no hay nadie de apoyo libre el gruero escanea él mismo (30 s por pallet). La misma configuración se aplica en los turnos A, B y C.")


@st.cache_data(show_spinner=False)
def _tobe_analitico(demanda: float, combo: str) -> dict:
    from modelo import capacity as cp, polivalencia as pv, polivalencia_grilla as pg
    pv.DESCARGA_S["valor"] = cp.seg_recepcion_por_pallet(1.0)
    return {"as_is": pg.analitico(demanda, "N+N"), "to_be": pg.analitico(demanda, combo)}


def _tabla_calculo_tobe(demanda: float, combo: str):
    """Opción B (decisión del usuario): el cálculo rápido del TO-BE con sus decimales, dentro de un desplegable."""
    import math
    from modelo import polivalencia_grilla as pg
    an = _tobe_analitico(float(demanda), combo)
    frac = sum(pg.MODOS[m][1] for m in combo.split("+"))
    filas, tot = [], {"ai": 0.0, "ai_n": 0, "tb": 0.0, "tb_n": 0}
    for t in pg.TURNOS:
        ai, tb = an["as_is"][t], an["to_be"][t]
        ai_n, tb_n = max(1, math.ceil(ai - 1e-9)), max(1, math.ceil(tb - 1e-9))
        tot["ai"] += ai; tot["tb"] += tb; tot["ai_n"] += ai_n; tot["tb_n"] += tb_n
        filas.append({"Turno": t, "Grueros si el gruero hace todo": f"{ai:.2f} → {ai_n}".replace(".", ","), "Grueros con este escenario": f"{tb:.2f} → {tb_n}".replace(".", ","),
                      "Diferencia": tb_n - ai_n, "Personas de apoyo (equivalentes)": f"{frac:g}".replace(".", ",")})
    filas.append({"Turno": "Día", "Grueros si el gruero hace todo": f"{tot['ai']:.2f} → {tot['ai_n']}".replace(".", ","), "Grueros con este escenario": f"{tot['tb']:.2f} → {tot['tb_n']}".replace(".", ","),
                  "Diferencia": tot["tb_n"] - tot["ai_n"], "Personas de apoyo (equivalentes)": f"{3 * frac:g}".replace(".", ",")})
    # 2026-11-24: la tabla del cálculo rápido ya no se muestra (decisión del usuario); solo da los grueros de «Mínima del modelo» en los escenarios con apoyo
    return {t: max(1, math.ceil(an["to_be"][t] - 1e-9)) for t in pg.TURNOS}


def _seccion_simulacion(demanda, ajustes, est, params, ajustes_tobe=None, opt=None):
    """Simulación SimPy del día completo ALIMENTADA por el modelo que el usuario acaba de confirmar (mismos volúmenes, tiempos, utilización y ventana efectiva). Tres pasos (2026-11-23):
    1) cómo se trabaja en el patio (hoy o un escenario TO-BE con personas de apoyo), 2) qué dotación (real, mínima del modelo, mínima validada o a mano), 3) simular.
    Cada corrida queda en una tabla de comparación."""
    from modelo import polivalencia_grilla as pg, simulacion_dia as sd, simulacion_turno_c as sc_
    st.markdown("---")
    st.markdown("### Simular el día (SimPy)")
    st.caption(f"Demanda: **{demanda:,.0f} cajas** (la de arriba). Los volúmenes y tiempos son los del modelo de capacidad de arriba.".replace(",", "."))

    # Paso 1: escenario
    st.markdown("**1. ¿Cómo se trabaja en el patio?**")
    if st.session_state.get("sim_esc") not in _ESCENARIOS_PATIO:      # nombres cambiados el 2026-11-24
        st.session_state.pop("sim_esc", None)
    c_esc, c_info = st.columns([5, 1.2])
    nombre = c_esc.radio("Escenario", list(_ESCENARIOS_PATIO), horizontal=True, key="sim_esc", label_visibility="collapsed")
    if c_info.button("¿Qué hace cada quien?", key="sim_btn_info", width="stretch"):
        _dialogo_escenarios_patio()
    combo = _ESCENARIOS_PATIO[nombre][0]
    if combo == "custom":
        c1, c2 = st.columns(2)
        orden = list(_MODOS_APOYO)
        m1 = c1.selectbox("Persona de apoyo 1", orden, index=4, format_func=_MODOS_APOYO.get, key="sim_p1")
        m2 = c2.selectbox("Persona de apoyo 2", orden, index=0, format_func=_MODOS_APOYO.get, key="sim_p2")
        combo = "+".join(sorted((m1, m2), key=orden.index))
        if combo == "N+N":
            combo = None

    # Paso 2: dotación
    st.markdown("**2. ¿Qué dotación simular?**")
    prop = {(f["turno"], f["recurso"]): f["propuesta"] for f in est["filas"] if f["tipo"] == "rol"}
    fuentes = ["Real (la de «Dotación actual»)", "Mínima del modelo de capacidad", "Manual (escribe la que quieras)"]       # 2026-11-24: sin la «mínima validada»; 2026-11-25: + manual (decisión del usuario)
    if st.session_state.get("sim_dot") not in fuentes:
        st.session_state.pop("sim_dot", None)
    fuente = st.radio("Dotación", fuentes, horizontal=True, key="sim_dot", label_visibility="collapsed")
    milp = {t: {r_: max(1, int(opt["n"][(t, r_)])) for r_ in ("Operario", "Gruero")} for t in sd.TURNOS} if opt else None
    if milp is None:
        _, d0 = sd.preparar_dia(demanda, params, ajustes)
        milp = {t: {r_: max(1, int(d0[t][r_])) for r_ in ("Operario", "Gruero")} for t in sd.TURNOS}
    grueros_tobe = _tabla_calculo_tobe(demanda, combo) if combo else None
    real0 = {t: {r_: max(1, int(prop[(t, r_)])) for r_ in ("Operario", "Gruero")} for t in sd.TURNOS}
    if fuente.startswith("Real"):
        dot_base = real0
    elif fuente.startswith("Manual"):
        if "sim_manual" not in st.session_state:                   # la manual parte de la real; se cambia en su ventana emergente
            st.session_state["sim_manual"] = {t: dict(real0[t]) for t in sd.TURNOS}
        if st.button("✏️ Escribir la dotación a probar", key="sim_btn_manual"):
            _dialogo_dotacion_manual(real0)
        dot_base = st.session_state["sim_manual"]
    else:
        dot_base = {t: {"Operario": milp[t]["Operario"], "Gruero": grueros_tobe[t] if combo else milp[t]["Gruero"]} for t in sd.TURNOS}
    # Operarios de A y B no se muestran: no deciden la viabilidad (en A y B solo deciden los grueros); se simulan con los de la fuente elegida (la real en «Manual»).
    st.markdown("**Dotación a simular**")
    st.table(pd.DataFrame({f"Turno {t}": [str(dot_base[t]["Gruero"]), str(dot_base[t]["Operario"]) if t == "C" else "—"] for t in ("A", "B", "C")},
                          index=["Grueros", "Pickers (operarios)"]))
    st.caption(("Para cambiarla, pulsa «Escribir la dotación a probar». " if fuente.startswith("Manual") else "Para probar otra, elige «Manual». ")
               + "Los operarios de A y B van con «—»: no deciden si el día es viable (en A y B solo cuentan los grueros)."
               + (" Personas de apoyo: 2 por turno, aparte de esta tabla." if combo else ""))
    sobre = [t for t in ("A", "B", "C") if dot_base[t]["Gruero"] > _gruas_cd()]
    if sobre:
        st.error(f"**No se puede simular:** el CD tiene solo {_gruas_cd()} grúas y " + ", ".join(f"el turno {t} tiene {dot_base[t]['Gruero']} grueros" for t in sobre)
                 + f". Cada gruero necesita una grúa: usa como máximo {_gruas_cd()} por turno.")

    # Paso 3: simular
    with st.expander("Opciones avanzadas (réplicas y umbral)"):
        c1, c2 = st.columns(2)
        reps = c1.number_input("Réplicas", 100, 5000, value=200, step=100, key="sim_reps",
                               help="200 sirven para explorar (error del % viable cerca del 90%: ±4 puntos); los resultados finales usan 1.000 (±1,9). La semilla es fija: subir las réplicas extiende la corrida, no la re-sortea.")
        umbral = c2.number_input("Umbral de viabilidad (% mínimo de réplicas en que el día cumple)", 50.0, 100.0, value=round(100.0 * sc_.SUPUESTOS["umbral_viabilidad"]["valor"], 1), step=1.0,
                                 key="sim_umbral") / 100.0
    usar_tobe = False
    if MOSTRAR_TOBE and ajustes_tobe is not None:
        usar_tobe = st.radio("Tiempos a simular", ["Actuales (AS-IS)", "TO-BE (los de la columna TO-BE)"], horizontal=True, key="sim_tiempos").startswith("TO-BE")
    aj_sim = ajustes_tobe if usar_tobe else ajustes
    seg_rep = 0.3 if combo else 0.2
    pulsado = st.button("▶ Simular", type="primary", key="sim_run", help=f"≈ {seg_rep:.1f} s por réplica".replace(".", ","), disabled=bool(sobre))
    if pulsado:                                                    # una simulación nueva parte de la dotación de la tabla, sin personas agregadas
        st.session_state["sim_extra"] = {}
    extras = dict(st.session_state.get("sim_extra", {}))
    sig = (demanda, aj_sim, combo, tuple(sorted((t, r_, n) for t, d in dot_base.items() for r_, n in d.items())), int(reps), umbral, tuple(sorted(extras.items())))
    if pulsado or st.session_state.pop("sim_rerun", False):
        horas = ajustes.horas_efectivas()
        sd0 = (sc_.HORAS_TURNO, dict(sd.FIN), sc_.TURNO)
        try:
            sd.configurar_ventana(horas)
            sc_.AVISOS.clear()
            dot = {t: dict(dot_base[t]) for t in sd.TURNOS}
            for (t_x, rol_x), n_x in extras.items():               # personas agregadas con los botones «Agregar 1 …»
                dot[t_x][rol_x] = dot[t_x][rol_x] + int(n_x)
            barra = st.progress(0.0, text=f"Simulando {int(reps)} réplicas (≈ {seg_rep * int(reps) / 60:.1f} min)…")
            prog = lambda i, n: barra.progress(i / n, text=f"Réplica {i} de {n}")
            if combo is None:
                preps, _ = sd.preparar_dia(demanda, params, aj_sim)
                r = sd.simular_dia(preps, dot, sc_.SUPUESTOS["gruas"]["valor"], int(reps), umbral, demanda=demanda, charla_min=ajustes.charla_min, colacion_min=ajustes.colacion_min, progreso=prog)
            else:
                rr = pg.simular(float(demanda), combo, None, int(reps), umbral, params, progreso=prog, dot=dot, ajustes=aj_sim)
                r = rr["agregado"]
                r["uso_apoyo"] = rr["uso_apoyo"]
            barra.empty()
            fin = {t: sd.FIN[t] for t in sd.TURNOS}
            etiqueta = nombre if nombre != "Personalizado" else f"Personalizado: {' + '.join(_MODOS_APOYO[m].lower() for m in (combo or 'N+N').split('+'))}"
            st.session_state["sim_res"] = (sig, r, sorted(set(sc_.AVISOS)), horas, fin, etiqueta, combo)
            comp = list(st.session_state.get("sim_comp", []))
            tr = sum(sum(r["turnos"][t]["traspaso_h"].values()) for t in sd.TURNOS)
            ua = r.get("uso_apoyo")
            comp.append({"#": len(comp) + 1, "Escenario": etiqueta, "Cajas": f"{demanda:,.0f}".replace(",", "."),
                         "Grueros A / B / C": " / ".join(str(r["dot"][t]["Gruero"]) for t in ("A", "B", "C")), "Grueros del día": sum(r["dot"][t]["Gruero"] for t in sd.TURNOS),
                         "Pickers turno C": r["dot"]["C"]["Operario"],
                         "Apoyo por turno": "—" if not combo else f"{sum(pg.MODOS[m][1] for m in combo.split('+')):g}".replace(".", ",") + " persona(s) equivalente(s)",
                         "Réplicas": int(reps), "% días viables": round(100 * r["pct_dia_ok"], 1), "¿Día viable?": "Sí" if r["viable"] else "NO",
                         "Traspaso (h-persona)": round(tr, 1), "Uso del apoyo % (A / B / C)": "—" if not ua else " / ".join("—" if ua[t] is None else str(round(100 * ua[t])) for t in ("A", "B", "C"))})
            st.session_state["sim_comp"] = comp
        finally:
            sc_.HORAS_TURNO, sc_.TURNO = sd0[0], sd0[2]
            sd.FIN.clear(); sd.FIN.update(sd0[1])
    if "sim_res" not in st.session_state:
        st.info("Elige el escenario y la dotación y pulsa «Simular».")
        return
    sig0, r, avisos, horas, fin, etiqueta, combo_r = st.session_state["sim_res"]
    st.markdown(f"#### Resultado: {etiqueta}")
    if sig0 != sig:
        st.warning("El resultado de abajo es de una configuración anterior: vuelve a pulsar «Simular» para usar la actual.")
    m = st.columns(3)
    m[0].metric("¿Día viable?", "Sí" if r["viable"] else "NO", f"{100 * r['pct_dia_ok']:.0f}% de réplicas cumplen (umbral {100 * r['umbral']:.0f}%)", delta_color="off")
    t99 = np.array(r.get("t99_lista") or [r["t99_picking"]])
    rango_pk = f"{sd.hora_real_picking('C', float(np.percentile(t99, 10)), ajustes)} a {sd.hora_real_picking('C', float(np.percentile(t99, 90)), ajustes)}"
    m[1].metric("Hora 99% del picking (C)", sd.hora_real_picking("C", r["t99_picking"], ajustes), f"rango p10 a p90: {rango_pk}", delta_color="off")
    tr_tot = sum(sum(r["turnos"][t]["traspaso_h"].values()) for t in sd.TURNOS)
    m[2].metric("Trabajo que pasa al turno siguiente", f"{tr_tot:.1f} h-persona".replace(".", ","), "media por día simulado", delta_color="off")
    filas = []
    ua = r.get("uso_apoyo")
    for t in sd.TURNOS:
        x = r["turnos"][t]
        f_ = {"Turno": t, "Pickers": str(r["dot"][t]["Operario"]) if t == "C" else "—", "Grueros": r["dot"][t]["Gruero"], "Decide": " + ".join(x["roles_criterio"]), "% réplicas que cumple": round(100 * x["pct_en_ventana"]),
              "Traspasa (h-persona, media)": round(sum(x["traspaso_h"].values()), 2), "Traspasa (h-persona, p90)": round(sum(x["traspaso_h_p90"].values()), 2),
              "Tareas traspasadas (media)": round(sum(x["traspaso_n"].values()), 1), "Término del picking, p10 a p90": rango_pk if t == "C" else "—", "Uso pickers %": str(round(100 * x["util_op"])) if t == "C" else "—", "Uso grueros %": round(100 * x["util_gr"])}
        if ua:
            f_["Uso del apoyo %"] = "—" if ua[t] is None else round(100 * ua[t])
        filas.append(f_)
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    e = r["espera_min"]
    st.caption(f"Trabajo que no alcanza a empezar al cierre de la ventana: pasa al turno siguiente (C→A, A→B; B no traspasa; el picking del C lo terminan los mismos pickers, como muestra el WMS); cada rol cumple si lo que suelta no pasa del {100 * (r.get('traspaso_tol') or 0):.0f}% de las horas de ese rol en el turno que lo recibe (los pickers del C y el B, si terminan en su ventana). "
               "Recepción y carga del turno A llegan según su distribución en el log del WMS (30 días); el resto del trabajo aparece al inicio del turno.")
    st.caption(f"Ventana efectiva por turno: {horas:.2f} h (las horas se muestran en reloj sumando charla y colación). Espera media por pedido: operario {e['operario']:.1f} min · gruero {e['gruero']:.1f} min · grúa {e['grua']:.1f} min. Uso de grúas (24 h): {100 * r['util_grua']:.0f}%."
               + (" Uso del apoyo = fracción de su tiempo disponible en el patio que pasa trabajando." if ua else ""))
    avisos = [a_ for a_ in avisos if not a_.startswith("recepcion_directo_stage.")]     # 2026-11-01: sin los avisos de recepción directo a stage
    if avisos:
        with st.expander("Avisos de la simulación"):
            for a_ in avisos:
                st.write("- " + a_)
    _agregar_personas(r, sig0 == sig, sd)
    _tabla_comparacion()


def _tabla_comparacion():
    """Cada corrida de «Simular» queda aquí, para comparar escenarios y dotaciones (p. ej. hoy contra 1 polivalente con la misma demanda)."""
    comp = st.session_state.get("sim_comp", [])
    if not comp:
        return
    st.markdown("#### Comparación de las corridas")
    df = pd.DataFrame(comp)
    st.dataframe(df, hide_index=True, width="stretch")
    ult = comp[-8:]
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.45 * len(ult)))
    y = np.arange(len(ult))[::-1]
    col = ["#3DBE6B" if c["¿Día viable?"] == "Sí" else "#C0504D" for c in ult]
    ax.barh(y, [c["Grueros del día"] for c in ult], color=col, edgecolor="#13281D")
    for y_, c in zip(y, ult):
        ax.text(c["Grueros del día"] + 0.15, y_, f"{c['Grueros del día']} grueros ({c['Grueros A / B / C']}) · {c['% días viables']:.0f}% viable".replace(".", ","), va="center", fontsize=8)
    ax.set_yticks(y, [f"#{c['#']} {c['Escenario']} · {c['Cajas']} cajas" for c in ult], fontsize=8)
    ax.set_xlabel("grueros del día (A + B + C)", fontsize=8)
    ax.set_xlim(0, max(c["Grueros del día"] for c in ult) * 1.6 + 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Verde = día viable, rojo = no viable. Para comparar escenarios, usa la misma demanda y las mismas réplicas: todas las corridas usan las mismas semillas, así la diferencia se debe al escenario y no al azar.")
    if st.button("Limpiar la comparación", key="sim_comp_limpiar"):
        st.session_state["sim_comp"] = []
        st.rerun()


@st.cache_resource
def _parametros():
    return cargar_parametros()


def _alternar_pct(clave: str):
    st.session_state[clave] = not st.session_state.get(clave, False)


def _aplicar_pct(claves: dict, pk: dict, total: float, modo: str):
    """Redistribuye el total del día entre los turnos según los % pedidos (suman 100%) y cierra el panel de porcentajes."""
    for t_, k_ in claves.items():
        if t_ in pk:
            st.session_state[k_] = round(total * st.session_state[pk[t_]] / 100.0, 1)
    for k_ in pk.values():
        st.session_state.pop(k_, None)
    st.session_state[modo] = False


@st.cache_data(show_spinner=False, max_entries=64)
def _estimar_cache(demanda, ajustes, prop_items, _params):
    """capacity.estimar con caché: el mismo (demanda, ajustes, propuesta) no se recalcula."""
    return capacity.estimar(demanda, ajustes, dict(prop_items), _params)


def _reset_estimador():
    for k in [k for k in st.session_state if str(k).startswith("est_")]:
        del st.session_state[k]


def _dotacion_confirmada() -> dict:
    """Dotación actual confirmada {(turno, rol): personas}; por defecto, la real del modelo (config.personas_turno)."""
    if "est_dot" not in st.session_state:
        st.session_state["est_dot"] = {(t_, r_): int(config.personas_turno(t_, r_)) for t_ in config.TURNOS for r_ in config.ROLES_OPTIMIZACION
                                       if (t_, r_) not in SIN_ESTIMAR}
    return st.session_state["est_dot"]


@st.dialog("Dotación a probar", width="large")
def _dialogo_dotacion_manual(real0: dict):
    """Ventana emergente de la opción «Manual»: grueros A, B, C (1 a 6, uno por grúa) y pickers del C. Nada cambia hasta «Confirmar»."""
    actual = st.session_state["sim_manual"]
    g = _gruas_cd()
    st.caption(f"Escribe la dotación que quieres simular. Grueros: de 1 a {g} por turno (hay {g} grúas). Pickers: solo el turno C (en A y B no deciden el día).")
    with st.form("sim_form_manual", border=False):
        cols = st.columns(3)
        nuevo = {}
        for c_, t_ in zip(cols, ("A", "B", "C")):
            c_.markdown(f"**Turno {t_}**")
            gr = int(c_.number_input(f"Grueros · turno {t_}", min_value=1, max_value=g, value=min(g, max(1, int(actual[t_]["Gruero"]))), step=1))
            op = int(c_.number_input("Pickers · turno C", min_value=1, max_value=60, value=max(1, int(actual[t_]["Operario"])), step=1)) if t_ == "C" else int(real0[t_]["Operario"])
            nuevo[t_] = {"Gruero": gr, "Operario": op}
        b1, b2, _ = st.columns([1, 1.4, 2])
        ok = b1.form_submit_button("Confirmar", type="primary")
        real = b2.form_submit_button("Volver a la real")
    if ok or real:
        st.session_state["sim_manual"] = nuevo if ok else {t: dict(real0[t]) for t in ("A", "B", "C")}
        st.rerun()


@st.dialog("Dotación actual", width="large")
def _dialogo_dotacion():
    """Ventana emergente: se editan todas las personas y nada se recalcula hasta «Confirmar» (formulario, sin recargas por campo)."""
    actual = _dotacion_confirmada()
    st.caption("Personas que hay hoy por rol y turno. Cambian la columna «Real», las brechas, el tope de la dotación óptima y la dotación que se simula "
               "con «Real». Al confirmar se recalcula todo de una vez.")
    with st.form("est_form_dot", border=False):
        cols_d = st.columns(3)
        nuevo = {}
        for c_, t_ in zip(cols_d, config.TURNOS):
            c_.markdown(f"**Turno {t_}**")
            for rol_ in config.ROLES_OPTIMIZACION:
                if (t_, rol_) in SIN_ESTIMAR:
                    continue
                nuevo[(t_, rol_)] = int(c_.number_input(f"{'Pickers' if rol_ == 'Operario' else 'Grueros'} · turno {t_}", min_value=0, max_value=_gruas_cd() if rol_ == "Gruero" else 60,
                                                        value=min(int(actual[(t_, rol_)]), _gruas_cd()) if rol_ == "Gruero" else int(actual[(t_, rol_)]), step=1,
                                                        help=f"Máximo {_gruas_cd()} grueros por turno: hay {_gruas_cd()} grúas en el CD." if rol_ == "Gruero" else None))
        b1, b2, _ = st.columns([1, 1.4, 2])
        ok = b1.form_submit_button("Confirmar", type="primary")
        real = b2.form_submit_button("Volver a la real")
    if ok or real:
        st.session_state["est_dot"] = nuevo if ok else {k: int(config.personas_turno(*k)) for k in actual}
        st.session_state["est_dot_nuevo"] = True
        st.rerun()


def _sync(origen: str, destino: str):
    st.session_state[destino] = st.session_state[origen]


def _demanda_de_entrada(prefijo: str, etiqueta: str, escenarios: dict, dias: list[dict], por_defecto: str = "p50"):
    """Widgets de demanda: escenario Baja / Media / Alta (con sus cajas) o número libre. Devuelve (demanda, descripción, fill rate real=None)."""
    niveles = {"polo bajo": "Baja", "p50": "Media", "p90": "Alta"}
    modo = st.radio(f"{etiqueta}: cómo elegir la demanda", ["Escenario", "Número libre"], horizontal=True, key=f"{prefijo}_modo")
    if modo == "Escenario":
        claves = [k for k in niveles if k in escenarios]
        esc = st.radio("Escenario de demanda", claves, index=claves.index(por_defecto) if por_defecto in claves else 0, horizontal=True,
                       format_func=lambda k: f"{niveles[k]} ({escenarios[k]:,.0f} cajas)".replace(",", "."), key=f"{prefijo}_esc")
        return float(escenarios[esc]), niveles[esc], None
    if f"{prefijo}_num" not in st.session_state:
        st.session_state[f"{prefijo}_num"] = 24_204
        st.session_state[f"{prefijo}_slider"] = 24_204
    c1, c2 = st.columns([3, 1])
    c1.slider("Cajas facturadas", 0, DEMANDA_MAX, step=100, key=f"{prefijo}_slider",
              on_change=_sync, args=(f"{prefijo}_slider", f"{prefijo}_num"))
    c2.number_input("o escribir", 0, DEMANDA_MAX, step=100, key=f"{prefijo}_num", help="Entre 0 y 70.000 cajas vendidas por día.",
                    on_change=_sync, args=(f"{prefijo}_num", f"{prefijo}_slider"))
    return float(st.session_state[f"{prefijo}_num"]), "número libre", None


_COLOR_TAREA = {"traslado_stage": "#6C9BD2", "traslado_mixtos": "#5FA8A0", "carga_cl": "#C9A227", "recogida_pallets": "#8FA3B8", "retorno_fletero_gruero": "#9C84C4", "traslados_operario": "#A07F5E", "traslados_gruero": "#A07F5E", "otras_actividades_operario": "#C3CCC6", "otras_actividades_gruero": "#C3CCC6", "retorno_fletero": "#7A6FA8", "merma": "#4D5B52", "picking": "#3DBE6B", "reabasto_caja": "#E0892B", "reabasto_caja_grua": "#F2B84B", "reabasto_pallet": "#B65F8E",
                "recepcion_directo_stage": "#3AA17E", "almacenamiento": "#7DBE6B", "carga_camion": "#8C9A91"}


@st.cache_data(show_spinner="Calculando la dotación…")
def _optimizar_cache(demanda, ajustes, _params):
    from modelo import optimizacion
    return optimizacion.resolver(demanda, ajustes, _params)


# Operario de A y B: la dotación NO se estima (acordado con Milton y Felipe): se informan solo las horas-hombre de las tareas que sí se pueden calcular.
SIN_ESTIMAR = (("A", "Operario"), ("B", "Operario"))


# TEMPORAL (grabación del video): False oculta la columna «TO-BE (s)» de los tiempos y la opción «Tiempos a simular» de la simulación. Poner True para volver a mostrarlas.
MOSTRAR_TOBE = False


DEMANDA_MAX = 70_000                 # control de entrada (2026-11-25): cajas vendidas por día entre 0 y 70.000


def _gruas_cd() -> int:
    """Grúas del CD (simulacion_turno_c.SUPUESTOS[«gruas»] = 6): ningún turno puede tener más grueros que grúas (cada gruero opera con su grúa)."""
    from modelo import simulacion_turno_c as sc_
    return int(sc_.SUPUESTOS["gruas"]["valor"])


def _panel_optimizacion(demanda, ajustes, params):
    """Bloque «Dotación óptima (PuLP)»: el MILP de modelo/optimizacion.py sobre la demanda y los ajustes confirmados."""
    st.markdown("#### Dotación óptima")
    try:
        r = _optimizar_cache(demanda, ajustes, params)
    except Exception as e:                 # PuLP no instalado o sin solver
        st.info(f"No se pudo resolver el modelo de optimización: {e}")
        return None
    # 2026-10-24 (reunión): la dotación ENTERA es el resultado principal; la fraccional (carga en personas-equivalentes) queda solo como referencia.
    ho = r["horas_operario_conocidas"]
    for rol in config.ROLES_OPTIMIZACION:
        for c_, s in zip(st.columns(len(config.TURNOS)), config.TURNOS):
            if (s, rol) in SIN_ESTIMAR:
                c_.metric(f"{rol} · turno {s}", f"{ho[s]:.1f} h-hombre".replace(".", ","), help="Horas-persona (horas de trabajo ÷ utilización) de las tareas calculables: reabasto de cajas, retorno de fletero, merma y picking diurno. No se estima la dotación de este rol y turno.")
                continue
            n_, R_ = r["n"].get((s, rol)), r["tope"][(s, rol)]
            c_.metric(f"{rol} · turno {s}", f"{n_} persona{'' if n_ == 1 else 's'}", delta=(f"{n_ - R_:+d} contra la real ({R_})" if n_ != R_ else f"igual a la real ({R_})"),
                      delta_color="inverse" if n_ != R_ else "off", help=f"Referencia: carga de {r['N'][(s, rol)]:.2f} personas-equivalentes (horas ÷ horas disponibles por persona)".replace(".", ","))
    filas = [{"Turno": s, "Rol": rol, "Dotación óptima (personas)": r["n"].get((s, rol)), "Dotación real": r["tope"][(s, rol)],
              "Diferencia": r["n"].get((s, rol), 0) - r["tope"][(s, rol)], "Referencia: carga fraccional": round(r["N"][(s, rol)], 2)}
             for s in config.TURNOS for rol in config.ROLES_OPTIMIZACION if (s, rol) not in SIN_ESTIMAR]
    with st.expander("Tabla con la carga fraccional (referencia)"):
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    sobre = [s for s in config.TURNOS if (s, "Gruero") in r["n"] and r["n"][(s, "Gruero")] > _gruas_cd()]
    if sobre:
        st.error("**No se puede operar así:** " + "; ".join(f"el turno {s} necesita {r['n'][(s, 'Gruero')]} grueros" for s in sobre)
                 + f", y el CD tiene solo {_gruas_cd()} grúas (cada gruero necesita una). Con esta demanda el patio no alcanza sin cambiar la forma de trabajar o mover trabajo a otro turno.")
    nota = ""
    if r["tope_relajado"]:
        nota += "El tope de dotación no alcanza: se resolvió sin la cota superior y la brecha es lo que falta respecto de la dotación real."
    if r["gruas_relajadas"]:
        exceso = {s: r["horas_grua"][s] - r["horas_grua_disponibles"][s] for s in config.TURNOS if r["horas_grua"][s] > r["horas_grua_disponibles"][s] + 1e-6}
        nota += " Las grúas no alcanzan en " + ", ".join(f"el turno {s} (faltan {h:.1f} h de grúa)".replace(".", ",") for s, h in exceso.items()) + "."
    if nota:
        st.caption(nota.strip())
    return r


def _panel_carga_por_tarea(est: dict):
    """Porcentaje de la carga de trabajo por tarea: reparte las horas-persona de cada (turno, rol) entre sus procesos. No calcula nada nuevo."""
    st.markdown("#### Carga de trabajo por tarea (% de las horas-persona)")
    st.caption("Cada barra es el 100% del trabajo de un rol en un turno, repartido entre las tareas que lo componen; el N parcial de cada tarea suma el N de la tabla principal. "
               "Es carga PROMEDIO (no modela colas).")
    base = st.radio("Porcentaje sobre", ["la carga del rol en ese turno", "la carga de todo el turno (suma de roles)"], horizontal=True, key="est_pct_base")
    col = "pct_rol" if base.startswith("la carga del rol") else "pct_turno"
    filas = [r for r in est["carga_por_tarea"] if r["horas_persona"] > 1e-9]
    if not filas:
        st.info("Sin horas de trabajo con los volúmenes elegidos.")
        return
    tabla = pd.DataFrame([{"Turno": r["turno"], "Rol": r["rol"], "Tarea": r["tarea"], "Función": r["funcion"],
                           "Volumen por día": f"{r['volumen']:,.1f} {r['unidad']}".replace(",", "§").replace(".", ",").replace("§", "."),
                           "Tiempo por unidad (s)": round(r["tiempo_estandar_s"], 1), "Horas-persona": round(r["horas_persona"], 1),
                           "N parcial": None if (r["turno"], r["rol"]) in SIN_ESTIMAR else round(r["n_parcial"], 2), "% de la carga": round(100 * r[col], 1)} for r in filas])
    grupos = []
    for r in filas:
        k = (r["turno"], r["rol"]) if col == "pct_rol" else (r["turno"], "todos los roles")
        if k not in grupos:
            grupos.append(k)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(grupos) + 1.2))
    for i, g in enumerate(grupos):
        izq = 0.0
        for r in filas:
            if ((r["turno"], r["rol"]) if col == "pct_rol" else (r["turno"], "todos los roles")) != g:
                continue
            w = 100 * r[col]
            ax.barh(i, w, left=izq, color=_COLOR_TAREA.get(r["proceso"], "#777777"), edgecolor="#13281D", height=0.62)
            if w >= 7:
                ax.text(izq + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8, color="white")
            izq += w
    ax.set_yticks(range(len(grupos)), [f"turno {t} · {rol}" for t, rol in grupos], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de las horas-persona", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    usados = [p for p in _COLOR_TAREA if any(r["proceso"] == p for r in filas)]
    nombres = {r["proceso"]: r["tarea"] for r in filas}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=_COLOR_TAREA[p]) for p in usados], [nombres[p] for p in usados], fontsize=7, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Ver la tabla por tarea"):
        st.dataframe(tabla.style.format(precision=2), hide_index=True, width="stretch")


def _panel_carga_dia(est: dict, opt: dict | None = None):
    """Carga de trabajo del ROL en el día y del PROCESO en el día (suma de los tres turnos), con sus gráficos. Sale de `est['carga_por_tarea']`: no calcula nada nuevo."""
    filas = [r for r in est["carga_por_tarea"] if r["horas_persona"] > 1e-9]
    if not filas:
        return
    df = pd.DataFrame(filas)
    total_h = df.horas_persona.sum()
    rol = df.groupby("rol").agg(horas=("horas_persona", "sum"), n=("n_parcial", "sum")).reset_index()
    for t_ in config.TURNOS:
        rol[f"N turno {t_}"] = [df[(df.rol == r_) & (df.turno == t_)].n_parcial.sum() for r_ in rol.rol]
    for t_, r_ in SIN_ESTIMAR:                      # sin dotación estimada de Operario A y B: solo horas
        rol.loc[rol.rol == r_, f"N turno {t_}"] = np.nan
    rol["n"] = rol[[f"N turno {t_}" for t_ in config.TURNOS]].sum(axis=1, skipna=True)
    rol["% de la carga del día"] = 100 * rol.horas / total_h
    st.markdown("#### Carga de trabajo del día")
    pr = df.groupby(["proceso", "tarea", "rol"]).agg(vol=("volumen", "sum"), horas=("horas_persona", "sum"), n=("n_parcial", "sum")).reset_index().sort_values("horas", ascending=False)
    pr["pct"] = 100 * pr.horas / total_h
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    base = np.zeros(len(rol))
    for t_, col_ in zip(config.TURNOS, ("#B7E4C7", "#5FD38A", "#1F9D55")):
        vals_ = np.array([(0 if (t_, r_) in SIN_ESTIMAR else (opt["n"].get((t_, r_), 0) if opt else rol.loc[i_, f"N turno {t_}"])) for i_, r_ in enumerate(rol.rol)], dtype=float)     # personas enteras (PuLP)
        ax.bar(rol.rol, vals_, bottom=base, color=col_, label=f"turno {t_}", edgecolor="#13281D")
        base += vals_
    for i_, b_ in enumerate(base):
        ax.text(i_, b_ + 0.15, f"{b_:g}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("personas (dotación entera)" if opt else "personas-equivalentes (N)", fontsize=8)
    ax.set_title("Carga del rol en el día, por turno", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    q = pr.sort_values("horas", ascending=True)
    ax.barh(q.tarea, q.pct, color=[_COLOR_TAREA.get(x, "#777777") for x in q.proceso])
    for i_, v_ in enumerate(q.pct):
        ax.text(v_ + 0.5, i_, f"{v_:.0f}%", va="center", fontsize=7)
    ax.set_xlabel("% de las horas-persona del día", fontsize=8)
    ax.set_title("Carga por proceso en el día", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    c2.pyplot(fig)
    plt.close(fig)
    with st.expander("Ver la tabla por proceso"):
        st.dataframe(pr.drop(columns="proceso").rename(columns={"tarea": "Proceso", "rol": "Rol", "vol": "Volumen por día", "horas": "Horas-persona del día", "n": "N del día", "pct": "% de la carga del día"}).round(2),
                     hide_index=True, width="stretch")


@st.dialog("Manual de usuario", width="large")
def _dialogo_manual():
    """Manual corto de la pestaña «Capacidad y simulación»: flujo normal, qué hace cada apartado y qué significa cada cosa."""
    st.markdown("""
**¿Para qué sirve?** Dices cuántas cajas se venden en el día y la herramienta responde **cuántos grueros y pickers necesita cada turno**, y luego lo **prueba simulando el día**.

#### Flujo normal (5 pasos)
1. **Elige la demanda**: Baja, Media o Alta, o escribe un número de cajas.
2. **Revisa la dotación actual** (cuadro de la derecha). Si no es la de hoy, pulsa **Editar dotación actual** y confirma.
3. **Pulsa ✅ Confirmar y calcular.** Nada cambia hasta que lo pulses.
4. **Lee la dotación óptima**: personas por rol y turno, comparadas con la real.
5. **Simula el día** al final de la página: elige cómo se trabaja en el patio, la dotación y pulsa **▶ Simular**.

#### Qué hace cada apartado
| Apartado | Qué muestra |
|---|---|
| **Demanda** | Cajas facturadas del día. Baja, Media y Alta salen de 821 días reales de venta. |
| **Dotación actual** | Personas que hay hoy. Se usa para comparar y para simular «Real». |
| **Parámetros editables** (opcional) | Tareas por día de cada proceso y turno, y tiempo por tarea. Sirve para probar «qué pasaría si». Si no tocas nada, se usa el modelo. |
| **Dotación óptima** | La **menor cantidad de personas enteras** que alcanza para el trabajo del turno. Verde/rojo = cuánto sobra o falta contra la real. |
| **Gráfico de barras** | Necesario (verde) contra real (gris), por rol y turno. |
| **Carga de trabajo por tarea / del día** | En qué se van las horas de cada rol: picking, carga, recepción, reabastos, etc. |
| **Simular el día** | Repite el día muchas veces con su variabilidad (camiones que llegan a distintas horas, tareas que duran más o menos) y dice si la dotación saca el trabajo. |
| **Comparación de las corridas** | Cada simulación queda guardada para compararla con las siguientes. |

#### Qué significa cada cosa
- **Turnos**: A 08–16 · B 16–24 · C 00–08 (el picking es de noche, en el C).
- **Gruero**: opera la grúa (recepción, almacenamiento, reabasto, carga de camiones). **Picker/operario**: arma los pedidos.
- **Horas de un turno**: 8 h menos colación y charla = 7 h 10 min. Nadie trabaja el 100% de ese tiempo: la **utilización** es la parte ocupada (0,80; en picking, la medida).
- **Carga fraccional** (p. ej. 2,6 personas): el trabajo dividido por las horas de una persona. La dotación óptima la redondea hacia arriba.
- **Réplica**: un día simulado. 200 réplicas = 200 días distintos con el mismo volumen.
- **Día viable**: el día cumple en al menos el **90% de las réplicas** (umbral). Que no llegue al 90% no quiere decir que el trabajo no salga: quiere decir que algunos días se atrasa.
- **Trabajo que pasa al turno siguiente**: lo que no alcanzó a empezar y queda para el turno siguiente (C→A, A→B). Se acepta hasta el 5% de las horas del turno que lo recibe.
- **Hora 99% del picking**: hora en que el turno C termina el 99% de los pedidos. Bien si es antes de las 08:00.
- **Uso %**: parte del turno en que los grueros o pickers están trabajando.
- **Escenarios del patio**: el gruero hace todo, o hay **apoyo** (escanea y traslada, sin grúa) o **polivalentes** (además descargan, cargan y hacen el retorno de fletero). Ver el botón **¿Qué hace cada quien?** en la simulación.
- **➕ Agregar 1 persona**: si el día no es viable, prueba con una persona más en el rol que falla.
- **Dotación a simular**: la real, la mínima del modelo o **Manual** (se escribe en una ventana y se confirma). La tabla es solo para mirar.
- **Límites**: la demanda va de 0 a 70.000 cajas. Ningún turno puede tener más de **6 grueros**, porque hay 6 grúas en el CD: si el modelo pide más, aparece un aviso rojo y no se deja simular.

#### Consejos
- Para comparar escenarios, usa **la misma demanda y las mismas réplicas** (los días simulados son los mismos y la diferencia se debe solo al escenario).
- 200 réplicas sirven para explorar; los resultados finales usan 1.000 (en «Opciones avanzadas»).
- **Restablecer valores** vuelve todo a lo del modelo.
""")


def tab_capacidad():
    st.subheader("Estimador de capacidad necesaria")
    st.caption("Ingresar una demanda (cajas facturadas) y obtener la dotación necesaria por turno y rol.")
    if not (RAIZ / "salidas" / "limpio" / "demanda_diaria.csv").exists():
        st.warning("Falta `salidas/limpio/demanda_diaria.csv` con septiembre: correr `python -m modelo.sales_loader`.")
        return
    escenarios = capacity.escenarios_estimador()
    dias = capacity.dias_historicos()
    params = _parametros()
    defs = {d["clave"]: d for d in capacity.ajustes_por_defecto()}

    b_man, b_res, _ = st.columns([1, 1, 4])
    if b_man.button("📖 Manual de usuario", key="est_manual", width="stretch"):
        _dialogo_manual()
    b_res.button("Restablecer valores", on_click=_reset_estimador, key="est_reset", width="stretch")
    with st.container(border=True):
        c_dem, c_dot = st.columns([3, 1.1])
        with c_dem:
            demanda, desc, fill_real = _demanda_de_entrada("est_a", "Demanda", escenarios, dias)
        with c_dot:
            dconf = _dotacion_confirmada()
            st.markdown("**Dotación actual**")
            for rol_ in config.ROLES_OPTIMIZACION:
                tt_ = [t_ for t_ in config.TURNOS if (t_, rol_) in dconf]
                nom_ = "Pickers" if rol_ == "Operario" else f"{rol_}s"
                st.caption(f"{nom_} {'/'.join(tt_)}: " + " / ".join(str(dconf[(t_, rol_)]) for t_ in tt_))
            if st.button("Editar dotación actual", key="est_btn_dot", width="stretch"):
                _dialogo_dotacion()
        sem = None
        regimen = "combinado"                      # 2026-10-23: el combinado es el definitivo (decisión del usuario); no se elige
        with st.expander("Parámetros editables (valor por defecto y fuente)"):
            ef_p = float(defs["eficiencia_picking"]["valor"])      # utilización de picking y del resto, y tiempo del turno: fijos en los valores del modelo (no se editan en la pantalla)
            ef_r = float(defs["eficiencia_resto"]["valor"])
            charla_m, colacion_m = 20.0, 30.0
            h_ef = capacity.Ajustes(charla_min=charla_m, colacion_min=colacion_m).horas_efectivas()
            st.markdown("**Horas efectivas del turno**")
            st.caption(f"8 h − {colacion_m:.0f} min de colación − {charla_m:.0f} min de charla = **{h_ef:.2f} h** de presencia productiva por persona "
                       f"(7 h 10). Horas de tareas por persona = {h_ef:.2f} h × utilización: resto de procesos (ρ = {ef_r:.2f}) **{h_ef * ef_r:.2f} h**; "
                       f"picking (η = {ef_p:.3f} sobre 7 h 50, equivale a {capacity.Ajustes().eficiencia_picking_efectiva():.3f} sobre 7 h 10) **{h_ef * capacity.Ajustes().eficiencia_picking_efectiva():.2f} h**.".replace(".", ","))
            tr_ov, ot_ov = [], []
            parciales_on, hojas_sel = True, ()      # reabasto de cajas = log dirigido + movimientos parciales; hojas de planificadores no se suman (duplicarían)
            st.markdown("**Tareas por día (editable)**")
            st.caption("**Total del día** = suma de los tres turnos. Escribe las tareas de **A, B y C** directamente (sumar o restar cambia el total y los porcentajes solos), "
                       "o usa **Cambiar %** para fijar el reparto: pones el % de A, B y C (deben sumar 100%) y se redistribuye el total. Cada proceso va en su unidad (líneas, tareas, camiones, trailers).")
            vm = capacity.volumenes_modelo(demanda, capacity.Ajustes(regimen=regimen, hojas=hojas_sel, con_parciales=parciales_on, semana_dotacion=sem), params)
            vol_ov = []
            tot_dia = 0.0
            sufijo = f"{regimen}_{int(demanda)}"
            hdr = st.columns([2.1, 1.0, 0.95, 0.95, 0.95, 1.5, 0.9])
            for col_, txt in zip(hdr, ("**Proceso (unidad)**", "**Total del día**", "**Turno A**", "**Turno B**", "**Turno C**", "**% A · B · C**", "")):
                col_.markdown(txt)
            for p_ in ("picking", "reabasto_pallet", "reabasto_caja", "recepcion_chico", "recepcion_grande", "almacenamiento", "carga_camion", "carga_cl", "recogida_pallets", "traslado_mixtos", "retorno_fletero", "retorno_fletero_gruero", "merma"):
                por_t = {r["turno"]: r for r in vm if r["proceso"] == p_}
                if not por_t:
                    continue
                r0 = next(iter(por_t.values()))
                claves = {t_: f"est_tv_{p_}_{t_}_{sufijo}" for t_ in config.TURNOS}
                for t_ in por_t:
                    if claves[t_] not in st.session_state:
                        st.session_state[claves[t_]] = round(float(por_t[t_]["modelo"]), 1)
                fila_ = st.columns([2.1, 1.0, 0.95, 0.95, 0.95, 1.5, 0.9])
                fila_[0].markdown(f"{r0['nombre']} ({r0['unidad']})")
                vals = {}
                for i_, t_ in enumerate(config.TURNOS):
                    if t_ in por_t:
                        vals[t_] = fila_[2 + i_].number_input(f"{p_} turno {t_}", min_value=0.0, step=1.0, key=claves[t_], label_visibility="collapsed")
                    else:
                        vals[t_] = 0.0
                        fila_[2 + i_].caption("no ocurre")
                total = sum(vals.values())
                tot_dia += total
                fila_[1].markdown(f"**{total:,.1f}**".replace(",", "§").replace(".", ",").replace("§", "."))
                fila_[5].caption(" · ".join(f"{100 * vals[t_] / total:.0f}%" if total > 0 else "—" for t_ in config.TURNOS))
                fila_[6].button("Cambiar %", key=f"est_btnpct_{p_}_{sufijo}", on_click=_alternar_pct, args=(f"est_modopct_{p_}_{sufijo}",))
                if st.session_state.get(f"est_modopct_{p_}_{sufijo}"):
                    cp_ = st.columns([2.1, 1.0, 0.95, 0.95, 0.95, 1.5, 0.9])
                    cp_[0].caption(f"Nuevo reparto del total ({total:,.1f}): % de cada turno, debe sumar 100%".replace(",", "§").replace(".", ",").replace("§", "."))
                    pk_ = {t_: f"est_pctnuevo_{p_}_{t_}_{sufijo}" for t_ in por_t}
                    for t_ in por_t:
                        if pk_[t_] not in st.session_state:
                            st.session_state[pk_[t_]] = round(100 * vals[t_] / total, 1) if total > 0 else 0.0
                    nuevos = {}
                    for i_, t_ in enumerate(config.TURNOS):
                        if t_ in por_t:
                            nuevos[t_] = cp_[2 + i_].number_input(f"% {p_} {t_}", min_value=0.0, max_value=100.0, step=1.0, key=pk_[t_], label_visibility="collapsed")
                    suma_ = sum(nuevos.values())
                    ok_ = abs(suma_ - 100.0) < 0.05
                    cp_[5].caption(f"Suma: {suma_:.1f}%" + ("" if ok_ else " — debe ser 100%"))
                    cp_[6].button("Aplicar", key=f"est_aplicar_{p_}_{sufijo}", disabled=not ok_, type="primary",
                                  on_click=_aplicar_pct, args=(claves, pk_, total, f"est_modopct_{p_}_{sufijo}"))
                if any(abs(vals[t_] - round(float(por_t[t_]["modelo"]), 1)) > 1e-9 for t_ in por_t):
                    for t_ in por_t:
                        vol_ov.append((p_, t_, float(vals[t_])))
            st.metric("Total de tareas del día (suma de todas las unidades)", f"{tot_dia:,.0f}".replace(",", "."))
            st.markdown("**Tiempo estándar por tarea (segundos por unidad)**" if not MOSTRAR_TOBE else
                        "**Tiempo estándar por tarea (segundos por unidad) — editable.** Vacío = no cambia. **TO-BE** es el tiempo que "
                        "debería tomar con la propuesta de mejora y se compara contra el actual en el resultado. Es un «qué pasaría si», no una medición.")
            st.caption("Escribe otro tiempo para probar «qué pasaría si» (vale para los tres turnos). El gris de la derecha es el valor del modelo; **Restablecer valores** lo devuelve.")
            tm_rows = capacity.tiempos_modelo(params, capacity.Ajustes())
            hdr = st.columns([2.2, 1.1, 1.0, 1.2] if MOSTRAR_TOBE else [2.2, 1.1, 1.0])
            for col_, txt in zip(hdr, ("**Proceso (rol)**", "**Tiempo (s)**", "**Modelo**", "**TO-BE (s)**")[:4 if MOSTRAR_TOBE else 3]):
                col_.markdown(txt)
            t_edit, t_tobe = {}, {}
            for r_ in tm_rows:
                fila_ = st.columns([2.2, 1.1, 1.0, 1.2] if MOSTRAR_TOBE else [2.2, 1.1, 1.0])
                fila_[0].markdown(f"{r_['nombre']} ({r_['rol']})")
                seg_ = r_["t_C"] if len({round(r_[f"t_{t_}"], 6) for t_ in config.TURNOS}) == 1 else None      # el modelo usa el mismo tiempo en los tres turnos
                ref_ = seg_ if seg_ is not None else r_["t_A"]
                k_ = f"est_tiempo_{r_['proceso']}"
                if k_ not in st.session_state:
                    st.session_state[k_] = round(float(ref_), 1)
                v_ = fila_[1].number_input(f"Tiempo {r_['nombre']}", min_value=0.0, step=5.0, key=k_, label_visibility="collapsed")
                fila_[2].caption(f"{ref_:,.1f} s".replace(",", "§").replace(".", ",").replace("§", ".") if seg_ is not None
                                 else " · ".join(f"{t_} {r_[f't_{t_}']:,.0f} s".replace(",", ".") for t_ in config.TURNOS))
                if abs(float(v_) - round(float(ref_), 1)) > 1e-9:
                    t_edit[r_["proceso"]] = float(v_)
                fila_ = fila_[:2] + fila_[3:]          # la columna TO-BE (si se muestra) queda en la posición 2
                w_ = fila_[2].number_input(f"Tiempo TO-BE {r_['nombre']}", min_value=0.0, value=None, step=5.0, key=f"est_tb_{r_['proceso']}", label_visibility="collapsed") if MOSTRAR_TOBE else None
                if w_ is not None:
                    t_tobe[r_["proceso"]] = float(w_)
    t_dic = {}                                # 2026-10-21: el picking usa los 6,92 min por lista de Diana (capacity.TIEMPOS_FINALES)
    t_dic.update(t_edit)                      # lo que el usuario escribe en la grilla manda
    t_over = tuple(t_dic.items())
    dot_ov = [(t_, r_, v_) for (t_, r_), v_ in _dotacion_confirmada().items() if int(v_) != int(config.personas_turno(t_, r_))]     # la del diálogo «Dotación actual»
    ajustes = capacity.Ajustes(regimen=regimen, eficiencia_picking=ef_p, eficiencia_resto=ef_r, tiempos_override=t_over, semana_dotacion=sem,
                                hojas=hojas_sel, con_parciales=parciales_on, volumenes_turno=tuple(vol_ov),
                                charla_min=charla_m, colacion_min=colacion_m, traslados_min=tuple(tr_ov), otras_h=tuple(ot_ov), picking_excedente_a=False, dotacion_real=tuple(dot_ov), recepcion_por_camion=True, recepcion_pct_planificador=0.0)
    # CONFIRMAR: los cambios de los parámetros NO recalculan el resultado hasta pulsar el botón (se muestra el último resultado confirmado).
    tobe_items = tuple(sorted(t_tobe.items()))
    actual = (demanda, ajustes, tobe_items, desc, fill_real)
    confirmar = st.button("✅ Confirmar y calcular", type="primary", key="est_confirmar")
    if confirmar or "est_conf" not in st.session_state:
        st.session_state["est_conf"] = actual
    elif st.session_state.pop("est_dot_nuevo", False):     # la dotación se confirma en su ventana: se aplica sola, sin tocar otros cambios pendientes
        d0, a0, tb0, ds0, fr0 = st.session_state["est_conf"]
        st.session_state["est_conf"] = (d0, replace(a0, dotacion_real=tuple(dot_ov)), tb0, ds0, fr0)
    st.session_state.pop("est_dot_nuevo", None)
    if st.session_state["est_conf"] != actual:
        st.warning("Hay cambios sin confirmar: abajo se ve el último resultado confirmado. Pulsa «Confirmar y calcular» para aplicarlos al modelo.")
    demanda, ajustes, tobe_items, desc, fill_real = st.session_state["est_conf"]
    prop_items = ()
    est = _estimar_cache(demanda, ajustes, prop_items, params)
    ajustes_tobe = replace(ajustes, tiempos_override=tuple({**dict(ajustes.tiempos_override), **dict(tobe_items)}.items())) if tobe_items else None
    # TO-BE de la recepción: el planificador escanea siempre (el gruero solo descarga)
    ajustes_tobe = replace(ajustes_tobe or ajustes, recepcion_pct_planificador=1.0)
    est_tobe = _estimar_cache(demanda, ajustes_tobe, prop_items, params) if ajustes_tobe else None

    st.markdown(f"### Resultado: {demanda:,.0f} cajas facturadas ({desc})".replace(",", "."))
    if fill_real is not None:
        st.caption(f"Fill rate real de ese día: {fill_real:.1%}".replace(".", ","))
    for a in est["advertencias"]:
        if any(x in a for x in ("Cota inferior", "cota inferior", "Régimen", "régimen", "regimen")):
            continue
        if "qué pasaría si" in a:              # avisos de volúmenes / tiempo de turno editados: ocultos por ahora (se vuelven a mostrar quitando esta línea)
            continue
        st.warning(a)
    opt = _panel_optimizacion(demanda, ajustes, params)       # bloque PuLP justo debajo de la tabla; devuelve la dotación entera n[(turno, rol)]
    roles = [f for f in est["filas"] if f["tipo"] == "rol" and (f["turno"], f["recurso"]) not in SIN_ESTIMAR]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(roles))
    nec = [(opt["n"][(f["turno"], f["recurso"])] if opt else f["necesarios"]) for f in roles]      # personas enteras (PuLP), no la fracción del estimador
    ax.bar(x - 0.2, nec, 0.38, color="#3DBE6B", label="necesario (entero)" if opt else "necesario", edgecolor="#13281D")
    for x_, v_ in zip(x, nec):
        ax.text(x_ - 0.2, v_ + 0.1, f"{v_:g}", ha="center", fontsize=8, fontweight="bold")
    ax.bar(x + 0.2, [f["reales"] for f in roles], 0.38, color="#9aa5b1", label="real", edgecolor="#13281D")
    ax.set_xticks(x, [f"{f['recurso']}\nturno {f['turno']}" for f in roles], fontsize=8)
    ax.set_ylabel("personas", fontsize=8)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    st.pyplot(fig)
    plt.close(fig)

    _panel_carga_por_tarea(est)
    _panel_carga_dia(est, opt)

    _seccion_simulacion(demanda, ajustes, est, params, ajustes_tobe, opt)

    csv_ =pd.DataFrame(capacity.estimacion_a_filas(est)).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar resultados (CSV, con los supuestos en columnas)", csv_,
                       file_name=f"estimacion_{int(demanda)}_cajas.csv", mime="text/csv", key="est_csv")


_CSS = """
<style>
:root{--verde:#3DBE6B;--verde-osc:#8FE0A9;--panel:#13281D;--borde:#27493A;}
.block-container{padding-top:4.2rem;max-width:1500px;}
[data-testid="stAppDeployButton"],footer{display:none !important;}
h2,h3,h4{color:var(--verde-osc);}
h4{border-bottom:2px solid var(--borde);padding-bottom:4px;margin-top:1.2rem;}
.ccu-hero{background:linear-gradient(120deg,#052A1A 0%,#0B5A33 55%,#1F8F4E 100%);color:#fff;padding:20px 28px 16px;border-radius:14px;margin-bottom:14px;border:1px solid #1E5C3A;box-shadow:0 6px 18px rgba(0,0,0,.45);}
.ccu-hero h1{font-size:1.65rem;margin:0;color:#fff;font-weight:700;letter-spacing:.2px;padding:0;}
.ccu-hero p{margin:4px 0 0;opacity:.92;font-size:.95rem;}
.ccu-chip{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.28);border-radius:999px;padding:2px 12px;margin:10px 8px 0 0;font-size:.78rem;}
/* casillas numéricas visibles: el fondo del campo era igual al del panel y solo se veían los botones − y + */
[data-testid="stNumberInput"] [data-baseweb="input"],[data-testid="stNumberInput"] [data-baseweb="base-input"]{background:#1E3A2B !important;border:1px solid #4F7D66 !important;border-radius:6px !important;}
[data-testid="stNumberInput"] input{background:#1E3A2B !important;color:#E4EEE8 !important;}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{border-color:#3DBE6B !important;}
[data-testid="stNumberInput"] input::placeholder{color:#7F9A8B !important;opacity:1;}
.st-key-seccion [role="radiogroup"]{gap:6px;flex-wrap:wrap;}
.st-key-seccion label{background:var(--panel);border:1px solid var(--borde);border-radius:999px;padding:3px 14px;cursor:pointer;transition:background .15s;}
.st-key-seccion label:hover{background:#1B3B2A;}
.st-key-seccion label > div:first-child{display:none;}
.st-key-seccion label:has(input:checked){background:var(--verde);border-color:var(--verde);}
.st-key-seccion label:has(input:checked) p{color:#06210F;font-weight:700;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--borde);border-left:5px solid var(--verde);border-radius:10px;padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,.35);}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--borde);border-radius:10px;}
[data-testid="stDataFrame"]{border:1px solid var(--borde);border-radius:8px;overflow:hidden;}
.stButton>button,.stDownloadButton>button{border-radius:8px;font-weight:600;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:12px;}
</style>
"""


def _estilo_global():
    """Identidad visual oscura (verdes corporativos sobre fondo verde casi negro) y paleta de los gráficos. Solo CSS ligero: no afecta el rendimiento."""
    from cycler import cycler
    st.markdown(_CSS, unsafe_allow_html=True)
    plt.rcParams.update({"axes.prop_cycle": cycler(color=["#3DBE6B", "#9BE564", "#5FD38A", "#F0A04B", "#4FC3A1", "#A99BE0", "#9DB0A5"]),
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#5B7A69", "axes.grid": True, "grid.color": "#24402F", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlecolor": "#BFEBCF", "axes.labelcolor": "#C9DCCF", "xtick.color": "#C9DCCF", "ytick.color": "#C9DCCF",
                         "text.color": "#E4EEE8", "figure.facecolor": "#13281D", "axes.facecolor": "#13281D", "savefig.facecolor": "#13281D", "legend.frameon": False,
                         "legend.labelcolor": "#C9DCCF"})


def _encabezado():
    st.markdown("""<div class="ccu-hero"><h1>Modelo de capacidad y simulación operacional</h1>
<p>Centro de Distribución Coquimbo · Transportes CCU · Capstone EIC 2026-2 (UCN)</p></div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="CCU Coquimbo — Capacidad y simulación", page_icon="🟢", layout="wide")
    _estilo_global()
    _encabezado()
    tab_capacidad()


if __name__ == "__main__":
    main()
