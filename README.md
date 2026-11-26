# Capacidad y simulación — CD Coquimbo

App de Streamlit del modelo de capacidad y la simulación de eventos discretos del Centro de Distribución Coquimbo (Transportes CCU), Capstone EIC 2026-2, UCN.

Hace lo mismo que la pestaña «Capacidad y simulación» del dashboard del proyecto:

1. **Modelo de capacidad.** Dada una demanda (cajas facturadas por día), calcula las horas de trabajo de cada tarea y las personas necesarias por turno (A, B, C) y rol (gruero, operario).
2. **Dotación mínima entera.** Un modelo de optimización (MILP, PuLP + CBC) entrega la dotación entera mínima que cumple la carga, con la restricción de 6 grúas.
3. **Simulación del día.** Simulación de eventos discretos (SimPy) del día completo con tiempos variables, prioridades, llegadas de camiones y traspaso del trabajo no iniciado al turno siguiente. Dice si la dotación es viable (al menos 90% de las réplicas cumple; el umbral se puede cambiar en «Opciones avanzadas»).

Se pueden editar la demanda, los tiempos, los volúmenes, la utilización y la dotación.

## Cómo correrla

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Se abre en el navegador (por defecto http://localhost:8501). Una simulación de 200 réplicas tarda alrededor de 30 a 60 segundos.

## Cómo publicarla en la nube (Streamlit Community Cloud, gratis)

1. Subir esta carpeta a un repositorio de GitHub (puede ser privado):
   ```bash
   git init && git add . && git commit -m "App de capacidad y simulación"
   git branch -M main
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```
2. Entrar a https://share.streamlit.io con la cuenta de GitHub → **Create app** → elegir el repositorio, rama `main` y archivo principal `app.py`.
3. En **Advanced settings**, elegir Python **3.12 o 3.13** (las versiones de `requirements.txt` existen para las dos). No necesita secretos.
4. **Deploy**. La primera vez instala las bibliotecas (unos minutos). Después queda en una dirección `https://<nombre>.streamlit.app`.

Si el repositorio es privado, en la app (Settings → Sharing) se elige quién puede verla. Al actualizar el modelo: volver a exportar, `git commit` y `git push`; la app se actualiza sola.

En la nube gratuita la simulación es más lenta que en un computador (1 núcleo compartido): 200 réplicas pueden tardar 1 a 2 minutos. Conviene no pasar de 1.000 réplicas ahí.

## Qué contiene

| Carpeta o archivo | Contenido |
|---|---|
| `app.py` | La interfaz (las funciones de la pestaña, copiadas del dashboard del proyecto). |
| `modelo/` | Los módulos del modelo de capacidad, el MILP y la simulación. |
| `salidas/limpio/` | Datos ya procesados que usa el modelo: parámetros de tiempos, volúmenes por turno, demanda diaria, llegadas de camiones. |
| `datos/congelado.json` | Valores tomados de las planillas originales del proyecto (que no se incluyen). |

No incluye datos crudos del WMS, ventas por línea ni planillas de terreno, ni nombres de personas: los usuarios del WMS aparecen como códigos.

## Cómo se actualiza

Esta carpeta se genera desde el repositorio del proyecto con `python -m modelo.exportar_app`. Si cambia el modelo o los datos, se vuelve a generar ahí; no conviene editar `modelo/` a mano en esta carpeta.
