# app_extended.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

import os

SQL_USER_STATS = """
WITH ultima_col AS (
  SELECT proyecto_id, MAX(orden) AS max_orden
  FROM columna_kanban GROUP BY proyecto_id
),
fecha_inicio AS (
  SELECT tarea_id, MIN(cambiado_en) AS fecha_inicio
  FROM historial_estado_tarea GROUP BY tarea_id
),
fecha_fin AS (
  SELECT h.tarea_id, MIN(h.cambiado_en) AS fecha_fin
  FROM historial_estado_tarea h
  JOIN columna_kanban c ON c.id = h.a_columna_id
  JOIN ultima_col uc ON uc.proyecto_id = c.proyecto_id AND uc.max_orden = c.orden
  GROUP BY h.tarea_id
)
SELECT
  COUNT(*) AS user_tasks_count,
  AVG(CASE WHEN DATE(ff.fecha_fin) > t.fecha_limite THEN 1 ELSE 0 END) AS user_delay_rate,
  AVG(TIMESTAMPDIFF(DAY, fi.fecha_inicio, ff.fecha_fin)) AS user_avg_days_real
FROM tarea t
JOIN fecha_inicio fi ON fi.tarea_id = t.id
JOIN fecha_fin ff ON ff.tarea_id = t.id
WHERE t.asignado_a = :uid;

"""

def fetch_user_stats(uid:int):
    try:
        eng = get_engine()
        df = pd.read_sql(SQL_USER_STATS, eng, params={"uid": uid})
        if df.empty or pd.isna(df.loc[0, 'user_tasks_count']):
            return 0.0, 0.0, 0.0
        r = df.loc[0]
        return float(r['user_tasks_count'] or 0), float(r['user_delay_rate'] or 0), float(r['user_avg_days_real'] or 0)
    except Exception:
        return 0.0, 0.0, 0.0
    
bundle_path = "models_bundle.joblib"
if os.path.exists(bundle_path):
    bundle = joblib.load(bundle_path)
    # Extraer
    clf = bundle.get('clf', None)
    regr = bundle.get('regr', None)
    user_enc = bundle.get('user_encoder', None)
    user_cats = bundle.get('user_cats', None)
    print("✅ Cargado models_bundle.joblib")
else:
    # Fallback: cargar modelos individuales (si existen)
    print("⚠️ models_bundle.joblib no encontrado — intento cargar archivos individuales...")
    if os.path.exists("modelo_retraso_mlp.joblib") and os.path.exists("modelo_esfuerzo_mlp.joblib") and os.path.exists("encoder_usuarios.joblib"):
        clf = joblib.load("modelo_retraso_mlp.joblib")
        regr = joblib.load("modelo_esfuerzo_mlp.joblib")
        user_enc = joblib.load("encoder_usuarios.joblib")
        user_cats = getattr(user_enc, 'categories_', None)
        print("✅ Cargados modelo_retraso_mlp.joblib, modelo_esfuerzo_mlp.joblib y encoder_usuarios.joblib")
    else:
        missing = []
        for f in ["models_bundle.joblib", "modelo_retraso_mlp.joblib", "modelo_esfuerzo_mlp.joblib", "encoder_usuarios.joblib"]:
            if not os.path.exists(f):
                missing.append(f)
        raise FileNotFoundError(f"No se encontraron archivos necesarios: {missing}. Genera/coloca los modelos antes de ejecutar el diagnóstico.")




def build_features(prioridad, puntos, dias_estimados, asignado_id, titulo="", descripcion=""):
    """
    Construye features alineadas con el entrenamiento.
    Features: puntos, dias_estimados, longitud_texto, prioridad (one-hot), asignado_id (one-hot)
    """
    user_tasks_count, user_delay_rate, user_avg_days_real = (0.0, 0.0, 0.0)
    if asignado_id is not None and asignado_id != -1:
        user_tasks_count, user_delay_rate, user_avg_days_real = fetch_user_stats(asignado_id)

    X_base = pd.DataFrame({
        'puntos': [puntos],
        'dias_estimados': [dias_estimados],
        'longitud_texto': [longitud_texto],
        'user_tasks_count': [user_tasks_count],
        'user_delay_rate': [user_delay_rate],
        'user_avg_days_real': [user_avg_days_real]
    })

    # 2. Codificación de prioridad (one-hot)
    # Crear las 3 columnas posibles: prioridad_Alta, prioridad_Baja, prioridad_Media
    prioridades_posibles = ['Alta', 'Baja', 'Media']
    X_cat = pd.DataFrame(0, index=[0], columns=[f'prioridad_{p}' for p in prioridades_posibles])
    
    # Activar la prioridad correspondiente
    if prioridad in prioridades_posibles:
        X_cat[f'prioridad_{prioridad}'] = 1
    else:
        # Default a Media si no es válida
        X_cat['prioridad_Media'] = 1

    # 3. Codificación del usuario (one-hot)
    try:
        uenc = user_enc.transform(pd.DataFrame({'asignado_id':[asignado_id]}))
        df_uenc = pd.DataFrame(uenc, columns=[f"user_{i}" for i in range(uenc.shape[1])])
    except:
        # Si el usuario no existe en el encoder, crear vector de ceros
        n_users = len(user_enc.categories_[0])
        df_uenc = pd.DataFrame(0, index=[0], columns=[f"user_{i}" for i in range(n_users)])

    # 4. Concatenar todas las features
    X = pd.concat([X_base, X_cat, df_uenc], axis=1)

    # 5. Convertir todos los nombres de columna a string
    X.columns = X.columns.astype(str)

    # 6. Alinear con las columnas del modelo entrenado (rellenar faltantes con 0)
    X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
    
    # 7. Limpiar infinitos y NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X

class TaskIn(BaseModel):
    asignado_id: int | None = -1
    prioridad: str
    puntos: int
    dias_estimados: int | None = None
    titulo: str = ""
    descripcion: str = ""

@app.post("/predict_extended")
def predict_extended(t: TaskIn, candidates: list[int] | None = None):
    """
    Endpoint original mantenido.
    Predice riesgo y esfuerzo para una tarea, opcionalmente evaluando candidatos.
    """
    # Si no se especifica dias_estimados, usar puntos
    dias_estimados = t.dias_estimados if t.dias_estimados is not None else float(t.puntos)
    
    # Construir features para el usuario actual
    X = build_features(t.prioridad, t.puntos, dias_estimados, t.asignado_id, t.titulo, t.descripcion)
    
    # Predicción de probabilidad de retraso
    prob = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else None
    
    # Predicción de esfuerzo (días reales)
    effort = float(regr.predict(X)[0])
    
    # Fecha sugerida de entrega
    suggested_due = (datetime.now() + timedelta(days=round(effort))).strftime("%Y-%m-%d")

    # Generar sugerencia basada en la probabilidad de retraso
    suggestion = "mantener"
    if prob is not None:
        if prob > 0.8:
            suggestion = "reasignar"
        elif prob > 0.6:
            suggestion = "subir_prioridad"

    result = {
        "for_user": {
            "asignado_id": t.asignado_id,
            "risk": prob,
            "effort_days": effort,
            "suggested_due_date": suggested_due,
            "suggestion": suggestion
        }
    }

    # Si se pasan candidatos, evaluar cada uno y recomendar el mejor
    if candidates:
        candidates_res = []
        for u in candidates:
            Xu = build_features(t.prioridad, t.puntos, dias_estimados, u, t.titulo, t.descripcion)
            prob_u = float(clf.predict_proba(Xu)[0][1])
            effort_u = float(regr.predict(Xu)[0])
            candidates_res.append({
                "asignado_id": u,
                "risk": prob_u,
                "effort_days": effort_u,
                "suggested_due_date": (datetime.now() + timedelta(days=round(effort_u))).strftime("%Y-%m-%d")
            })
        
        # Ordenar por menor riesgo, luego por menor esfuerzo
        candidates_res = sorted(candidates_res, key=lambda x: (x['risk'], x['effort_days']))
        result['candidates'] = candidates_res
        
        # Sugerir reasignar al primer candidato si su riesgo es significativamente menor
        best = candidates_res[0]
        if best['risk'] + 0.15 < result['for_user']['risk']:  # umbral heurístico
            result['recommend_reassign_to'] = best['asignado_id']

    return result
