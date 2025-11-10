import pandas as pd
import numpy as np
import joblib
from db import get_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Mejor para datos pequeños
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')

# ========================================
# SQL QUERY - Extrae datos históricos
# ========================================
SQL = """
WITH ultima_col AS (
  SELECT c.proyecto_id, MAX(c.orden) AS max_orden
  FROM columna_kanban c
  GROUP BY c.proyecto_id
),
fecha_inicio AS (
  SELECT h.tarea_id, MIN(h.cambiado_en) AS fecha_inicio
  FROM historial_estado_tarea h
  GROUP BY h.tarea_id
),
fecha_fin AS (
  SELECT h.tarea_id, MIN(h.cambiado_en) AS fecha_fin
  FROM historial_estado_tarea h
  JOIN columna_kanban c ON c.id = h.a_columna_id
  JOIN ultima_col uc ON uc.proyecto_id = c.proyecto_id AND uc.max_orden = c.orden
  GROUP BY h.tarea_id
)
SELECT
  t.id AS tarea_id,
  COALESCE(t.asignado_a, -1) AS asignado_id,
  COALESCE(t.prioridad, 'Media') AS prioridad,
  COALESCE(t.puntos, 1) AS puntos,
  COALESCE(t.titulo, '') AS titulo,
  COALESCE(t.descripcion, '') AS descripcion,
  fi.fecha_inicio,
  ff.fecha_fin,
  t.fecha_limite,
  CASE
    WHEN ff.fecha_fin IS NOT NULL AND fi.fecha_inicio IS NOT NULL 
    THEN TIMESTAMPDIFF(DAY, fi.fecha_inicio, ff.fecha_fin)
    ELSE NULL
  END AS dias_reales
FROM tarea t
LEFT JOIN fecha_inicio fi ON fi.tarea_id = t.id
LEFT JOIN fecha_fin ff ON ff.tarea_id = t.id
WHERE t.fecha_limite IS NOT NULL
  AND ff.fecha_fin IS NOT NULL
  AND fi.fecha_inicio IS NOT NULL;
"""

def load_data():
    """Carga los datos desde la base de datos"""
    print("📊 Cargando datos desde la base de datos...")
    engine = get_engine()
    df = pd.read_sql(SQL, engine)
    print(f"   ✓ {len(df)} tareas cargadas")
    return df

def preprocess_data(df):
    """
    Preprocesa los datos para el entrenamiento:
    - Calcula features derivadas
    - Crea target de clasificación (retraso)
    - Limpia datos
    """
    print("\n🔧 Preprocesando datos...")
    
    # Feature: días estimados (basado en puntos)
    df['dias_estimados'] = df['puntos'].astype(float)
    
    # Feature: texto combinado
    df['texto_completo'] = df['titulo'].fillna('') + ' ' + df['descripcion'].fillna('')
    
    # Feature: longitud del texto (puede indicar complejidad)
    df['longitud_texto'] = df['texto_completo'].str.len()
    
    # Target de clasificación: ¿Hubo retraso?
    # Retraso = fecha_fin > fecha_limite
    df['fecha_fin_dt'] = pd.to_datetime(df['fecha_fin'])
    df['fecha_limite_dt'] = pd.to_datetime(df['fecha_limite'])
    df['retraso'] = (df['fecha_fin_dt'].dt.date > df['fecha_limite_dt'].dt.date).astype(int)
    
    # Target de regresión: días reales
    df['dias_reales'] = df['dias_reales'].astype(float)
    
    # Limpiar datos incompletos
    df_clean = df.dropna(subset=['dias_reales', 'asignado_id', 'prioridad'])
    
    print(f"   ✓ {len(df_clean)} tareas válidas para entrenamiento")
    print(f"   ✓ Distribución de retrasos:")
    print(f"      - Sin retraso: {(df_clean['retraso'] == 0).sum()} ({(df_clean['retraso'] == 0).sum()/len(df_clean)*100:.1f}%)")
    print(f"      - Con retraso: {(df_clean['retraso'] == 1).sum()} ({(df_clean['retraso'] == 1).sum()/len(df_clean)*100:.1f}%)")
    
    return df_clean

def build_features(df):
    """
    Construye la matriz de features para el modelo, incluyendo features históricas por usuario.
    """
    print("\n🏗️  Construyendo features...")

    # Features numéricas base + features históricas
    num_cols = ['puntos', 'dias_estimados', 'longitud_texto',
                'user_tasks_count', 'user_delay_rate', 'user_avg_days_real']
    # Asegurar existencia y tipos
    for c in ['puntos','dias_estimados','longitud_texto','user_tasks_count','user_delay_rate','user_avg_days_real']:
        if c not in df.columns:
            df[c] = 0
    X_num = df[num_cols].copy().astype(float)

    # One-hot encoding para prioridad (mantener columnas consistentes)
    X_prioridad = pd.get_dummies(df['prioridad'], prefix='prioridad', drop_first=False)
    for p in ['prioridad_Alta','prioridad_Baja','prioridad_Media']:
        if p not in X_prioridad.columns:
            X_prioridad[p] = 0.0
    X_prioridad = X_prioridad[['prioridad_Alta','prioridad_Baja','prioridad_Media']]

    # One-hot encoding para usuario (mantener comportamiento actual)
    user_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_user = user_encoder.fit_transform(df[['asignado_id']])
    X_user_df = pd.DataFrame(X_user,
                              columns=[f'user_{i}' for i in range(X_user.shape[1])],
                              index=df.index)

    # Concatenar
    X = pd.concat([X_num, X_prioridad, X_user_df], axis=1)
    X.columns = X.columns.astype(str)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"   ✓ {X.shape[1]} features creadas")
    return X, user_encoder


def train_models(X, y_clf, y_reg):
    print("\n🤖 Entrenando modelos...")

    # Evitar stratify si y_clf tiene una sola clase
    strat = y_clf if y_clf.nunique() > 1 else None
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=strat
    )

    print(f"   ✓ Train: {len(X_train)} muestras")
    print(f"   ✓ Test:  {len(X_test)} muestras")

    # Opción A: RandomForest (más estable y explicable)
    print("\n   🎯 Entrenando RandomForest clasificador (robusto con pocos datos)...")
    rf_clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_clf.fit(X_train, y_clf_train)

    # Calibrar probabilidades si quieres (opcional pero recomendable)

    # Intentar calibrar; si falla por falta de muestras/estratos, usar el RandomForest sin calibrar
    try:
        # elegir cv dinámico (no más que min clases en y_clf_train)
        unique_classes, counts = np.unique(y_clf_train, return_counts=True)
        min_class_count = counts.min() if len(counts) > 0 else 0
        # cv no mayor a 5 y no mayor que min_class_count (cada fold necesita al menos 1 muestra por clase)
        cv_used = min(5, max(2, int(min_class_count)))
        if cv_used < 2:
            raise ValueError("No hay suficientes muestras por clase para calibración con CV")
        # Usar el nombre de parámetro correcto 'estimator' (no base_estimator)
        clf = CalibratedClassifierCV(estimator=rf_clf, cv=cv_used).fit(X_train, y_clf_train)
        print(f"   ✓ Clasificador calibrado con cv={cv_used}")
    except Exception as e:
        print("⚠️ No se pudo calibrar el clasificador (falló CalibratedClassifierCV). Usando RandomForest sin calibrar. Error:", e)
        clf = rf_clf  # ya fue ajustado arriba


    # --- Evaluar clasificador: predicciones y probabilidades (si aplica) ---
    print("\n   📈 Resultados del Clasificador:")
    try:
        y_clf_pred = clf.predict(X_test)
    except Exception as e:
        # Si clf es CalibratedClassifierCV envuelto o RF, esto debería funcionar; si no, intentamos usar el estimador base
        print("⚠️ Error al predecir con el clasificador principal:", e)
        # si existe rf_clf como fallback, usarlo
        try:
            y_clf_pred = rf_clf.predict(X_test)
            print("   ✓ Usando rf_clf.predict como fallback para obtener predicciones.")
        except Exception as e2:
            raise RuntimeError("No se pudo obtener predicciones del clasificador: {}".format(e2))

    # Intentar obtener probabilidades (opcional)
    y_clf_proba = None
    try:
        if hasattr(clf, "predict_proba"):
            y_clf_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            # fallback a decision_function (no es probabilidad)
            y_clf_proba = clf.decision_function(X_test)
    except Exception as e:
        print("⚠️ No se pudieron obtener probabilidades:", e)
        y_clf_proba = None

    # Imprimir métricas
    print(classification_report(y_clf_test, y_clf_pred, target_names=['Sin retraso', 'Con retraso'], digits=3))
    cm = confusion_matrix(y_clf_test, y_clf_pred)
    print("   Matriz de confusión:")
    print(f"      [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"       [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

    # Estadísticas adicionales sobre probabilidades si existen
    if y_clf_proba is not None:
        import numpy as _np
        print("\n   Estadísticas de probabilidades predichas (si aplica):")
        print(f"      - media: {_np.mean(y_clf_proba):.3f}, mediana: {_np.median(y_clf_proba):.3f}, std: {_np.std(y_clf_proba):.3f}")
        # Opcional: mostrar calibración rápida
        try:
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_clf_test, y_clf_proba, n_bins=5, strategy='uniform')
            print("      - calibration curve (true_prob vs pred_prob):")
            for t, p in zip(prob_true, prob_pred):
                print(f"         true: {t:.3f}, pred: {p:.3f}")
        except Exception:
            pass

    # Si el clasificador interno tiene feature_importances_, muéstralas (útil con RandomForest)
    base_est = getattr(clf, 'base_estimator', None)
    if base_est is None:
        # Si no hay base_estimator, puede que clf sea ya el estimator (rf_clf fallback)
        base_est = clf
    try:
        if hasattr(base_est, 'feature_importances_'):
            import pandas as _pd
            fi = base_est.feature_importances_
            cols = X.columns
            imp_df = _pd.Series(fi, index=cols).sort_values(ascending=False)
            print("\n   🔎 Feature importances (top 10):")
            print(imp_df.head(10).to_string())
    except Exception as e:
        print("⚠️ No se pudieron calcular feature importances:", e)


    # ===== Regresor =====
    print("\n   📊 Entrenando regresor de esfuerzo (MLPRegressor)...")
    regr = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=False)
    regr.fit(X_train, y_reg_train)
    y_reg_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    print("\n   📈 Resultados del Regresor:")
    print(f"      MAE (Error Absoluto Medio): {mae:.2f} días")

    return clf, regr


def save_models(clf, regr, user_encoder):
    print("\n💾 Guardando modelos...")
    bundle = {
        'clf': clf,
        'regr': regr,
        'user_encoder': user_encoder,
        'user_cats': list(user_encoder.categories_[0])
    }
    joblib.dump(bundle, "models_bundle.joblib")
    print("   ✓ Modelos guardados en models_bundle.joblib")


def main():
    """Función principal"""
    print("=" * 60)
    print("   ENTRENAMIENTO DE MODELOS DE PREDICCIÓN DE TAREAS")
    print("=" * 60)
    
    # 1. Cargar datos
    df = load_data()
    
    if df.empty:
        raise SystemExit("❌ No hay datos disponibles. Ejecuta el seeder primero.")
    
    # 2. Preprocesar
    df_clean = preprocess_data(df)
    # === Estadísticas por usuario (features de historial) ===
    print("\n🔢 Calculando estadísticas históricas por usuario...")
    user_stats = df_clean.groupby('asignado_id').agg(
        user_tasks_count = ('tarea_id', 'count'),
        user_delays = ('retraso', 'sum'),
        user_delay_rate = ('retraso', 'mean'),
        user_avg_days_real = ('dias_reales', 'mean')
    ).fillna(0)

    # Normalizar/llenar NaNs
    user_stats['user_delay_rate'] = user_stats['user_delay_rate'].fillna(0.0)
    user_stats['user_avg_days_real'] = user_stats['user_avg_days_real'].fillna(df_clean['dias_reales'].mean())

    # Merge de vuelta al df_clean
    df_clean = df_clean.merge(user_stats, left_on='asignado_id', right_index=True, how='left')
    df_clean[['user_tasks_count','user_delays','user_delay_rate','user_avg_days_real']] = df_clean[['user_tasks_count','user_delays','user_delay_rate','user_avg_days_real']].fillna(0)
    print("   ✓ Estadísticas por usuario añadidas")

    
    if len(df_clean) < 10:
        raise SystemExit("❌ Datos insuficientes para entrenar. Se necesitan al menos 10 tareas completadas.")
    
    # 3. Construir features
    X, user_encoder = build_features(df_clean)
    
    # 4. Targets
    y_clf = df_clean['retraso'].astype(int)
    y_reg = df_clean['dias_reales'].astype(float)
    
    # 5. Entrenar
    clf, regr = train_models(X, y_clf, y_reg)
    
    # 6. Guardar
    save_models(clf, regr, user_encoder)
    
    print("\n" + "=" * 60)
    print("   ✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()