import pandas as pd
import joblib
import numpy as np
from db import get_engine
from sklearn.utils import check_array
# Cargar modelos
# Intentar cargar bundle si existe; si no, cargar los tres archivos individuales como fallback.
import os

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




print("=" * 70)
print("DIAGNÓSTICO DEL MODELO")
print("=" * 70)

# 1. Verificar features del modelo
print("\n📊 Features esperadas por el modelo:")
print(f"Total: {len(clf.feature_names_in_)}")
print("\nPrimeras 20 features:")
for i, feat in enumerate(clf.feature_names_in_[:20]):
    print(f"  {i+1}. {feat}")

# 2. Verificar usuarios en el encoder
print("\n👥 Usuarios en el encoder:")
usuarios_encoder = user_enc.categories_[0]
print(f"Total usuarios: {len(usuarios_encoder)}")
print(f"IDs: {list(usuarios_encoder)}")

# 3. Consultar tarea real de la BD
print("\n🔍 Consultando tarea ID=1 de la base de datos:")
engine = get_engine()
query = """
SELECT 
    t.id,
    t.titulo,
    t.asignado_a,
    u.name as usuario_nombre,
    t.prioridad,
    t.puntos,
    t.titulo,
    t.descripcion,
    t.fecha_limite,
    (SELECT MIN(h.cambiado_en) FROM historial_estado_tarea h WHERE h.tarea_id = t.id) as fecha_inicio,
    (SELECT MIN(h2.cambiado_en)
     FROM historial_estado_tarea h2
     JOIN columna_kanban c2 ON c2.id = h2.a_columna_id
     JOIN (SELECT proyecto_id, MAX(orden) AS max_orden FROM columna_kanban GROUP BY proyecto_id) uc 
         ON uc.proyecto_id = c2.proyecto_id AND uc.max_orden = c2.orden
     WHERE h2.tarea_id = t.id) as fecha_fin
FROM tarea t
LEFT JOIN usuario u ON u.id = t.asignado_a
WHERE t.id = 1
"""
df_tarea = pd.read_sql(query, engine)

if df_tarea.empty:
    print("❌ No se encontró la tarea ID=1")
else:
    tarea = df_tarea.iloc[0]
    print(f"\nTarea: {tarea['titulo']}")
    print(f"Asignado a: ID={tarea['asignado_a']}, Nombre={tarea['usuario_nombre']}")
    print(f"Prioridad: {tarea['prioridad']}")
    print(f"Puntos: {tarea['puntos']}")
    print(f"Fecha inicio: {tarea['fecha_inicio']}")
    print(f"Fecha fin: {tarea['fecha_fin']}")
    print(f"Fecha límite: {tarea['fecha_limite']}")
    
    # Calcular si tuvo retraso
    if pd.notna(tarea['fecha_fin']) and pd.notna(tarea['fecha_limite']):
        fecha_fin = pd.to_datetime(tarea['fecha_fin']).date()
        fecha_limite = pd.to_datetime(tarea['fecha_limite']).date()
        tuvo_retraso = fecha_fin > fecha_limite
        dias_retraso = (fecha_fin - fecha_limite).days
        print(f"\n¿Tuvo retraso? {'SÍ' if tuvo_retraso else 'NO'}")
        print(f"Días de retraso: {dias_retraso}")

# 4. Verificar historial del usuario
print("\n📈 Historial del usuario asignado (ID=2):")
query_historial = """
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
    t.id,
    t.titulo,
    t.asignado_a,
    t.puntos,
    t.prioridad,
    fi.fecha_inicio,
    ff.fecha_fin,
    t.fecha_limite,
    CASE
        WHEN ff.fecha_fin IS NOT NULL AND t.fecha_limite IS NOT NULL
        THEN CASE WHEN DATE(ff.fecha_fin) > DATE(t.fecha_limite) THEN 1 ELSE 0 END
        ELSE NULL
    END AS retraso
FROM tarea t
LEFT JOIN fecha_inicio fi ON fi.tarea_id = t.id
LEFT JOIN fecha_fin ff ON ff.tarea_id = t.id
WHERE t.asignado_a = 2
    AND ff.fecha_fin IS NOT NULL
ORDER BY t.id
"""
df_historial = pd.read_sql(query_historial, engine)
# Tasa de retraso por usuario (empírica)
if not df_historial.empty:
    tasas_por_usuario = df_historial.groupby('asignado_a')['retraso'].agg(['sum','count'])
    tasas_por_usuario['rate'] = tasas_por_usuario['sum'] / tasas_por_usuario['count']
    print("\n📌 Tasa empírica de retrasos por usuario (usuarios con al menos 1 tarea completada):")
    print(tasas_por_usuario[['sum','count','rate']].sort_values('rate', ascending=False).to_string())
    print("\nTasa de retraso global (dataset usado para historial): {:.2f}%".format(df_historial['retraso'].mean()*100))

if not df_historial.empty:
    print(f"Total tareas completadas: {len(df_historial)}")
    retrasos = df_historial['retraso'].sum()
    tasa = retrasos / len(df_historial) * 100
    print(f"Tareas con retraso: {retrasos} ({tasa:.1f}%)")
    print("\nDetalle de tareas:")
    print(df_historial[['id', 'titulo', 'puntos', 'prioridad', 'retraso']].to_string(index=False))
else:
    print("No hay tareas completadas para este usuario")

# 5. Simular predicción manualmente
print("\n🤖 Simulando predicción con los datos reales:")

def build_features_debug(prioridad, puntos, asignado_id, titulo="", descripcion=""):
    texto_completo = f"{titulo} {descripcion}".strip()
    longitud_texto = len(texto_completo)
    
    X_base = pd.DataFrame({
        'puntos': [puntos],
        'dias_estimados': [float(puntos)],
        'longitud_texto': [longitud_texto]
    })
    
    prioridades_posibles = ['Alta', 'Baja', 'Media']
    X_cat = pd.DataFrame(0, index=[0], columns=[f'prioridad_{p}' for p in prioridades_posibles])
    if prioridad in prioridades_posibles:
        X_cat[f'prioridad_{prioridad}'] = 1
    else:
        X_cat['prioridad_Media'] = 1
    
    try:
        uenc = user_enc.transform([[asignado_id]])
        df_uenc = pd.DataFrame(uenc, columns=[f"user_{i}" for i in range(uenc.shape[1])])
    except:
        n_users = len(user_enc.categories_[0])
        df_uenc = pd.DataFrame(0, index=[0], columns=[f"user_{i}" for i in range(n_users)])
    
    X = pd.concat([X_base, X_cat, df_uenc], axis=1)
    X.columns = X.columns.astype(str)
    
    print(f"\n📋 Features construidas:")
    print(f"  - puntos: {puntos}")
    print(f"  - dias_estimados: {float(puntos)}")
    print(f"  - longitud_texto: {longitud_texto}")
    print(f"  - prioridad: {prioridad}")
    print(f"  - asignado_id: {asignado_id}")
    print(f"  - Columnas prioridad activas: {[col for col in X_cat.columns if X_cat[col].iloc[0] == 1]}")
    print(f"  - Columnas usuario activas: {[col for col in df_uenc.columns if df_uenc[col].iloc[0] == 1]}")
    
    X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X

if not df_tarea.empty:
    tarea = df_tarea.iloc[0]
    X_test = build_features_debug(
        tarea['prioridad'],
        int(tarea['puntos']),
        int(tarea['asignado_a']),
        str(tarea['titulo']),
        str(tarea['descripcion'])
    )
    
    print("\n🔬 Vector de features final usado para la predicción (columnas y valores):")
    pd.options.display.max_rows = 200
    print(X_test.T)   # Transpuesta para ver "nombre_columna: valor"
    print("\nEncoder usuarios - categories_:", user_enc.categories_)
    print("Feature names esperadas por el modelo (len={}):".format(len(clf.feature_names_in_)))
    print(list(clf.feature_names_in_))

    cats = list(user_enc.categories_[0])
    print("\nMapeo esperado columnas usuario:")
    for i, uid in enumerate(cats):
        print(f"  user_{i} -> usuario_id {uid}")

    print(f"\n🎯 Predicción:")
    prob = clf.predict_proba(X_test)[0][1]
    effort = regr.predict(X_test)[0]
    print(f"  - Probabilidad de retraso: {prob:.6f} ({prob*100:.2f}%)")
    print(f"  - Esfuerzo estimado: {effort:.2f} días")
    
    if prob < 0.3:
        print("\n⚠️  PROBLEMA DETECTADO: La probabilidad es muy baja")
        print("Posibles causas:")
        print("  1. El modelo no se entrenó correctamente")
        print("  2. Los datos de entrenamiento no tienen el patrón esperado")
        print("  3. El usuario no está en los datos de entrenamiento")
        print("\n💡 Solución: Re-ejecutar el entrenamiento")
        print("  python train_extended.py")

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print("\n✅ Siguiente paso:")
print("  1. Verifica que el seeder haya generado correctamente los datos")
print("  2. Re-ejecuta el entrenamiento: python train_extended.py")
print("  3. Verifica que el modelo aprenda los patrones de retraso por usuario")
print("=" * 70)

def build_features_for_df(df_src, user_encoder):
    """
    Construye features para un DataFrame de tareas (puede ser muchas filas).
    Maneja casos donde las columnas titulo/descripcion podrían faltar o ser escalares.
    """
    df_work = df_src.copy()

    # Asegurar índices y longitud
    n = len(df_work)
    if n == 0:
        return pd.DataFrame(columns=list(clf.feature_names_in_))

    # Helpers para obtener Series seguras
    def safe_series(df, col):
        if col in df.columns:
            s = df[col]
            # si es un valor escalar (no Series), convertir a Series repetida
            if not hasattr(s, 'fillna'):
                return pd.Series([str(s)] * n, index=df.index)
            return s.fillna('').astype(str)
        else:
            return pd.Series([''] * n, index=df.index)

    titulo_s = safe_series(df_work, 'titulo')
    descripcion_s = safe_series(df_work, 'descripcion')

    # Features numéricas base
    df_work['puntos'] = pd.to_numeric(df_work['puntos'], errors='coerce').fillna(0).astype(float)
    df_work['dias_estimados'] = df_work['puntos'].astype(float)
    df_work['texto_completo'] = (titulo_s + ' ' + descripcion_s).str.strip()
    df_work['longitud_texto'] = df_work['texto_completo'].str.len().astype(float)

    # One-hot prioridad (asegurar las 3 columnas)
    prioridades = ['Alta', 'Baja', 'Media']
    for p in prioridades:
        df_work[f'prioridad_{p}'] = (df_work.get('prioridad', '').astype(str) == p).astype(float)

    # Construir X_num con orden conocido
    X_num = df_work[['puntos', 'dias_estimados', 'longitud_texto',
                     'prioridad_Alta', 'prioridad_Baja', 'prioridad_Media']].copy()

    # Codificar usuarios con el encoder — usar DataFrame con la misma columna usada en training
    try:
        asignados = df_work.get('asignado_a', df_work.get('asignado_id', pd.Series([-1]*n)))
        udf = pd.DataFrame({'asignado_id': asignados.astype(int).values}, index=df_work.index)
        uenc = user_encoder.transform(udf)
        ucols = [f'user_{i}' for i in range(uenc.shape[1])]
        df_uenc = pd.DataFrame(uenc, columns=ucols, index=df_work.index)
    except Exception as e:
        # Fallback: vector de ceros si falla el encoder
        n_users = len(user_encoder.categories_[0])
        df_uenc = pd.DataFrame(0, index=df_work.index, columns=[f'user_{i}' for i in range(n_users)])
        print("Warning: error al codificar usuarios:", e)

    # Concatenar y reindexar a las columnas esperadas por el modelo
    X_full = pd.concat([X_num, df_uenc], axis=1)
    X_full = X_full.reindex(columns=list(clf.feature_names_in_), fill_value=0)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_full
