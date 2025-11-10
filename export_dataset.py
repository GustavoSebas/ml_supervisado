import argparse
import os
import pandas as pd
from db import get_engine
from dotenv import load_dotenv
load_dotenv()

def query_tasks(engine):
    """
    Consulta que obtiene todas las tareas con su historial completo.
    Calcula:
    - fecha_inicio: Primera entrada en el historial (primer movimiento)
    - fecha_fin: Primera vez que la tarea llegó a la columna final (máximo orden)
    """
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
        t.titulo,
        t.descripcion,
        COALESCE(t.asignado_a, -1) AS asignado_id,
        COALESCE(t.prioridad, 'Media') AS prioridad,
        COALESCE(t.puntos, 1) AS puntos,
        t.fecha_limite,
        t.creado_en,
        fi.fecha_inicio,
        ff.fecha_fin,
        -- Calcular días reales (solo si está completada)
        CASE
            WHEN ff.fecha_fin IS NOT NULL AND fi.fecha_inicio IS NOT NULL 
            THEN TIMESTAMPDIFF(DAY, fi.fecha_inicio, ff.fecha_fin)
            ELSE NULL
        END AS dias_reales_calculados
    FROM tarea t
    LEFT JOIN fecha_inicio fi ON fi.tarea_id = t.id
    LEFT JOIN fecha_fin ff ON ff.tarea_id = t.id
    WHERE t.fecha_limite IS NOT NULL
    ORDER BY t.id;
    """
    print("📊 Ejecutando consulta SQL...")
    return pd.read_sql(SQL, engine)

def compute_features(df):
    """
    Calcula todas las features necesarias para ML:
    1. Features básicas (días estimados, días reales, retraso)
    2. Features derivadas (longitud de texto, estado)
    3. Features agregadas por usuario (historial del usuario)
    """
    print("🔧 Calculando features...")
    
    # ========== 1. NORMALIZAR FECHAS ==========
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    df['fecha_fin'] = pd.to_datetime(df['fecha_fin'])
    df['fecha_limite'] = pd.to_datetime(df['fecha_limite'])
    df['creado_en'] = pd.to_datetime(df['creado_en'])
    
    # Si no hay fecha_inicio, usar creado_en
    df['fecha_inicio'] = df['fecha_inicio'].fillna(df['creado_en'])
    
    # ========== 2. FEATURES BÁSICAS ==========
    
    # Días estimados: basado en puntos (1 punto = 1 día)
    df['dias_estimados'] = df['puntos'].astype(float)
    
    # Días reales: usar el calculado de SQL si existe, sino calcular
    if 'dias_reales_calculados' in df.columns:
        df['dias_reales'] = df['dias_reales_calculados'].astype(float)
    else:
        df['dias_reales'] = (df['fecha_fin'] - df['fecha_inicio']).dt.total_seconds() / (24 * 3600)
    
    # ========== 3. CALCULAR RETRASO ==========
    # IMPORTANTE: Retraso se determina comparando fecha_fin con fecha_limite
    # NO por dias_reales vs dias_estimados
    
    def calcular_retraso(row):
        """
        Retraso = 1 si fecha_fin > fecha_limite
        Retraso = 0 si fecha_fin <= fecha_limite o no está terminada
        """
        if pd.isna(row['fecha_fin']) or pd.isna(row['fecha_limite']):
            return None  # No se puede determinar (tarea no completada)
        
        # Comparar fechas (convertir a date para comparación precisa)
        fecha_fin_date = row['fecha_fin'].date()
        fecha_limite_date = row['fecha_limite'].date()
        
        return 1 if fecha_fin_date > fecha_limite_date else 0
    
    df['retraso'] = df.apply(calcular_retraso, axis=1)
    
    # Retraso en días (cuántos días tarde/temprano)
    def calcular_dias_retraso(row):
        if pd.isna(row['fecha_fin']) or pd.isna(row['fecha_limite']):
            return None
        
        delta = (row['fecha_fin'].date() - row['fecha_limite'].date()).days
        return delta  # Positivo = tarde, Negativo = temprano, 0 = a tiempo
    
    df['retraso_dias'] = df.apply(calcular_dias_retraso, axis=1)
    
    # ========== 4. ESTADO DE COMPLETITUD ==========
    df['completada'] = df['fecha_fin'].notna().astype(int)
    df['en_tiempo'] = df['retraso'].apply(lambda x: 1 if x == 0 else 0 if x == 1 else None)
    
    # ========== 5. FEATURES DE TEXTO ==========
    df['texto_completo'] = (df['titulo'].fillna('') + ' ' + df['descripcion'].fillna('')).str.strip()
    df['longitud_texto'] = df['texto_completo'].str.len()
    df['tiene_descripcion'] = (df['descripcion'].fillna('').str.len() > 0).astype(int)
    
    # ========== 6. FEATURES AGREGADAS POR USUARIO ==========
    # Solo considerar tareas completadas para calcular historial del usuario
    df_completadas = df[df['completada'] == 1].copy()
    
    if len(df_completadas) > 0:
        agg_usuario = df_completadas.groupby('asignado_id').agg(
            tareas_completadas=('tarea_id', 'count'),
            avg_dias_reales=('dias_reales', 'mean'),
            avg_puntos=('puntos', 'mean'),
            tasa_retraso=('retraso', lambda x: x.mean() if x.notna().sum() > 0 else 0),
            total_retrasos=('retraso', 'sum'),
            avg_retraso_dias=('retraso_dias', lambda x: x.mean() if x.notna().sum() > 0 else 0)
        ).reset_index()
        
        # Unir con el dataframe principal
        df = df.merge(agg_usuario, on='asignado_id', how='left', suffixes=('', '_agg'))
    else:
        # Si no hay tareas completadas, crear columnas con ceros
        df['tareas_completadas'] = 0
        df['avg_dias_reales'] = 0.0
        df['avg_puntos'] = 0.0
        df['tasa_retraso'] = 0.0
        df['total_retrasos'] = 0
        df['avg_retraso_dias'] = 0.0
    
    # Rellenar NaN para usuarios sin historial
    for col in ['tareas_completadas', 'avg_dias_reales', 'avg_puntos', 
                'tasa_retraso', 'total_retrasos', 'avg_retraso_dias']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # ========== 7. FEATURES ADICIONALES ==========
    # Diferencia entre estimado y real (solo para completadas)
    df['diferencia_estimado_real'] = df['dias_reales'] - df['dias_estimados']
    
    # Días desde creación hasta inicio
    df['dias_hasta_inicio'] = (df['fecha_inicio'] - df['creado_en']).dt.total_seconds() / (24 * 3600)
    
    # Días disponibles (desde creación hasta límite)
    df['dias_disponibles'] = (df['fecha_limite'] - df['creado_en']).dt.total_seconds() / (24 * 3600)
    
    return df

def generate_report(df):
    """Genera un reporte estadístico de los datos"""
    print("\n" + "="*70)
    print("📈 REPORTE ESTADÍSTICO")
    print("="*70)
    
    total = len(df)
    completadas = df['completada'].sum()
    en_progreso = total - completadas
    
    print(f"\n📊 ESTADO GENERAL:")
    print(f"   Total de tareas:      {total}")
    print(f"   Completadas:          {completadas} ({completadas/total*100:.1f}%)")
    print(f"   En progreso:          {en_progreso} ({en_progreso/total*100:.1f}%)")
    
    if completadas > 0:
        df_comp = df[df['completada'] == 1]
        con_retraso = df_comp['retraso'].sum()
        sin_retraso = len(df_comp) - con_retraso
        
        print(f"\n⏰ ANÁLISIS DE RETRASOS (solo tareas completadas):")
        print(f"   Sin retraso:          {sin_retraso} ({sin_retraso/completadas*100:.1f}%)")
        print(f"   Con retraso:          {con_retraso} ({con_retraso/completadas*100:.1f}%)")
        print(f"   Promedio días reales: {df_comp['dias_reales'].mean():.2f} días")
        print(f"   Promedio retraso:     {df_comp['retraso_dias'].mean():.2f} días")
        
        # Análisis por usuario
        if 'asignado_id' in df_comp.columns:
            usuarios = df_comp.groupby('asignado_id').agg({
                'tarea_id': 'count',
                'retraso': lambda x: (x == 1).sum(),
                'dias_reales': 'mean'
            }).round(2)
            usuarios.columns = ['Total tareas', 'Con retraso', 'Promedio días']
            usuarios['% Retraso'] = (usuarios['Con retraso'] / usuarios['Total tareas'] * 100).round(1)
            
            print(f"\n👥 ANÁLISIS POR USUARIO:")
            print(usuarios.to_string())
        
        # Análisis por prioridad
        if 'prioridad' in df_comp.columns:
            prioridad = df_comp.groupby('prioridad').agg({
                'tarea_id': 'count',
                'retraso': lambda x: (x == 1).sum(),
                'dias_reales': 'mean'
            }).round(2)
            prioridad.columns = ['Total', 'Retrasos', 'Promedio días']
            
            print(f"\n🎯 ANÁLISIS POR PRIORIDAD:")
            print(prioridad.to_string())
    
    print("\n" + "="*70)

def save_to_csv(df, output_path):
    """Guarda el dataframe a CSV con las columnas más relevantes"""
    print(f"\n💾 Guardando datos en CSV...")
    
    # Seleccionar columnas para exportar
    columnas_export = [
        # Identificación
        'tarea_id', 'titulo', 'descripcion',
        
        # Asignación y configuración
        'asignado_id', 'prioridad', 'puntos', 
        
        # Fechas
        'creado_en', 'fecha_inicio', 'fecha_fin', 'fecha_limite',
        
        # Métricas principales
        'dias_estimados', 'dias_reales', 'retraso', 'retraso_dias',
        
        # Estado
        'completada', 'en_tiempo',
        
        # Features de texto
        'texto_completo', 'longitud_texto', 'tiene_descripcion',
        
        # Features agregadas del usuario
        'tareas_completadas', 'avg_dias_reales', 'avg_puntos',
        'tasa_retraso', 'total_retrasos', 'avg_retraso_dias',
        
        # Features adicionales
        'diferencia_estimado_real', 'dias_hasta_inicio', 'dias_disponibles'
    ]
    
    # Filtrar solo las columnas que existen
    columnas_existentes = [col for col in columnas_export if col in df.columns]
    df_export = df[columnas_existentes].copy()
    
    # Guardar con formato de fecha legible
    df_export.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    
    print(f"   ✓ Archivo guardado: {output_path}")
    print(f"   ✓ Filas exportadas: {len(df_export)}")
    print(f"   ✓ Columnas: {len(columnas_existentes)}")

def main(output_path, limit=None):
    """Función principal"""
    print("="*70)
    print("   EXPORTACIÓN DE TAREAS HISTÓRICAS PARA MACHINE LEARNING")
    print("="*70)
    
    # 1. Conectar y consultar
    engine = get_engine()
    print("\n🔌 Conectando a la base de datos...")
    df = query_tasks(engine)
    
    if df.empty:
        print("\n❌ No se obtuvieron datos.")
        print("   Verifica que:")
        print("   - La base de datos esté accesible")
        print("   - Existan tareas con fecha_limite")
        print("   - El seeder se haya ejecutado correctamente")
        return
    
    print(f"   ✓ {len(df)} tareas obtenidas")
    
    # 2. Limitar si se especificó (para pruebas)
    if limit is not None:
        df = df.head(limit)
        print(f"   ⚠️  Limitando a {limit} registros (modo prueba)")
    
    # 3. Calcular features
    df_processed = compute_features(df)
    
    # 4. Generar reporte
    generate_report(df_processed)
    
    # 5. Guardar CSV
    save_to_csv(df_processed, output_path)
    
    print("\n" + "="*70)
    print("   ✅ EXPORTACIÓN COMPLETADA EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exportar tareas históricas desde MySQL a CSV para Machine Learning."
    )
    parser.add_argument(
        "--output", "-o", 
        default="tareas_historicas.csv", 
        help="Ruta de salida del archivo CSV (default: tareas_historicas.csv)"
    )
    parser.add_argument(
        "--limit", "-n", 
        type=int, 
        default=None, 
        help="Limitar número de filas para pruebas (opcional)"
    )
    args = parser.parse_args()
    
    main(args.output, args.limit)