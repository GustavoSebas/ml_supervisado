import pandas as pd
import joblib
from db import get_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

SQL = """
WITH ultima_col AS (
  SELECT c.proyecto_id, MAX(c.orden) AS max_orden
  FROM columna_kanban c
  GROUP BY c.proyecto_id
),
fecha_fin AS (
  SELECT
    h.tarea_id,
    MIN(h.cambiado_en) AS fecha_fin
  FROM historial_estado_tarea h
  JOIN columna_kanban c
    ON c.id = h.a_columna_id
  JOIN tarea t
    ON t.id = h.tarea_id
   AND t.proyecto_id = c.proyecto_id
  JOIN ultima_col uc
    ON uc.proyecto_id = c.proyecto_id
   AND uc.max_orden = c.orden
  GROUP BY h.tarea_id
)
SELECT
  t.id                       AS tarea_id,
  COALESCE(u.id, -1)         AS asignado_id,
  COALESCE(t.prioridad, '')  AS prioridad,
  COALESCE(t.puntos, 0)      AS puntos,
  COALESCE(t.titulo, '')     AS titulo,
  COALESCE(t.descripcion, '')AS descripcion,
  t.fecha_limite,
  ff.fecha_fin,
  CASE
    WHEN ff.fecha_fin IS NOT NULL
     AND t.fecha_limite IS NOT NULL
     AND DATE(ff.fecha_fin) > t.fecha_limite
    THEN 1 ELSE 0
  END AS retraso
FROM tarea t
LEFT JOIN fecha_fin ff ON ff.tarea_id = t.id
LEFT JOIN Usuario u    ON u.id = t.asignado_a
WHERE t.fecha_limite IS NOT NULL;
"""

def load_df():
    engine = get_engine()
    return pd.read_sql(SQL, engine)

def build_pipe():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["prioridad"]),
        ("num", SimpleImputer(strategy="median"), ["asignado_id","puntos","dias_hasta_vencer"]),
        ("txt", TfidfVectorizer(max_features=4000), "texto"),
    ])
    return Pipeline([("prep", pre), ("model", GradientBoostingClassifier())])

if __name__ == "__main__":
    df = load_df()
    df["dias_hasta_vencer"] = (
        pd.to_datetime(df["fecha_limite"]) -
        pd.to_datetime(df["fecha_fin"]).fillna(pd.to_datetime(df["fecha_limite"]))
    ).dt.days
    df["texto"] = (df["titulo"].fillna("") + " " + df["descripcion"].fillna(""))


    y = df["retraso"].astype(int)

    X = df[["asignado_id","prioridad","puntos","dias_hasta_vencer","texto"]]


    if y.nunique() < 2:
        raise SystemExit("No hay suficientes clases en 'retraso' (necesito 0 y 1).")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipe().fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print("F1:", round(f1_score(yte, pred), 3))
    joblib.dump(pipe, "modelo_retraso.joblib")
