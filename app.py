from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
import pandas as pd   # <-- importante

class TareaIn(BaseModel):
    asignado_id: int | None = None
    prioridad: str | None = ""
    puntos: int | None = 0
    dias_hasta_vencer: int | None = 0
    titulo: str | None = ""
    descripcion: str | None = ""

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "./modelo_retraso.joblib")
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(t: TareaIn):
    try:
        texto = f"{t.titulo or ''} {t.descripcion or ''}"

        payload = {
            "asignado_id": t.asignado_id if t.asignado_id is not None else -1,
            "prioridad": t.prioridad or "",
            "puntos": t.puntos or 0,
            "dias_hasta_vencer": t.dias_hasta_vencer or 0,
            "texto": texto
        }
        X = pd.DataFrame([payload])   # <-- aquí está la clave

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        else:
            # fallback raro (por si cambiaste de modelo)
            import math
            score = float(model.decision_function(X)[0])
            proba = 1.0 / (1.0 + math.exp(-score))

        suggestion = "subir_prioridad" if proba > 0.7 else ("revisar" if proba > 0.5 else "ok")
        return {"risk": proba, "suggestion": suggestion}
    except Exception as e:
        # para depurar más fácil
        raise HTTPException(status_code=500, detail=str(e))
