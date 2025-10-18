from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(title="Virtual Diabetes Clinic - Triage API")

pipeline = None
meta = None
try:
    model_bundle = joblib.load(MODEL_PATH)
    pipeline = model_bundle.get("pipeline")
    meta = model_bundle.get("meta", {})
    if pipeline is None:
        raise ValueError("Loaded artifact does not contain 'pipeline'")
    model_version = meta.get("kind", "unknown")
except Exception as e:
    print("Error loading model:", e)
    pipeline = None
    meta = None
    model_version = None


class DiabetesPayload(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    """Return model health and version."""
    if pipeline is None:
        return {"status": "error", "model_version": None}
    return {
        "status": "ok",
        "model_version": model_version,
    }


@app.post("/predict")
def predict(payload: DiabetesPayload):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    try:
        data = np.array([
            [
                payload.age,
                payload.sex,
                payload.bmi,
                payload.bp,
                payload.s1,
                payload.s2,
                payload.s3,
                payload.s4,
                payload.s5,
                payload.s6
            ]
        ])
        pred = pipeline.predict(data)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
