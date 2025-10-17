# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np
from typing import Optional

# Default relative path; can be overridden with MODEL_PATH env var
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(title="Virtual Diabetes Clinic - Triage API")

# Try to load pipeline at import/startup. If loading fails, keep app running
pipeline = None
meta = None
try:
    model_bundle = joblib.load(MODEL_PATH)
    # Expecting {"pipeline": pipeline, "meta": {...}}
    pipeline = model_bundle.get("pipeline")
    meta = model_bundle.get("meta", {})
    if pipeline is None:
        raise ValueError("Loaded artifact does not contain 'pipeline'")
    model_version = meta.get("kind", "unknown")
except Exception as e:
    # Print exception (visible in logs). Keep pipeline=None so endpoints can return JSON errors.
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
    """
    Return model health and version. If model failed to load, return status "error".
    """
    if pipeline is None:
        return {"status": "error", "model_version": None}
    return {"status": "ok", "model_version": model_version}

@app.post("/predict")
def predict(payload: DiabetesPayload):
    """
    Accepts JSON with the 10 diabetes features and returns a numeric prediction:
    { "prediction": <float> }
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    try:
        # maintain feature order exactly as dataset: age, sex, bmi, bp, s1..s6
        data = np.array([[
            payload.age, payload.sex, payload.bmi, payload.bp,
            payload.s1, payload.s2, payload.s3, payload.s4,
            payload.s5, payload.s6
        ]])
        pred = pipeline.predict(data)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        # Return JSON error for bad inputs or internal errors
        raise HTTPException(status_code=400, detail=str(e))
