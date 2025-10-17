# src/model.py
from pathlib import Path
import joblib
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42

def load_data():
    """
    Load the sklearn diabetes dataset as a pandas DataFrame (as_frame=True).
    Returns:
        X (DataFrame): features (10 columns)
        y (Series): target (progression index)
    """
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    return X, y

def build_pipeline(kind: str = "linear") -> Pipeline:
    """
    Build a sklearn Pipeline with a StandardScaler and the requested model.
    Supported kinds: "linear" (LinearRegression)
    """
    if kind == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    return pipe

def train_and_save(kind: str = "linear", out_path: str = "artifacts/model.joblib") -> Dict[str, Any]:
    """
    Train the pipeline deterministically and save a joblib artifact that contains
    {"pipeline": pipeline, "meta": {"kind": kind, "rmse": rmse}}.

    Returns a dict with out_path, rmse, meta, and the trained pipeline.
    """
    np.random.seed(RANDOM_STATE)
    X, y = load_data()

    # deterministic train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(kind=kind)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))

    # ensure output directory exists
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"kind": kind, "rmse": rmse}
    joblib.dump({"pipeline": pipe, "meta": meta}, str(out_path))

    return {"out_path": str(out_path), "rmse": rmse, "meta": meta, "pipeline": pipe}

def load_model(path: str = "artifacts/model.joblib") -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load a saved model artifact and return (pipeline, meta).
    Raises exceptions if loading fails.
    """
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError(f"Invalid model artifact format at {path}")
    pipeline = obj["pipeline"]
    meta = obj.get("meta", {})
    return pipeline, meta
