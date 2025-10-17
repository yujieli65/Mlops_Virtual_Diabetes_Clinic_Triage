from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42


def load_data():
    """Load scikit-learn diabetes dataset."""
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    return X, y


def train_and_save(kind="linear", out_path="artifacts/model.joblib"):
    """Train model and save pipeline + metadata."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    if kind == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))

    # 保存 artifact
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": {"kind": kind}}, out_path)

    return {"out_path": out_path, "rmse": rmse, "meta": {"kind": kind}}
