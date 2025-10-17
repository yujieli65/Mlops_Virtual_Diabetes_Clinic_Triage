# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_predict():
    payload = {
      "age": 0.02,"sex": -0.04464164,"bmi": 0.06169621,"bp": -0.02187235,
      "s1": -0.03482149,"s2": 0.04340185,"s3": -0.00259226,"s4": 0.00371151,
      "s5": 0.01466408,"s6": -0.01813516
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
