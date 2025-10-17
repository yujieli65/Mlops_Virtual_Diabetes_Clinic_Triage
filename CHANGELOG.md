# CHANGELOG.md

## v0.1
- **Model**: Baseline pipeline using StandardScaler + LinearRegression
- **Metrics**: RMSE on held-out test split ≈ 53.85
- **API**: FastAPI endpoints:
  - GET /health → returns model health and version
  - POST /predict → returns numeric prediction for diabetes progression
- **Docker**: Self-contained image built as diabetes-api:v0.1 including trained model (artifacts/model.joblib)
- **Reproducibility**: Deterministic train/test split using random_state=42
