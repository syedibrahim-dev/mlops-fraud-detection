"""
FastAPI serving endpoint for fraud detection model.
Exposes /predict, /health, and /metrics (Prometheus).
"""

import os
import time
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

# ── Prometheus metrics ──────────────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["result"]          # labels: fraud / legit
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
FRAUD_RATE = Gauge(
    "fraud_rate_last_100",
    "Fraud rate over last 100 predictions"
)
MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently loaded model version",
    ["version"]
)

# ── Continual Learning metrics (updated after each retraining window) ────────
CL_PLASTICITY = Gauge(
    "cl_plasticity_score",
    "Plasticity: F1 on newest window (learning ability)",
    ["strategy"]
)
CL_STABILITY = Gauge(
    "cl_stability_score",
    "Stability: avg F1 on old windows (retention of past knowledge)",
    ["strategy"]
)
CL_FORGETTING = Gauge(
    "cl_forgetting_rate",
    "Forgetting: avg performance drop on old windows",
    ["strategy"]
)
DRIFT_DETECTED = Gauge(
    "drift_detected",
    "1 if drift was detected in latest window, 0 otherwise"
)
PSI_RATIO = Gauge(
    "drift_psi_ratio",
    "Fraction of features that fired PSI threshold"
)
KS_RATIO = Gauge(
    "drift_ks_ratio",
    "Fraction of features that fired KS threshold"
)

# Rolling window for fraud rate calculation
_recent_predictions: list[int] = []
MODEL_PATH = os.getenv("MODEL_PATH", "data/model.json")
MODEL_VERSION_TAG = os.getenv("MODEL_VERSION", "1")

app = FastAPI(title="Fraud Detection API")

# ── Load model ──────────────────────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
MODEL_VERSION.labels(version=MODEL_VERSION_TAG).set(1)

FEATURE_COUNT = 30  # 28 PCA features + Time + Amount


class PredictRequest(BaseModel):
    features: list[float]   # expects 30 values


class PredictResponse(BaseModel):
    fraud: bool
    probability: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION_TAG}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FEATURE_COUNT} features, got {len(req.features)}"
        )

    start = time.time()
    x = np.array(req.features).reshape(1, -1)
    proba = float(model.predict_proba(x)[0][1])
    is_fraud = proba >= 0.5
    elapsed = time.time() - start

    # Update Prometheus metrics
    REQUEST_LATENCY.observe(elapsed)
    label = "fraud" if is_fraud else "legit"
    PREDICTION_COUNT.labels(result=label).inc()

    # Update rolling fraud rate
    _recent_predictions.append(int(is_fraud))
    if len(_recent_predictions) > 100:
        _recent_predictions.pop(0)
    FRAUD_RATE.set(sum(_recent_predictions) / len(_recent_predictions))

    return PredictResponse(
        fraud=is_fraud,
        probability=round(proba, 4),
        model_version=MODEL_VERSION_TAG
    )


@app.post("/update_cl_metrics")
def update_cl_metrics(payload: dict):
    """
    Called by the retraining pipeline after each CL window to push
    plasticity/stability scores into Prometheus.
    """
    for strategy in ["finetune", "experience_replay", "sliding_window"]:
        if strategy in payload:
            s = payload[strategy]
            CL_PLASTICITY.labels(strategy=strategy).set(s.get("plasticity", 0))
            CL_STABILITY.labels(strategy=strategy).set(s.get("stability", 0))
            CL_FORGETTING.labels(strategy=strategy).set(s.get("forgetting", 0))

    if "drift" in payload:
        d = payload["drift"]
        DRIFT_DETECTED.set(1 if d.get("drift_confirmed") else 0)
        PSI_RATIO.set(d.get("psi_ratio", 0))
        KS_RATIO.set(d.get("ks_ratio", 0))

    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
