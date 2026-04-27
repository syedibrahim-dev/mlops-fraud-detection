"""
Basic tests for the serving API.
Uses a dummy model so no real artifact is needed in CI.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
import xgboost as xgb
import os, tempfile

# Create a tiny dummy model before importing serve
_tmp = tempfile.mkdtemp()
_model_path = os.path.join(_tmp, "model.json")
_dummy = xgb.XGBClassifier(n_estimators=1)
_dummy.fit(np.zeros((4, 30)), [0, 1, 0, 1])
_dummy.save_model(_model_path)

os.environ["MODEL_PATH"] = _model_path
os.environ["MODEL_VERSION"] = "test"

from src.serve import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_legit():
    r = client.post("/predict", json={"features": [0.0] * 30})
    assert r.status_code == 200
    body = r.json()
    assert "fraud" in body
    assert "probability" in body
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_wrong_feature_count():
    r = client.post("/predict", json={"features": [0.0] * 10})
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"fraud_predictions_total" in r.content
