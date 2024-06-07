import os

import pytest
from fastapi.testclient import TestClient
from src.serve.main import app

client = TestClient(app)
''' 
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}

def test_create_metric_limit():
    response = client.post("/metric-limit", json={"value": 0.5})
    assert response.status_code == 200
    assert response.json()["message"] == "Metric limit added successfully"

def test_get_latest_metric_limit():
    response = client.get("/metric-limit/latest")
    assert response.status_code == 200

def test_predict():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "prediction" in response.json() or "error" in response.json()

def test_predict_regression():
    response = client.get("/predict/regression")
    assert response.status_code == 200
    assert "prediction" in response.json() or "error" in response.json()

def test_historical_prices():
    response = client.get("/historical-prices")
    assert response.status_code == 200
    assert "prices" in response.json() or "error" in response.json()

def test_get_latest_validation_result():
    response = client.get("/latest-validation-result")
    assert response.status_code == 200
    assert response.json() is not None

def test_metrics_history():
    response = client.get("/metrics-history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_production_metrics_history():
    response = client.get("/production-metrics-history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
'''