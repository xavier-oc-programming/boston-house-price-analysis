from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

AVERAGE_PROPERTY = {
    "CRIM": 3.61, "ZN": 11.36, "INDUS": 11.14, "CHAS": 0.07,
    "NOX": 0.55, "RM": 6.28, "AGE": 68.57, "DIS": 3.80,
    "RAD": 9.55, "TAX": 408.24, "PTRATIO": 18.46,
    "B": 356.67, "LSTAT": 12.65,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_average_property():
    response = client.post("/predict", json=AVERAGE_PROPERTY)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price_dollars" in data
        assert "predicted_price_formatted" in data
        assert 10000 < data["predicted_price_dollars"] < 50000


def test_predict_missing_feature():
    incomplete = {k: v for k, v in AVERAGE_PROPERTY.items() if k != "RM"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_feature_stats():
    response = client.get("/api/feature-stats")
    assert response.status_code == 200
    data = response.json()
    assert "RM" in data
    assert "PRICE" not in data


def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "test_r2_log" in data


def test_feature_descriptions():
    response = client.get("/api/feature-descriptions")
    assert response.status_code == 200
    data = response.json()
    assert "RM" in data
    assert "label" in data["RM"]
