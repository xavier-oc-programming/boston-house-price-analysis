import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import predictor

app = FastAPI(
    title="Boston House Price Predictor",
    description="Predicts 1970s Boston property values using multivariable linear regression on 13 socio-economic features. Log-transformed target achieves test r²=0.74. Supply property characteristics and receive a dollar estimate.",
    version="1.0.0",
)

_INDEX_PATH = Path("templates/index.html")

MODEL_LOADED = False
MODEL_R2 = 0.74

try:
    predictor._load_models()
    with open("models/model_metrics.json") as f:
        _metrics = json.load(f)
    MODEL_R2 = _metrics.get("test_r2_log", 0.74)
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

FEATURE_DESCRIPTIONS = {
    "CRIM": {"label": "Crime Rate", "description": "Per capita crime rate by town"},
    "ZN": {"label": "Residential Zoning", "description": "% land zoned for large lots"},
    "INDUS": {"label": "Industrial Land", "description": "% non-retail business acres"},
    "CHAS": {
        "label": "Charles River",
        "description": "1 = borders river, 0 = does not",
        "type": "binary",
    },
    "NOX": {"label": "Pollution (NOX)", "description": "Nitric oxides concentration"},
    "RM": {"label": "Rooms", "description": "Average rooms per dwelling"},
    "AGE": {"label": "Property Age", "description": "% units built before 1940"},
    "DIS": {
        "label": "Distance to Employment",
        "description": "Weighted distance to employment centres",
    },
    "RAD": {
        "label": "Highway Access",
        "description": "Index of highway accessibility",
    },
    "TAX": {
        "label": "Property Tax Rate",
        "description": "Full-value tax rate per $10,000",
    },
    "PTRATIO": {
        "label": "Pupil-Teacher Ratio",
        "description": "Pupils per teacher by town",
    },
    "B": {
        "label": "B Index",
        "description": "1000(Bk-0.63)² where Bk = % Black residents",
    },
    "LSTAT": {
        "label": "Lower Status %",
        "description": "% population lower status",
    },
}


class PredictionRequest(BaseModel):
    CRIM: float = Field(..., ge=0, description="Per capita crime rate")
    ZN: float = Field(..., ge=0, le=100)
    INDUS: float = Field(..., ge=0, le=100)
    CHAS: float = Field(..., ge=0, le=1)
    NOX: float = Field(..., ge=0, le=1)
    RM: float = Field(..., ge=1, le=15)
    AGE: float = Field(..., ge=0, le=100)
    DIS: float = Field(..., ge=0)
    RAD: float = Field(..., ge=1, le=24)
    TAX: float = Field(..., ge=0)
    PTRATIO: float = Field(..., ge=0, le=30)
    B: float = Field(..., ge=0, le=400)
    LSTAT: float = Field(..., ge=0, le=40)


class PredictionResponse(BaseModel):
    predicted_price_dollars: float
    predicted_price_formatted: str
    features_used: dict
    model_r2: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_r2: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_PATH.read_text())


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_LOADED,
        model_r2=MODEL_R2 if MODEL_LOADED else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    """
    Predict Boston property price from 13 socio-economic features.

    Example request (average Boston property):
    {
      "CRIM": 3.61, "ZN": 11.36, "INDUS": 11.14, "CHAS": 0.07,
      "NOX": 0.55, "RM": 6.28, "AGE": 68.57, "DIS": 3.80,
      "RAD": 9.55, "TAX": 408.24, "PTRATIO": 18.46,
      "B": 356.67, "LSTAT": 12.65
    }
    Expected: ~$20,703
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = body.model_dump()
    result = predictor.predict_price(features)

    dollars = result["predicted_price_dollars"]
    formatted = f"${dollars:,.0f}"

    return PredictionResponse(
        predicted_price_dollars=dollars,
        predicted_price_formatted=formatted,
        features_used=result["features_used"],
        model_r2=MODEL_R2,
    )


@app.get("/api/feature-stats")
def feature_stats():
    return predictor.get_feature_stats()


@app.get("/api/model-info")
def model_info():
    with open("models/model_metrics.json") as f:
        return json.load(f)


@app.get("/api/feature-descriptions")
def feature_descriptions():
    return FEATURE_DESCRIPTIONS
