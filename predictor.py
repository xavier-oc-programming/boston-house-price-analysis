import json
import os

import joblib
import numpy as np

_scaler = None
_model = None
_feature_names = None

MODELS_DIR = "models"


def _load_models():
    global _scaler, _model, _feature_names
    if _scaler is None:
        _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
        with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
            _feature_names = json.load(f)


def predict_price(features: dict) -> dict:
    _load_models()

    values = np.array([features[f] for f in _feature_names]).reshape(1, -1)

    scaled = _scaler.transform(values)

    log_prediction = float(_model.predict(scaled)[0])

    # Prices in the dataset are in $1,000s. The model predicts log(price_in_thousands).
    # To get dollars: exp(prediction) gives price in thousands, multiply by 1000.
    predicted_thousands = float(np.exp(log_prediction))
    predicted_dollars = predicted_thousands * 1000

    return {
        "predicted_price_dollars": predicted_dollars,
        "predicted_price_thousands": predicted_thousands,
        "log_prediction": log_prediction,
        "features_used": features,
    }


def get_feature_stats() -> dict:
    with open(os.path.join(MODELS_DIR, "feature_stats.json")) as f:
        return json.load(f)
