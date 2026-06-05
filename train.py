import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/boston.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH, index_col=0)

X = data.drop(columns=["PRICE"])
y = np.log(data["PRICE"])

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_r2 = model.score(X_train_scaled, y_train)
test_r2_log = model.score(X_test_scaled, y_test)

y_test_pred_log = model.predict(X_test_scaled)
y_test_pred_dollars = np.exp(y_test_pred_log) * 1000
y_test_actual_dollars = np.exp(y_test) * 1000

ss_res = np.sum((y_test_actual_dollars - y_test_pred_dollars) ** 2)
ss_tot = np.sum((y_test_actual_dollars - y_test_actual_dollars.mean()) ** 2)
test_r2_original = 1 - ss_res / ss_tot

print(f"Train r²:             {train_r2:.4f}")
print(f"Test r² (log scale):  {test_r2_log:.4f}")
print(f"Test r² ($ scale):    {test_r2_original:.4f}")

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))

with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f)

metrics = {
    "train_r2": round(train_r2, 4),
    "test_r2_log": round(test_r2_log, 4),
    "test_r2_original": round(test_r2_original, 4),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": feature_names,
    "target_transform": "log",
    "note": "Log-transformed target. Predictions are exponentiated before returning dollar values.",
}
with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

feature_stats = {}
for col in feature_names:
    feature_stats[col] = {
        "mean": float(X_train[col].mean()),
        "min": float(X_train[col].min()),
        "max": float(X_train[col].max()),
        "std": float(X_train[col].std()),
    }
with open(os.path.join(MODELS_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

coef_df = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)
print("\nTop 3 positive coefficients:")
for feat, val in coef_df.head(3).items():
    print(f"  {feat}: {val:.4f}")

print("\nTop 3 negative coefficients:")
for feat, val in coef_df.tail(3).items():
    print(f"  {feat}: {val:.4f}")

mean_vals = np.array([feature_stats[f]["mean"] for f in feature_names]).reshape(1, -1)
mean_scaled = scaler.transform(mean_vals)
avg_log_pred = model.predict(mean_scaled)[0]
avg_price = np.exp(avg_log_pred) * 1000
print(f"\nPredicted price — average property: ${avg_price:,.0f}")

premium_vals = mean_vals.copy()
feat_idx = {f: i for i, f in enumerate(feature_names)}
premium_vals[0, feat_idx["RM"]] = 8
premium_vals[0, feat_idx["CHAS"]] = 1
premium_vals[0, feat_idx["LSTAT"]] = float(np.percentile(X_train["LSTAT"], 5))
premium_vals[0, feat_idx["NOX"]] = float(np.percentile(X_train["NOX"], 25))
premium_vals[0, feat_idx["CRIM"]] = float(np.percentile(X_train["CRIM"], 25))
premium_scaled = scaler.transform(premium_vals)
premium_log_pred = model.predict(premium_scaled)[0]
premium_price = np.exp(premium_log_pred) * 1000
print(f"Predicted price — premium property:  ${premium_price:,.0f}")

print("\nModel files saved to models/")
