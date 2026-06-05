![Python](https://img.shields.io/badge/Python-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Docker](https://img.shields.io/badge/Docker-blue)
![Azure App Service](https://img.shields.io/badge/Azure_App_Service-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-black)
![pandas](https://img.shields.io/badge/pandas-blue)
![NumPy](https://img.shields.io/badge/NumPy-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-orange)
![Plotly](https://img.shields.io/badge/Plotly-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-orange)

# Boston House Price Predictor

A property valuation web app trained on 506 Boston census tracts with 13 socio-economic and geographic features. Adjust room count, pollution levels, crime rate, or river proximity via sliders and get a live dollar price estimate in return. The model uses a log-transformed target (test r² = 0.74) and runs on Azure App Service.

**[boston-house-price-xoc.azurewebsites.net](https://boston-house-price-xoc.azurewebsites.net)** — live predictor  
**[/docs](https://boston-house-price-xoc.azurewebsites.net/docs)** — interactive API docs  
**[Rendered notebook](https://xavier-oc-programming.github.io/boston-house-price-analysis/notebook_web_render/)** — full analysis with outputs

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [How It Works](#2-how-it-works)
3. [API Reference](#3-api-reference)
4. [Key Findings](#4-key-findings)
5. [Dataset Schema](#5-dataset-schema)
6. [Architecture](#6-architecture)
7. [Analysis Flow](#7-analysis-flow)
8. [Visualisations](#8-visualisations)
9. [Dependencies](#9-dependencies)
10. [Deployment](#10-deployment)
11. [Background](#11-background)

---

## 1. Quick Start

### Run the web app locally

```bash
git clone https://github.com/xavier-oc-programming/boston-house-price-analysis.git
cd boston-house-price-analysis
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) — sliders update the price estimate in real time.

### Retrain the model

```bash
python train.py
```

Saves `models/scaler.pkl`, `models/model.pkl`, and supporting JSON files.

### Run the tests

```bash
pytest tests/ -v
```

### Run the analysis notebook

```bash
jupyter notebook
```

Open `notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb`.

---

## 2. How It Works

The app exposes all 13 Boston Housing features as interactive sliders. As you move a slider, the frontend sends a `POST /predict` request and updates the price display within 300 ms.

**Model pipeline:**

```
data/boston.csv
    │
    ├── Features X: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
    ├── Target y: np.log(PRICE)          ← log transform reduces skew 1.46 → 0.09
    │
    ├── train_test_split(test_size=0.2, random_state=10)
    ├── StandardScaler.fit(X_train)
    ├── LinearRegression.fit(X_train_scaled, log_y_train)
    │
    └── Inference: np.exp(model.predict(scaler.transform(input))) × 1000
                  └── converts log prediction back to dollars
```

**Model results:**

| | r² |
|--|--|
| Train | 0.79 |
| Test (log scale) | 0.74 |
| Test (dollar scale) | 0.73 |

Two models were compared during analysis — raw linear regression (test r² = 0.67) and log-transformed target (test r² = 0.74). The log transformation is kept for deployment.

**Preset scenarios:**

| Preset | Description |
|--------|-------------|
| Average property | All 13 features at training-set mean |
| Premium property | 8 rooms, river-front, low poverty, low pollution |
| Budget property | High poverty, high crime, high pollution, 5 rooms |

---

## 3. API Reference

Base URL: `https://boston-house-price-xoc.azurewebsites.net`  
Interactive docs: `/docs`

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_r2": 0.7447
}
```

### `POST /predict`

Supply all 13 features, receive a dollar estimate.

**Request body (average Boston property):**

```json
{
  "CRIM": 3.61, "ZN": 11.36, "INDUS": 11.14, "CHAS": 0.07,
  "NOX": 0.55, "RM": 6.28, "AGE": 68.57, "DIS": 3.80,
  "RAD": 9.55, "TAX": 408.24, "PTRATIO": 18.46,
  "B": 356.67, "LSTAT": 12.65
}
```

**Response:**

```json
{
  "predicted_price_dollars": 20198.42,
  "predicted_price_formatted": "$20,198",
  "features_used": { ... },
  "model_r2": 0.7447
}
```

### `GET /api/feature-stats`

Returns `mean`, `min`, `max`, `std` for each feature from the training set. Used by the frontend to set slider ranges and defaults.

### `GET /api/model-info`

Returns `models/model_metrics.json` — r² values, feature list, target transform, train/test sizes.

### `GET /api/feature-descriptions`

Returns plain-English labels and descriptions for each feature. Used by the frontend to label sliders.

---

## 4. Key Findings

- **Each additional room adds ~$3,109** to the estimated property value
- **Log transformation reduces residual skew from 1.46 to 0.09**, bringing the error distribution close to normal
- **Test-set r² improves from 0.67 to 0.74** after log-transforming the price target
- **Pollution (NOX) is the strongest negative predictor** — log model coefficient: −0.70
- **River proximity (CHAS) is a positive predictor** — log model coefficient: +0.08
- **Average Boston property estimated at ~$20,198** using mean values across all 13 features
- **8-room river-front property in a low-poverty area estimated at ~$37,110**
- **Only 35 of 506 properties** in the dataset border the Charles River
- **506 observations, no missing values, no duplicates** — dataset is analysis-ready as supplied

---

## 5. Dataset Schema

### `data/boston.csv`

| Column  | Type    | Description |
|---------|---------|-------------|
| CRIM    | float64 | Per capita crime rate by town |
| ZN      | float64 | Proportion of residential land zoned for lots > 25,000 sq ft |
| INDUS   | float64 | Proportion of non-retail business acres per town |
| CHAS    | float64 | Charles River dummy variable (1 = bounds river, 0 = does not) |
| NOX     | float64 | Nitric oxides concentration (parts per 10 million) |
| RM      | float64 | Average number of rooms per dwelling |
| AGE     | float64 | Proportion of owner-occupied units built prior to 1940 |
| DIS     | float64 | Weighted distance to five Boston employment centres |
| RAD     | float64 | Index of accessibility to radial highways |
| TAX     | float64 | Full-value property-tax rate per $10,000 |
| PTRATIO | float64 | Pupil-to-teacher ratio by town |
| B       | float64 | 1000(Bk − 0.63)² where Bk is the proportion of Black residents by town |
| LSTAT   | float64 | % lower status of the population |
| PRICE   | float64 | **Target** — median home value in $1,000s |

506 rows. No missing values. No duplicates.

---

## 6. Architecture

```
boston-house-price-analysis/
│
├── main.py                     ← FastAPI app — routes, Pydantic models, startup
├── predictor.py                ← prediction module — loads pkl, returns dollar price
├── train.py                    ← standalone training script — run once to generate models/
├── conftest.py                 ← pytest root path fix
├── Dockerfile                  ← container definition
├── startup.txt                 ← Azure App Service startup command (gunicorn)
├── requirements.txt
│
├── models/                     ← committed — API and CI load these without retraining
│   ├── scaler.pkl
│   ├── model.pkl
│   ├── feature_names.json
│   ├── feature_stats.json      ← slider ranges and defaults for the frontend
│   └── model_metrics.json      ← r² values, feature list, target transform
│
├── templates/
│   └── index.html              ← single-page frontend — sliders, price display, feature impact
│
├── tests/
│   └── test_api.py             ← 6 pytest tests covering all routes
│
├── .github/
│   └── workflows/
│       ├── publish_notebook.yml  ← renders notebook to GitHub Pages on commit
│       └── ci.yml                ← runs pytest on every push to main
│
├── notebooks/
│   └── analysis/
│       ├── A_02_Multivariable_Regression_Start.ipynb
│       └── A_03_Multivariable_Regression_Complete.ipynb
│
├── data/
│   └── boston.csv
│
├── plots/                      ← 19 charts saved at 150 dpi
├── notebook_web_render/        ← rendered HTML notebook (GitHub Pages)
└── docs/
    └── COURSE_NOTES.md
```

---

## 7. Analysis Flow

Full analysis in `notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb`.

```
pipeline
    │
    │  ── [Ingestion] ────────────────────────────────────────────────
    ├── pd.read_csv('../../data/boston.csv', index_col=0)  →  data (506 × 14)
    │
    │  ── [Exploration] ──────────────────────────────────────────────
    ├── .shape / .info() / .describe()     →  structure and summary stats
    ├── .isna() / .duplicated()            →  confirm clean data
    ├── .mean() on PTRATIO, PRICE, RM      →  descriptive answers
    │
    │  ── [Visualisation] ────────────────────────────────────────────
    ├── sns.displot(PRICE, kde=True)        →  price distribution
    ├── sns.displot(DIS / RM, kde=True)     →  commute and rooms distributions
    ├── plt.hist(RAD)                       →  highway accessibility histogram
    ├── px.bar(CHAS value_counts)           →  river access bar chart
    ├── sns.pairplot(data)                  →  all pairwise scatter plots
    ├── sns.jointplot(DIS, NOX)             →  distance vs. pollution
    ├── sns.jointplot(INDUS, NOX)           →  industry vs. pollution
    ├── sns.jointplot(LSTAT, PRICE)         →  poverty vs. price
    ├── sns.jointplot(RM, PRICE)            →  rooms vs. price
    │
    │  ── [Modelling — v1] ───────────────────────────────────────────
    ├── train_test_split(test_size=0.2, random_state=10)
    ├── LinearRegression().fit(X_train, y_train)
    ├── .score() on training set            →  r² = 0.75
    ├── pd.DataFrame(regr.coef_)            →  coefficient table
    ├── scatter(y_train, predicted_vals)    →  actual vs. predicted
    ├── scatter(predicted_vals, residuals)  →  residuals vs. predicted
    ├── sns.displot(residuals, kde=True)    →  residual distribution (skew 1.46)
    │
    │  ── [Transformation] ───────────────────────────────────────────
    ├── data['PRICE'].skew()                →  confirm positive skew
    ├── np.log(data['PRICE'])               →  log-transform target
    ├── plt.scatter(PRICE, log PRICE)       →  visualise compression
    │
    │  ── [Modelling — v2] ───────────────────────────────────────────
    ├── LinearRegression().fit(X_train, log_y_train)
    ├── .score() on training set            →  r² = 0.79
    ├── residual plots (log model)          →  skew drops to 0.09
    ├── regr.score(X_test, y_test) × 2     →  test r²: 0.67 → 0.74
    │
    │  ── [Valuation] ────────────────────────────────────────────────
    ├── features.mean()                     →  average property baseline
    ├── log_regr.predict(property_stats)    →  log price estimate
    └── np.exp(log_estimate) × 1000        →  dollar value
```

---

## 8. Visualisations

All charts saved to `plots/` at 150 dpi.

| File | Description |
|------|-------------|
| `price_distribution.png` | Distribution of median home values with KDE |
| `distance_distribution.png` | Distribution of weighted distance to employment centres |
| `rooms_distribution.png` | Distribution of average rooms per dwelling |
| `highway_access_histogram.png` | Histogram of highway accessibility index (RAD) |
| `river_access_bar.png` | Count of properties bordering the Charles River |
| `pairplot.png` | Pairwise scatter plots across all 14 columns |
| `jointplot_dis_nox.png` | Distance to employment vs. nitric oxide pollution |
| `jointplot_indus_nox.png` | Industrial land proportion vs. pollution |
| `jointplot_lstat_rm.png` | Poverty level vs. number of rooms |
| `jointplot_lstat_price.png` | Poverty level vs. home price |
| `jointplot_rm_price.png` | Rooms vs. home price |
| `actual_vs_predicted.png` | Raw model: actual vs. predicted prices |
| `residuals_vs_predicted.png` | Raw model: residuals vs. predicted prices |
| `residual_distribution.png` | Raw model residual distribution (skew 1.46) |
| `price_distribution_normal.png` | Raw price distribution showing positive skew |
| `log_price_distribution.png` | Log-transformed price distribution (skew near zero) |
| `price_vs_log_price.png` | Original prices vs. log prices — shows compression effect |
| `log_actual_vs_predicted.png` | Log model: actual vs. predicted log prices |
| `log_residuals_vs_predicted.png` | Log model: residuals vs. predicted (skew 0.09) |
| `log_residual_distribution.png` | Log model residual distribution |

---

## 9. Dependencies

| Module | Used in | Purpose |
|--------|---------|---------|
| pandas | train.py, notebooks | Data loading, cleaning, descriptive statistics |
| numpy | train.py, predictor.py, notebooks | Log transform, exponentiation, array ops |
| scikit-learn | train.py, predictor.py | `StandardScaler`, `LinearRegression`, `train_test_split` |
| joblib | train.py, predictor.py | Serialise and load scaler/model pkl files |
| fastapi | main.py | REST API framework — routes, Pydantic validation |
| uvicorn | local dev | ASGI server for local development (`--reload`) |
| gunicorn | Azure deployment | Production ASGI server with UvicornWorker |
| pydantic | main.py | Request/response models with field validation |
| jinja2 | main.py | HTML template rendering for the frontend |
| httpx | tests | HTTP client used by FastAPI TestClient |
| pytest | tests/ | API test runner |
| matplotlib | notebooks | Histogram, residual scatter, log-price scatter plots |
| seaborn | notebooks | Distribution plots, joint plots, pair plot |
| plotly | notebooks | Interactive bar chart for Charles River access counts |
| notebook | local dev | Jupyter notebook server |

---

## 10. Deployment

FastAPI app deployed to Azure App Service (Free tier F1) via zip deploy.

### Docker

```bash
docker build -t boston-house-price .
docker run -p 8000:8000 boston-house-price
```

### Azure deployment

```bash
az group create --name boston-house-price-rg --location westeurope
az appservice plan create --name boston-house-price-plan --resource-group boston-house-price-rg --sku F1 --is-linux
az webapp create --name boston-house-price-xoc --resource-group boston-house-price-rg --plan boston-house-price-plan --runtime "PYTHON:3.11"
az webapp config set --name boston-house-price-xoc --resource-group boston-house-price-rg --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set --name boston-house-price-xoc --resource-group boston-house-price-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*" -x "*.ipynb_checkpoints*"
az webapp deployment source config-zip --name boston-house-price-xoc --resource-group boston-house-price-rg --src deploy.zip
```

Live: https://boston-house-price-xoc.azurewebsites.net  
API docs: https://boston-house-price-xoc.azurewebsites.net/docs

---

## 11. Background

100 Days of Code: The Complete Python Pro Bootcamp — Day 81: Multivariable Regression & Valuation Model.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the full brief and key results.

Rendered notebook (outputs and charts, no code):  
https://xavier-oc-programming.github.io/boston-house-price-analysis/notebook_web_render/

Regenerated automatically via GitHub Actions on every commit to the analysis notebook. To regenerate manually:

```bash
jupyter nbconvert --to html --no-input \
  --output index.html \
  notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb
mv index.html notebook_web_render/index.html
```
