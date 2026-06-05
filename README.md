![Python](https://img.shields.io/badge/Python-blue)
![pandas](https://img.shields.io/badge/pandas-blue)
![NumPy](https://img.shields.io/badge/NumPy-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-orange)
![Plotly](https://img.shields.io/badge/Plotly-purple)
![scikit--learn](https://img.shields.io/badge/scikit--learn-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-black)
![Azure App Service](https://img.shields.io/badge/Azure_App_Service-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Docker](https://img.shields.io/badge/Docker-blue)

# Boston House Price Analysis

Which neighbourhood characteristics actually drive residential property values — and by how much? This analysis uses the Boston Housing Dataset (Harrison & Rubinfeld, 1978) to answer that question for 506 census tracts across 1970s Boston, building a multivariable linear regression model across 13 socio-economic and geographic features.

The raw price data carries a positive skew (skewness 1.46) that biases the residuals. Applying a log transformation to the target reduces residual skew to 0.09 and raises the test-set r² from 0.67 to 0.74 — a meaningful improvement in out-of-sample accuracy. The final model identifies number of rooms (`RM`) as the strongest positive predictor, valuing each additional room at roughly $3,109, while pollution (`NOX`), poverty (`LSTAT`), and crime (`CRIM`) each depress prices.

The model produces a working valuation function: supply any combination of property characteristics and get a dollar estimate in return. An average Boston property across all 13 features is estimated at $20,703. An 8-room river-front property in a low-poverty area is estimated at $25,792.

**Live predictor → [boston-house-price-xoc.azurewebsites.net](https://boston-house-price-xoc.azurewebsites.net)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://boston-house-price-xoc.azurewebsites.net/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb**

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Analysis Flow](#2-analysis-flow)
3. [Key Findings](#3-key-findings)
4. [Dataset Schema](#4-dataset-schema)
5. [Architecture](#5-architecture)
6. [Visualisations](#6-visualisations)
7. [Operations Reference](#7-operations-reference)
8. [Background](#8-background)
9. [Dependencies](#9-dependencies)
10. [Portfolio Integration](#10-portfolio-integration)

---

## 1. Quick Start

```bash
git clone https://github.com/xavier-oc-programming/boston-house-price-analysis.git
cd boston-house-price-analysis
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb` to run the full analysis with outputs.

---

## 2. Analysis Flow

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

## 3. Key Findings

- **Each additional room adds ~$3,109** to the estimated property value (raw model coefficient: 3.11)
- **Log transformation reduces residual skew from 1.46 to 0.09**, bringing the error distribution close to normal
- **Test-set r² improves from 0.67 to 0.74** after log-transforming the price target
- **River proximity (CHAS) is a positive predictor** — log model coefficient: +0.08
- **Pollution (NOX) is the strongest negative predictor** — log model coefficient: −0.70
- **Average Boston property estimated at $20,703** using mean values across all 13 features
- **8-room river-front property in a low-poverty area estimated at $25,792**
- **Only 35 of 506 properties** in the dataset border the Charles River
- **506 observations, no missing values, no duplicates** — dataset is analysis-ready as supplied

---

## 4. Dataset Schema

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

**Computed at runtime:**

| Column    | Created in | Description |
|-----------|------------|-------------|
| log PRICE | analysis notebooks | `np.log(data['PRICE'])` — log-transformed target for model v2 |

---

## 5. Architecture

```
boston-house-price-analysis/
│
├── notebooks/
│   ├── analysis/
│   │   ├── A_02_Multivariable_Regression_Start.ipynb
│   │   └── A_03_Multivariable_Regression_Complete.ipynb
│   └── concepts/
│       ├── 00__Overview.ipynb
│       └── 01__Solution_and_Learning_Points.ipynb
│
├── data/
│   └── boston.csv
│
├── plots/
│   └── [charts saved at 150 dpi]
│
├── notebook_web_render/
│   └── index.html
│
├── docs/
│   └── COURSE_NOTES.md
│
├── .github/
│   └── workflows/
│       └── publish_notebook.yml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Visualisations

All charts are saved to `plots/` at 150 dpi.

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

## 7. Operations Reference

| Value | Location | Description |
|-------|----------|-------------|
| `'../../data/boston.csv'` | analysis notebooks, load cell | Relative path from `notebooks/analysis/` to the dataset |
| `index_col=0` | `pd.read_csv()` | First CSV column is the row index, not a feature |
| `pd.options.display.float_format = '{:,.2f}'.format` | analysis notebooks, setup cell | Display floats with 2 decimal places throughout |
| `test_size=0.2` | `train_test_split()` | 80/20 train/test split |
| `random_state=10` | `train_test_split()` | Fixed seed for reproducible split |
| `q=0.75` / `q=0.25` | `data.NOX.quantile()` / `data.LSTAT.quantile()` | Valuation scenario: high pollution, low poverty quartiles |

---

## 8. Background

100 Days of Code: The Complete Python Pro Bootcamp — Day 81: Multivariable Regression & Valuation Model.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the full brief and key results.

---

## 9. Dependencies

| Module | Used in | Purpose |
|--------|---------|---------|
| pandas | analysis notebooks | Data loading, cleaning, descriptive statistics, DataFrame display |
| numpy | analysis notebooks | Log transformation (`np.log`), exponentiation (`np.exp`), array ops |
| matplotlib | analysis notebooks | Histogram, residual scatter, log-price scatter plots |
| seaborn | analysis notebooks | Distribution plots (`displot`), joint plots, pair plot |
| plotly | analysis notebooks | Interactive bar chart for Charles River access counts |
| scikit-learn | analysis notebooks | `LinearRegression`, `train_test_split` |
| notebook | local dev | Jupyter notebook server |

---

## 10. Portfolio Integration

Rendered notebook (outputs and charts only, no code):  
https://xavier-oc-programming.github.io/boston-house-price-analysis/notebook_web_render/

Regenerated automatically via GitHub Actions on every commit to `notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb`.

To regenerate manually:

```bash
jupyter nbconvert --to html --no-input \
  --output index.html \
  notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb
mv index.html notebook_web_render/index.html
```

---

## 11. Deployment

FastAPI app deployed to Azure App Service (Free tier F1) via zip deploy.

### Train and run locally

```bash
pip install -r requirements.txt
python train.py                      # generates models/*.pkl and models/*.json
uvicorn main:app --reload            # http://localhost:8000
pytest tests/ -v                     # run API tests
```

### Docker

```bash
docker build -t boston-house-price .
docker run -p 8000:8000 boston-house-price
```

### Azure deployment

```bash
az group create --name boston-house-price-rg --location westeurope
az appservice plan create --name boston-house-price-plan --resource-group boston-house-price-rg --sku B1 --is-linux
# Scale to F1 via portal after creation
az webapp create --name boston-house-price-xoc --resource-group boston-house-price-rg --plan boston-house-price-plan --runtime "PYTHON:3.11"
az webapp config set --name boston-house-price-xoc --resource-group boston-house-price-rg --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set --name boston-house-price-xoc --resource-group boston-house-price-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true
cd boston-house-price-analysis && zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*" -x "*.ipynb_checkpoints*"
az webapp deployment source config-zip --name boston-house-price-xoc --resource-group boston-house-price-rg --src deploy.zip
```

Live: https://boston-house-price-xoc.azurewebsites.net  
API docs: https://boston-house-price-xoc.azurewebsites.net/docs
