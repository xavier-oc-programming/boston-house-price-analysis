![Python](https://img.shields.io/badge/Python-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Azure App Service](https://img.shields.io/badge/Azure_App_Service-blue)
![Docker](https://img.shields.io/badge/Docker-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-black)

# Boston House Price Predictor

Adjust rooms, pollution, crime rate, and river proximity via sliders — get a live dollar price estimate in return. The model is a log-transformed linear regression trained on 506 1970s Boston census tracts (test r² = 0.74).

**[Live app →](https://boston-house-price-xoc.azurewebsites.net)** &nbsp;·&nbsp; **[API docs →](https://boston-house-price-xoc.azurewebsites.net/docs)** &nbsp;·&nbsp; **[Rendered notebook →](https://xavier-oc-programming.github.io/boston-house-price-analysis/notebook_web_render/)**

---

## What's in this repo

This project has two parts that build on each other:

**Part 1 — Analysis** (`notebooks/`)  
Exploratory data analysis and model development in Jupyter. Compares a raw linear regression (test r² = 0.67) against a log-transformed model (test r² = 0.74). Produces the trained model that the web app deploys.

**Part 2 — Web App** (`main.py`, `predictor.py`, `train.py`, `templates/`)  
FastAPI app that wraps the trained model in a REST API and serves an interactive slider UI. Deployed to Azure App Service.

---

## Part 1 — Analysis

### Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook
# open notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb
```

### What the notebook covers

```
[Ingestion]      pd.read_csv → data (506 × 14)
[Exploration]    .describe(), .isna(), .duplicated()
[Visualisation]  price distribution, pairplot, jointplots (DIS/NOX, LSTAT/PRICE, RM/PRICE)
[Model v1]       LinearRegression on raw PRICE → train r² 0.75, test r² 0.67, residual skew 1.46
[Transformation] np.log(PRICE) → residual skew drops to 0.09
[Model v2]       LinearRegression on log(PRICE) → train r² 0.79, test r² 0.74
[Valuation]      np.exp(prediction) × 1000 → dollar estimate for any property
```

### Key findings

- **Each additional room adds ~$3,109** to the estimated value
- **Log transformation reduces residual skew from 1.46 to 0.09** — error distribution becomes near-normal
- **Test r² improves from 0.67 to 0.74** after log-transforming the target
- **Pollution (NOX) is the strongest negative predictor** — coefficient −0.70
- **River proximity (CHAS) adds a positive premium** — coefficient +0.08
- **Average Boston property: ~$20,198** · **8-room river-front, low-poverty property: ~$37,110**

---

## Part 2 — Web App

### How the model is served

The model is split across three files with a clear separation of concerns:

```
train.py          Runs once. Fits StandardScaler + LinearRegression on
                  log(PRICE),saves scaler.pkl, model.pkl, and three JSON files to models/.

predictor.py      Loaded at API startup. Lazy-loads the pkl files on first call
                  arranges features in the correct order, scales the input, predicts log price,
                  then returns np.exp(prediction) × 1000 to convert back to dollars.

main.py           FastAPI layer. Validates the incoming request with Pydantic,
                  calls predictor predict_price(), formats the response. Also serves
                  the slider UI via Jinja2 template.
```

The `models/` directory is committed to the repo so the API and CI can load the model without needing to retrain. To regenerate (e.g. after changing the training data), run `python train.py` and commit the updated files.

### Run locally

```bash
pip install -r requirements.txt
python train.py          # only needed if models/ are missing or stale
uvicorn main:app --reload
# open http://localhost:8000
```

### Run tests

```bash
pytest tests/ -v
```

### API endpoints

| Method | Route                       | Description                                                 |
| ------ | --------------------------- | ----------------------------------------------------------- |
| `GET`  | `/`                         | Slider UI                                                   |
| `GET`  | `/health`                   | Model status and r²                                         |
| `POST` | `/predict`                  | Price prediction from 13 features                           |
| `GET`  | `/api/feature-stats`        | Min / max / mean / std per feature (used for slider ranges) |
| `GET`  | `/api/model-info`           | r² values, feature list, target transform                   |
| `GET`  | `/api/feature-descriptions` | Plain-English labels for each feature                       |

**Example `POST /predict` request:**

```json
{
  "CRIM": 3.61,
  "ZN": 11.36,
  "INDUS": 11.14,
  "CHAS": 0.07,
  "NOX": 0.55,
  "RM": 6.28,
  "AGE": 68.57,
  "DIS": 3.8,
  "RAD": 9.55,
  "TAX": 408.24,
  "PTRATIO": 18.46,
  "B": 356.67,
  "LSTAT": 12.65
}
```

**Response:**

```json
{
  "predicted_price_dollars": 20198.42,
  "predicted_price_formatted": "$20,198",
  "model_r2": 0.7447
}
```

---

## Dataset

`data/boston.csv` — 506 rows, 14 columns, no missing values, no duplicates.

| Column    | Description                                   |
| --------- | --------------------------------------------- |
| CRIM      | Per capita crime rate by town                 |
| ZN        | % residential land zoned for large lots       |
| INDUS     | % non-retail business acres                   |
| CHAS      | Charles River dummy (1 = borders river)       |
| NOX       | Nitric oxides concentration                   |
| RM        | Average rooms per dwelling                    |
| AGE       | % units built before 1940                     |
| DIS       | Weighted distance to employment centres       |
| RAD       | Highway accessibility index                   |
| TAX       | Property tax rate per $10,000                 |
| PTRATIO   | Pupils per teacher                            |
| B         | 1000(Bk − 0.63)² where Bk = % Black residents |
| LSTAT     | % lower-status population                     |
| **PRICE** | **Target — median home value in $1,000s**     |

---

## File structure

```
├── main.py              FastAPI app — routes and Pydantic models
├── predictor.py         Loads pkl files, runs inference, returns dollar price
├── train.py             One-off training script — run to regenerate models/
├── Dockerfile
├── startup.txt          Azure App Service startup command
├── requirements.txt
│
├── models/              Committed — API and CI load these without retraining
│   ├── scaler.pkl
│   ├── model.pkl
│   ├── feature_names.json
│   ├── feature_stats.json
│   └── model_metrics.json
│
├── templates/
│   └── index.html       Single-page slider UI (inline CSS + JS)
│
├── tests/
│   └── test_api.py      6 pytest tests — health, predict, validation, stats
│
├── .github/workflows/
│   ├── ci.yml                 See "CI workflows" section below.
│   └── publish_notebook.yml   See "CI workflows" section below.
│
├── notebooks/
│   └── analysis/
│       ├── A_02_Multivariable_Regression_Start.ipynb
│       └── A_03_Multivariable_Regression_Complete.ipynb
│
├── data/boston.csv
├── plots/               19 charts at 150 dpi (generated by notebook)
├── docs/COURSE_NOTES.md
└── notebook_web_render/ Rendered HTML notebook (served via GitHub Pages)
```

---

## Deployment

**Platform:** Azure App Service, Free tier (F1), Linux, Python 3.11  
**Method:** zip deploy — the repo is zipped locally and uploaded directly to Azure. Azure then runs `pip install -r requirements.txt` on the server (`SCM_DO_BUILD_DURING_DEPLOYMENT=true`) and starts the app with the gunicorn command set in `startup.txt`.

**Why gunicorn + uvicorn worker?** FastAPI is an ASGI app. Gunicorn manages the worker process lifecycle; the UvicornWorker gives it ASGI support. Azure App Service expects a long-running process bound to port 8000 — the `--timeout 600` prevents the free tier from killing slow cold starts.

**Why `models/` is committed:** Azure App Service does not run `train.py` on deployment. The pkl files must already be present in the zip so `predictor.py` can load them at startup. If you retrain locally, commit the updated `models/` files before re-deploying.

### Steps

```bash
# 1. Create the resource group and F1 plan
az group create --name boston-house-price-rg --location westeurope
az appservice plan create --name boston-house-price-plan --resource-group boston-house-price-rg --sku F1 --is-linux

# 2. Create the web app (Python 3.11 runtime)
az webapp create --name boston-house-price-xoc --resource-group boston-house-price-rg --plan boston-house-price-plan --runtime "PYTHON:3.11"

# 3. Set the startup command and tell Azure to pip install on deploy
az webapp config set --name boston-house-price-xoc --resource-group boston-house-price-rg --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set --name boston-house-price-xoc --resource-group boston-house-price-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

# 4. Zip and deploy
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*" -x "*.ipynb_checkpoints*"
az webapp deployment source config-zip --name boston-house-price-xoc --resource-group boston-house-price-rg --src deploy.zip
```

---

## CI workflows

There are two independent workflows in `.github/workflows/`. They never conflict — each triggers on a different condition.

### `ci.yml` — API tests

Runs on every push to `main` and on every pull request targeting `main`.

**Steps:**
1. Check out the repo
2. Set up Python 3.11
3. `pip install -r requirements.txt`
4. `pytest tests/ -v` — runs the 6 API tests in `tests/test_api.py`

**Why no retrain step?** The `models/` directory (`scaler.pkl`, `model.pkl`, and the JSON files) is committed directly to the repo. When the CI runner checks out the code, the pkl files are already there. `predictor.py` loads them at startup, so the tests can hit `POST /predict` and get real predictions without needing to run `train.py` first.

**What it tests:** health endpoint, predict endpoint (average property, expected price range), missing-feature validation (expects 422), feature stats, model info, feature descriptions.

---

### `publish_notebook.yml` — rendered notebook

Triggers only when `notebooks/analysis/A_03_Multivariable_Regression_Complete.ipynb` is changed in a push to `main`. Does not run on every commit — only when the notebook file itself is modified.

**Steps:**
1. Check out the repo
2. Set up Python 3.11
3. `pip install jupyter nbconvert`
4. Run `jupyter nbconvert --to html --no-input` — converts the notebook to HTML, stripping all code cells so only outputs and markdown are visible
5. Move the output to `notebook_web_render/index.html`
6. Commit and push the updated HTML back to `main` with a bot commit

**Why commit back to main instead of gh-pages?** The GitHub Pages source for this repo is set to serve `notebook_web_render/` from `main` directly, so the bot pushes the rendered HTML there rather than maintaining a separate branch.

---

## Background

100 Days of Code: The Complete Python Pro Bootcamp — Day 81.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the original brief.
