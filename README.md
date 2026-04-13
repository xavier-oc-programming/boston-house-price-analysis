# boston-house-price-analysis

Multivariable regression model predicting 1970s Boston house prices from 13 neighbourhood features.

This project investigates which socio-economic and geographic characteristics drive residential property values in Boston, Massachusetts. Using the canonical Boston Housing Dataset (Harrison & Rubinfeld, 1978), it answers questions such as: how much does each extra room add to a home's value? Does proximity to the Charles River carry a price premium? How do crime, pollution, and poverty levels depress prices? A multivariable linear regression model is trained on 13 features to produce price estimates for any given property configuration.

The dataset contains 506 observations collected in the 1970s, each representing a Boston census tract. After loading and cleaning the data (no missing values or duplicates), the analysis explores distributions and pairwise relationships, then splits the data 80/20 into training and test sets. A first regression model is evaluated using residual plots, revealing positive skew in the errors. A log transformation is applied to the target variable to reduce that skew, and a second model is trained — achieving meaningfully higher r² on both training and test data.

No external APIs or credentials are required. All data is included in `data/boston.csv` and all analysis runs locally in Jupyter notebooks.

---

## Table of Contents

1. [Quick start](#1-quick-start)
2. [Analysis flow](#2-analysis-flow)
3. [Features](#3-features)
4. [Dataset schema](#4-dataset-schema)
5. [Architecture](#5-architecture)
6. [Notebook reference](#6-notebook-reference)
7. [Configuration reference](#7-configuration-reference)
8. [Course context](#8-course-context)
9. [Dependencies](#9-dependencies)

---

## 1. Quick start

```bash
git clone https://github.com/xavier-oc-programming/boston-house-price-analysis.git
cd boston-house-price-analysis
pip install -r requirements.txt
jupyter notebook
```

Open `practice/A_03_Multivariable_Regression_Complete.ipynb` to see the full analysis with outputs.  
Open `practice/A_02_Multivariable_Regression_Start.ipynb` to work through the exercises yourself.

---

## 2. Analysis flow

```
pipeline
    │
    │  ── [Ingestion] ────────────────────────────────────────────────
    ├── pd.read_csv('../data/boston.csv', index_col=0)  →  data (506 × 14)
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
    ├── .score() on training set            →  r² ≈ 0.75
    ├── pd.DataFrame(regr.coef_)            →  coefficient table
    ├── scatter(y_train, predicted_vals)    →  actual vs. predicted
    ├── scatter(predicted_vals, residuals)  →  residuals vs. predicted
    ├── sns.displot(residuals, kde=True)    →  residual distribution
    │
    │  ── [Transformation] ───────────────────────────────────────────
    ├── data['PRICE'].skew()                →  confirm positive skew
    ├── np.log(data['PRICE'])               →  log-transform target
    ├── plt.scatter(PRICE, log PRICE)       →  visualise compression
    │
    │  ── [Modelling — v2] ───────────────────────────────────────────
    ├── LinearRegression().fit(X_train, log_y_train)
    ├── .score() on training set            →  r² ≈ 0.79
    ├── residual plots (log model)          →  near-normal, less skew
    ├── regr.score(X_test, y_test) × 2     →  compare test r²
    │
    │  ── [Valuation] ────────────────────────────────────────────────
    ├── features.mean()                     →  average property baseline
    ├── log_regr.predict(property_stats)    →  log price estimate
    └── np.exp(log_estimate) × 1000        →  dollar value
```

---

## 3. Features

- Identifies the strongest positive predictor of house price: number of rooms (`RM`)
- Quantifies the price premium for Charles River proximity (`CHAS`)
- Shows that pollution (`NOX`), crime (`CRIM`), and poverty (`LSTAT`) all depress prices
- Demonstrates that distance from employment centres (`DIS`) relates to lower pollution
- Reveals right-skew in the raw price distribution and fixes it via log transformation
- Achieves r² ≈ 0.74 on held-out test data (log model), versus ≈ 0.67 for the raw model
- Produces a working valuation function: specify any property characteristics → get a dollar estimate

---

## 4. Dataset schema

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

| Column   | Created in | Description |
|----------|------------|-------------|
| log PRICE | practice notebooks | `np.log(data['PRICE'])` — log-transformed target for model v2 |

---

## 5. Architecture

```
boston-house-price-analysis/
│
├── theory/
│   ├── 00__Overview.ipynb                         # Day goals, dataset schema, analysis plan, key methods
│   └── 01__Solution_and_Learning_Points.ipynb     # Results, residual diagnosis, log-transform rationale
│
├── practice/
│   ├── A_02_Multivariable_Regression_Start.ipynb  # Exercise template — blank cells with challenge prompts
│   └── A_03_Multivariable_Regression_Complete.ipynb # Full solution with outputs — start here to read
│
├── data/
│   └── boston.csv                                 # Boston Housing Dataset — 506 rows, 14 columns
│
├── docs/
│   └── COURSE_NOTES.md                            # Course brief, dataset description, key results
│
├── requirements.txt                               # Pinned dependencies
├── .gitignore
└── README.md
```

---

## 6. Notebook reference

### theory/

| Notebook | Key methods covered | Question answered |
|----------|--------------------|--------------------|
| 00__Overview.ipynb | `pd.read_csv`, `describe`, `displot`, `jointplot`, `pairplot`, `train_test_split`, `LinearRegression`, `np.log`, `np.exp` | What is the project, what will be built, and how does the analysis flow? |
| 01__Solution_and_Learning_Points.ipynb | Coefficient interpretation, residual diagnostics, log transformation rationale, valuation example | What did the analysis find and what are the key takeaways? |

### practice/

| Notebook | Key methods covered | Question answered |
|----------|--------------------|--------------------|
| A_02_Multivariable_Regression_Start.ipynb | All of the above | Exercise template — prompts only, student fills in solutions |
| A_03_Multivariable_Regression_Complete.ipynb | `read_csv`, `describe`, `isna`, `duplicated`, `displot`, `jointplot`, `pairplot`, `px.bar`, `train_test_split`, `LinearRegression.fit`, `.score`, `.coef_`, `.intercept_`, residual scatter, `np.log`, valuation with `.predict` + `np.exp` | Which neighbourhood features drive Boston house prices? How well can a linear model predict them? |

---

## 7. Configuration reference

| Value | Location | Description |
|-------|----------|-------------|
| `'../data/boston.csv'` | practice notebooks, load cell | Relative path from `practice/` to the dataset |
| `index_col=0` | `pd.read_csv()` | First CSV column is the row index, not a feature |
| `pd.options.display.float_format = '{:,.2f}'.format` | practice notebooks, setup cell | Display floats with 2 decimal places throughout |
| `test_size=0.2` | `train_test_split()` | 80/20 train/test split |
| `random_state=10` | `train_test_split()` | Fixed seed for reproducible split |
| `q=0.75` / `q=0.25` | `data.NOX.quantile()` / `data.LSTAT.quantile()` | Valuation scenario: high pollution, low poverty quartiles |

---

## 8. Course context

100 Days of Code: The Complete Python Pro Bootcamp — Day 81: Multivariable Regression & Valuation Model.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the full exercise brief and key results.

---

## 9. Dependencies

| Module | Used in | Purpose |
|--------|---------|---------|
| pandas | practice notebooks | Data loading, cleaning, descriptive statistics, DataFrame display |
| numpy | practice notebooks | Log transformation (`np.log`), exponentiation (`np.exp`), array ops |
| matplotlib | practice notebooks | Histogram, residual scatter, log-price scatter plots |
| seaborn | practice notebooks | Distribution plots (`displot`), joint plots, pair plot |
| plotly | practice notebooks | Interactive bar chart for Charles River access counts |
| scikit-learn | practice notebooks | `LinearRegression`, `train_test_split` |
| notebook | local dev | Jupyter notebook server |
