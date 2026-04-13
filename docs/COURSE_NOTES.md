# Course Notes — Day 81: Multivariable Regression & Valuation Model

**Course:** 100 Days of Code: The Complete Python Pro Bootcamp  
**Day:** 81  
**Topics:** Multivariable linear regression, sklearn, residual analysis, log transformation, valuation models

---

## Exercise Brief

Build a multivariable regression model to estimate the median value of residential homes in Boston, Massachusetts using 13 socio-economic and geographic features recorded in the 1970s.

The goal is to:
1. Explore relationships between features and the target (`PRICE`)
2. Diagnose regression assumptions via residual plots
3. Improve model performance using log transformation on the target
4. Use the fitted model to make real property valuations

---

## Dataset

**File:** `boston.csv`  
**Source:** UCI Machine Learning Repository / Boston Housing Dataset (Harrison & Rubinfeld, 1978)  
**Rows:** 506 observations  
**Target:** `PRICE` — median home value in thousands of 1970s USD

| Feature  | Description |
|----------|-------------|
| CRIM     | Per capita crime rate by town |
| ZN       | Proportion of residential land zoned for lots over 25,000 sq ft |
| INDUS    | Proportion of non-retail business acres per town |
| CHAS     | Charles River dummy variable (1 if tract bounds river, 0 otherwise) |
| NOX      | Nitric oxides concentration (parts per 10 million) |
| RM       | Average number of rooms per dwelling |
| AGE      | Proportion of owner-occupied units built prior to 1940 |
| DIS      | Weighted distance to five Boston employment centres |
| RAD      | Index of accessibility to radial highways |
| TAX      | Full-value property-tax rate per $10,000 |
| PTRATIO  | Pupil-to-teacher ratio by town |
| B        | 1000(Bk − 0.63)² where Bk is the proportion of Black residents by town |
| LSTAT    | % lower status of the population |
| PRICE    | Median value of owner-occupied homes in $1,000s (target) |

---

## Key Concepts Covered

- **Multivariable linear regression:** fitting a model with 13 predictors simultaneously
- **Train/test split:** 80/20 split to evaluate out-of-sample performance
- **Residual analysis:** checking for patterns, skew, and heteroscedasticity
- **Log transformation:** applying `np.log(PRICE)` to normalise the target distribution and reduce skew
- **Coefficient interpretation:** understanding the sign and magnitude of each feature's contribution
- **Valuation model:** using the fitted model to predict the value of a specific property configuration
- **sklearn API:** `LinearRegression()`, `.fit()`, `.score()`, `.predict()`, `.coef_`, `.intercept_`

---

## Key Results

- Original model r²: ~0.75 (training), ~0.67 (test)
- Log-transformed model r²: ~0.79 (training), ~0.74 (test)
- Key positive predictor: `RM` (rooms) — each extra room adds ~$3,000 to value
- Key negative predictors: `LSTAT` (poverty), `CRIM` (crime), `NOX` (pollution)
- Charles River proximity (`CHAS = 1`) adds a meaningful premium
