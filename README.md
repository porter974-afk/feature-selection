# 50 Startups Multiple Linear Regression (CRISP-DM)

https://porter974-afk.github.io/feature-selection/ml-projects.html

![50 startup](50startup.png)

An end-to-end Machine Learning project using **Scikit-Learn** to model and predict startup profitability based on operational expenditures (`R&D Spend`, `Administration`, `Marketing Spend`) and geographic location (`State`). The project strictly adheres to the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** lifecycle.

---

## CRISP-DM Lifecycle Summary

```
+-------------------------------------------------------------------------------+
|                               CRISP-DM PHASES                                 |
+-------------------------------------------------------------------------------+
|  1. Business Understanding   --> Strategic capital allocation & ROI analysis  |
|  2. Data Understanding       --> 50 startups, EDA, VIF multicollinearity      |
|  3. Data Preparation         --> OneHotEncoding (drop='first'), Train/Test    |
|  4. Modeling                 --> OLS Linear Regression, Ridge, Lasso, Reduced |
|  5. Evaluation               --> R2 = 0.90-0.95, Gauss-Markov diagnostics     |
|  6. Deployment               --> Pipeline serialization (.pkl) & CLI tool     |
+-------------------------------------------------------------------------------+
```

---

## 1. Business Understanding
- **Context**: Venture Capitalists and executives evaluate early-stage startups and require quantitative models to predict profitability and allocate investment capital effectively.
- **Key Business Questions**:
  1. *Which expenditure category produces the greatest marginal return on investment (ROI)?*
  2. *Does corporate overhead / administrative spending meaningfully increase profitability?*
  3. *Does operating in New York, California, or Florida confer a statistically significant profit advantage?*
- **Data Mining Goal**: Build a robust Multiple Linear Regression model with $R^2 \ge 0.85$ and low Mean Absolute Error (MAE), verifying that econometric assumptions (normality, homoscedasticity, no autocorrelation) hold.

---

## 2. Data Understanding
- **Dataset**: `50_Startups.csv` (50 companies, 5 columns).
- **Attributes**:
  - `R&D Spend` (continuous numeric): Research and development investment ($).
  - `Administration` (continuous numeric): Administrative and executive overhead ($).
  - `Marketing Spend` (continuous numeric): Advertising and promotion expenditure ($).
  - `State` (nominal categorical): Operational headquarters (`California`, `Florida`, `New York`).
  - `Profit` (continuous numeric target): Annual net profit ($).
- **Exploratory Insights**:
  - **No missing values** across any column.
  - **Zero Spend**: 2 startups spent $0 on R&D; 3 startups spent $0 on Marketing.
  - **Correlations**:
    - `R&D Spend` vs `Profit`: **0.973** (exceptionally strong linear correlation).
    - `Marketing Spend` vs `Profit`: **0.748** (strong positive correlation).
    - `Marketing Spend` vs `R&D Spend`: **0.724** (indicates mild multicollinearity).
    - `Administration` vs `Profit`: **0.201** (weak correlation).

---

## 3. Data Preparation
- **Train-Test Split**: 80% Training set (40 startups), 20% Holdout Testing set (10 startups) with `random_state=42`.
- **Categorical Encoding**:
  - `OneHotEncoder(drop='first')` applied to `State`.
  - California serves as the reference category; dummy columns are created for Florida and New York. This prevents the **Dummy Variable Trap** (perfect multicollinearity where dummy columns sum to 1).
- **Leakage-Free Architecture**: Implemented using `sklearn.compose.ColumnTransformer` embedded within `sklearn.pipeline.Pipeline`.

---

## 4. Modeling
Four candidate architectures were trained and cross-validated (5-Fold CV):
1. **Multiple Linear Regression (Full OLS)**: Includes all 3 expenditure features + State dummies.
2. **Ridge Regression (L2 Regularized)**: Penalizes large weights to mitigate collinearity between R&D and Marketing.
3. **Lasso Regression (L1 Regularized)**: Drives negligible coefficients to zero.
4. **Reduced Linear Model**: Pruned to the two statistically significant drivers (`R&D Spend` + `Marketing Spend`).

### Fitted Linear Regression Equation:
$$\text{Profit} = \$54,028.04 + (0.8056 \times \text{R\&D Spend}) - (0.0688 \times \text{Administration}) + (0.0299 \times \text{Marketing Spend}) + (938.79 \times \text{State\_Florida}) + (6.99 \times \text{State\_New York})$$

### Sequential Forward Feature Selection (SFS) Analysis
As demonstrated in the diagnostic curves below, we iteratively evaluated models with $k \in \{1, 2, 3, 4, 5\}$ features:

| $k$ Features | Feature Added | Holdout Test RMSE ($) | Holdout Test $R^2$ | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | `R&D Spend` | $8,274.87 | 0.94646 | Baseline Driver |
| **2** | `Marketing Spend` | **$8,198.80** | **0.94744** | **Global Optimum (Minimum RMSE, Peak $R^2$)** |
| **3** | `State_New York` | $8,309.06 | 0.94602 | Performance degrades (noise added) |
| **4** | `State_Florida` | $8,409.92 | 0.94470 | Performance degrades |
| **5** | `Administration` | $9,137.99 | 0.93471 | Severe error spike (+11.5% RMSE) |

- **Optimal Subset**: $k = 2$ (`['R&D Spend', 'Marketing Spend']`).
- **Optimal Model Equation**:
  $$\text{Profit} = \$46,346.34 + 0.7886 \times (\text{R\&D Spend}) + 0.0326 \times (\text{Marketing Spend})$$

---

## 5. Evaluation


### Model Comparison Table (Test Set: 20% Holdout)

| Model | Train $R^2$ | Test $R^2$ | Test Adj. $R^2$ | Test RMSE ($) | Test MAE ($) | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression (Full)** | 0.9537 | 0.8987 | 0.7721 | $9,055.96 | $6,961.48 | 10.28% |
| **Ridge Regression (L2)** | 0.9534 | 0.8975 | 0.7693 | $9,112.25 | $7,180.40 | 10.37% |
| **Lasso Regression (L1)** | 0.9522 | 0.9166 | 0.8124 | $8,216.99 | $6,560.58 | 9.59% |
| **Reduced Model (R&D + Marketing)** | 0.9519 | **0.9168** | **0.8931** | **$8,206.33** | **$6,469.18** | **8.84%** |

### Gauss-Markov Assumption Verification
1. **Normality of Residuals**: Shapiro-Wilk test $p = 0.836 > 0.05$. Residuals follow an ideal normal distribution.
2. **Independence of Errors**: Durbin-Watson statistic $= 1.821 \approx 2.0$, confirming no autocorrelation.
3. **Homoscedasticity**: Residual vs. fitted value plot displays random scatter around zero with constant variance.
4. **Multicollinearity**: VIF for R&D (2.47) and Marketing (2.33) are well below the threshold of 5.0.

### Business & Investment Takeaways
- **R&D is King**: For every **$1.00** invested in R&D, profit increases by **~$0.81** ($p < 0.001$).
- **Marketing Adds Modest Value**: For every **$1.00** in Marketing, profit increases by **~$0.03**.
- **Admin Spend is Non-Contributing**: Administration spend has no statistically significant relationship with profit ($p > 0.60$).
- **Geography is Neutral**: Operating in NY or Florida versus California shows no significant profit variance ($p > 0.90$).

---

## 6. Deployment

### Project Directory Structure
```
50-startup multi/
├── 50_Startups.csv                 # Raw dataset
├── main.py                         # Master CRISP-DM execution script
├── predict.py                      # Standalone CLI / inference tool
├── README.md                       # Comprehensive project documentation
├── src/
│   ├── __init__.py
│   ├── eda.py                      # Exploratory data analysis & VIF calculation
│   ├── pipeline.py                 # Scikit-learn Pipeline & transformer setup
│   └── diagnostics.py              # Statistical metrics & residual diagnostics
├── outputs/
│   ├── correlation_heatmap.png     # Correlation matrix visualization
│   ├── eda_distributions.png       # Feature & target distributions
│   ├── state_profit_distribution.png# Profit boxplot by state
│   ├── residuals_diagnostics.png   # 4-panel residual diagnostic charts
│   └── model_coefficients.png      # Feature coefficient bar chart
└── model/
    ├── startup_profit_pipeline.pkl # Production serialized pipeline
    └── model_metadata.json         # Feature schema, coefficients, and metrics
```

---

## Quick Start & Usage

### 1. Run Complete CRISP-DM Pipeline
```bash
python main.py
```
This executes all 6 phases, displays statistics, generates diagnostic plots in `outputs/`, and serializes the production model into `model/`.

### 2. Run Single Prediction via CLI
```bash
python predict.py --rd 160000 --admin 130000 --marketing 300000 --state "California"
# Output: Predicted Profit: $182,943.06
```

### 3. Run Interactive Prediction Mode
```bash
python predict.py
```

### 4. Run Batch Scoring on CSV File
```bash
python predict.py --batch new_startups.csv --output scored_startups.csv
```
