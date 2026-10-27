# HW2 Flood Prediction：模型分析與部署紀錄

官方資料來源：[Kaggle Playground Series S4E5 train.csv](https://www.kaggle.com/competitions/playground-series-s4e5/data?select=train.csv)。使用實際下載的官方競賽資料（該競賽提供合成資料），不是自行編造數值。共 1,117,957 筆、22 欄，排除 id 與 FloodProbability 後為 **20 個原始輸入特徵**。網站保留 50 Startups、Boston Housing 與自訂 CSV 入口。

## 資料下載與重現

1. 登入 Kaggle，至上述官方資料頁，依平台要求接受競賽規則，再下載 train.csv（若下載 ZIP，先解壓縮）。
2. 在專案根目錄建立 datasets/，將原始檔放為 datasets/train.csv；不可改名為 train.csv/train.csv。
3. 安裝專案所需 Python 套件後，從專案根目錄執行 `python flood-prediction/main.py` 可重現運算。本次部署修正沒有執行訓練。

完整 train.csv 與下載 ZIP 不納入 Git；原始資料保留本機。網站只需預先產生的 JS、PNG、JSON/CSV 結果，不需在 Pages 執行 Python。

## 模型與資料切分

使用 sklearn LinearRegression；random_state=42，80% 訓練 894,365 筆、20% 獨立測試 223,592 筆。訓練內再分成 proper training 715,492 筆及內部驗證／校準 178,873 筆。訓練與指標計算使用全部資料，沒有抽樣。

RFE 在 proper training 擬合，完整淘汰順序評估 k=1–20；10 特徵為固定比較基準，不是最佳子集。最新選取順序：

1. `MonsoonIntensity`
2. `Siltation`
3. `RiverManagement`
4. `DamsQuality`
5. `ClimateChange`
6. `PopulationScore`
7. `TopographyDrainage`
8. `Landslides`
9. `Deforestation`
10. `WetlandLoss`

驗證集 RMSE 最小為 k=20。測試曲線僅作描述，不用於選擇 k。

## 實測結果

數值以 model/model_metadata.json、outputs/model_evaluation_metrics.csv 及 outputs/rfe_feature_selection_metrics.csv 為準。

| 模型 | RMSE | MAE | R² | 95% 區間覆蓋率 | 平均區間寬度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LinearRegression 20 特徵 | 0.02008 | 0.01579 | 0.84488 | 94.97% | 0.08012 |
| RFE 10 特徵 | 0.04026 | 0.03249 | 0.37642 | 94.95% | 0.15501 |

點預測模型使用完整 80% 訓練集。Split Conformal 區間另外使用 proper-training 模型，校準及測試使用同一模型，校準後不重新擬合；殘差分位數採有限樣本修正與 method=higher。覆蓋率是獨立測試集的實測比例，不是每個樣本的個別保證。

本資料中保留 20 特徵比固定 RFE 10 特徵有較低誤差及較高 R²；不能單憑此結果推論特徵具有因果效果或統計顯著性。95% 區間實測覆蓋率接近目標，亦不表示所有迴歸假設均已成立。

## 三張核心圖（300 dpi）

![LinearRegression 實際值與預測值：20 特徵與 RFE 10 特徵的雙面板 Hexbin 圖；紅虛線為 y=x。](flood-prediction/outputs/actual_vs_predicted_comparison.png)

`flood-prediction/outputs/actual_vs_predicted_comparison.png`：LinearRegression 實際值與預測值：20 特徵與 RFE 10 特徵的雙面板 Hexbin 圖；紅虛線為 y=x。

![RFE 1–20 特徵的 RMSE 與 R²，並列內部驗證及獨立測試結果；最佳特徵數只由驗證集決定。](flood-prediction/outputs/feature_selection_curve.png)

`flood-prediction/outputs/feature_selection_curve.png`：RFE 1–20 特徵的 RMSE 與 R²，並列內部驗證及獨立測試結果；最佳特徵數只由驗證集決定。

![95% Split Conformal 預測區間、預測曲線與區間內外實際值；圖中標示完整測試集覆蓋率及平均寬度，橫軸不是時間。](flood-prediction/outputs/prediction_interval_95.png)

`flood-prediction/outputs/prediction_interval_95.png`：95% Split Conformal 預測區間、預測曲線與區間內外實際值；圖中標示完整測試集覆蓋率及平均寬度，橫軸不是時間。

散佈圖只為顯示，使用 RandomState(42).choice(test_rows, 40000, replace=False) 無放回抽樣；所有指標仍使用完整測試集。區間圖依 proper-model 預測排序，取 180 個等間距排名位置顯示；沒有時間序列、分箱平均線或下方直方圖。

完整 RFE 1–20 結果：[RFE CSV](flood-prediction/outputs/rfe_feature_selection_metrics.csv)；模型指標：[模型 CSV](flood-prediction/outputs/model_evaluation_metrics.csv)。資料 SHA-256 與切分、抽樣資訊位於 [metadata](flood-prediction/model/model_metadata.json)。

## 測試與部署狀態（2026-10-27，Asia/Taipei）

- Python：前輪已完成完整運算；本輪只比對 metadata 與兩份 CSV 的所有相關指標、k=1–20 完整性，以及 PNG 解碼和 300 dpi，未重訓或修改成果數據。
- localhost:8000：前輪 HTTP 檢查入口、三個專案與直接引用的本機資源皆為 200；使用者本輪已確認完成 localhost 圖表驗收。這是使用者回報的手動驗收，不是本代理新增的瀏覽器測試。
- 專案切換／CSV：保留既有實作，ml-projects.html、50 Startups 與 Boston Housing 程式未修改。先前代理未能自動操作瀏覽器或讀取 Console；不能將靜態檢查解讀為互動／Console 全面通過。
- GitHub Pages：尚待推送後由使用者手動開啟確認；不可宣稱公開網站已驗證成功。

正式網址：[Flood Prediction](https://porter974-afk.github.io/feature-selection/ml-projects.html?project=flood)。公開驗收請確認三圖可見、三個題目切換、CSV URL 及本機 CSV 匯入，並在 F12 Console／Network 檢查 JavaScript 錯誤、404 與 CDN／CORS 問題。CSV 測試可用專案的 50_Startups.csv；不需點擊訓練。

---

# 50 Startups Multiple Linear Regression (CRISP-DM)

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
