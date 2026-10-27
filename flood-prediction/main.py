"""
=============================================================================
Kaggle Flood Prediction CRISP-DM Pipeline (Playground Series S4E5)
=============================================================================
Dataset: train.csv (1,117,957 samples, 20 features, 1 target: FloodProbability)
Target:  FloodProbability (Continuous float: [0.285, 0.725])
Methods: Multiple Linear Regression (Full 20) + RFE (Select 10) +
         95% Conformal Prediction Interval Calibration
=============================================================================
"""
import os, sys, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

DATASET_PATH = "../datasets/train.csv"
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "datasets/train.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Cannot find dataset at {DATASET_PATH}")

FEATURE_DESCRIPTIONS = {
    "MonsoonIntensity": "季風降雨強度 (Monsoon Intensity)",
    "TopographyDrainage": "地形排水坡度條件 (Topography Drainage)",
    "RiverManagement": "河川整治與防洪設施 (River Management)",
    "Deforestation": "森林砍伐與植被破壞 (Deforestation)",
    "Urbanization": "都市化與不透水鋪面比率 (Urbanization)",
    "ClimateChange": "氣候變遷降雨極端化 (Climate Change)",
    "DamsQuality": "水庫與防洪大壩品質 (Dams Quality)",
    "Siltation": "河床淤積沉積程度 (Siltation)",
    "AgriculturalPractices": "農業耕作模式防蝕程度 (Agricultural Practices)",
    "Encroachments": "行水區違規侵占違建 (Encroachments)",
    "IneffectiveDisasterPreparedness": "防災整備不足程度 (Ineffective Disaster Preparedness)",
    "DrainageSystems": "下水道與排水系統效能 (Drainage Systems)",
    "CoastalVulnerability": "沿海風暴潮脆弱性 (Coastal Vulnerability)",
    "Landslides": "山坡崩塌土石流風險 (Landslides)",
    "Watersheds": "集水區保育治理程度 (Watersheds)",
    "DeterioratingInfrastructure": "水利基礎設施老化率 (Deteriorating Infrastructure)",
    "PopulationScore": "受災潛勢區人口密度 (Population Score)",
    "WetlandLoss": "天然滯洪濕地流失率 (Wetland Loss)",
    "InadequatePlanning": "防洪規劃與分洪不足 (Inadequate Planning)",
    "PoliticalFactors": "防洪政策落實度與治理 (Political Factors)"
}

print("="*75)
print("CRISP-DM PHASE 1 & 2: BUSINESS & DATA UNDERSTANDING")
print("="*75)
print(f"Loading official Kaggle dataset: {DATASET_PATH}...")
df_raw = pd.read_csv(DATASET_PATH)
print(f"Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# Exclude id and FloodProbability
target_col = "FloodProbability"
id_col = "id"
feature_cols = [c for c in df_raw.columns if c not in [id_col, target_col]]

print(f"\nTarget column: {target_col}")
print(f"Excluded column: {id_col}")
print(f"Total raw input features count: {len(feature_cols)}")
assert len(feature_cols) == 20, f"Expected 20 features, got {len(feature_cols)}"
print("Validation PASSED: Exactly 20 raw input features verified (13 < 20 < 40).")

# Missing values check
missing_total = df_raw.isnull().sum().sum()
print(f"Total missing values: {missing_total}")
print("\nTarget summary stats:")
print(df_raw[target_col].describe().round(5))

# =========================================================
# PHASE 3: DATA PREPARATION
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 3: DATA PREPARATION (Fixed random_state=42)")
print("="*75)

X = df_raw[feature_cols]
y = df_raw[target_col]

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train set: {X_train.shape[0]:,} samples (80.0%)")
print(f"Test set:  {X_test.shape[0]:,} samples (20.0%)")

# Internal calibration split for 95% Conformal Prediction Interval
# Split training data into: proper training (80% of train) and calibration set (20% of train)
X_train_prop, X_cal, y_train_prop, y_cal = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)
print(f"  Proper training subset: {X_train_prop.shape[0]:,} samples")
print(f"  Calibration subset:     {X_cal.shape[0]:,} samples")

# =========================================================
# PHASE 4: MODELING (Full 20 vs RFE 10)
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 4: MODELING (LinearRegression + RFE 10 Features)")
print("="*75)

# 1. Full 20 Features Model
print("Training Full Multiple Linear Regression (20 features)...")
full_model = LinearRegression()
full_model.fit(X_train, y_train)

# 2. RFE (Recursive Feature Elimination: Select 10 features)
print("Executing RFE (Selecting 10 features from 20)...")
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=10, step=1)
rfe_selector.fit(X_train, y_train)

rfe_selected_mask = rfe_selector.support_
rfe_features = [f for f, s in zip(feature_cols, rfe_selected_mask) if s]
rfe_rankings = {f: int(r) for f, r in zip(feature_cols, rfe_selector.ranking_)}

print(f"\nRFE Selected 10 Features: {rfe_features}")

# Train RFE 10 model
rfe_model = LinearRegression()
rfe_model.fit(X_train[rfe_features], y_train)

# Conformal calibration models (trained on proper train subset)
full_cal_model = LinearRegression().fit(X_train_prop, y_train_prop)
rfe_cal_model = LinearRegression().fit(X_train_prop[rfe_features], y_train_prop)

# Calibration nonconformity residuals: |y - y_pred|
cal_res_full = np.abs(y_cal - full_cal_model.predict(X_cal))
cal_res_rfe = np.abs(y_cal - rfe_cal_model.predict(X_cal[rfe_features]))

# 95% prediction interval quantile: ceil((n+1)*0.95)/n
n_cal = len(y_cal)
alpha = 0.05
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_95_full = float(np.quantile(cal_res_full, min(1.0, q_level)))
q_95_rfe = float(np.quantile(cal_res_rfe, min(1.0, q_level)))

print(f"95% Conformal Prediction Half-width (Full 20): ±{q_95_full:.5f} (Width: {2*q_95_full:.5f})")
print(f"95% Conformal Prediction Half-width (RFE 10):  ±{q_95_rfe:.5f} (Width: {2*q_95_rfe:.5f})")

# =========================================================
# PHASE 5: EVALUATION (Metrics + Diagnostics)
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 5: EVALUATION (Test Set: 223,592 samples)")
print("="*75)

def evaluate_model(model, X_te, y_te, q_95, name="Model"):
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, y_pred)
    
    # 95% interval coverage
    lower_bound = y_pred - q_95
    upper_bound = y_pred + q_95
    in_interval = (y_te >= lower_bound) & (y_te <= upper_bound)
    coverage = float(np.mean(in_interval)) * 100.0
    avg_width = float(2 * q_95)
    
    print(f"[{name}]")
    print(f"  Test R^2:              {r2:.5f} ({r2*100:.3f}%)")
    print(f"  Test RMSE:             {rmse:.5f}")
    print(f"  Test MSE:              {mse:.7f}")
    print(f"  Test MAE:              {mae:.5f}")
    print(f"  95% PI Test Coverage:  {coverage:.2f}% (Target: 95.0%)")
    print(f"  95% PI Avg Width:      {avg_width:.5f}")
    
    return {
        "name": name,
        "r2": round(float(r2), 5),
        "rmse": round(float(rmse), 5),
        "mse": round(float(mse), 7),
        "mae": round(float(mae), 5),
        "pi_half_width": round(float(q_95), 5),
        "pi_avg_width": round(avg_width, 5),
        "pi_coverage_pct": round(coverage, 2),
        "y_pred": y_pred
    }

eval_full = evaluate_model(full_model, X_test, y_test, q_95_full, "Full OLS (20 Features)")
eval_rfe = evaluate_model(rfe_model, X_test[rfe_features], y_test, q_95_rfe, "RFE Selected (10 Features)")

# =========================================================
# GENERATING PUBLICATION PLOTS
# =========================================================
print("\nGenerating diagnostic plots in outputs/ ...")

# 1. Actual vs Predicted Scatter Plot
print("  [Plot 1/5] actual_vs_predicted.png ...")
sample_idx = np.random.RandomState(42).choice(len(y_test), size=15000, replace=False)
y_test_sample = y_test.iloc[sample_idx].values
y_pred_sample = eval_full["y_pred"][sample_idx]

fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
hb = ax.hexbin(y_test_sample, y_pred_sample, gridsize=50, cmap='YlGnBu', mincnt=1, bins='log')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Log10(Sample Density)', fontsize=10)
min_v = min(y_test_sample.min(), y_pred_sample.min())
max_v = max(y_test_sample.max(), y_pred_sample.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal Fit Line (y = x)')
ax.set_title(f'Kaggle Flood Prediction - Actual vs Predicted\n(Full OLS 20 Features, Test R² = {eval_full["r2"]:.4f})', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Actual Flood Probability (Ground Truth)', fontsize=11)
ax.set_ylabel('Model Predicted Flood Probability', fontsize=11)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=180)
plt.close()

# 2. Residual Diagnostics Plot
print("  [Plot 2/5] residual_diagnostics.png ...")
residuals = y_test_sample - y_pred_sample
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
ax1.scatter(y_pred_sample, residuals, alpha=0.15, s=8, color='#3b82f6', edgecolor='none')
ax1.axhline(0, color='red', linestyle='--', lw=1.8)
ax1.set_title('Residuals vs Fitted Values', fontsize=11, fontweight='bold')
ax1.set_xlabel('Fitted Values (Predicted FloodProbability)', fontsize=10)
ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)

sns.histplot(residuals, kde=True, ax=ax2, color='#6366f1', bins=40, stat='density')
ax2.axvline(0, color='red', linestyle='--', lw=1.8)
ax2.set_title(f'Residual Distribution (Mean: {np.mean(residuals):.5f}, Std: {np.std(residuals):.5f})', fontsize=11, fontweight='bold')
ax2.set_xlabel('Residual Error', fontsize=10)
plt.suptitle('Kaggle Flood Prediction - Linear Regression Residual Diagnostics', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("outputs/residual_diagnostics.png", dpi=180)
plt.close()

# 3. 95% Prediction Interval Plot
print("  [Plot 3/5] prediction_interval_95.png ...")
sort_sub_idx = np.random.RandomState(42).choice(len(y_test), size=120, replace=False)
y_sub_act = y_test.iloc[sort_sub_idx].values
y_sub_pred = eval_full["y_pred"][sort_sub_idx]
order = np.argsort(y_sub_pred)
y_sub_act = y_sub_act[order]
y_sub_pred = y_sub_pred[order]
lower_pi = y_sub_pred - q_95_full
upper_pi = y_sub_pred + q_95_full

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
x_axis = np.arange(len(order))
ax.fill_between(x_axis, lower_pi, upper_pi, color='#818cf8', alpha=0.3, label=f'95% Conformal Prediction Interval (±{q_95_full:.4f})')
ax.plot(x_axis, y_sub_pred, color='#4338ca', lw=2.2, label='Predicted FloodProbability')
ax.scatter(x_axis, y_sub_act, color='#ef4444', s=24, zorder=5, label='Actual FloodProbability (Ground Truth)')
ax.set_title(f'Kaggle Flood Prediction - 95% Prediction Intervals (Calibration Test Coverage: {eval_full["pi_coverage_pct"]}%)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Sorted Test Sample Index (Representative Subset n=120)', fontsize=11)
ax.set_ylabel('Flood Probability', fontsize=11)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("outputs/prediction_interval_95.png", dpi=180)
plt.close()

# 4. Correlation Heatmap
print("  [Plot 4/5] correlation_heatmap.png ...")
sample_corr_df = df_raw.sample(n=50000, random_state=42)[[target_col] + feature_cols]
corr_matrix = sample_corr_df.corr()
fig, ax = plt.subplots(figsize=(14, 12), dpi=140)
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size": 7.5},
            cbar_kws={'label': 'Pearson Correlation'}, ax=ax, linewidths=0.5)
ax.set_title('Kaggle Flood Prediction - Pearson Correlation Heatmap (20 Features + Target)', fontsize=13, fontweight='bold', pad=14)
plt.xticks(rotation=45, ha='right', fontsize=8.5)
plt.yticks(fontsize=8.5)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=180)
plt.close()

# 5. RFE Selection & Model Comparison Plot
print("  [Plot 5/5] model_comparison.png ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Coef comparison for RFE 10 features
rfe_coef_dict = dict(zip(rfe_features, rfe_model.coef_))
sorted_rfe = sorted(rfe_coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
bars = ax1.barh([x[0] for x in sorted_rfe][::-1], [x[1] for x in sorted_rfe][::-1], color='#3b82f6')
ax1.set_title('RFE Top 10 Features - Regression Coefficients (β)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Coefficient Weight (Marginal Impact on Probability)', fontsize=10)
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 0.0001 if w >= 0 else w - 0.0008, bar.get_y() + bar.get_height()/4, f'{w:.4f}', fontsize=8)

# Metric comparison table / bar chart
metrics_names = ['Test R²', 'Test RMSE', 'Test MAE']
full_vals = [eval_full['r2'], eval_full['rmse'], eval_full['mae']]
rfe_vals  = [eval_rfe['r2'], eval_rfe['rmse'], eval_rfe['mae']]

x_m = np.arange(len(metrics_names))
bw = 0.35
ax2.bar(x_m - bw/2, full_vals, width=bw, label='Full OLS (20 Features)', color='#6366f1')
ax2.bar(x_m + bw/2, rfe_vals, width=bw, label='RFE Optimal (10 Features)', color='#10b981')
ax2.set_xticks(x_m)
ax2.set_xticklabels(metrics_names, fontsize=10)
ax2.set_title('Model Performance Comparison: Full 20 vs RFE 10', fontsize=11, fontweight='bold')
ax2.set_ylabel('Metric Value', fontsize=10)
ax2.legend(frameon=True)
for i in range(len(metrics_names)):
    ax2.text(i - bw/2, full_vals[i] + 0.008, f'{full_vals[i]:.4f}', ha='center', fontsize=8.5, fontweight='bold')
    ax2.text(i + bw/2, rfe_vals[i] + 0.008, f'{rfe_vals[i]:.4f}', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=180)
plt.close()

# Save models
print("\nSaving trained models to model/ ...")
joblib.dump(full_model, "model/flood_full_ols.pkl")
joblib.dump(rfe_model, "model/flood_rfe_10.pkl")

# Metadata JSON
metadata = {
    "project": "Kaggle_Flood_Prediction_CRISP_DM",
    "dataset_source": "Kaggle Playground Series s4e5",
    "dataset_shape": {
        "total_rows": int(df_raw.shape[0]),
        "total_cols": int(df_raw.shape[1]),
        "features_count": len(feature_cols),
        "target_column": target_col,
        "excluded_columns": [id_col, target_col]
    },
    "full_ols": {
        "features_count": 20,
        "features": feature_cols,
        "intercept": float(full_model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(feature_cols, full_model.coef_)},
        "metrics": {k: v for k, v in eval_full.items() if k != "y_pred"}
    },
    "rfe_10": {
        "features_count": 10,
        "selected_features": rfe_features,
        "feature_rankings": rfe_rankings,
        "intercept": float(rfe_model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(rfe_features, rfe_model.coef_)},
        "metrics": {k: v for k, v in eval_rfe.items() if k != "y_pred"}
    },
    "conformal_prediction_95": {
        "confidence_level": 0.95,
        "calibration_samples": int(len(y_cal)),
        "full_half_width": float(q_95_full),
        "full_avg_width": float(2 * q_95_full),
        "full_coverage_pct": float(eval_full["pi_coverage_pct"]),
        "rfe_half_width": float(q_95_rfe),
        "rfe_avg_width": float(2 * q_95_rfe),
        "rfe_coverage_pct": float(eval_rfe["pi_coverage_pct"])
    }
}

with open("model/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("  [Saved] model/model_metadata.json")

# Export dataset.js for interactive web table (500 representative test samples)
print("  Generating dataset.js (500 representative samples for web table)...")
web_sample_idx = np.random.RandomState(42).choice(len(X_test), size=500, replace=False)
web_sample_df = X_test.iloc[web_sample_idx].copy()
web_sample_df[target_col] = y_test.iloc[web_sample_idx].values
web_sample_df["PRED_FULL"] = np.round(full_model.predict(web_sample_df[feature_cols]), 4)
web_sample_df["PRED_RFE"] = np.round(rfe_model.predict(web_sample_df[rfe_features]), 4)

rows_js = []
for _, row in web_sample_df.iterrows():
    r = {col: round(float(row[col]), 4) for col in web_sample_df.columns}
    rows_js.append(json.dumps(r, ensure_ascii=False))

dataset_js = "const FLOOD_DATA = [\n  " + ",\n  ".join(rows_js) + "\n];\n"
with open("dataset.js", "w", encoding="utf-8") as f:
    f.write(dataset_js)
print("  [Saved] dataset.js")

# Export model_config.js for client-side simulator and charts
print("  Generating model_config.js...")
model_config_js = f"""// Kaggle Flood Prediction ML Model Configuration (auto-generated by main.py)
const FLOOD_MODEL = {{
  // Metadata
  projectName: "Kaggle Flood Prediction",
  competition: "Playground Series s4e5",
  totalSamples: {int(df_raw.shape[0])},
  totalFeatures: 20,
  targetColumn: "{target_col}",
  
  // Full OLS (20 features)
  fullFeatures: {json.dumps(feature_cols)},
  fullIntercept: {full_model.intercept_},
  fullCoefficients: {json.dumps({f: float(c) for f, c in zip(feature_cols, full_model.coef_)})},
  evalFull: {json.dumps({k: v for k, v in eval_full.items() if k != 'y_pred'})},
  
  // RFE (10 features)
  rfeFeatures: {json.dumps(rfe_features)},
  rfeRankings: {json.dumps(rfe_rankings)},
  rfeIntercept: {rfe_model.intercept_},
  rfeCoefficients: {json.dumps({f: float(c) for f, c in zip(rfe_features, rfe_model.coef_)})},
  evalRfe: {json.dumps({k: v for k, v in eval_rfe.items() if k != 'y_pred'})},
  
  // 95% Conformal Prediction Interval
  conformalPI: {{
    confidence: 0.95,
    q95Full: {q_95_full},
    q95Rfe: {q_95_rfe},
    fullWidth: {2 * q_95_full},
    rfeWidth: {2 * q_95_rfe},
    fullCoverage: {eval_full['pi_coverage_pct']},
    rfeCoverage: {eval_rfe['pi_coverage_pct']}
  }},
  
  // Feature Descriptions
  featureDescriptions: {json.dumps(FEATURE_DESCRIPTIONS, ensure_ascii=False)}
}};
"""

with open("model_config.js", "w", encoding="utf-8") as f:
    f.write(model_config_js)
print("  [Saved] model_config.js")

print("\n" + "="*75)
print("CRISP-DM PIPELINE COMPLETED SUCCESSFULLY!")
print("="*75)
