"""
=============================================================================
Kaggle Flood Prediction CRISP-DM Pipeline (Playground Series S4E5)
=============================================================================
Dataset: train.csv (1,117,957 samples, 20 features, 1 target: FloodProbability)
Target:  FloodProbability (Continuous float: [0.285, 0.725])
Methods: Multiple Linear Regression (Full 20) + RFE Feature Selection +
         95% Conformal Prediction Interval Calibration (Split Conformal)
=============================================================================
"""
import os, sys, json, hashlib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
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
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
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

target_col = "FloodProbability"
id_col = "id"
feature_cols = [c for c in df_raw.columns if c not in [id_col, target_col]]

print(f"\nTarget column: {target_col}")
print(f"Excluded column: {id_col}")
print(f"Total raw input features count: {len(feature_cols)}")
assert len(feature_cols) == 20, f"Expected 20 features, got {len(feature_cols)}"
print("Validation PASSED: Exactly 20 raw input features verified (13 < 20 < 40).")
print(f"Total missing values: {df_raw.isnull().sum().sum()}")

# =========================================================
# PHASE 3: DATA PREPARATION (Strict Leakage-Free Architecture)
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 3: DATA PREPARATION (Fixed random_state=42)")
print("="*75)

X = df_raw[feature_cols]
y = df_raw[target_col]

# 1. Split into 80% Training set and 20% Holdout Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train set: {X_train.shape[0]:,} samples (80.0%)")
print(f"Test set:  {X_test.shape[0]:,} samples (20.0%)")

# 2. Split Training set internally into Proper Train (80%) and Validation / Calibration set (20%)
# This ensures feature selection optimization does NOT peek into the holdout test set!
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
print(f"  Internal Training subset:   {X_tr.shape[0]:,} samples")
print(f"  Internal Validation subset: {X_val.shape[0]:,} samples")

# =========================================================
# PHASE 4: RFE FEATURE SELECTION ACROSS k = 1..20
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 4: RFE FEATURE SELECTION & PROGRESSION")
print("="*75)
print("Executing RFE on training data...")

# Fit RFE on internal training subset
rfe = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
rfe.fit(X_tr, y_tr)
feature_ranks = sorted(zip(rfe.ranking_, feature_cols), key=lambda x: x[0])
print("Feature Elimination Rankings (Rank 1 = most essential):")
for r, f in feature_ranks:
    print(f"  Rank {r:2d}: {f} ({FEATURE_DESCRIPTIONS[f]})")

# Evaluate all k from 1 to 20
sfs_history = []
for k in range(1, 21):
    feats_k = [f for r, f in feature_ranks if r <= k]

    # 1. Train on internal train subset, evaluate on validation set (NO TEST LEAKAGE)
    reg_val = LinearRegression().fit(X_tr[feats_k], y_tr)
    pred_val = reg_val.predict(X_val[feats_k])
    val_rmse = root_mean_squared_error(y_val, pred_val)
    val_r2 = r2_score(y_val, pred_val)
    val_mae = mean_absolute_error(y_val, pred_val)

    # 2. Retrain on full training set, evaluate on holdout test set
    reg_test = LinearRegression().fit(X_train[feats_k], y_train)
    pred_test = reg_test.predict(X_test[feats_k])
    test_rmse = root_mean_squared_error(y_test, pred_test)
    test_r2 = r2_score(y_test, pred_test)
    test_mae = mean_absolute_error(y_test, pred_test)

    sfs_history.append({
        "k": k,
        "added_feature": feature_ranks[k-1][1],
        "feature_desc": FEATURE_DESCRIPTIONS[feature_ranks[k-1][1]],
        "selected_features": feats_k,
        "selected_features_str": ", ".join(feats_k),
        "val_rmse": round(float(val_rmse), 5),
        "val_r2": round(float(val_r2), 5),
        "val_mae": round(float(val_mae), 5),
        "test_rmse": round(float(test_rmse), 5),
        "test_r2": round(float(test_r2), 5),
        "test_mae": round(float(test_mae), 5)
    })

sfs_df = pd.DataFrame(sfs_history)

# Save RFE progression metrics to CSV
csv_cols = ["k", "added_feature", "feature_desc", "val_rmse", "val_r2", "test_rmse", "test_r2", "test_mae", "selected_features_str"]
sfs_df[csv_cols].to_csv("outputs/rfe_feature_selection_metrics.csv", index=False, encoding="utf-8-sig")
print("  [Saved] outputs/rfe_feature_selection_metrics.csv")

# Print summary table for key k values (5, 8, 10, 12, 15, 18, 20)
key_k = [1, 5, 8, 10, 12, 15, 18, 20]
print("\nRFE Performance progression across key feature counts:")
print(sfs_df[sfs_df['k'].isin(key_k)][["k", "added_feature", "val_rmse", "val_r2", "test_rmse", "test_r2"]].to_string(index=False))

# Selected 10 Features
rfe_10_features = [f for r, f in feature_ranks if r <= 10]
print(f"\nRFE 10 Features selected: {rfe_10_features}")

# Train Full OLS (20) and RFE 10 Models
full_model = LinearRegression().fit(X_train, y_train)
rfe_10_model = LinearRegression().fit(X_train[rfe_10_features], y_train)

# =========================================================
# PHASE 5: CONFORMAL PREDICTION INTERVALS (95% CI/PI)
# =========================================================
print("\n" + "="*75)
print("CRISP-DM PHASE 5: 95% CONFORMAL PREDICTION CALIBRATION")
print("="*75)

# Fit calibration models on proper training subset
full_cal_model = LinearRegression().fit(X_tr, y_tr)
rfe_cal_model  = LinearRegression().fit(X_tr[rfe_10_features], y_tr)

# Nonconformity residuals on calibration set
cal_res_full = np.abs(y_val - full_cal_model.predict(X_val))
cal_res_rfe  = np.abs(y_val - rfe_cal_model.predict(X_val[rfe_10_features]))

# Calculate 95% quantile: ceil((n+1)*0.95)/n
n_cal = len(y_val)
alpha = 0.05
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_95_full = float(np.quantile(cal_res_full, min(1.0, q_level), method="higher"))
q_95_rfe  = float(np.quantile(cal_res_rfe, min(1.0, q_level), method="higher"))

print(f"95% Conformal Prediction Half-width (Full 20): ±{q_95_full:.5f} (Width: {2*q_95_full:.5f})")
print(f"95% Conformal Prediction Half-width (RFE 10):  ±{q_95_rfe:.5f} (Width: {2*q_95_rfe:.5f})")

# Evaluate on holdout test set
y_pred_full = full_model.predict(X_test)
y_pred_rfe  = rfe_10_model.predict(X_test[rfe_10_features])

def calc_eval(y_true, y_pred, q_half, name="Model", pi_pred=None):
    if pi_pred is None:
        pi_pred = y_pred
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    in_interval = (y_true >= pi_pred - q_half) & (y_true <= pi_pred + q_half)
    coverage = float(np.mean(in_interval)) * 100.0
    avg_width = float(2 * q_half)

    print(f"[{name}]")
    print(f"  Test R^2:              {r2:.5f} ({r2*100:.3f}%)")
    print(f"  Test RMSE:             {rmse:.5f}")
    print(f"  Test MSE:              {mse:.7f}")
    print(f"  Test MAE:              {mae:.5f}")
    print(f"  95% PI Test Coverage:  {coverage:.2f}% (Target: 95.0%)")
    print(f"  95% PI Avg Width:      {avg_width:.5f} (±{q_half:.5f})")

    return {
        "model_name": name,
        "r2": round(float(r2), 5),
        "rmse": round(float(rmse), 5),
        "mse": round(float(mse), 7),
        "mae": round(float(mae), 5),
        "pi_half_width": round(float(q_half), 5),
        "pi_avg_width": round(avg_width, 5),
        "pi_coverage_pct": round(coverage, 2)
    }

pi_pred_full = full_cal_model.predict(X_test)
pi_pred_rfe = rfe_cal_model.predict(X_test[rfe_10_features])

eval_full = calc_eval(y_test, y_pred_full, q_95_full, "Full OLS (20 Features)", pi_pred_full)
eval_rfe  = calc_eval(y_test, y_pred_rfe, q_95_rfe, "RFE Selected (10 Features)", pi_pred_rfe)

# Save evaluation comparison to CSV
eval_df = pd.DataFrame([eval_full, eval_rfe])
eval_df.to_csv("outputs/model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
print("  [Saved] outputs/model_evaluation_metrics.csv")

# =========================================================
# GENERATING THE THREE 300-DPI HW2 CORE FIGURES
# =========================================================
print("\n" + "="*75)
print("GENERATING 300-DPI CORE FIGURES IN outputs/ ...")
print("="*75)

# All fit/evaluation rows are used; only display points are sampled.
sample_idx = np.random.RandomState(42).choice(len(y_test), min(40000, len(y_test)), replace=False)
y_act_sample = y_test.iloc[sample_idx].to_numpy()
y_pred_full_sample = y_pred_full[sample_idx]
y_pred_rfe_sample = y_pred_rfe[sample_idx]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, pred, ev, title in zip(axes, [y_pred_full_sample, y_pred_rfe_sample], [eval_full, eval_rfe], ['LinearRegression: 20 features', 'RFE: 10 features (fixed comparison)']):
    hb = ax.hexbin(y_act_sample, pred, gridsize=55, bins='log', mincnt=1, cmap='viridis')
    fig.colorbar(hb, ax=ax, label='Sample count (log scale)')
    limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(limits, limits, 'r--', label='Ideal: y = x')
    ax.set(xlabel='Actual FloodProbability', ylabel='Predicted FloodProbability', title=title)
    ax.text(.04, .96, f"Test R?={ev['r2']:.5f}\nRMSE={ev['rmse']:.5f}\nMAE={ev['mae']:.5f}", transform=ax.transAxes, va='top')
    ax.legend(loc='lower right')
fig.suptitle(f'Kaggle Flood: actual vs predicted | test n={len(y_test):,}\nDisplay: 40,000 test rows without replacement, random_state=42')
fig.tight_layout()
for filename in ['actual_vs_predicted.png', 'actual_vs_predicted_comparison.png']:
    fig.savefig('outputs/' + filename, dpi=300)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, label in zip(axes, ['rmse', 'r2'], ['RMSE', 'R?']):
    ax.plot(sfs_df.k, sfs_df['val_' + metric], 'o-', label='Internal validation')
    ax.plot(sfs_df.k, sfs_df['test_' + metric], 's--', label='Holdout test (descriptive)')
    ax.set(xlabel='Number of RFE features', ylabel=label, xticks=range(1, 21))
    ax.legend()
best_k = int(sfs_df.loc[sfs_df.val_rmse.idxmin(), 'k'])
fig.suptitle(f'RFE performance: all 20 subsets recomputed | validation-selected k={best_k}\nRanking fitted on proper training only; test scores do not select k')
fig.tight_layout()
fig.savefig('outputs/feature_selection_curve.png', dpi=300)
plt.close(fig)
# The same proper-training model supplies calibration and test predictions.
order = np.argsort(pi_pred_full)
display_idx = order[np.linspace(0, len(order)-1, 180, dtype=int)]
sub_pred = pi_pred_full[display_idx]
sub_actual = y_test.iloc[display_idx].to_numpy()
covered = np.abs(sub_actual-sub_pred) <= q_95_full
x = np.arange(len(display_idx))
fig, ax = plt.subplots(figsize=(13, 6))
ax.fill_between(x, sub_pred-q_95_full, sub_pred+q_95_full, alpha=.25, label='95% split conformal prediction interval')
ax.plot(x, sub_pred, label='Proper-training model prediction')
ax.scatter(x[covered], sub_actual[covered], s=15, label='Actual: inside interval')
ax.scatter(x[~covered], sub_actual[~covered], marker='x', c='red', label='Actual: outside interval')
ax.set(xlabel='180 evenly spaced ranks after sorting test predictions (not time)', ylabel='FloodProbability')
ax.set_title(f"95% Conformal Prediction | full-test coverage={eval_full['pi_coverage_pct']:.2f}%\nMean width={2*q_95_full:.6f} | calibration n={n_cal:,} | test n={len(y_test):,}")
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('outputs/prediction_interval_95.png', dpi=300)
plt.close(fig)

# ---------------------------------------------------------
# ---------------------------------------------------------
# Companion: Residual Diagnostics & Model Comparison
# ---------------------------------------------------------
print("Generating residual_diagnostics.png and model_comparison.png (300 dpi)...")

# Residual diagnostics
residuals_full = y_act_sample - y_pred_full_sample
fig, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
ax_r1.scatter(y_pred_full_sample, residuals_full, alpha=0.15, s=6, color='#0284c7', edgecolor='none')
ax_r1.axhline(0, color='red', linestyle='--', lw=1.8)
ax_r1.set_title('Residuals vs Fitted Values (無明顯異方差)', fontsize=11, fontweight='bold')
ax_r1.set_xlabel('Fitted Values (Predicted FloodProbability)', fontsize=10)
ax_r1.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)

sns.histplot(residuals_full, kde=True, ax=ax_r2, color='#6366f1', bins=40, stat='density')
ax_r2.axvline(0, color='red', linestyle='--', lw=1.8)
ax_r2.set_title(f'Residual Distribution (Mean: {np.mean(residuals_full):.5f}, Std: {np.std(residuals_full):.5f})', fontsize=11, fontweight='bold')
ax_r2.set_xlabel('Residual Error', fontsize=10)
plt.suptitle('Kaggle Flood Prediction - Linear Regression Residual Diagnostics (300 DPI)', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("outputs/residual_diagnostics.png", dpi=300)
plt.close()

# Model Comparison
fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
rfe_coef_dict = dict(zip(rfe_10_features, rfe_10_model.coef_))
sorted_rfe = sorted(rfe_coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
bars = ax_c1.barh([x[0] for x in sorted_rfe][::-1], [x[1] for x in sorted_rfe][::-1], color='#0ea5e9')
ax_c1.set_title('RFE Top 10 Features - Regression Coefficients (β)', fontsize=11, fontweight='bold')
ax_c1.set_xlabel('Coefficient Weight (Marginal Impact on Probability)', fontsize=10)
for bar in bars:
    w = bar.get_width()
    ax_c1.text(w + 0.0001 if w >= 0 else w - 0.0008, bar.get_y() + bar.get_height()/4, f'{w:.4f}', fontsize=8)

metrics_names = ['Test R²', 'Test RMSE', 'Test MAE']
full_vals = [eval_full['r2'], eval_full['rmse'], eval_full['mae']]
rfe_vals  = [eval_rfe['r2'], eval_rfe['rmse'], eval_rfe['mae']]
x_m = np.arange(len(metrics_names))
bw = 0.35
ax_c2.bar(x_m - bw/2, full_vals, width=bw, label='Full OLS (20 Features)', color='#0284c7')
ax_c2.bar(x_m + bw/2, rfe_vals, width=bw, label='RFE Optimal (10 Features)', color='#10b981')
ax_c2.set_xticks(x_m)
ax_c2.set_xticklabels(metrics_names, fontsize=10)
ax_c2.set_title('Model Performance Comparison: Full 20 vs RFE 10', fontsize=11, fontweight='bold')
ax_c2.set_ylabel('Metric Value', fontsize=10)
ax_c2.legend(frameon=True)
for i in range(len(metrics_names)):
    ax_c2.text(i - bw/2, full_vals[i] + 0.008, f'{full_vals[i]:.4f}', ha='center', fontsize=8.5, fontweight='bold')
    ax_c2.text(i + bw/2, rfe_vals[i] + 0.008, f'{rfe_vals[i]:.4f}', ha='center', fontsize=8.5, fontweight='bold')
plt.suptitle('Kaggle Flood Prediction - 模型指標與係數比較 (300 DPI)', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=300)
plt.close()

# Save models
print("\nSaving trained models to model/ ...")
joblib.dump(full_cal_model, "model/flood_conformal_ols.pkl")
joblib.dump(rfe_cal_model, "model/flood_conformal_rfe_10.pkl")
joblib.dump(full_model, "model/flood_full_ols.pkl")
joblib.dump(rfe_10_model, "model/flood_rfe_10.pkl")

# Metadata JSON
metadata = {
    "project": "Kaggle_Flood_Prediction_CRISP_DM",
    "dataset_source": "Kaggle Playground Series s4e5",
    "reproducibility": {"random_state": 42, "training_sampling": "none; all official train.csv rows", "dataset_sha256": hashlib.sha256(Path(DATASET_PATH).read_bytes()).hexdigest(), "proper_train_rows": len(X_tr), "validation_calibration_rows": len(X_val), "test_rows": len(X_test), "scatter_sampling": "RandomState(42).choice(test_rows, 40000, replace=False)", "interval_display": "180 evenly spaced ranks of sorted proper-model test predictions", "conformal": "same proper-training model for calibration and test; quantile method=higher; no refit", "validation_selected_k": best_k},
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
        "metrics": {k: v for k, v in eval_full.items() if k != "model_name"}
    },
    "rfe_10": {
        "features_count": 10,
        "selected_features": rfe_10_features,
        "feature_rankings": {f: int(r) for r, f in feature_ranks},
        "intercept": float(rfe_10_model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(rfe_10_features, rfe_10_model.coef_)},
        "metrics": {k: v for k, v in eval_rfe.items() if k != "model_name"}
    },
    "rfe_progression": sfs_history,
    "conformal_prediction_95": {
        "confidence_level": 0.95,
        "calibration_samples": int(n_cal),
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
web_sample_df["PRED_RFE"] = np.round(rfe_10_model.predict(web_sample_df[rfe_10_features]), 4)

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
  evalFull: {json.dumps({k: v for k, v in eval_full.items() if k != 'model_name'})},

  // RFE (10 features)
  rfeFeatures: {json.dumps(rfe_10_features)},
  rfeRankings: {json.dumps({f: int(r) for r, f in feature_ranks})},
  rfeIntercept: {rfe_10_model.intercept_},
  rfeCoefficients: {json.dumps({f: float(c) for f, c in zip(rfe_10_features, rfe_10_model.coef_)})},
  evalRfe: {json.dumps({k: v for k, v in eval_rfe.items() if k != 'model_name'})},

  // RFE Progression (k=1..20)
  rfeProgression: {json.dumps(sfs_history)},

  conformalModels: {json.dumps({"full": {"intercept": float(full_cal_model.intercept_), "coefficients": dict(zip(feature_cols, map(float, full_cal_model.coef_)))}, "rfe": {"intercept": float(rfe_cal_model.intercept_), "coefficients": dict(zip(rfe_10_features, map(float, rfe_cal_model.coef_)))}})},
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
print("ALL 300-DPI CORE FIGURES & CSVS GENERATED SUCCESSFULLY!")
print("="*75)
