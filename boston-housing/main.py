"""
=============================================================================
Boston House Price CRISP-DM Complete Pipeline (Kaggle HousingData.csv)
=============================================================================
Dataset: HousingData.csv (506 samples, 13 features, with NA values)
Target:  MEDV - Median value of owner-occupied homes ($1000s)
Methods: Multiple Linear Regression + Ridge + Lasso + SFS Feature Selection
=============================================================================
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("src", exist_ok=True)

FEATURE_DESCRIPTIONS = {
    "CRIM":    "各鎮人均犯罪率 (per capita crime rate)",
    "ZN":      "25,000 sq.ft.以上住宅用地比例",
    "INDUS":   "非零售商業用地佔比 (%)",
    "CHAS":    "查爾斯河臨河虛擬變數 (1=臨河, 0=否)",
    "NOX":     "一氧化氮濃度 (parts per 10M)",
    "RM":      "每戶平均房間數",
    "AGE":     "1940年前建成的自有住宅佔比 (%)",
    "DIS":     "至波士頓五大就業中心加權距離",
    "RAD":     "高速公路可及性指數",
    "TAX":     "每萬美元財產稅率 ($)",
    "PTRATIO": "各鎮師生比",
    "B":       "1000(Bk-0.63)², Bk=黑人人口比例",
    "LSTAT":   "低收入戶人口佔比 (%)",
    "MEDV":    "自有住宅中位數價格 ($1000s) [目標變數]"
}

# =========================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =========================================================
print("="*70)
print("CRISP-DM PHASE 1: BUSINESS UNDERSTANDING")
print("="*70)
print("""
[Business Context]
Real estate investors and urban planners need quantitative models to
accurately price homes and identify key drivers of residential property value.

[Key Business Questions]
1. Which features most strongly predict Boston suburban house prices?
2. How much does each additional room (RM) add to property value?
3. Does crime rate (CRIM) and low-income ratio (LSTAT) depress prices?
4. What is the minimal optimal feature set (CRISP-DM forward SFS)?

[Success Criteria]
- Test R2 >= 0.70 | RMSE < $5,000 | Gauss-Markov assumptions verified
""")

# =========================================================
# PHASE 2: DATA UNDERSTANDING
# =========================================================
print("="*70)
print("CRISP-DM PHASE 2: DATA UNDERSTANDING")
print("="*70)

df_raw = pd.read_csv("../feature-selection/HousingData.csv")
print(f"\nRaw dataset: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"Missing values per column:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Impute NA with median (industry standard for Boston dataset)
df = df_raw.copy()
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Imputed {col}: {df_raw[col].isnull().sum()} NAs -> median={median_val:.4f}")

print(f"\nClean dataset: {df.shape[0]} rows x {df.shape[1]} cols, Missing={df.isnull().sum().sum()}")
print("\nMEDV (target) statistics:")
print(df['MEDV'].describe().round(3))

correlations = df.corr()['MEDV'].drop('MEDV').sort_values(key=abs, ascending=False)
print("\nTop correlations with MEDV:")
for f, c in correlations.items():
    print(f"  {f:10s}: {c:+.4f}")

# EDA Plots
print("\nGenerating EDA plots...")
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
numeric_feats = [c for c in df.columns if c != 'MEDV']
for i, feat in enumerate(numeric_feats):
    axes[i].hist(df[feat], bins=25, color='#2563eb', alpha=0.75, edgecolor='white')
    axes[i].set_title(feat, fontsize=10, fontweight='bold')
    axes[i].set_xlabel(FEATURE_DESCRIPTIONS.get(feat, feat)[:20], fontsize=7)
# MEDV histogram
axes[13].hist(df['MEDV'], bins=25, color='#10b981', alpha=0.75, edgecolor='white')
axes[13].set_title('MEDV (Target)', fontsize=10, fontweight='bold')
axes[14].axis('off')
plt.suptitle('Boston Housing - Feature Distributions (EDA)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/eda_distributions.png")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 11))
corr_mat = df.corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            annot_kws={"size": 8}, linewidths=0.5, mask=mask)
ax.set_title('Boston Housing - Pearson Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/correlation_heatmap.png")

# Scatter: RM, LSTAT, PTRATIO vs MEDV
top3 = ['RM', 'LSTAT', 'PTRATIO']
colors = ['#2563eb', '#ef4444', '#f59e0b']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (feat, color) in enumerate(zip(top3, colors)):
    axes[i].scatter(df[feat], df['MEDV'], alpha=0.35, s=18, color=color)
    try:
        m, b = np.polyfit(df[feat].astype(float), df['MEDV'].astype(float), 1)
        xl = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(xl, m*xl+b, 'k--', lw=2, label=f'r={df[feat].corr(df["MEDV"]):.3f}')
    except Exception:
        axes[i].text(0.5, 0.05, f'r={df[feat].corr(df["MEDV"]):.3f}', transform=axes[i].transAxes, ha='center')
    axes[i].set_xlabel(feat, fontsize=11); axes[i].set_ylabel('MEDV ($1000s)', fontsize=10)
    axes[i].set_title(f'{feat} vs MEDV', fontsize=12, fontweight='bold')
    axes[i].legend(fontsize=10, loc='upper right')
plt.suptitle('Boston Housing - Top Features vs House Price', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/scatter_top_features.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/scatter_top_features.png")

# State distribution (CHAS)
fig, ax = plt.subplots(figsize=(7, 4))
df.groupby('CHAS')['MEDV'].mean().plot(kind='bar', ax=ax, color=['#6366f1','#10b981'], edgecolor='white')
ax.set_title('Average MEDV by CHAS (Charles River)', fontsize=12, fontweight='bold')
ax.set_xlabel('CHAS (0=Non-River, 1=Riverside)'); ax.set_ylabel('Mean MEDV ($1000s)')
ax.set_xticklabels(['Non-River', 'Riverside'], rotation=0)
plt.tight_layout()
plt.savefig('outputs/chas_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/chas_distribution.png")

# =========================================================
# PHASE 3: DATA PREPARATION
# =========================================================
print("\n" + "="*70)
print("CRISP-DM PHASE 3: DATA PREPARATION")
print("="*70)

X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

# VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data.sort_values("VIF", ascending=False)
print("\nVariance Inflation Factors (VIF):")
print(vif_data.to_string(index=False))

# =========================================================
# PHASE 4: MODELING + SEQUENTIAL FORWARD FEATURE SELECTION
# =========================================================
print("\n" + "="*70)
print("CRISP-DM PHASE 4: MODELING + SEQUENTIAL FEATURE SELECTION")
print("="*70)

# --- 4.0 SFS ---
print("\n[4.0] Sequential Forward Feature Selection...")
all_features = list(X.columns)
selected = []
remaining = list(all_features)
sfs_history = []

for step in range(1, len(all_features)+1):
    best_r2, best_rmse, best_feat = -np.inf, None, None
    for feat in remaining:
        cand = selected + [feat]
        reg = LinearRegression().fit(X_train[cand], y_train)
        pred = reg.predict(X_test[cand])
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        if r2 > best_r2:
            best_r2, best_rmse, best_feat = r2, rmse, feat
    selected.append(best_feat)
    remaining.remove(best_feat)
    sfs_history.append({"k": step, "added_feature": best_feat,
                         "selected_features": list(selected),
                         "rmse": round(best_rmse, 4), "r2": round(best_r2, 6)})

sfs_df = pd.DataFrame(sfs_history)
print(sfs_df[["k","added_feature","rmse","r2"]].to_string(index=False))

best_idx = sfs_df["rmse"].idxmin()
best_row = sfs_df.loc[best_idx]
optimal_k = int(best_row['k'])
optimal_features = list(best_row['selected_features'])
print(f"\n>>> Optimal k = {optimal_k}: {optimal_features}")
print(f">>> Min Test RMSE: ${best_row['rmse']:.4f}k | Peak R2: {best_row['r2']:.5f}")

# SFS Curve (matching screenshot style)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(sfs_df['k'], sfs_df['rmse'], marker='o', color='#1f77b4', lw=1.8, markersize=6)
axes[0].set_title('RMSE by Number of Features', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('RMSE')
axes[0].set_xticks(sfs_df['k'])
axes[1].plot(sfs_df['k'], sfs_df['r2'], marker='o', color='#1f77b4', lw=1.8, markersize=6)
axes[1].set_title('R-squared by Number of Features', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Number of Features'); axes[1].set_ylabel('R-squared')
axes[1].set_xticks(sfs_df['k'])
plt.tight_layout()
plt.savefig('outputs/feature_selection_curve.png', dpi=300)
plt.close()
print("  [Saved] outputs/feature_selection_curve.png")

# --- 4.1 Four Candidate Models ---
print("\n[4.1] Training 4 candidate models...")
alphas = np.logspace(-3, 3, 50)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train); X_te_sc = scaler.transform(X_test)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_def = {
    "Multiple Linear Regression (Full OLS)": ("linear", False),
    "Ridge Regression (L2 Regularized)":     ("ridge",  True),
    "Lasso Regression (L1 Regularized)":     ("lasso",  True),
    f"Reduced SFS Model (k={optimal_k})":    ("reduced", False),
}

fitted = {}; eval_rows = []; cv_metrics_all = {}

for name, (mtype, scaled) in models_def.items():
    Xtr = X_tr_sc if scaled else X_train.values
    Xte = X_te_sc  if scaled else X_test.values
    if mtype == "reduced":
        Xtr = X_train[optimal_features].values
        Xte = X_test[optimal_features].values
        n_feats = optimal_k
    else:
        n_feats = X.shape[1]

    if   mtype == "linear":  model = LinearRegression()
    elif mtype == "ridge":   model = RidgeCV(alphas=alphas, cv=5)
    elif mtype == "lasso":   model = LassoCV(cv=5, max_iter=20000, random_state=42)
    else:                    model = LinearRegression()

    model.fit(Xtr, y_train)
    y_tr_p = model.predict(Xtr); y_te_p = model.predict(Xte)
    r2_tr = r2_score(y_train, y_tr_p); r2_te = r2_score(y_test, y_te_p)
    n_te = len(y_test)
    adj_r2 = 1-(1-r2_te)*(n_te-1)/(n_te-n_feats-1)
    rmse = np.sqrt(mean_squared_error(y_test, y_te_p))
    mae  = mean_absolute_error(y_test, y_te_p)
    mape = mean_absolute_percentage_error(y_test, y_te_p)*100

    cv_r = cross_validate(LinearRegression() if mtype in ("linear","reduced") else model,
                          Xtr, y_train,
                          scoring={"r2":"r2","rmse":"neg_root_mean_squared_error"},
                          cv=kf)
    cv_metrics_all[name] = {"cv_r2_mean": float(np.mean(cv_r['test_r2'])),
                             "cv_rmse_mean": float(-np.mean(cv_r['test_rmse']))}

    fitted[name] = {"model":model,"y_te_p":y_te_p,"y_tr_p":y_tr_p,"scaled":scaled,"mtype":mtype}
    eval_rows.append({"Model":name,"Train R2":f"{r2_tr:.4f}","Test R2":f"{r2_te:.4f}",
                      "Adj R2":f"{adj_r2:.4f}","RMSE":f"{rmse:.4f}","MAE":f"{mae:.4f}","MAPE":f"{mape:.2f}%"})
    print(f"  [OK] {name}")

print("\nModel Evaluation:")
print(pd.DataFrame(eval_rows).to_string(index=False))

# =========================================================
# PHASE 5: EVALUATION
# =========================================================
print("\n" + "="*70)
print("CRISP-DM PHASE 5: EVALUATION")
print("="*70)

full_ols = fitted["Multiple Linear Regression (Full OLS)"]["model"]
y_te_p  = fitted["Multiple Linear Regression (Full OLS)"]["y_te_p"]
y_tr_p  = fitted["Multiple Linear Regression (Full OLS)"]["y_tr_p"]
residuals = y_test.values - y_te_p

shap_stat, shap_p = stats.shapiro(residuals)
dw_val = float(durbin_watson(residuals))
print(f"\nShapiro-Wilk: stat={shap_stat:.4f}, p={shap_p:.4f} -> {'Normal' if shap_p>0.05 else 'Non-normal (clipped data)'}")
print(f"Durbin-Watson: {dw_val:.4f} -> {'No autocorrelation' if 1.5<=dw_val<=2.5 else 'Possible autocorrelation'}")

# Residual 4-panel
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
all_a = np.concatenate([y_train.values, y_test.values])
all_p = np.concatenate([y_tr_p, y_te_p])
mn, mx = all_a.min()*0.9, all_a.max()*1.05

axes[0,0].scatter(y_train, y_tr_p, color='#2563eb', alpha=0.55, s=25, label='Train')
axes[0,0].scatter(y_test, y_te_p,  color='#ef4444', alpha=0.85, s=35, marker='^', label='Test')
axes[0,0].plot([mn,mx],[mn,mx],'k--',lw=1.5, label='Perfect Fit')
axes[0,0].set_xlabel('Actual MEDV ($1000s)'); axes[0,0].set_ylabel('Predicted MEDV')
axes[0,0].set_title('Actual vs Predicted', fontweight='bold'); axes[0,0].legend(fontsize=9)

axes[0,1].scatter(y_te_p, residuals, color='#8b5cf6', alpha=0.6, s=25)
axes[0,1].axhline(0, color='red', lw=1.5, ls='--')
axes[0,1].set_xlabel('Fitted Values'); axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residuals vs Fitted', fontweight='bold')

axes[1,0].hist(residuals, bins=25, color='#10b981', alpha=0.75, edgecolor='white')
xr = np.linspace(residuals.min(), residuals.max(), 100)
mu, sig = np.mean(residuals), np.std(residuals)
axes[1,0].plot(xr, stats.norm.pdf(xr,mu,sig)*len(residuals)*(residuals.max()-residuals.min())/25, 'r-', lw=2)
axes[1,0].set_title('Residual Distribution', fontweight='bold'); axes[1,0].set_xlabel('Residual')

(osm, osr),(slope,intercept,r) = stats.probplot(residuals, dist='norm')
axes[1,1].scatter(osm, osr, color='#f59e0b', s=15, alpha=0.8)
axes[1,1].plot(osm, slope*np.array(osm)+intercept, 'r--', lw=1.5)
axes[1,1].set_title('Normal Q-Q Plot', fontweight='bold')
axes[1,1].set_xlabel('Theoretical Quantiles'); axes[1,1].set_ylabel('Sample Quantiles')

plt.suptitle('Boston Housing - Full OLS Regression Diagnostics', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/residuals_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/residuals_diagnostics.png")

# Coefficients bar chart
coef_df = pd.DataFrame({'Feature':X.columns,'Coefficient':full_ols.coef_}).sort_values('Coefficient')
bar_colors = ['#ef4444' if c<0 else '#10b981' for c in coef_df['Coefficient']]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=bar_colors, edgecolor='white', height=0.65)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Boston Housing - Feature Coefficients (Full OLS)', fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient Value')
for bar, val in zip(ax.patches, coef_df['Coefficient']):
    ax.text(val + (0.05 if val>=0 else -0.05), bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left' if val>=0 else 'right', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/model_coefficients.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [Saved] outputs/model_coefficients.png")

# =========================================================
# PHASE 6: DEPLOYMENT
# =========================================================
print("\n" + "="*70)
print("CRISP-DM PHASE 6: DEPLOYMENT")
print("="*70)

# Model parameters
coef_dict = dict(zip(X.columns.tolist(), [float(c) for c in full_ols.coef_]))
intercept_val = float(full_ols.intercept_)

opt_model = fitted[f"Reduced SFS Model (k={optimal_k})"]["model"]
opt_coef  = dict(zip(optimal_features, [float(c) for c in opt_model.coef_]))
opt_intercept = float(opt_model.intercept_)

# Save models
joblib.dump(full_ols, 'model/boston_full_ols.pkl')
joblib.dump(opt_model, 'model/boston_sfs_optimal.pkl')
print("  [Saved] model/boston_full_ols.pkl")
print("  [Saved] model/boston_sfs_optimal.pkl")

# metadata JSON
full_r2_te = float(r2_score(y_test, y_te_p))
metadata = {
    "project": "Boston_HousePrice_CRISP_DM",
    "data_source": "Kaggle HousingData.csv",
    "dataset_size": df.shape[0],
    "features_total": len(X.columns),
    "na_imputation": "median",
    "train_test_split": "80/20, random_state=42",
    "full_ols": {
        "intercept": intercept_val,
        "coefficients": coef_dict,
        "eval": eval_rows[0],
        "gauss_markov": {"shapiro_p": float(shap_p), "durbin_watson": float(dw_val)}
    },
    "optimal_sfs": {
        "optimal_k": optimal_k,
        "features": optimal_features,
        "intercept": opt_intercept,
        "coefficients": opt_coef,
        "test_r2": float(best_row['r2']),
        "test_rmse": float(best_row['rmse'])
    },
    "sfs_history": sfs_df.to_dict(orient='records'),
    "all_models_eval": eval_rows
}
with open('model/model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("  [Saved] model/model_metadata.json")

# dataset.js (for web app)
print("  Generating dataset.js...")
df_web = df_raw.copy()
for col in df_web.columns:
    df_web[col] = df_web[col].fillna(df_web[col].median())

# Compute ML prediction for each row
preds_full, preds_opt = [], []
for _, row in df_web.iterrows():
    xf = np.array([[row[c] for c in X.columns]])
    xo = np.array([[row[c] for c in optimal_features]])
    preds_full.append(round(float(full_ols.predict(xf)[0]), 2))
    preds_opt.append(round(float(opt_model.predict(xo)[0]), 2))

df_web['PRED_FULL'] = preds_full
df_web['PRED_OPT']  = preds_opt

rows_js = []
for _, row in df_web.iterrows():
    r = {col: round(float(row[col]), 4) for col in df_web.columns}
    rows_js.append(json.dumps(r, ensure_ascii=False))
dataset_js = "const BOSTON_DATA = [\n  " + ",\n  ".join(rows_js) + "\n];\n"
with open('dataset.js', 'w', encoding='utf-8') as f:
    f.write(dataset_js)
print("  [Saved] dataset.js")

# model_config.js (for web app)
model_config_js = f"""// Boston Housing ML Model Configuration (auto-generated by main.py)
const BOSTON_MODEL = {{
  // Full OLS (13 features)
  intercept: {intercept_val},
  coefficients: {json.dumps(coef_dict)},
  // Optimal SFS (k={optimal_k} features)
  optimalK: {optimal_k},
  optimalFeatures: {json.dumps(optimal_features)},
  optimalIntercept: {opt_intercept},
  optimalCoefficients: {json.dumps(opt_coef)},
  // SFS progression
  sfsHistory: {json.dumps(sfs_df.to_dict(orient='records'))},
  // Feature metadata
  featureDescriptions: {json.dumps(FEATURE_DESCRIPTIONS, ensure_ascii=False)},
  // Evaluation metrics
  evalFull: {json.dumps(eval_rows[0])},
  evalOptimal: {json.dumps(eval_rows[3])},
  allModels: {json.dumps(eval_rows)},
  // Gauss-Markov
  shapiroP: {float(shap_p)},
  durbinWatson: {float(dw_val)}
}};
"""
with open('model_config.js', 'w', encoding='utf-8') as f:
    f.write(model_config_js)
print("  [Saved] model_config.js")

# Final summary
print("\n" + "="*70)
print("CRISP-DM PIPELINE COMPLETE - BOSTON HOUSE PRICE (Kaggle)")
print("="*70)
print(f"Full OLS Intercept: {intercept_val:.4f}")
print("Top 5 Coefficients (by abs value):")
for feat, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {feat:10s}: {coef:+.4f}")
print(f"\nOptimal SFS (k={optimal_k}): {optimal_features}")
print(f"  Intercept: {opt_intercept:.4f}")
for feat, coef in opt_coef.items():
    print(f"  {feat:10s}: {coef:+.4f}")
print(f"\nTest R2 (Full OLS):     {eval_rows[0]['Test R2']}")
print(f"Test RMSE (Full OLS):   {eval_rows[0]['RMSE']}")
print(f"Test R2 (SFS Optimal):  {eval_rows[3]['Test R2']}")
print(f"Test RMSE (SFS):        {eval_rows[3]['RMSE']}")
print("="*70)
