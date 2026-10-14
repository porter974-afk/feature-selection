"""
Module: diagnostics.py
CRISP-DM Phase 5: Evaluation
Provides statistical evaluation metrics, linear regression assumption tests,
and diagnostic visualization generation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_adjusted_r2(r2, n, p):
    """
    Computes Adjusted R-squared:
    Adjusted R2 = 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
    where n = number of samples, p = number of predictors
    """
    if n - p - 1 <= 0:
        return r2
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def compute_regression_metrics(y_true, y_pred, num_features):
    """
    Calculates comprehensive regression evaluation metrics:
    R^2, Adjusted R^2, RMSE, MSE, MAE, and MAPE.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = calculate_adjusted_r2(r2, n, num_features)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "samples": n,
        "features": num_features,
        "r2": float(r2),
        "adjusted_r2": float(adj_r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape)
    }

def test_regression_assumptions(y_true, y_pred):
    """
    Tests the fundamental Gauss-Markov and Classical Linear Regression Assumptions:
    1. Normality of residuals (Shapiro-Wilk test)
    2. Independence of errors (Durbin-Watson statistic, target ~2.0)
    3. Mean of residuals (should be approximately zero)
    """
    residuals = y_true - y_pred
    mean_resid = float(np.mean(residuals))
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Durbin-Watson statistic for autocorrelation
    dw_stat = float(durbin_watson(residuals))

    return {
        "mean_residual": mean_resid,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p_value": float(shapiro_p),
        "is_residual_normal": bool(shapiro_p > 0.05),
        "durbin_watson": dw_stat,
        "autocorrelation_status": "No autocorrelation (ideal)" if 1.5 <= dw_stat <= 2.5 else "Potential autocorrelation"
    }

def plot_regression_diagnostics(y_train, y_train_pred, y_test, y_test_pred, output_dir="outputs"):
    """
    Generates a 4-panel diagnostic figure evaluating:
    1. Actual vs. Predicted (Train & Test)
    2. Residuals vs. Fitted Values (Homoscedasticity test)
    3. Residual Histogram & Normal Distribution Fit
    4. Normal Q-Q Plot of Residuals
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_train_pred, color="#2563eb", alpha=0.7, edgecolors="k", s=55, label="Train Data")
    ax1.scatter(y_test, y_test_pred, color="#ef4444", alpha=0.9, edgecolors="k", s=70, marker="^", label="Test Data")
    
    all_vals = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
    min_val, max_val = all_vals.min() * 0.95, all_vals.max() * 1.05
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Fit (y = x)")
    ax1.set_title("1. Actual vs. Predicted Profit ($)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual Profit ($)")
    ax1.set_ylabel("Predicted Profit ($)")
    ax1.legend(frameon=True)

    # Panel 2: Residuals vs Fitted (Homoscedasticity)
    ax2 = axes[0, 1]
    ax2.scatter(y_train_pred, train_residuals, color="#2563eb", alpha=0.7, edgecolors="k", s=55, label="Train Residuals")
    ax2.scatter(y_test_pred, test_residuals, color="#ef4444", alpha=0.9, edgecolors="k", s=70, marker="^", label="Test Residuals")
    ax2.axhline(0, color="black", linestyle="--", lw=1.8)
    ax2.set_title("2. Residuals vs. Fitted (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fitted (Predicted) Profit ($)")
    ax2.set_ylabel("Residuals ($)")
    ax2.legend(frameon=True)

    # Panel 3: Histogram of Residuals (Normality)
    ax3 = axes[1, 0]
    all_resids = np.concatenate([train_residuals, test_residuals])
    sns.histplot(all_resids, kde=True, color="#10b981", bins=12, ax=ax3)
    ax3.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error Line")
    ax3.set_title("3. Distribution of Residuals (Normality Check)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Residual Error ($)")
    ax3.set_ylabel("Density / Count")
    ax3.legend(frameon=True)

    # Panel 4: Normal Q-Q Plot
    ax4 = axes[1, 1]
    stats.probplot(all_resids, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor("#8b5cf6")
    ax4.get_lines()[0].set_markeredgecolor("k")
    ax4.get_lines()[0].set_alpha(0.8)
    ax4.get_lines()[1].set_color("#dc2626")
    ax4.get_lines()[1].set_linewidth(2)
    ax4.set_title("4. Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")

    plt.tight_layout()
    diag_path = os.path.join(output_dir, "residuals_diagnostics.png")
    fig.savefig(diag_path, dpi=300)
    plt.close(fig)
    return diag_path

def plot_coefficients_bar(coef_dict, output_dir="outputs"):
    """
    Plots horizontal bar chart of estimated regression coefficients.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = list(coef_dict.keys())
    values = list(coef_dict.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
    bars = ax.barh(features, values, color=colors, edgecolor="black", alpha=0.85)

    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_title("Multiple Linear Regression Model Coefficients", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient Magnitude (Marginal Effect on Profit $)")
    ax.set_ylabel("Predictor Variable")

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        xpos = width + (max(values) * 0.02 if width >= 0 else -abs(max(values)) * 0.08)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{width:,.2f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    coef_path = os.path.join(output_dir, "model_coefficients.png")
    fig.savefig(coef_path, dpi=300)
    plt.close(fig)
    return coef_path
