"""
Module: eda.py
CRISP-DM Phase 2: Data Understanding
Provides tools to inspect, summarize, and visualize the 50 Startups dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(filepath="50_Startups.csv"):
    """Loads startup dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def generate_data_summary(df):
    """
    Returns a dictionary of core exploratory data metrics:
    shape, columns, dtypes, missing values, zero spend counts, descriptive stats.
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "state_counts": df['State'].value_counts().to_dict(),
        "zero_spends": {
            "R&D Spend": (df["R&D Spend"] == 0).sum(),
            "Administration": (df["Administration"] == 0).sum(),
            "Marketing Spend": (df["Marketing Spend"] == 0).sum(),
        },
        "describe": df.describe().round(2).to_dict()
    }
    return summary

def compute_vif(df, numeric_cols=None):
    """
    Calculates Variance Inflation Factor (VIF) for numeric features to check multicollinearity.
    Rule of thumb: VIF > 5 or 10 indicates high multicollinearity.
    """
    if numeric_cols is None:
        numeric_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    
    X_num = df[numeric_cols].copy()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(len(numeric_cols))]
    return vif_data

def save_eda_plots(df, output_dir="outputs"):
    """
    Generates and saves exploratory visual charts:
    1. Distributions of numerical features and target
    2. Correlation heatmap
    3. Pairplot showing bivariate relationships and state grouping
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    
    # Set overall visual style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. Feature & Target Distributions (Histograms + KDE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ["#2b5c8f", "#3b82f6", "#10b981", "#8b5cf6"]
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=12)
        axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(f"{col} ($)")
        axes[i].set_ylabel("Frequency")
    
    dist_path = os.path.join(output_dir, "eda_distributions.png")
    fig.savefig(dist_path, dpi=300)
    plt.close(fig)

    # 2. Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Pearson Correlation Matrix (Numerical Features & Target)", fontsize=13, fontweight="bold", pad=15)
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(corr_path, dpi=300)
    plt.close(fig)

    # 3. Boxplot by State
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="State", y="Profit", data=df, palette="Set2", ax=ax, width=0.45)
    sns.stripplot(x="State", y="Profit", data=df, color="black", alpha=0.6, jitter=0.2, ax=ax)
    ax.set_title("Startup Profit Distribution Across States", fontsize=13, fontweight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("Profit ($)")
    boxplot_path = os.path.join(output_dir, "state_profit_distribution.png")
    fig.savefig(boxplot_path, dpi=300)
    plt.close(fig)

    return {
        "distributions": dist_path,
        "correlation": corr_path,
        "state_boxplot": boxplot_path
    }
