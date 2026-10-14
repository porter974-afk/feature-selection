"""
Module: feature_selection.py
CRISP-DM Phase 3 & 4: Sequential Forward Feature Selection (SFS)
Reproduces and validates the optimal feature subset based on RMSE and R-squared curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def perform_forward_selection(data_path="50_Startups.csv", output_dir="outputs", random_state=0):
    """
    Executes greedy forward feature selection to identify the optimal number of features
    that minimizes RMSE and maximizes R-squared on the holdout test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and encode data
    df = pd.read_csv(data_path)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=float)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']

    # 2. Train/Test split matching the benchmark
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    all_features = list(X.columns)
    selected_features = []
    remaining_features = list(all_features)

    history = []

    for step in range(1, len(all_features) + 1):
        best_score = -float('inf')
        best_feature = None
        best_rmse = None

        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            reg = LinearRegression().fit(X_train[candidate_features], y_train)
            pred = reg.predict(X_test[candidate_features])
            
            score_r2 = r2_score(y_test, pred)
            score_rmse = root_mean_squared_error(y_test, pred)

            if score_r2 > best_score:
                best_score = score_r2
                best_feature = feature
                best_rmse = score_rmse

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        history.append({
            "k": step,
            "added_feature": best_feature,
            "selected_features": list(selected_features),
            "rmse": best_rmse,
            "r2": best_score
        })

    history_df = pd.DataFrame(history)

    # 3. Generate the exact diagnostic plot from the user's screenshot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Left Plot: RMSE by Number of Features
    axes[0].plot(history_df['k'], history_df['rmse'], marker='o', color='#1f77b4', lw=1.8, markersize=6)
    axes[0].set_title('RMSE by Number of Features', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Number of Features', fontsize=10)
    axes[0].set_ylabel('RMSE', fontsize=10)
    axes[0].set_xticks(history_df['k'])

    # Right Plot: R-squared by Number of Features
    axes[1].plot(history_df['k'], history_df['r2'], marker='o', color='#1f77b4', lw=1.8, markersize=6)
    axes[1].set_title('R-squared by Number of Features', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Number of Features', fontsize=10)
    axes[1].set_ylabel('R-squared', fontsize=10)
    axes[1].set_xticks(history_df['k'])

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "feature_selection_curve.png")
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)

    # 4. Train the Optimal 2-Feature Model
    optimal_features = history_df.loc[history_df['rmse'].idxmin(), 'selected_features']
    optimal_model = LinearRegression().fit(X_train[optimal_features], y_train)
    opt_pred = optimal_model.predict(X_test[optimal_features])

    optimal_summary = {
        "optimal_k": len(optimal_features),
        "optimal_features": optimal_features,
        "intercept": float(optimal_model.intercept_),
        "coefficients": dict(zip(optimal_features, optimal_model.coef_)),
        "test_r2": float(r2_score(y_test, opt_pred)),
        "test_rmse": float(root_mean_squared_error(y_test, opt_pred)),
        "test_mae": float(mean_absolute_error(y_test, opt_pred)),
        "chart_path": chart_path,
        "history": history
    }

    return history_df, optimal_summary

if __name__ == "__main__":
    df_res, summary = perform_forward_selection()
    print("\n=== Sequential Feature Selection Results ===")
    print(df_res[['k', 'added_feature', 'rmse', 'r2']].to_string(index=False))
    print(f"\nOptimal Features ({summary['optimal_k']}):", summary['optimal_features'])
    print(f"Optimal Test RMSE: ${summary['test_rmse']:,.2f}")
    print(f"Optimal Test R^2:  {summary['test_r2']:.5f}")
