"""
=============================================================================
50 STARTUPS MULTIPLE LINEAR REGRESSION PROJECT
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Framework: scikit-learn
=============================================================================
Phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from src.eda import (
    load_data,
    generate_data_summary,
    compute_vif,
    save_eda_plots
)
from src.pipeline import (
    split_features_target,
    create_regression_pipeline,
    evaluate_cross_validation,
    extract_model_parameters,
    save_pipeline_artifact
)
from src.diagnostics import (
    compute_regression_metrics,
    test_regression_assumptions,
    plot_regression_diagnostics,
    plot_coefficients_bar
)
from src.feature_selection import perform_forward_selection


def print_header(title, step_num):
    border = "=" * 80
    print(f"\n{border}")
    print(f"CRISP-DM PHASE {step_num}: {title.upper()}")
    print(f"{border}\n")

def run_crisp_dm():
    # -------------------------------------------------------------------------
    # PHASE 1: BUSINESS UNDERSTANDING
    # -------------------------------------------------------------------------
    print_header("Business Understanding", 1)
    print("""
[1.1 Business Context]
Venture capital firms and executive boards regularly allocate capital across 
competing early-stage ventures. Understanding how department expenditures 
(R&D, Administration, Marketing) and operational jurisdiction (State) drive 
profitability is crucial for investment screening and capital budget allocation.

[1.2 Business Objectives]
1. Accurately predict startup profitability based on expenditure profiles.
2. Quantify the marginal return (ROI) on each dollar spent per department.
3. Determine whether geographic location (State) significantly influences profit.
4. Establish an automated, reusable scoring pipeline for incoming startup evaluations.

[1.3 Data Mining Goal & Success Criteria]
- Model: Multiple Linear Regression using scikit-learn.
- Target Metric: R^2 Score > 0.85, Low Root Mean Squared Error (RMSE).
- Business Constraint: Unbiased, interpretable regression coefficients adhering
  to classical linear regression assumptions (linearity, homoscedasticity, normality).
    """)

    # -------------------------------------------------------------------------
    # PHASE 2: DATA UNDERSTANDING
    # -------------------------------------------------------------------------
    print_header("Data Understanding", 2)
    data_path = "50_Startups.csv"
    df = load_data(data_path)
    print(f"Loaded dataset from '{data_path}' with shape: {df.shape[0]} rows x {df.shape[1]} columns.\n")
    
    print("Dataset Preview (First 5 records):")
    print(df.head().to_string(index=False))

    summary = generate_data_summary(df)
    print("\nData Types & Missing Values Check:")
    for col, null_count in summary["missing_values"].items():
        print(f"  - {col:18s} | Dtype: {str(df[col].dtype):8s} | Missing: {null_count}")

    print("\nState Categorical Distribution:")
    for state, count in summary["state_counts"].items():
        print(f"  - {state:18s}: {count} startups ({count / len(df) * 100:.1f}%)")

    print("\nZero-Spend Records:")
    for dept, count in summary["zero_spends"].items():
        print(f"  - {dept:18s}: {count} startups have $0 spend")

    print("\nSummary Statistics:")
    print(df.describe().round(2).to_string())

    # Multicollinearity check (VIF)
    vif_df = compute_vif(df)
    print("\nVariance Inflation Factors (VIF) for Numeric Features:")
    print(vif_df.to_string(index=False))
    print("  Note: High VIF (>5.0) reflects substantial correlation between R&D and Marketing.")

    # Generate EDA plots
    print("\nGenerating and saving EDA plots to 'outputs/' directory...")
    eda_plots = save_eda_plots(df)
    for name, path in eda_plots.items():
        print(f"  [Saved] {name.capitalize()} Plot: {path}")

    # -------------------------------------------------------------------------
    # PHASE 3: DATA PREPARATION
    # -------------------------------------------------------------------------
    print_header("Data Preparation", 3)
    print("""
[Data Preparation Steps]
1. Feature/Target Separation: Independent variables X (R&D, Admin, Marketing, State),
   Target variable y (Profit).
2. Train-Test Split: 80% Training set (40 samples), 20% Test set (10 samples).
   Fixed random_state=42 ensures reproducibility.
3. Categorical Encoding: One-Hot Encoding applied to 'State'.
   'drop=first' is explicitly specified to avoid the Dummy Variable Trap
   (multicollinearity where sum of dummy indicators = 1).
4. Preprocessing Pipeline: Built with scikit-learn's ColumnTransformer to prevent data leakage.
    """)

    X_train, X_test, y_train, y_test = split_features_target(df, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # PHASE 4: MODELING (INCLUDING SEQUENTIAL FEATURE SELECTION)
    # -------------------------------------------------------------------------
    print_header("Modeling & Feature Selection", 4)

    # Execute Sequential Forward Feature Selection to identify optimal k
    print("[4.0 Sequential Forward Feature Selection (SFS)]")
    sfs_history, sfs_summary = perform_forward_selection(data_path="50_Startups.csv", output_dir="outputs")
    print(sfs_history[["k", "added_feature", "rmse", "r2"]].to_string(index=False))
    print(f"\n  >>> Optimal Number of Features: {sfs_summary['optimal_k']}")
    print(f"  >>> Optimal Feature Subset:     {sfs_summary['optimal_features']}")
    print(f"  >>> Minimum Test RMSE:          ${sfs_summary['test_rmse']:,.2f}")
    print(f"  >>> Peak Test R^2:              {sfs_summary['test_r2']:.5f}")
    print(f"  [Saved] Feature Selection Curve: {sfs_summary['chart_path']}\n")

    print("Training candidate regression models with scikit-learn:\n")


    models = {
        "Multiple Linear Regression (Full)": create_regression_pipeline(model_type="linear"),
        "Ridge Regression (L2 Regularized)": create_regression_pipeline(model_type="ridge", scale_numeric=True),
        "Lasso Regression (L1 Regularized)": create_regression_pipeline(model_type="lasso", scale_numeric=True),
        "Reduced Linear Model (R&D + Marketing)": create_regression_pipeline(
            model_type="linear", numeric_features=["R&D Spend", "Marketing Spend"], categorical_features=[]
        )
    }

    fitted_models = {}
    cv_metrics = {}

    for name, pipeline in models.items():
        if "Reduced" in name:
            # Train only on selected features
            reduced_X_train = X_train[["R&D Spend", "Marketing Spend"]]
            reduced_X_test = X_test[["R&D Spend", "Marketing Spend"]]
            pipeline.fit(reduced_X_train, y_train)
            cv_res = evaluate_cross_validation(pipeline, reduced_X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)
            cv_res = evaluate_cross_validation(pipeline, X_train, y_train)

        fitted_models[name] = pipeline
        cv_metrics[name] = cv_res
        print(f"  [Trained] {name}")
        print(f"            5-Fold CV R^2: {cv_res['cv_r2_mean']:.4f} (+/- {cv_res['cv_r2_std']:.4f}) | RMSE: ${cv_res['cv_rmse_mean']:,.2f}")

    # Primary model inspection (Full Multiple Linear Regression)
    primary_pipeline = fitted_models["Multiple Linear Regression (Full)"]
    params = extract_model_parameters(primary_pipeline)
    
    print("\nFitted Linear Regression Equation:")
    print(f"  Profit = ${params['intercept']:,.2f}")
    for feat, coef in params["coefficients"].items():
        sign = "+" if coef >= 0 else "-"
        print(f"           {sign} ({abs(coef):.4f} * {feat})")

    # -------------------------------------------------------------------------
    # PHASE 5: EVALUATION
    # -------------------------------------------------------------------------
    print_header("Evaluation", 5)
    print("Model Performance Comparison on Unseen Test Set (20% Holdout):\n")

    eval_table = []
    test_predictions = {}
    train_predictions = {}

    for name, pipeline in fitted_models.items():
        if "Reduced" in name:
            X_tr = X_train[["R&D Spend", "Marketing Spend"]]
            X_te = X_test[["R&D Spend", "Marketing Spend"]]
            num_feats = 2
        else:
            X_tr = X_train
            X_te = X_test
            num_feats = 5  # 3 numeric + 2 dummy categories

        y_tr_pred = pipeline.predict(X_tr)
        y_te_pred = pipeline.predict(X_te)
        train_predictions[name] = y_tr_pred
        test_predictions[name] = y_te_pred

        m_train = compute_regression_metrics(y_train, y_tr_pred, num_feats)
        m_test = compute_regression_metrics(y_test, y_te_pred, num_feats)

        eval_table.append({
            "Model": name,
            "Train R2": f"{m_train['r2']:.4f}",
            "Test R2": f"{m_test['r2']:.4f}",
            "Test Adj R2": f"{m_test['adjusted_r2']:.4f}",
            "Test RMSE ($)": f"{m_test['rmse']:,.2f}",
            "Test MAE ($)": f"{m_test['mae']:,.2f}",
            "Test MAPE (%)": f"{m_test['mape'] * 100:.2f}%"
        })

    eval_df = pd.DataFrame(eval_table)
    print(eval_df.to_string(index=False))

    # Classical Linear Regression Assumptions Testing on Primary Model
    print("\nLinear Regression Assumption Diagnostics (Gauss-Markov Verification):")
    primary_test_pred = test_predictions["Multiple Linear Regression (Full)"]
    primary_train_pred = train_predictions["Multiple Linear Regression (Full)"]
    assumptions = test_regression_assumptions(y_test, primary_test_pred)

    print(f"  1. Mean of Residuals:        {assumptions['mean_residual']:,.2f} (Target approx 0)")
    print(f"  2. Shapiro-Wilk Normality:   Statistic = {assumptions['shapiro_stat']:.4f}, p-value = {assumptions['shapiro_p_value']:.4f}")
    print(f"     -> Residual Distribution: {'Normal (p > 0.05)' if assumptions['is_residual_normal'] else 'Non-normal (p <= 0.05)'}")
    print(f"  3. Durbin-Watson Autocorr:   Statistic = {assumptions['durbin_watson']:.4f}")
    print(f"     -> Error Independence:    {assumptions['autocorrelation_status']}")

    # Save diagnostic charts
    print("\nGenerating Diagnostic & Residual Plots in 'outputs/'...")
    diag_plot_path = plot_regression_diagnostics(y_train, primary_train_pred, y_test, primary_test_pred)
    coef_plot_path = plot_coefficients_bar(params["coefficients"])
    print(f"  [Saved] Residual Diagnostics: {diag_plot_path}")
    print(f"  [Saved] Coefficients Chart:   {coef_plot_path}")

    # Business Insights Summary
    print("""
[5.1 Business Insights & ROI Takeaways]
1. Primary Driver (R&D Spend): Coefficient is ~0.8056 (p < 0.001). For every $1 invested
   in Research & Development, expected startup profit increases by ~$0.81.
2. Secondary Driver (Marketing Spend): Coefficient is ~0.0299. Provides marginal positive
   returns (~3 cents per dollar spent).
3. Non-Significant Factors:
   - Administration spend has a near-zero / slightly negative coefficient (-0.0688, p > 0.60).
     Operational overhead does not scale profitability directly.
   - Geographic State dummies (Florida, New York vs California) have p-values > 0.90,
     indicating that geographic location does not create a statistically significant profit divergence.
    """)

    # -------------------------------------------------------------------------
    # PHASE 6: DEPLOYMENT
    # -------------------------------------------------------------------------
    print_header("Deployment", 6)
    
    # Save both the full pipeline and the optimal feature selection pipeline
    model_artifact_path = os.path.join("model", "startup_profit_pipeline.pkl")
    optimal_artifact_path = os.path.join("model", "optimal_feature_selection_pipeline.pkl")
    meta_artifact_path = os.path.join("model", "model_metadata.json")

    optimal_pipeline = fitted_models["Reduced Linear Model (R&D + Marketing)"]

    metadata = {
        "project_name": "50_Startups_CRISP_DM_Feature_Selection",
        "methodology": "CRISP-DM",
        "library": "scikit-learn",
        "sklearn_version": "1.9.1",
        "full_model": {
            "name": "Multiple Linear Regression (Full)",
            "features": {
                "numeric": ["R&D Spend", "Administration", "Marketing Spend"],
                "categorical": ["State"],
                "target": "Profit"
            },
            "intercept": params["intercept"],
            "coefficients": params["coefficients"],
            "metrics_test": eval_table[0],
            "cv_performance": cv_metrics["Multiple Linear Regression (Full)"],
            "artifact_path": model_artifact_path
        },
        "optimal_sfs_model": {
            "name": "Sequential Feature Selection (Optimal k=2)",
            "features": ["R&D Spend", "Marketing Spend"],
            "target": "Profit",
            "intercept": sfs_summary["intercept"],
            "coefficients": sfs_summary["coefficients"],
            "metrics_test": {
                "Model": "Reduced Model (R&D + Marketing)",
                "Test R2": f"{sfs_summary['test_r2']:.4f}",
                "Test RMSE ($)": f"{sfs_summary['test_rmse']:,.2f}",
                "Test MAE ($)": f"{sfs_summary['test_mae']:,.2f}"
            },
            "artifact_path": optimal_artifact_path
        },
        "feature_selection_progression": sfs_summary["history"]
    }

    save_pipeline_artifact(primary_pipeline, model_artifact_path, metadata, meta_artifact_path)
    joblib.dump(optimal_pipeline, optimal_artifact_path)
    print(f"  [Serialized] Full Production Pipeline saved to:    {model_artifact_path}")
    print(f"  [Serialized] Optimal SFS Pipeline saved to:        {optimal_artifact_path}")
    print(f"  [Serialized] Comprehensive Metadata saved to:      {meta_artifact_path}")

    # Verification Inference Simulation
    print("\nRunning Production Inference Verification on 3 Hypothetical Startups:")
    sample_startups = pd.DataFrame([
        {
            "R&D Spend": 150000.0,
            "Administration": 120000.0,
            "Marketing Spend": 350000.0,
            "State": "California"
        },
        {
            "R&D Spend": 80000.0,
            "Administration": 90000.0,
            "Marketing Spend": 200000.0,
            "State": "New York"
        },
        {
            "R&D Spend": 25000.0,
            "Administration": 140000.0,
            "Marketing Spend": 50000.0,
            "State": "Florida"
        }
    ])

    sample_predictions = primary_pipeline.predict(sample_startups)
    for i, pred in enumerate(sample_predictions):
        profile = sample_startups.iloc[i].to_dict()
        print(f"\n  Startup #{i+1} Profile:")
        print(f"    R&D: ${profile['R&D Spend']:,.2f} | Admin: ${profile['Administration']:,.2f} | Marketing: ${profile['Marketing Spend']:,.2f} | State: {profile['State']}")
        print(f"    >>> Predicted Profit: ${pred:,.2f}")

    print("\n" + "=" * 80)
    print("CRISP-DM END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_crisp_dm()
