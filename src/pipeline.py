"""
Module: pipeline.py
CRISP-DM Phase 3: Data Preparation & Phase 4: Modeling
Builds end-to-end scikit-learn preprocessing and regression pipelines.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

def split_features_target(df, target_col="Profit", test_size=0.2, random_state=42):
    """
    Separates independent features and target variable,
    then executes an 80/20 train/test split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_preprocessing_transformer(numeric_features=None, categorical_features=None, scale_numeric=False):
    """
    Constructs a ColumnTransformer to handle:
    - Categorical features: One-Hot Encoding with drop='first' (prevents dummy variable trap)
    - Numeric features: Passthrough or StandardScaler
    """
    if numeric_features is None:
        numeric_features = ["R&D Spend", "Administration", "Marketing Spend"]
    if categorical_features is None:
        categorical_features = ["State"]

    transformers = [
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        )
    ]

    if scale_numeric:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )
    else:
        transformers.append(
            ("num", "passthrough", numeric_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor

def create_regression_pipeline(model_type="linear", scale_numeric=False, numeric_features=None, categorical_features=None):
    """
    Instantiates an end-to-end scikit-learn Pipeline combining preprocessor and regressor.
    Supported model_type: 'linear', 'ridge', 'lasso'.
    """
    preprocessor = build_preprocessing_transformer(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric
    )

    if model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "ridge":
        # RidgeCV automatically tests an array of alpha penalty strengths via internal CV
        alphas = np.logspace(-3, 3, 50)
        regressor = RidgeCV(alphas=alphas, cv=5)
    elif model_type == "lasso":
        # LassoCV automatically tunes alpha via internal CV
        regressor = LassoCV(cv=5, random_state=42, max_iter=10000)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    return pipeline

def evaluate_cross_validation(pipeline, X_train, y_train, cv_folds=5):
    """
    Performs K-Fold cross validation on the training set to prevent data leakage.
    Returns mean and std of R^2, RMSE, and MAE.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "r2": "r2",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error"
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=scoring, return_train_score=True)
    
    summary = {
        "cv_r2_mean": float(np.mean(cv_results["test_r2"])),
        "cv_r2_std": float(np.std(cv_results["test_r2"])),
        "cv_rmse_mean": float(-np.mean(cv_results["test_neg_root_mean_squared_error"])),
        "cv_rmse_std": float(np.std(cv_results["test_neg_root_mean_squared_error"])),
        "cv_mae_mean": float(-np.mean(cv_results["test_neg_mean_absolute_error"])),
        "cv_mae_std": float(np.std(cv_results["test_neg_mean_absolute_error"]))
    }
    return summary

def extract_model_parameters(pipeline):
    """
    Extracts regression intercept, feature names, and estimated coefficients from fitted pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    
    feature_names = preprocessor.get_feature_names_out()
    # Clean up prefixes for display
    clean_feature_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    coef_dict = dict(zip(clean_feature_names, regressor.coef_))
    intercept = float(regressor.intercept_)
    
    return {
        "intercept": intercept,
        "coefficients": coef_dict,
        "feature_names": clean_feature_names
    }

def save_pipeline_artifact(pipeline, filepath="model/startup_profit_pipeline.pkl", metadata=None, meta_filepath="model/model_metadata.json"):
    """
    Serializes fitted pipeline and metadata to disk for deployment.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    
    if metadata and meta_filepath:
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
