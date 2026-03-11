"""
train.py — Train and compare multiple ML models with MLflow tracking.

Models trained:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM

Each model is wrapped in a full Pipeline (feature_pipeline + classifier),
logged to MLflow, and serialized with joblib.
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, MODELS_CONFIG, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, BEST_MODEL_NAME, RANDOM_STATE
)
from src.feature_engineering import build_feature_pipeline
from src.evaluate import evaluate_model


CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "xgboost":             XGBClassifier,
    "lightgbm":            LGBMClassifier,
}


def build_model_pipeline(model_name: str) -> Pipeline:
    """Combine feature pipeline + classifier into one sklearn Pipeline."""
    if model_name not in CLASSIFIER_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(CLASSIFIER_MAP.keys())}"
        )

    feature_pipe = build_feature_pipeline()
    clf = CLASSIFIER_MAP[model_name](**MODELS_CONFIG[model_name])

    pipeline = Pipeline([
        ("features",   feature_pipe),
        ("classifier", clf),
    ])

    return pipeline


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> dict:
    """
    Train all models, log to MLflow, save to disk.
    Returns a dict of {model_name: metrics}.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results = {}

    for model_name in CLASSIFIER_MAP:
        logger.info(f"\n{'='*55}")
        logger.info(f"Training: {model_name.upper()}")
        logger.info(f"{'='*55}")

        with mlflow.start_run(run_name=model_name):

            # ── Build & train ────────────────────────────────────────
            pipeline = build_model_pipeline(model_name)

            # XGBoost needs eval_set passed differently — use fit_params
            if model_name == "xgboost":
                # Transform validation data through feature step
                feature_pipe = pipeline.named_steps["features"]
                X_train_t = feature_pipe.fit_transform(X_train)
                X_val_t   = feature_pipe.transform(X_val)
                clf = pipeline.named_steps["classifier"]
                clf.fit(
                    X_train_t, y_train,
                    eval_set=[(X_val_t, y_val)],
                    verbose=False
                )
            else:
                pipeline.fit(X_train, y_train)

            # ── Evaluate ─────────────────────────────────────────────
            metrics_val  = evaluate_model(pipeline, X_val,  y_val,  "val")
            metrics_test = evaluate_model(pipeline, X_test, y_test, "test")

            # ── MLflow logging ────────────────────────────────────────
            mlflow.log_params(MODELS_CONFIG[model_name])
            mlflow.log_metrics({
                f"val_{k}":  v for k, v in metrics_val.items()
            })
            mlflow.log_metrics({
                f"test_{k}": v for k, v in metrics_test.items()
            })
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=f"model_{model_name}",
                registered_model_name=model_name
            )

            # ── Save locally ──────────────────────────────────────────
            save_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
            joblib.dump(pipeline, save_path)
            logger.info(f"Model saved → {save_path}")

            results[model_name] = {
                "val":  metrics_val,
                "test": metrics_test,
                "pipeline": pipeline
            }

    return results


def print_leaderboard(results: dict) -> str:
    """Print a sorted leaderboard comparing all models."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":          name,
            "Val ROC-AUC":    res["val"]["roc_auc"],
            "Val F1":         res["val"]["f1"],
            "Test ROC-AUC":   res["test"]["roc_auc"],
            "Test F1":        res["test"]["f1"],
            "Test Precision": res["test"]["precision"],
            "Test Recall":    res["test"]["recall"],
        })

    board = (
        pd.DataFrame(rows)
        .sort_values("Test ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    board.index += 1

    print("\n" + "="*70)
    print("MODEL LEADERBOARD")
    print("="*70)
    print(board.to_string(float_format=lambda x: f"{x:.4f}"))
    print("="*70 + "\n")

    return board.iloc[0]["Model"]


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor, split_data

    logger.info("Loading and preparing data …")
    df = load_data()
    df = DataPreprocessor().clean(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    best = print_leaderboard(results)
    logger.info(f"\nBest model: {best}")
