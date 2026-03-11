"""
config.py — Central configuration for CRM Churn Prediction Project
All parameters in one place for easy tuning and reproducibility.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
PLOTS_DIR = LOGS_DIR / "plots"

RAW_DATA_PATH       = DATA_DIR / "telco_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "telco_churn_processed.csv"

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "CRM-Churn-Prediction"

# ─── Data ─────────────────────────────────────────────────────────────────────
TARGET_COLUMN  = "Churn"
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
VAL_SIZE       = 0.1

# Columns to drop (IDs, not features)
DROP_COLUMNS = ["customerID"]

# Categorical columns (for encoding)
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Numerical columns (for scaling)
NUMERICAL_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

# ─── Model Hyperparameters ────────────────────────────────────────────────────
MODELS_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "use_label_encoder": False
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbose": -1
    }
}

# Best model to use for API deployment
BEST_MODEL_NAME = "xgboost"
BEST_MODEL_PATH = MODELS_DIR / f"{BEST_MODEL_NAME}_pipeline.joblib"

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Thresholds ───────────────────────────────────────────────────────────────
CHURN_RISK_THRESHOLDS = {
    "low":    (0.0,  0.33),
    "medium": (0.33, 0.66),
    "high":   (0.66, 1.0)
}
