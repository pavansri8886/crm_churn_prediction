"""
run_pipeline.py
---------------
ONE command runs the entire CRM churn prediction pipeline:

    python run_pipeline.py

Steps:
  1. Load real Kaggle CSV + data quality audit
  2. EDA  — generate all exploratory plots
  3. Preprocess — clean, encode, stratified split
  4. Train — 4 models with MLflow tracking
  5. Evaluate — ROC curves, confusion matrix, SHAP
"""

import sys
from pathlib import Path
from loguru import logger

# ── Logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
Path("logs").mkdir(parents=True, exist_ok=True)
Path("logs/plots").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
logger.add(
    "logs/pipeline.log",
    rotation="1 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from config import RAW_DATA_PATH, BEST_MODEL_NAME

from src.data_loader   import load_data
from src.eda           import run_eda
from src.preprocessing import preprocess
from src.train         import train_all_models, print_leaderboard
from src.evaluate      import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_shap_importance,
    find_optimal_threshold,
)


# ── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("CRM CHURN PREDICTION PIPELINE — START")
    logger.info("=" * 60)

    # ── Step 1: Load data + quality audit ─────────────────────────────────────
    logger.info("STEP 1/5 — Loading data & running quality audit")
    df = load_data(RAW_DATA_PATH)

    # ── Step 2: EDA — all exploratory plots ───────────────────────────────────
    logger.info("STEP 2/5 — Exploratory Data Analysis")
    run_eda(df, plots_dir="logs/plots")

    # ── Step 3: Preprocess — clean + stratified split ─────────────────────────
    logger.info("STEP 3/5 — Preprocessing & stratified split")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

    # ── Step 4: Train all 4 models ────────────────────────────────────────────
    logger.info("STEP 4/5 — Training models (MLflow tracking)")
    results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    best_model_name = print_leaderboard(results)

    # ── Step 5: Evaluate best model ───────────────────────────────────────────
    logger.info("STEP 5/5 — Evaluation & plots")

    best_pipeline = results[best_model_name]["pipeline"]

    # ROC curves — all models overlaid
    all_pipelines = {name: res["pipeline"] for name, res in results.items()}
    plot_roc_curves(all_pipelines, X_test, y_test)

    # Confusion matrix — best model with optimal threshold
    optimal_thresh = find_optimal_threshold(best_pipeline, X_val, y_val)
    plot_confusion_matrix(best_pipeline, X_test, y_test,
                          model_name=best_model_name,
                          threshold=optimal_thresh)

    # SHAP feature importance — best model
    plot_shap_importance(best_pipeline, X_test,
                         model_name=best_model_name)

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE ✅  Best model: {best_model_name}")
    logger.info(f"  Plots   → logs/plots/")
    logger.info(f"  Models  → models/")
    logger.info(f"  MLflow  → python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info(f"  API     → python -m uvicorn api.main:app --reload --port 8000")
    logger.info(f"  Dashboard → python -m streamlit run dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
