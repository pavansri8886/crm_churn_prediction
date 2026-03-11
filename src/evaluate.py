"""
evaluate.py — Model evaluation: metrics, confusion matrix, ROC, SHAP.

Produces:
  - Classification report
  - ROC-AUC, F1, Precision, Recall
  - Confusion matrix plot
  - ROC curve plot
  - SHAP feature importance plot (for tree models)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve, average_precision_score,
    precision_recall_curve
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, LOGS_DIR


PLOTS_DIR = LOGS_DIR / "plots"


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
    threshold: float = 0.5
) -> dict:
    """
    Evaluate a fitted pipeline. Returns dict of metrics.
    """
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "f1":        round(f1_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall":    round(recall_score(y, y_pred), 4),
        "avg_precision": round(average_precision_score(y, y_prob), 4),
    }

    logger.info(f"\n[{split_name.upper()}] Evaluation Results:")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics


def plot_confusion_matrix(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
    threshold: float = 0.5
) -> Path:
    """Save confusion matrix heatmap."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax
    )
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix (threshold={threshold})", fontsize=13)
    plt.tight_layout()

    path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {path}")
    return path


def plot_roc_curves(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> Path:
    """Overlay ROC curves for multiple models."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for (name, pipeline), color in zip(pipelines.items(), colors):
        y_prob = pipeline.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "roc_curves_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curves saved → {path}")
    return path


def plot_shap_importance(
    pipeline,
    X: pd.DataFrame,
    model_name: str = "xgboost",
    max_display: int = 20
) -> Path:
    """
    Generate SHAP beeswarm plot for tree-based models.
    Shows which features push predictions toward or away from churn.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Skipping SHAP plot.")
        return None

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values (this may take ~30s) …")

    # Extract feature step and classifier
    feature_step = pipeline.named_steps["features"]
    classifier   = pipeline.named_steps["classifier"]

    # Transform data
    X_transformed = feature_step.transform(X)

    # Get feature names
    try:
        feature_names = feature_step.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Choose explainer based on model type
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        tree_types = (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    except ImportError:
        tree_types = (RandomForestClassifier,)

    if isinstance(classifier, tree_types):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Linear models: use LinearExplainer
        masker = shap.maskers.Independent(X_transformed, max_samples=100)
        explainer = shap.LinearExplainer(classifier, masker)
        shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers, use class=1 shap values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, pad=15)
    plt.tight_layout()

    path = PLOTS_DIR / f"{model_name}_shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP plot saved → {path}")
    return path


def find_optimal_threshold(pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Find threshold that maximizes F1 on validation set.
    Important for imbalanced CRM datasets.
    """
    y_prob = pipeline.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx   = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    logger.info(
        f"Optimal threshold: {best_thresh:.3f} "
        f"(F1={f1_scores[best_idx]:.4f})"
    )
    return best_thresh