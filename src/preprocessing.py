"""
src/preprocessing.py
--------------------
Cleans and prepares the raw Telco churn dataframe.
Each cleaning step is explicitly tied to one of the 8 dirty-data types
from the course so the logic is traceable and explainable.

Cleaning steps applied:
  1. Structural fix    — enforce correct schema
  2. Type coercion     — TotalCharges string → float
  3. Missing values    — median / mode imputation
  4. Duplicate removal — drop exact duplicate rows
  5. Invalid data fix  — clamp impossible values
  6. Inconsistent data — strip whitespace, normalise casing
  7. Outlier handling  — log & keep (IQR info stored for EDA)
  8. Target encoding   — Churn Yes/No → 1/0
  9. Stratified split  — preserves churn rate in every subset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

from config import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)


# ── Main entry point ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Run the full cleaning + split pipeline.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    logger.info("Starting preprocessing pipeline...")

    df = _step1_structural_fix(df)
    df = _step2_fix_data_types(df)
    df = _step3_handle_missing_values(df)
    df = _step4_remove_duplicates(df)
    df = _step5_fix_invalid_data(df)
    df = _step6_fix_inconsistent_data(df)
    _step7_log_outliers(df)          # log only — we keep outliers
    df = _step8_encode_target(df)

    X, y = _drop_id_and_split_xy(df)
    splits = _step9_stratified_split(X, y)

    logger.success("Preprocessing complete.")
    return splits


# ── Step 1 — Structural Fix ───────────────────────────────────────────────────
def _step1_structural_fix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has the correct columns.
    Structural errors arise from ETL pipelines and cannot be fixed by
    cleaning alone — but we can guard against them defensively here.
    """
    logger.info("[Step 1/9] Structural fix")

    from src.data_loader import EXPECTED_COLUMNS
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Check your CSV export or ETL pipeline."
        )

    # Keep only expected columns (drop any accidental extras)
    df = df[[c for c in EXPECTED_COLUMNS if c in df.columns]].copy()
    logger.success(f"  Schema validated — {df.shape[1]} columns retained")
    return df


# ── Step 2 — Data Type Issues ─────────────────────────────────────────────────
def _step2_fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to their correct dtype.
    TotalCharges is stored as 'object' (string) in the raw CSV because
    some entries contain only whitespace (new customers with no charges).
    """
    logger.info("[Step 2/9] Data type coercion")

    before_dtype = df["TotalCharges"].dtype

    # Force numeric — whitespace entries become NaN (handled in step 3)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    after_dtype  = df["TotalCharges"].dtype
    nan_produced = df["TotalCharges"].isnull().sum()

    logger.info(
        f"  TotalCharges: {before_dtype} → {after_dtype} "
        f"({nan_produced} whitespace entries became NaN)"
    )
    return df


# ── Step 3 — Missing Data ─────────────────────────────────────────────────────
def _step3_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three strategies from the course:
      a) Drop rows/columns where data is truly absent and unrecoverable
      b) Impute numerical columns with median (robust to outliers)
      c) Impute categorical columns with mode (most frequent value)

    We do NOT use mean imputation for TotalCharges because outliers
    (very high-spend customers) would bias the mean upward.
    """
    logger.info("[Step 3/9] Missing value imputation")

    # Numerical — median imputation
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(
                f"  '{col}': {n_missing} missing → imputed with median ({median_val:.2f})"
            )

    # Categorical — mode imputation
    for col in df.select_dtypes(include=["object"]).columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(
                f"  '{col}': {n_missing} missing → imputed with mode ('{mode_val}')"
            )

    remaining = df.isnull().sum().sum()
    if remaining == 0:
        logger.success("  No missing values remaining ✓")
    else:
        logger.warning(f"  {remaining} missing values still remain after imputation")

    return df


# ── Step 4 — Duplicate Data ───────────────────────────────────────────────────
def _step4_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.
    Duplicates inflate counts and distort statistics — they can come from
    combining data sources, double form submissions, or faulty pipelines.
    """
    logger.info("[Step 4/9] Duplicate removal")

    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    removed = before - after

    if removed > 0:
        logger.warning(f"  Removed {removed} duplicate rows")
    else:
        logger.success(f"  No duplicate rows found ✓ ({before:,} rows retained)")

    return df


# ── Step 5 — Invalid Data ─────────────────────────────────────────────────────
def _step5_fix_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix logically impossible values that result from processing errors
    (not collection errors — those are contaminated data).

    Example from the course: app session duration = -22 hours because
    finish_hour - start_hour wraps around midnight.
    Our equivalent: negative tenure or zero monthly charges.
    """
    logger.info("[Step 5/9] Invalid data correction")
    fixes = 0

    # tenure must be >= 0
    if "tenure" in df.columns:
        mask = df["tenure"] < 0
        if mask.sum() > 0:
            df.loc[mask, "tenure"] = 0
            logger.warning(f"  tenure: {mask.sum()} negative values clamped to 0")
            fixes += mask.sum()

    # MonthlyCharges must be > 0
    if "MonthlyCharges" in df.columns:
        mask = df["MonthlyCharges"] <= 0
        if mask.sum() > 0:
            median_mc = df.loc[~mask, "MonthlyCharges"].median()
            df.loc[mask, "MonthlyCharges"] = median_mc
            logger.warning(
                f"  MonthlyCharges: {mask.sum()} zero/negative values "
                f"replaced with median ({median_mc:.2f})"
            )
            fixes += mask.sum()

    # SeniorCitizen must be 0 or 1
    if "SeniorCitizen" in df.columns:
        mask = ~df["SeniorCitizen"].isin([0, 1])
        if mask.sum() > 0:
            df.loc[mask, "SeniorCitizen"] = 0
            logger.warning(f"  SeniorCitizen: {mask.sum()} invalid values set to 0")
            fixes += mask.sum()

    if fixes == 0:
        logger.success("  No invalid values found ✓")
    else:
        logger.info(f"  Total invalid values corrected: {fixes}")

    return df


# ── Step 6 — Inconsistent Data ────────────────────────────────────────────────
def _step6_fix_inconsistent_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise categorical values:
    - Strip leading/trailing whitespace
    - Consistent casing (Title Case for Yes/No columns)

    As the course notes: 'apples', 'Apples', 'APPLES' are three different
    things to a computer — we normalise them here.
    """
    logger.info("[Step 6/9] Inconsistent data normalisation")

    for col in df.select_dtypes(include=["object"]).columns:
        before = df[col].copy()

        # Strip whitespace
        df[col] = df[col].astype(str).str.strip()

        # Normalise common Yes/No variants
        df[col] = df[col].replace({
            "yes": "Yes", "YES": "Yes", "y": "Yes", "Y": "Yes",
            "no":  "No",  "NO":  "No",  "n": "No",  "N": "No",
        })

        changed = (before != df[col]).sum()
        if changed > 0:
            logger.info(f"  '{col}': {changed} values normalised")

    logger.success("  Categorical columns normalised ✓")
    return df


# ── Step 7 — Outlier Logging ──────────────────────────────────────────────────
def _step7_log_outliers(df: pd.DataFrame) -> None:
    """
    Log outlier information using IQR method.
    We DO NOT remove outliers here — the course notes three valid strategies:
      1. Remove them
      2. Segment them into a separate group
      3. Keep them but use robust statistical methods

    We choose strategy 3: keep outliers and use StandardScaler which
    is mean/std based. For a more robust alternative, RobustScaler
    (which uses IQR) could be used in feature_engineering.py.
    """
    logger.info("[Step 7/9] Outlier audit (IQR method — keep strategy)")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        n_out  = ((series < lower) | (series > upper)).sum()

        if n_out > 0:
            logger.info(
                f"  '{col}': {n_out} outliers (IQR bounds [{lower:.1f}, {upper:.1f}]) "
                f"— retained, will use StandardScaler"
            )
        else:
            logger.success(f"  '{col}': no outliers detected ✓")


# ── Step 8 — Target Encoding ──────────────────────────────────────────────────
def _step8_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target column:  'Yes' → 1,  'No' → 0
    Also encodes SeniorCitizen which is already 0/1 but needs confirmation.
    """
    logger.info("[Step 8/9] Target encoding")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
        unique_vals = df[TARGET_COLUMN].unique()
        logger.success(f"  Churn encoded → unique values: {sorted(unique_vals)}")

    return df


# ── Drop ID + split X/y ───────────────────────────────────────────────────────
def _drop_id_and_split_xy(df: pd.DataFrame):
    """Drop customerID (no predictive value) and separate features from target."""
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("  'customerID' dropped (no predictive value)")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    logger.info(f"  Features: {X.shape[1]} columns | Target: '{TARGET_COLUMN}'")
    return X, y


# ── Step 9 — Stratified Split ─────────────────────────────────────────────────
def _step9_stratified_split(X: pd.DataFrame, y: pd.Series):
    """
    Stratified 70 / 10 / 20 split.
    Stratification ensures the churn rate (~26%) is preserved in every subset —
    critical for imbalanced classification problems.
    """
    logger.info("[Step 9/9] Stratified train/validation/test split (70/10/20)")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    for name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        n    = len(y_split)
        pct  = n / len(y) * 100
        churn = y_split.mean() * 100
        logger.info(
            f"  {name:<12}: {n:>5,} samples ({pct:.0f}%) — churn rate: {churn:.1f}%"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
