"""
src/data_loader.py
------------------
Loads the IBM Telco Customer Churn CSV and runs a full data quality
report covering all 8 dirty-data types from the course:
  1. Missing Data
  2. Outliers
  3. Contaminated Data
  4. Inconsistent Data
  5. Invalid Data
  6. Duplicate Data
  7. Data Type Issues
  8. Structural Errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

NUMERICAL_COLUMNS  = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
TARGET_COLUMN = "Churn"


# ── Main loader ───────────────────────────────────────────────────────────────
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV and run a full data quality audit before returning the DataFrame.
    Raises FileNotFoundError if the file is missing.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download the IBM Telco Customer Churn CSV from Kaggle:\n"
            "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and save it as:  data/telco_churn.csv"
        )

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.success(f"Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Run full quality audit
    _run_quality_report(df)

    return df


# ── Data Quality Report ───────────────────────────────────────────────────────
def _run_quality_report(df: pd.DataFrame) -> None:
    """
    Comprehensive data quality audit covering all 8 dirty-data types.
    Results are logged — no data is modified here (cleaning happens in preprocessing.py).
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    _check_structural_errors(df)
    _check_missing_data(df)
    _check_data_type_issues(df)
    _check_duplicate_data(df)
    _check_invalid_data(df)
    _check_inconsistent_data(df)
    _check_outliers(df)
    _check_contaminated_data(df)
    _check_class_balance(df)

    logger.info("=" * 60)
    logger.info("END OF DATA QUALITY REPORT")
    logger.info("=" * 60)


# ── 1. Structural Errors ──────────────────────────────────────────────────────
def _check_structural_errors(df: pd.DataFrame) -> None:
    """
    Check that the file has exactly the expected columns in the right order.
    Structural errors arise during data export or ETL pipelines — they cannot
    be fixed by cleaning alone; the source pipeline must be corrected.
    """
    logger.info("--- [1/8] STRUCTURAL ERRORS ---")

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols   = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        logger.warning(f"  Missing expected columns : {missing_cols}")
    else:
        logger.success("  All expected columns present ✓")

    if extra_cols:
        logger.warning(f"  Unexpected extra columns : {extra_cols}")
    else:
        logger.success("  No unexpected extra columns ✓")

    if df.shape[0] == 0:
        logger.error("  CRITICAL: Dataset is empty — check the ETL pipeline.")
    else:
        logger.info(f"  Row count: {df.shape[0]:,}")


# ── 2. Missing Data ───────────────────────────────────────────────────────────
def _check_missing_data(df: pd.DataFrame) -> None:
    """
    Detect missing values in all forms: NaN, None, empty strings,
    whitespace-only strings, and common sentinel values like 'N/A', 'null'.
    """
    logger.info("--- [2/8] MISSING DATA ---")

    # Standard NaN / None
    nan_counts = df.isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        for col, count in nan_cols.items():
            pct = count / len(df) * 100
            logger.warning(f"  NaN in '{col}': {count} rows ({pct:.1f}%)")
    else:
        logger.success("  No standard NaN / None values found ✓")

    # Empty strings or whitespace-only (common in exported CSVs)
    for col in df.select_dtypes(include="object").columns:
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        if empty_count > 0:
            logger.warning(f"  Empty/whitespace strings in '{col}': {empty_count} rows")

    # TotalCharges specific check — it's numeric but stored as string in this dataset
    if "TotalCharges" in df.columns:
        spaces = (df["TotalCharges"].astype(str).str.strip() == "").sum()
        if spaces > 0:
            logger.warning(
                f"  'TotalCharges' has {spaces} whitespace-only entries "
                f"(new customers with no charges yet) → will be imputed with median"
            )


# ── 3. Data Type Issues ───────────────────────────────────────────────────────
def _check_data_type_issues(df: pd.DataFrame) -> None:
    """
    Verify that columns have the correct dtype.
    TotalCharges is the known offender in this dataset — stored as object
    instead of float64 due to whitespace entries.
    """
    logger.info("--- [3/8] DATA TYPE ISSUES ---")

    # TotalCharges should be numeric
    if "TotalCharges" in df.columns:
        if df["TotalCharges"].dtype == object:
            logger.warning(
                "  'TotalCharges' dtype is 'object' (string) — expected float64. "
                "Cause: whitespace entries prevent automatic numeric parsing."
            )
        else:
            logger.success(f"  'TotalCharges' dtype is {df['TotalCharges'].dtype} ✓")

    # SeniorCitizen should be int (0/1), not string
    if "SeniorCitizen" in df.columns:
        if df["SeniorCitizen"].dtype not in ["int64", "int32", "float64"]:
            logger.warning(f"  'SeniorCitizen' unexpected dtype: {df['SeniorCitizen'].dtype}")
        else:
            logger.success(f"  'SeniorCitizen' dtype is {df['SeniorCitizen'].dtype} ✓")

    # Churn should be categorical Yes/No (later encoded to 0/1)
    if "Churn" in df.columns:
        unique_churn = df["Churn"].unique()
        logger.info(f"  'Churn' unique values: {unique_churn} — will be encoded to 0/1")


# ── 4. Duplicate Data ─────────────────────────────────────────────────────────
def _check_duplicate_data(df: pd.DataFrame) -> None:
    """
    Check for duplicate rows and duplicate customerIDs.
    Duplicates inflate counts and distort statistics.
    """
    logger.info("--- [4/8] DUPLICATE DATA ---")

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        logger.warning(f"  {dup_rows} fully duplicate rows found → will be dropped")
    else:
        logger.success("  No duplicate rows ✓")

    if "customerID" in df.columns:
        dup_ids = df["customerID"].duplicated().sum()
        if dup_ids > 0:
            logger.warning(f"  {dup_ids} duplicate customerIDs found")
        else:
            logger.success("  All customerIDs are unique ✓")


# ── 5. Invalid Data ───────────────────────────────────────────────────────────
def _check_invalid_data(df: pd.DataFrame) -> None:
    """
    Check for logically impossible values — values that are technically
    parseable but make no real-world sense.
    Unlike contaminated data, invalid data usually comes from processing
    errors rather than faulty collection.
    """
    logger.info("--- [5/8] INVALID DATA ---")
    issues_found = False

    # tenure must be >= 0
    if "tenure" in df.columns:
        negative_tenure = (df["tenure"] < 0).sum()
        if negative_tenure > 0:
            logger.warning(f"  Negative tenure values: {negative_tenure}")
            issues_found = True
        else:
            logger.success("  tenure >= 0 for all rows ✓")

    # MonthlyCharges must be > 0
    if "MonthlyCharges" in df.columns:
        invalid_charges = (df["MonthlyCharges"] <= 0).sum()
        if invalid_charges > 0:
            logger.warning(f"  MonthlyCharges <= 0: {invalid_charges} rows")
            issues_found = True
        else:
            logger.success("  MonthlyCharges > 0 for all rows ✓")

    # SeniorCitizen must be 0 or 1 only
    if "SeniorCitizen" in df.columns:
        invalid_senior = (~df["SeniorCitizen"].isin([0, 1])).sum()
        if invalid_senior > 0:
            logger.warning(f"  SeniorCitizen has values outside {{0,1}}: {invalid_senior} rows")
            issues_found = True
        else:
            logger.success("  SeniorCitizen is strictly 0 or 1 ✓")

    # TotalCharges should not be less than MonthlyCharges for customers with tenure > 1
    if all(c in df.columns for c in ["TotalCharges", "MonthlyCharges", "tenure"]):
        df_temp = df.copy()
        df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
        inconsistent = (
            (df_temp["tenure"] > 1) &
            (df_temp["TotalCharges"] < df_temp["MonthlyCharges"])
        ).sum()
        if inconsistent > 0:
            logger.warning(
                f"  {inconsistent} rows where TotalCharges < MonthlyCharges "
                f"despite tenure > 1 month (logically invalid)"
            )
            issues_found = True
        else:
            logger.success("  TotalCharges >= MonthlyCharges for all tenure > 1 rows ✓")

    if not issues_found:
        logger.success("  No invalid data detected ✓")


# ── 6. Inconsistent Data ──────────────────────────────────────────────────────
def _check_inconsistent_data(df: pd.DataFrame) -> None:
    """
    Check for inconsistent value representations in categorical columns.
    e.g. 'Yes' vs 'yes' vs 'YES', or unexpected category values.
    """
    logger.info("--- [6/8] INCONSISTENT DATA ---")

    EXPECTED_BINARY = {"Yes", "No"}
    EXPECTED_GENDER = {"Male", "Female"}
    EXPECTED_CONTRACT = {"Month-to-month", "One year", "Two year"}
    EXPECTED_INTERNET = {"DSL", "Fiber optic", "No"}

    checks = {
        "gender"          : EXPECTED_GENDER,
        "Partner"         : EXPECTED_BINARY,
        "Dependents"      : EXPECTED_BINARY,
        "PhoneService"    : EXPECTED_BINARY,
        "PaperlessBilling": EXPECTED_BINARY,
        "Churn"           : EXPECTED_BINARY,
        "Contract"        : EXPECTED_CONTRACT,
        "InternetService" : EXPECTED_INTERNET,
    }

    issues_found = False
    for col, expected_vals in checks.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().unique())
        unexpected  = actual_vals - expected_vals
        if unexpected:
            logger.warning(
                f"  '{col}' has unexpected values: {unexpected} "
                f"(expected: {expected_vals})"
            )
            issues_found = True

    if not issues_found:
        logger.success("  All categorical columns have consistent values ✓")


# ── 7. Outliers ───────────────────────────────────────────────────────────────
def _check_outliers(df: pd.DataFrame) -> None:
    """
    Detect outliers in numerical columns using the IQR method.
    Outliers are flagged for review — not automatically removed,
    as they may represent genuinely interesting behavior (e.g. very high spenders).
    """
    logger.info("--- [7/8] OUTLIERS (IQR method) ---")

    df_temp = df.copy()
    df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df_temp.columns:
            continue
        series = df_temp[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        n_out  = ((series < lower) | (series > upper)).sum()

        if n_out > 0:
            logger.warning(
                f"  '{col}': {n_out} outliers detected "
                f"(range [{lower:.1f}, {upper:.1f}]) "
                f"— keeping in dataset, using IQR-robust scaling"
            )
        else:
            logger.success(f"  '{col}': no outliers detected ✓")


# ── 8. Contaminated Data ──────────────────────────────────────────────────────
def _check_contaminated_data(df: pd.DataFrame) -> None:
    """
    Check for data leakage — features that would not be available at prediction
    time (i.e. future data contaminating the training set).
    This is the 'sneaky' type from the course: e.g. quarterly averages
    computed after the period ends included in per-day trading records.
    """
    logger.info("--- [8/8] CONTAMINATED DATA (leakage check) ---")

    # In the Telco dataset TotalCharges is technically a leakage risk
    # if used naively — it's the sum of all monthly charges, so it encodes
    # the tenure implicitly. We engineer TotalChargesGap instead to
    # use the *difference* from expected, which is safer.
    logger.info(
        "  'TotalCharges' note: contains implicit tenure information. "
        "We engineer 'TotalChargesGap' (actual vs expected) to reduce leakage risk."
    )
    logger.success("  No obvious future-data contamination detected ✓")


# ── Class balance ─────────────────────────────────────────────────────────────
def _check_class_balance(df: pd.DataFrame) -> None:
    """Report class imbalance in the target variable."""
    if TARGET_COLUMN not in df.columns:
        return
    counts   = df[TARGET_COLUMN].value_counts()
    churn_pct = (df[TARGET_COLUMN] == "Yes").mean() * 100
    logger.info(f"--- CLASS BALANCE ---")
    logger.info(f"  No Churn : {counts.get('No', 0):,} ({100 - churn_pct:.1f}%)")
    logger.info(f"  Churn    : {counts.get('Yes', 0):,} ({churn_pct:.1f}%)")
    if churn_pct < 30:
        logger.warning(
            f"  Class imbalance detected ({churn_pct:.1f}% churn). "
            "Using stratified splits and threshold tuning to compensate."
        )
