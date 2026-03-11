"""
feature_engineering.py — Build sklearn-compatible FeaturePipeline.

Transformations:
  1. CRM-domain engineered features (business logic)
  2. One-hot encoding for categoricals
  3. Standard scaling for numericals
  4. Returns a fitted ColumnTransformer-based pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS


# ─── Custom transformer: CRM domain features ──────────────────────────────────

class CRMFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates business-meaningful features from raw CRM columns.
    These encode domain knowledge that tree models might not discover
    automatically from raw counts alone.
    """

    def fit(self, X, y=None):
        return self  # Stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Revenue per month of tenure (customer value density)
        if "MonthlyCharges" in df and "tenure" in df:
            df["ChargesPerTenureMonth"] = (
                df["MonthlyCharges"] / (df["tenure"] + 1)  # +1 avoids div/0
            )

        # 2. Is a long-term customer? (tenure > 24 months = 2 years)
        if "tenure" in df:
            df["IsLongTerm"]  = (df["tenure"] >= 24).astype(int)
            df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

        # 3. Service bundle score (more services → stickier customer)
        service_cols = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        existing = [c for c in service_cols if c in df.columns]
        if existing:
            df["ServiceBundleScore"] = (
                df[existing]
                .apply(lambda col: col.map({"Yes": 1}).fillna(0))
                .sum(axis=1)
            )

        # 4. Contract risk score
        contract_risk = {
            "Month-to-month": 3,
            "One year":        2,
            "Two year":        1
        }
        if "Contract" in df:
            df["ContractRiskScore"] = (
                df["Contract"].map(contract_risk).fillna(3).astype(int)
            )

        # 5. TotalCharges gap: expected vs actual (detects discounting / billing issues)
        if all(c in df.columns for c in ["TotalCharges", "MonthlyCharges", "tenure"]):
            df["ExpectedTotal"] = df["MonthlyCharges"] * df["tenure"]
            df["TotalChargesGap"] = df["ExpectedTotal"] - df["TotalCharges"].fillna(0)
            df.drop(columns=["ExpectedTotal"], inplace=True)

        # 6. High-risk combo flag: month-to-month + fiber optic + no security
        if all(c in df.columns for c in ["Contract", "InternetService", "OnlineSecurity"]):
            df["HighRiskCombo"] = (
                (df["Contract"]       == "Month-to-month") &
                (df["InternetService"] == "Fiber optic")   &
                (df["OnlineSecurity"]  == "No")
            ).astype(int)

        logger.debug(
            f"Feature engineering complete. "
            f"New features added: ChargesPerTenureMonth, IsLongTerm, "
            f"IsNewCustomer, ServiceBundleScore, ContractRiskScore, "
            f"TotalChargesGap, HighRiskCombo"
        )
        return df

    def get_feature_names_out(self, input_features=None):
        return None  # handled downstream


# ─── Build full preprocessing pipeline ────────────────────────────────────────

def build_feature_pipeline() -> Pipeline:
    """
    Returns a full sklearn Pipeline:
      1. CRMFeatureEngineer (domain features)
      2. ColumnTransformer
          - Numerical: impute → scale
          - Categorical: impute → one-hot encode
    """

    # After CRMFeatureEngineer, new numerical columns are added
    extended_numerical = NUMERICAL_COLUMNS + [
        "ChargesPerTenureMonth",
        "ServiceBundleScore",
        "ContractRiskScore",
        "TotalChargesGap",
        "IsLongTerm",
        "IsNewCustomer",
        "HighRiskCombo",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first"          # drop first to avoid dummy trap
        )),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,    extended_numerical),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )

    pipeline = Pipeline([
        ("feature_engineer", CRMFeatureEngineer()),
        ("preprocessor",     column_transformer),
    ])

    logger.info("Feature pipeline built.")
    return pipeline


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import DataPreprocessor, split_data

    df = load_data()
    df = DataPreprocessor().clean(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pipe = build_feature_pipeline()
    X_train_t = pipe.fit_transform(X_train)
    X_val_t   = pipe.transform(X_val)

    print(f"\nTransformed train shape: {X_train_t.shape}")
    print("Feature pipeline works correctly.")
