"""
tests/test_pipeline.py — Unit & integration tests.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ─── Data loader tests ────────────────────────────────────────────────────────

class TestDataLoader:

    def test_synthetic_data_generates_correct_shape(self):
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data(n=500)
        assert df.shape[0] == 500
        assert "Churn" in df.columns
        assert "customerID" in df.columns

    def test_churn_rate_is_realistic(self):
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data(n=2000, seed=42)
        churn_rate = (df["Churn"] == "Yes").mean()
        # Should be between 10% and 50%
        assert 0.10 <= churn_rate <= 0.50, f"Unrealistic churn rate: {churn_rate:.1%}"

    def test_required_columns_present(self):
        from src.data_loader import generate_synthetic_data
        from config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
        df = generate_synthetic_data(n=100)
        for col in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"


# ─── Preprocessing tests ──────────────────────────────────────────────────────

class TestPreprocessor:

    @pytest.fixture
    def sample_df(self):
        from src.data_loader import generate_synthetic_data
        return generate_synthetic_data(n=200, seed=42)

    def test_clean_removes_customer_id(self, sample_df):
        from src.preprocessing import DataPreprocessor
        df_clean = DataPreprocessor().clean(sample_df)
        assert "customerID" not in df_clean.columns

    def test_churn_is_binary_int(self, sample_df):
        from src.preprocessing import DataPreprocessor
        df_clean = DataPreprocessor().clean(sample_df)
        assert set(df_clean["Churn"].unique()).issubset({0, 1})

    def test_no_missing_after_cleaning(self, sample_df):
        from src.preprocessing import DataPreprocessor
        # Introduce artificial missing values
        sample_df.loc[sample_df.index[:10], "TotalCharges"] = np.nan
        df_clean = DataPreprocessor().clean(sample_df)
        assert df_clean.isnull().sum().sum() == 0

    def test_split_preserves_churn_ratio(self, sample_df):
        from src.preprocessing import DataPreprocessor, split_data
        df_clean = DataPreprocessor().clean(sample_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

        overall_rate = y_train.mean()
        # Each split should be within 5% of overall rate
        assert abs(y_val.mean()  - overall_rate) < 0.05
        assert abs(y_test.mean() - overall_rate) < 0.05


# ─── Feature engineering tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    @pytest.fixture
    def clean_data(self):
        from src.data_loader import generate_synthetic_data
        from src.preprocessing import DataPreprocessor, split_data
        df = generate_synthetic_data(n=500, seed=42)
        df = DataPreprocessor().clean(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        return X_train, X_val, y_train, y_val

    def test_feature_pipeline_transforms_without_error(self, clean_data):
        from src.feature_engineering import build_feature_pipeline
        X_train, X_val, y_train, y_val = clean_data
        pipe = build_feature_pipeline()
        X_t = pipe.fit_transform(X_train)
        assert X_t.shape[0] == len(X_train)
        assert X_t.shape[1] > 0

    def test_no_nans_after_transform(self, clean_data):
        from src.feature_engineering import build_feature_pipeline
        X_train, X_val, y_train, y_val = clean_data
        pipe = build_feature_pipeline()
        X_t = pipe.fit_transform(X_train)
        assert not np.isnan(X_t).any()

    def test_crm_features_added(self):
        from src.feature_engineering import CRMFeatureEngineer
        from src.data_loader import generate_synthetic_data
        from src.preprocessing import DataPreprocessor

        df = generate_synthetic_data(n=50, seed=0)
        df = DataPreprocessor().clean(df)
        X = df.drop(columns=["Churn"])

        eng = CRMFeatureEngineer()
        X_eng = eng.transform(X)

        for feature in ["IsLongTerm", "IsNewCustomer", "ServiceBundleScore",
                        "ContractRiskScore", "HighRiskCombo"]:
            assert feature in X_eng.columns, f"Missing engineered feature: {feature}"


# ─── Model prediction tests ───────────────────────────────────────────────────

class TestModelPrediction:

    @pytest.fixture(scope="class")
    def trained_pipeline(self):
        """Train a fast logistic regression for testing."""
        from src.data_loader import generate_synthetic_data
        from src.preprocessing import DataPreprocessor, split_data
        from src.train import build_model_pipeline

        df = generate_synthetic_data(n=1000, seed=42)
        df = DataPreprocessor().clean(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        pipe = build_model_pipeline("logistic_regression")
        pipe.fit(X_train, y_train)
        return pipe, X_test, y_test

    def test_predict_proba_shape(self, trained_pipeline):
        pipe, X_test, y_test = trained_pipeline
        probs = pipe.predict_proba(X_test)
        assert probs.shape == (len(X_test), 2)

    def test_proba_sum_to_one(self, trained_pipeline):
        pipe, X_test, y_test = trained_pipeline
        probs = pipe.predict_proba(X_test)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_roc_auc_above_random(self, trained_pipeline):
        from sklearn.metrics import roc_auc_score
        pipe, X_test, y_test = trained_pipeline
        probs = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        assert auc > 0.60, f"ROC-AUC too low: {auc:.4f} (should beat random)"
