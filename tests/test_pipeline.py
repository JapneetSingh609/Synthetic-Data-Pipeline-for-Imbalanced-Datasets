"""Unit tests for fraud detection pipeline components using standard library unittest."""

import sys
import unittest
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from fraud_detection import (
    FraudPreprocessor,
    generate_synthetic_transactions,
    resample_training_data,
    split_data,
    validate_schema,
)


class TestFraudPipeline(unittest.TestCase):
    """Test suite verifying core pipeline logic."""

    def test_generate_synthetic_transactions(self):
        """Verify synthetic transactions generation conforms to schema."""
        df = generate_synthetic_transactions(num_samples=100, fraud_rate=0.1, random_state=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn("is_fraud", df.columns)
        self.assertTrue(set(df["is_fraud"].unique()).issubset({0, 1}))

    def test_validate_schema_valid(self):
        """Verify schema validation passes for valid dataset."""
        df = generate_synthetic_transactions(num_samples=50, random_state=42)
        # Should not raise
        validate_schema(df, target_col="is_fraud")

    def test_validate_schema_invalid(self):
        """Verify schema validation raises ValueError if target column is absent."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        with self.assertRaises(ValueError):
            validate_schema(df, target_col="is_fraud")

    def test_preprocessor_no_leakage(self):
        """Verify preprocessor fits only on train and transforms both train and test."""
        df = generate_synthetic_transactions(num_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_col="is_fraud",
            drop_cols=["transaction_time"],
            test_size=0.2,
            random_state=42,
        )
        preprocessor = FraudPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        self.assertIsInstance(X_train_proc, np.ndarray)
        self.assertIsInstance(X_test_proc, np.ndarray)
        self.assertEqual(X_train_proc.shape[0], len(X_train))
        self.assertEqual(X_test_proc.shape[0], len(X_test))
        self.assertEqual(X_train_proc.shape[1], X_test_proc.shape[1])
        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertFalse(np.isnan(X_test_proc).any())

    def test_smote_resampling(self):
        """Verify SMOTE balances training classes."""
        df = generate_synthetic_transactions(num_samples=200, fraud_rate=0.05, random_state=42)
        X_train, _, y_train, _ = split_data(
            df,
            target_col="is_fraud",
            drop_cols=["transaction_time"],
            test_size=0.2,
            random_state=42,
        )
        preprocessor = FraudPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)

        X_res, y_res = resample_training_data(X_train_proc, y_train)
        classes, counts = np.unique(y_res, return_counts=True)
        self.assertEqual(len(classes), 2)
        self.assertEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
