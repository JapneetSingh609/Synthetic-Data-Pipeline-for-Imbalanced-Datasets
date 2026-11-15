"""Data loading, schema validation, and dataset splitting module."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Safely loads transaction dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded transactions DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed as a DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'. Please verify the path or "
            "provide sample data."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV from '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Loaded dataset from '{path}' is completely empty.")

    logger.info("Successfully loaded dataset from %s (%d rows, %d columns)", path, len(df), len(df.columns))
    return df


def validate_schema(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    required_cols: Optional[List[str]] = None,
) -> None:
    """Validates that the DataFrame conforms to schema expectations.

    Args:
        df: Input DataFrame to validate.
        target_col: Expected target column name.
        required_cols: Optional list of columns required to be present.

    Raises:
        ValueError: If target_col is missing or dataset has invalid structure.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

    unique_targets = df[target_col].dropna().unique()
    if len(unique_targets) < 2:
        raise ValueError(
            f"Target column '{target_col}' must have at least 2 distinct classes, found: {unique_targets}"
        )


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into stratified train and test partitions.

    Ensures the target and non-predictive columns are separated prior to splitting.

    Args:
        df: Full dataset.
        target_col: Name of the binary target label column.
        drop_cols: List of columns to exclude from feature matrix X.
        test_size: Proportion of dataset allocated to the test split.
        random_state: Random seed for reproducibility.
        stratify: Whether to preserve class proportions via stratified split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    y = df[target_col].copy()

    columns_to_drop = [target_col]
    if drop_cols:
        columns_to_drop.extend([col for col in drop_cols if col in df.columns])

    X = df.drop(columns=columns_to_drop)

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        "Split dataset into Train (%d samples) and Test (%d samples). "
        "Train fraud rate: %.2f%%, Test fraud rate: %.2f%%",
        len(X_train),
        len(X_test),
        (y_train.sum() / len(y_train)) * 100,
        (y_test.sum() / len(y_test)) * 100,
    )

    return X_train, X_test, y_train, y_test


def generate_synthetic_transactions(
    num_samples: int = 1500,
    fraud_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates a realistic synthetic transaction dataset matching original schema.

    Useful for offline testing, demos, and CI/CD without proprietary data.

    Args:
        num_samples: Total number of rows to generate.
        fraud_rate: Approximate proportion of fraudulent records (class imbalance).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic transaction dataset.
    """
    rng = np.random.default_rng(random_state)

    countries = ["US", "GB", "CA", "DE", "FR", "SG", "AU"]
    channels = ["web", "mobile_app", "pos", "api"]
    merchant_categories = ["retail", "travel", "gaming", "electronics", "grocery", "services"]

    amounts = np.round(rng.exponential(scale=75.0, size=num_samples) + 2.0, 2)
    device_risk_scores = np.round(rng.beta(a=2, b=5, size=num_samples), 3)
    account_age_days = rng.integers(low=1, high=1800, size=num_samples)
    transaction_distance_km = np.round(rng.exponential(scale=35.0, size=num_samples), 1)

    country_vals = rng.choice(countries, size=num_samples)
    bin_country_vals = rng.choice(countries, size=num_samples)
    channel_vals = rng.choice(channels, size=num_samples)
    merchant_vals = rng.choice(merchant_categories, size=num_samples)

    # Base fraud probability influenced by device risk, amount, and country mismatch
    mismatch = (country_vals != bin_country_vals).astype(float)
    latent_risk = (
        0.35 * (device_risk_scores > 0.6).astype(float)
        + 0.25 * (amounts > 250).astype(float)
        + 0.20 * mismatch
        + 0.20 * rng.uniform(0, 1, size=num_samples)
    )

    threshold = np.percentile(latent_risk, 100 * (1 - fraud_rate))
    is_fraud = (latent_risk >= threshold).astype(int)

    # Timestamp strings
    dates = pd.date_range("2026-01-01", periods=num_samples, freq="15min")
    transaction_times = dates.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(
        {
            "amount": amounts,
            "device_risk_score": device_risk_scores,
            "account_age_days": account_age_days,
            "transaction_distance_km": transaction_distance_km,
            "country": country_vals,
            "bin_country": bin_country_vals,
            "channel": channel_vals,
            "merchant_category": merchant_vals,
            "transaction_time": transaction_times,
            "is_fraud": is_fraud,
        }
    )
    return df
