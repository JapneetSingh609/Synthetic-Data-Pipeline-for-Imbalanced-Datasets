"""Preprocessing pipeline module for feature transformation, encoding, and scaling.

Strictly fits transformers on training data only to prevent data leakage.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import PreprocessorConfig

logger = logging.getLogger(__name__)


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Encapsulates feature preprocessing without data leakage.

    Handles numeric imputation and scaling (StandardScaler), plus categorical
    imputation and ordinal encoding with graceful handling of unseen categories.
    """

    def __init__(
        self,
        config: Optional[PreprocessorConfig] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ) -> None:
        """Initializes FraudPreprocessor.

        Args:
            config: Preprocessing configuration options.
            categorical_cols: Explicit list of categorical columns.
            numerical_cols: Explicit list of numerical columns.
        """
        self.config = config or PreprocessorConfig()
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.pipeline: Optional[ColumnTransformer] = None
        self._feature_names: List[str] = []

    def _resolve_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Resolves numerical and categorical columns from DataFrame if not specified."""
        if self.categorical_cols is not None and self.numerical_cols is not None:
            cat_cols = [c for c in self.categorical_cols if c in X.columns]
            num_cols = [c for c in self.numerical_cols if c in X.columns]
            return cat_cols, num_cols

        cat_cols = (
            [c for c in self.categorical_cols if c in X.columns]
            if self.categorical_cols is not None
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )

        num_cols = (
            [c for c in self.numerical_cols if c in X.columns]
            if self.numerical_cols is not None
            else [c for c in X.select_dtypes(include=[np.number]).columns if c not in cat_cols]
        )

        return cat_cols, num_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudPreprocessor":
        """Fits the transformers on the training feature set X.

        Args:
            X: Training features.
            y: Ignored (unsupervised preprocessing).

        Returns:
            self: Fitted preprocessor instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for column-based preprocessing.")

        cat_cols, num_cols = self._resolve_columns(X)
        self.categorical_cols = cat_cols
        self.numerical_cols = num_cols

        logger.info(
            "Fitting Preprocessor on %d samples: %d numeric features, %d categorical features.",
            len(X),
            len(num_cols),
            len(cat_cols),
        )

        transformers = []

        if num_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.config.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            num_pipeline = Pipeline(steps=num_steps)
            transformers.append(("num", num_pipeline, num_cols))

        if cat_cols:
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=self.config.unknown_value,
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.pipeline.fit(X)
        self._feature_names = num_cols + cat_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature set X using pre-fitted transformers.

        Args:
            X: Input DataFrame to transform.

        Returns:
            np.ndarray: Transformed numeric matrix.
        """
        if self.pipeline is None:
            raise RuntimeError("FraudPreprocessor must be fitted before transform is called.")

        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fits preprocessor on training data and transforms it in one step.

        Args:
            X: Training features.
            y: Ignored.

        Returns:
            np.ndarray: Transformed training matrix.
        """
        return self.fit(X, y).transform(X)

    @property
    def feature_names(self) -> List[str]:
        """Names of features in order of the transformed array columns."""
        return self._feature_names
