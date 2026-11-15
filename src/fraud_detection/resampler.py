"""Resampling module for handling class imbalance via SMOTE.

Strictly applied ONLY to the training dataset.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from .config import SMOTEConfig

logger = logging.getLogger(__name__)


def resample_training_data(
    X_train: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    config: Optional[SMOTEConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies SMOTE oversampling strictly to training partitions.

    Args:
        X_train: Feature matrix for training.
        y_train: Binary labels for training.
        config: SMOTE configuration settings.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled (X_train_resampled, y_train_resampled).

    Raises:
        ValueError: If input arrays are empty or contain only one class.
    """
    config = config or SMOTEConfig()

    y_array = np.asarray(y_train)
    classes, counts = np.unique(y_array, return_counts=True)

    if len(classes) < 2:
        raise ValueError("SMOTE requires at least two classes in y_train.")

    class_counts_before = dict(zip(classes, counts))
    logger.info("Class distribution BEFORE SMOTE: %s", class_counts_before)

    minority_count = min(counts)
    k_neighbors = config.k_neighbors
    # Ensure k_neighbors is strictly less than minority class count
    if minority_count <= k_neighbors:
        k_neighbors = max(1, minority_count - 1)
        logger.warning(
            "Minority class count (%d) is <= default k_neighbors (%d). "
            "Adjusting k_neighbors to %d.",
            minority_count,
            config.k_neighbors,
            k_neighbors,
        )

    smote = SMOTE(
        random_state=config.random_state,
        sampling_strategy=config.sampling_strategy,
        k_neighbors=k_neighbors,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_array)

    res_classes, res_counts = np.unique(y_resampled, return_counts=True)
    class_counts_after = dict(zip(res_classes, res_counts))
    logger.info("Class distribution AFTER SMOTE: %s", class_counts_after)

    return X_resampled, y_resampled
