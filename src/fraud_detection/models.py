"""Model definitions, hyperparameter tuning, and training orchestration module.

Provides initialization, cross-validated hyperparameter search (RandomizedSearchCV)
optimizing for fraud-specific metrics (F1-score), and training routines for
Logistic Regression, Decision Tree, and Random Forest classifiers.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .config import ModelConfig, TuningConfig

logger = logging.getLogger(__name__)


def get_models(config: Optional[ModelConfig] = None) -> Dict[str, BaseEstimator]:
    """Initializes and returns dictionary of candidate classification models.

    Args:
        config: Model hyperparameter configuration.

    Returns:
        Dict[str, BaseEstimator]: Mapping of model name to un-fitted estimator instance.
    """
    config = config or ModelConfig()

    models: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(**config.logistic_regression_params),
        "Decision Tree": DecisionTreeClassifier(**config.decision_tree_params),
        "Random Forest": RandomForestClassifier(**config.random_forest_params),
    }
    return models


def train_model(
    name: str,
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Trains an individual scikit-learn estimator with execution timing.

    Args:
        name: Readable model identifier.
        model: Model estimator to fit.
        X_train: Training features.
        y_train: Training target labels.

    Returns:
        BaseEstimator: Fitted model.
    """
    logger.info("Training %s...", name)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time
    logger.info("Finished training %s in %.3f seconds.", name, elapsed)
    return model


def train_all_models(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, BaseEstimator]:
    """Fits all models on the provided training set without hyperparameter search.

    Args:
        models: Dictionary of model name to un-fitted estimator.
        X_train: Resampled training feature matrix.
        y_train: Resampled training target array.

    Returns:
        Dict[str, BaseEstimator]: Dictionary of fitted model estimators.
    """
    fitted_models: Dict[str, BaseEstimator] = {}
    for name, model in models.items():
        fitted_models[name] = train_model(name, model, X_train, y_train)
    return fitted_models


def tune_model(
    name: str,
    base_estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    tuning_config: TuningConfig,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """Performs cross-validated hyperparameter tuning using RandomizedSearchCV.

    Args:
        name: Model identifier name.
        base_estimator: Base estimator instance to tune.
        param_distributions: Dictionary of hyperparameter search spaces.
        X_train: Resampled training feature matrix.
        y_train: Resampled training target array.
        tuning_config: Tuning configuration (scoring, n_iter, cv_folds).

    Returns:
        Tuple[BaseEstimator, Dict[str, Any], float]:
            Best estimator, best hyperparameters dictionary, best cross-validation score.
    """
    logger.info(
        "Starting hyperparameter tuning for %s (metric: %s, iterations: %d, CV: %d-fold)...",
        name,
        tuning_config.scoring,
        tuning_config.n_iter,
        tuning_config.cv_folds,
    )

    # Compute total combinations to cap n_iter if grid is smaller than n_iter
    total_combinations = 1
    for values in param_distributions.values():
        total_combinations *= len(values)
    actual_n_iter = min(tuning_config.n_iter, total_combinations)

    cv = StratifiedKFold(
        n_splits=tuning_config.cv_folds,
        shuffle=True,
        random_state=tuning_config.random_state,
    )

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=actual_n_iter,
        scoring=tuning_config.scoring,
        cv=cv,
        random_state=tuning_config.random_state,
        n_jobs=tuning_config.n_jobs,
        refit=True,
        verbose=1,
    )

    start_time = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)

    logger.info(
        "Tuned %s in %.2f seconds | Best CV %s: %.4f | Best Params: %s",
        name,
        elapsed,
        tuning_config.scoring.upper(),
        best_score,
        best_params,
    )

    return best_estimator, best_params, best_score


def train_and_tune_all_models(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    tuning_config: TuningConfig,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, Any]]]:
    """Trains and tunes all models, returning the best estimators and parameters.

    Args:
        models: Dictionary of model name to base estimator.
        X_train: Resampled training feature matrix.
        y_train: Resampled training target array.
        tuning_config: Tuning configuration and search spaces.

    Returns:
        Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, Any]]]:
            Dictionary of best estimators and dictionary of best hyperparameters.
    """
    grid_mapping = {
        "Logistic Regression": tuning_config.logistic_regression_grid,
        "Decision Tree": tuning_config.decision_tree_grid,
        "Random Forest": tuning_config.random_forest_grid,
    }

    best_models: Dict[str, BaseEstimator] = {}
    best_params_record: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        param_dist = grid_mapping.get(name)
        if tuning_config.enable_tuning and param_dist:
            best_estimator, best_params, best_score = tune_model(
                name=name,
                base_estimator=model,
                param_distributions=param_dist,
                X_train=X_train,
                y_train=y_train,
                tuning_config=tuning_config,
            )
            best_models[name] = best_estimator
            best_params_record[name] = {
                "best_cv_score": best_score,
                "scoring_metric": tuning_config.scoring,
                "parameters": best_params,
            }
        else:
            best_models[name] = train_model(name, model, X_train, y_train)
            best_params_record[name] = {"parameters": "default"}

    return best_models, best_params_record
