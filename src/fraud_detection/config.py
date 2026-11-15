"""Centralized configuration module for the Fraud Detection ML Pipeline.

Provides dataclasses for managing hyperparameters, seeds, data paths,
feature settings, hyperparameter tuning spaces, and pipeline parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for data ingestion and splitting."""

    data_path: Path = Path("transactions.csv")
    target_col: str = "is_fraud"
    # Columns to drop completely (identifiers or non-predictive timestamps)
    drop_cols: List[str] = field(
        default_factory=lambda: ["transaction_time", "transaction_id", "user_id"]
    )
    # Explicit categorical features; if None, preprocessor will auto-detect object/category types
    categorical_cols: Optional[List[str]] = field(
        default_factory=lambda: ["country", "bin_country", "channel", "merchant_category"]
    )
    # Explicit numeric features; if None, preprocessor will auto-detect numeric dtypes
    numerical_cols: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class PreprocessorConfig:
    """Configuration for feature encoding and scaling."""

    scale_numeric: bool = True
    handle_unknown_categorical: str = "use_encoded_value"  # for OrdinalEncoder
    unknown_value: int = -1


@dataclass
class SMOTEConfig:
    """Configuration for synthetic minority oversampling (SMOTE)."""

    random_state: int = 42
    sampling_strategy: str = "auto"
    k_neighbors: int = 5


@dataclass
class ModelConfig:
    """Hyperparameter configurations for all classification models."""

    random_state: int = 42
    logistic_regression_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_iter": 3000,
            "random_state": 42,
            "solver": "lbfgs",
            "class_weight": None,  # SMOTE addresses imbalance directly
        }
    )
    decision_tree_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "criterion": "gini",
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2,
        }
    )
    random_forest_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
            "criterion": "gini",
        }
    )


@dataclass
class TuningConfig:
    """Configuration for hyperparameter search and cross-validation."""

    enable_tuning: bool = True
    scoring: str = "f1"  # Optimize specifically for fraud F1-score
    n_iter: int = 5  # Number of parameter settings sampled in RandomizedSearchCV
    cv_folds: int = 3  # Stratified K-fold cross-validation folds
    n_jobs: int = -1  # Parallel search workers
    random_state: int = 42

    # Parameter search spaces
    logistic_regression_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": ["lbfgs"],
        }
    )
    decision_tree_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "max_depth": [10, 15, 20, 25, 30],
            "min_samples_split": [10, 20, 50, 100],
            "min_samples_leaf": [5, 10, 20, 50],
            "criterion": ["gini", "entropy"],
        }
    )
    random_forest_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "n_estimators": [100, 150, 200],
            "max_depth": [15, 20, 25, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", "log2"],
        }
    )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and reporting."""

    output_dir: Path = Path("output")
    save_plots: bool = True
    show_plots: bool = False
    roc_pr_plot_filename: str = "roc_pr_curves.png"
    confusion_matrix_filename: str = "confusion_matrices.png"
    metrics_comparison_filename: str = "metrics_comparison.png"
    best_params_filename: str = "best_hyperparameters.json"
    default_model_order: List[str] = field(
        default_factory=lambda: ["Logistic Regression", "Decision Tree", "Random Forest"]
    )


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    smote: SMOTEConfig = field(default_factory=SMOTEConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
