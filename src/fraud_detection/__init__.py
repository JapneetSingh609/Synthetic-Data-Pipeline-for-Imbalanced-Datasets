"""Fraud Detection Package.

Production-grade machine learning pipeline for detecting fraudulent transactions
in highly imbalanced tabular datasets.
"""

__version__ = "1.0.0"

from .config import PipelineConfig, DataConfig, ModelConfig, TuningConfig, PreprocessorConfig, SMOTEConfig
from .data_loader import load_data, validate_schema, split_data, generate_synthetic_transactions
from .preprocessor import FraudPreprocessor
from .resampler import resample_training_data
from .models import get_models, train_model, train_all_models, tune_model, train_and_tune_all_models
from .evaluate import (
    evaluate_model,
    evaluate_all,
    results_to_dataframe,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_roc_pr_curves,
)

__all__ = [
    "PipelineConfig",
    "DataConfig",
    "ModelConfig",
    "TuningConfig",
    "PreprocessorConfig",
    "SMOTEConfig",
    "load_data",
    "validate_schema",
    "split_data",
    "generate_synthetic_transactions",
    "FraudPreprocessor",
    "resample_training_data",
    "get_models",
    "train_model",
    "train_all_models",
    "tune_model",
    "train_and_tune_all_models",
    "evaluate_model",
    "evaluate_all",
    "results_to_dataframe",
    "plot_confusion_matrices",
    "plot_metrics_comparison",
    "plot_roc_pr_curves",
]

