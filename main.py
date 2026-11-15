"""Main orchestration entry point for the Fraud Detection Pipeline.

Executes data loading, stratified train-test splitting, leak-free preprocessing,
SMOTE resampling strictly on training data, cross-validated hyperparameter tuning,
multi-model evaluation, and headless visualization export.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src directory is on sys.path for direct script execution
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from fraud_detection import (
    FraudPreprocessor,
    PipelineConfig,
    evaluate_all,
    generate_synthetic_transactions,
    get_models,
    load_data,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_roc_pr_curves,
    resample_training_data,
    results_to_dataframe,
    split_data,
    train_and_tune_all_models,
    validate_schema,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraud_detection_pipeline")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Production-grade Fraud Detection ML Pipeline with Hyperparameter Tuning."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("transactions.csv"),
        help="Path to input transactions CSV dataset (default: transactions.csv)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="is_fraud",
        help="Target column name for fraud label (default: is_fraud)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data allocated to the test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting and modeling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save evaluation reports and plots (default: output/)",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save visualization plots to disk (default: True)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Interactively display visualization plots (blocks execution)",
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Perform cross-validated hyperparameter tuning (default: True)",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=5,
        help="Number of parameter combinations sampled in RandomizedSearchCV (default: 5)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for tuning (default: 3)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic transactions if data file is absent or for testing",
    )
    return parser.parse_args()


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Executes the complete machine learning pipeline end-to-end.

    Args:
        config: Master configuration for pipeline components.

    Returns:
        pd.DataFrame: Summary table comparing all model metrics.
    """
    logger.info("Initializing Fraud Detection Pipeline (src-layout)...")

    # Step 1: Ingest Data
    data_path = config.data.data_path
    if not data_path.exists():
        logger.warning("Dataset not found at '%s'. Generating synthetic sample data.", data_path)
        df = generate_synthetic_transactions(
            num_samples=2000,
            fraud_rate=0.05,
            random_state=config.data.random_state,
        )
        df.to_csv(data_path, index=False)
        logger.info("Generated sample transactions dataset at '%s'", data_path)
    else:
        df = load_data(data_path)

    # Step 2: Schema Validation
    validate_schema(df, target_col=config.data.target_col)

    # Step 3: Stratified Train-Test Split (BEFORE any transformations)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df,
        target_col=config.data.target_col,
        drop_cols=config.data.drop_cols,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=config.data.stratify,
    )

    # Step 4: Preprocessing (Strictly fit on X_train only to prevent data leakage)
    preprocessor = FraudPreprocessor(
        config=config.preprocessor,
        categorical_cols=config.data.categorical_cols,
        numerical_cols=config.data.numerical_cols,
    )
    X_train_preprocessed = preprocessor.fit_transform(X_train_raw)
    X_test_preprocessed = preprocessor.transform(X_test_raw)
    logger.info("Transformed features shape: Train=%s, Test=%s", X_train_preprocessed.shape, X_test_preprocessed.shape)

    # Step 5: Resample Training Data (SMOTE applied ONLY on training partition)
    X_train_resampled, y_train_resampled = resample_training_data(
        X_train_preprocessed,
        y_train,
        config=config.smote,
    )

    # Step 6: Model Training & Hyperparameter Tuning
    models = get_models(config=config.models)
    fitted_models, best_params_record = train_and_tune_all_models(
        models=models,
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        tuning_config=config.tuning,
    )

    # Step 7: Model Evaluation (Evaluated on clean, unresampled test data)
    model_order = config.evaluation.default_model_order
    eval_results = evaluate_all(fitted_models, X_test_preprocessed, y_test)

    # Print individual reports in progression order
    for name in model_order:
        if name in eval_results:
            eval_results[name].print_summary()

    # Step 8: Comparative Metrics Table (Standardized Progression Order)
    metrics_df = results_to_dataframe(eval_results, model_order=model_order)
    print("\n" + "=" * 80)
    print("FINAL MODEL PERFORMANCE COMPARISON (Baseline to Advanced Progression)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Step 9: Headless Visualizations & Artifact Export
    output_dir = config.evaluation.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics CSV
    metrics_csv_path = output_dir / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Exported metrics comparison table to %s", metrics_csv_path)

    # Save Best Hyperparameters JSON
    best_params_path = output_dir / config.evaluation.best_params_filename
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_record, f, indent=2)
    logger.info("Exported best hyperparameters to %s", best_params_path)

    if config.evaluation.save_plots:
        # Confusion Matrices
        plot_confusion_matrices(
            eval_results,
            output_path=output_dir / config.evaluation.confusion_matrix_filename,
            show=config.evaluation.show_plots,
            model_order=model_order,
        )

        # Metrics Bar Chart
        plot_metrics_comparison(
            metrics_df,
            output_path=output_dir / config.evaluation.metrics_comparison_filename,
            show=config.evaluation.show_plots,
            model_order=model_order,
        )

        # ROC & Precision-Recall Curves
        plot_roc_pr_curves(
            eval_results,
            y_test=y_test,
            output_path=output_dir / config.evaluation.roc_pr_plot_filename,
            show=config.evaluation.show_plots,
            model_order=model_order,
        )

    # Pick best model dynamically based on top F1-Score
    best_row = metrics_df.loc[metrics_df["F1-Score"].idxmax()]
    best_model_name = best_row["Model"]
    best_f1 = best_row["F1-Score"]
    best_recall = best_row["Recall (Fraud)"]
    best_precision = best_row["Precision (Fraud)"]
    logger.info(
        "Recommended Model: '%s' (F1-Score: %.4f, Recall: %.4f, Precision: %.4f)",
        best_model_name,
        best_f1,
        best_recall,
        best_precision,
    )
    return metrics_df


def main() -> None:
    """CLI handler."""
    args = parse_args()

    config = PipelineConfig()
    config.data.data_path = args.data_path
    config.data.target_col = args.target_col
    config.data.test_size = args.test_size
    config.data.random_state = args.random_state
    config.models.random_state = args.random_state
    config.tuning.enable_tuning = args.tune
    config.tuning.n_iter = args.tune_iter
    config.tuning.cv_folds = args.cv_folds
    config.tuning.random_state = args.random_state
    config.evaluation.output_dir = args.output_dir
    config.evaluation.save_plots = args.save_plots
    config.evaluation.show_plots = args.show_plots

    if args.generate_sample and not args.data_path.exists():
        logger.info("Generating synthetic transactions at '%s' as requested...", args.data_path)
        sample_df = generate_synthetic_transactions(
            num_samples=2000,
            fraud_rate=0.05,
            random_state=args.random_state,
        )
        sample_df.to_csv(args.data_path, index=False)

    try:
        run_pipeline(config)
    except Exception as err:
        logger.exception("Pipeline failed with error: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()