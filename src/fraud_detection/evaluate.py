"""Evaluation and reporting module for fraud detection models.

Prioritizes fraud-specific metrics (Precision, Recall, F1, PR-AUC, ROC-AUC,
Confusion Matrix) over uninformative raw accuracy on imbalanced data.
Visualizations are fully decoupled for non-blocking headless CI/CD execution.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Encapsulates all evaluation metrics for a single trained model."""

    model_name: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: np.ndarray
    classification_report_str: str
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None

    def print_summary(self) -> None:
        """Prints formatted summary report to standard output."""
        print(f"\n{'='*55}")
        print(f"Model: {self.model_name}")
        print(f"{'='*55}")
        print(f"Accuracy:          {self.accuracy:.4f}")
        print(f"Balanced Accuracy: {self.balanced_accuracy:.4f}")
        print(f"Precision (Fraud): {self.precision:.4f}")
        print(f"Recall (Fraud):    {self.recall:.4f}")
        print(f"F1-Score (Fraud):  {self.f1:.4f}")
        if self.roc_auc is not None:
            print(f"ROC-AUC:           {self.roc_auc:.4f}")
        if self.pr_auc is not None:
            print(f"PR-AUC (Avg Prec): {self.pr_auc:.4f}")
        print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
        print(self.confusion_matrix)
        print("\nDetailed Classification Report:")
        print(self.classification_report_str)


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: Union[pd.Series, np.ndarray],
) -> EvaluationResult:
    """Evaluates a single model on test features and returns structured metrics.

    Args:
        model_name: Identifier for the model.
        model: Fitted estimator.
        X_test: Unseen test features.
        y_test: True test ground truth labels.

    Returns:
        EvaluationResult: Structured metrics dataclass.
    """
    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC and PR-AUC
    y_prob: Optional[np.ndarray] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        # Class 1 probability
        if probs.shape[1] > 1:
            y_prob = probs[:, 1]
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        roc_auc = float(roc_auc_score(y_true, decision))
        pr_auc = float(average_precision_score(y_true, decision))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    result = EvaluationResult(
        model_name=model_name,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        classification_report_str=report,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return result


def evaluate_all(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, EvaluationResult]:
    """Evaluates all fitted models without variable overwriting bugs."""
    results: Dict[str, EvaluationResult] = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_test, y_test)
    return results


def results_to_dataframe(
    results: Dict[str, EvaluationResult],
    model_order: Optional[List[str]] = None,
    sort_by_metric: Optional[str] = None,
) -> pd.DataFrame:
    """Tabulates key evaluation metrics for easy side-by-side comparison.

    Args:
        results: Dictionary mapping model names to EvaluationResult.
        model_order: Optional list specifying explicit model sequence.
            Defaults to ["Logistic Regression", "Decision Tree", "Random Forest"].
        sort_by_metric: Optional metric column to sort by (e.g. 'F1-Score').
            If None, respects model_order (progression sequence).

    Returns:
        pd.DataFrame: Structured metrics comparison table.
    """
    order = model_order or ["Logistic Regression", "Decision Tree", "Random Forest"]
    ordered_results: List[EvaluationResult] = []
    seen = set()

    for name in order:
        if name in results:
            ordered_results.append(results[name])
            seen.add(name)
    for name, res in results.items():
        if name not in seen:
            ordered_results.append(res)

    records = []
    for res in ordered_results:
        records.append(
            {
                "Model": res.model_name,
                "Recall (Fraud)": res.recall,
                "Precision (Fraud)": res.precision,
                "F1-Score": res.f1,
                "ROC-AUC": res.roc_auc if res.roc_auc is not None else np.nan,
                "PR-AUC": res.pr_auc if res.pr_auc is not None else np.nan,
                "Balanced Accuracy": res.balanced_accuracy,
                "Accuracy": res.accuracy,
            }
        )
    df_metrics = pd.DataFrame(records)
    if sort_by_metric:
        df_metrics = df_metrics.sort_values(by=sort_by_metric, ascending=False).reset_index(drop=True)
    return df_metrics


# =====================================================================
# Headless Visualizations (Non-blocking, auto-closing figures)
# =====================================================================


def plot_confusion_matrices(
    results: Dict[str, EvaluationResult],
    output_path: Optional[Path] = None,
    show: bool = False,
    model_order: Optional[List[str]] = None,
) -> None:
    """Plots confusion matrices side-by-side in unified baseline-to-advanced order."""
    order = model_order or ["Logistic Regression", "Decision Tree", "Random Forest"]
    ordered_results = [results[m] for m in order if m in results] + [
        res for m, res in results.items() if m not in order
    ]
    n_models = len(ordered_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, ordered_results):
        sns.heatmap(
            res.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"],
        )
        ax.set_title(f"{res.model_name}\nF1: {res.f1:.3f} | Recall: {res.recall:.3f}", fontsize=11)
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved confusion matrices plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
    model_order: Optional[List[str]] = None,
) -> None:
    """Plots bar chart comparing Precision, Recall, and F1 in unified baseline-to-advanced order."""
    order = model_order or ["Logistic Regression", "Decision Tree", "Random Forest"]
    order = [m for m in order if m in metrics_df["Model"].values]

    melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Recall (Fraud)", "Precision (Fraud)", "F1-Score", "Accuracy"],
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#26A69A", "#42A5F5", "#AB47BC", "#78909C"]
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette, order=order, ax=ax)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_title("Model Comparison: Fraud Detection Metrics (Progression Order)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Metric", loc="upper right")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved metrics comparison plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_roc_pr_curves(
    results: Dict[str, EvaluationResult],
    y_test: Union[pd.Series, np.ndarray],
    output_path: Optional[Path] = None,
    show: bool = False,
    model_order: Optional[List[str]] = None,
) -> None:
    """Plots ROC and Precision-Recall curves side-by-side in unified baseline-to-advanced order."""
    order = model_order or ["Logistic Regression", "Decision Tree", "Random Forest"]
    ordered_results = [results[m] for m in order if m in results] + [
        res for m, res in results.items() if m not in order
    ]

    y_true = np.asarray(y_test)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # Progression colors: Blue (Baseline), Orange (Tree), Purple (Ensemble)
    color_map = {
        "Logistic Regression": "#1E88E5",
        "Decision Tree": "#FB8C00",
        "Random Forest": "#8E24AA",
    }
    fallback_colors = ["#004D40", "#D81B60", "#FFC107", "#00ACC1"]

    for idx, res in enumerate(ordered_results):
        if res.y_prob is None:
            continue
        color = color_map.get(res.model_name, fallback_colors[idx % len(fallback_colors)])

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, res.y_prob)
        ax_roc.plot(fpr, tpr, label=f"{res.model_name} (AUC = {res.roc_auc:.3f})", color=color, lw=2)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, res.y_prob)
        ax_pr.plot(recall, precision, label=f"{res.model_name} (PR-AUC = {res.pr_auc:.3f})", color=color, lw=2)

    # Reference lines
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    baseline_fraud = y_true.sum() / len(y_true)
    ax_pr.axhline(baseline_fraud, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_fraud:.2f})")
    ax_pr.set_title("Precision-Recall (PR) Curve", fontweight="bold")
    ax_pr.set_xlabel("Recall", fontweight="bold")
    ax_pr.set_ylabel("Precision", fontweight="bold")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved ROC & PR curves plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)

