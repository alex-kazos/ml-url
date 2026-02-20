"""
Utility functions for model training, evaluation, and visualisation.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")              # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return a dict of standard classification metrics."""
    metrics: Dict[str, float] = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Pretty-print the sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"]))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Save a confusion-matrix heatmap as a PNG file."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Save an ROC-curve plot as a PNG file."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curve       → {save_path}")


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of top-N feature importances (tree models)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importances — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved feature importance → {save_path}")
