"""
visualize.py
============
Data visualization generator for DeBERTa pipeline metrics.

Generates:
  - Combined loss, accuracy, and F1 training history plots.
  - Confusion Matrix heatmaps.
  - ROC (Receiver Operating Characteristic) curve.
  - Precision-Recall curve.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import matplotlib
matplotlib.use("Agg")  # Run headlessly to prevent Tkinter GUI errors on Windows.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_combined_curves(history: Dict[str, List[dict]], output_dir: Path) -> None:
    """
    Generate and save Loss, Accuracy, and F1 progression plots.

    Args:
        history    : Parsed training history.
        output_dir : Destination folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss Curves
    plt.figure(figsize=(7, 5))
    train_loss = history.get("train_loss", [])
    eval_loss = history.get("eval_loss", [])

    if train_loss:
        x_t = [entry["epoch"] for entry in train_loss]
        y_t = [entry["loss"] for entry in train_loss]
        plt.plot(x_t, y_t, label="Training Loss", color="#3b82f6", alpha=0.9, lw=2)

    if eval_loss:
        x_e = [entry["epoch"] for entry in eval_loss]
        y_e = [entry["loss"] for entry in eval_loss]
        plt.plot(x_e, y_e, label="Validation Loss", color="#ef4444", alpha=0.9, marker="o", lw=2)

    plt.title("Loss Progression", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()
    logger.info("Saved Loss Curves -> %s", output_dir / "loss_curves.png")

    # 2. Accuracy Curve
    plt.figure(figsize=(7, 5))
    eval_acc = history.get("eval_accuracy", [])
    if eval_acc:
        x_a = [entry["epoch"] for entry in eval_acc]
        y_a = [entry["accuracy"] for entry in eval_acc]
        plt.plot(x_a, y_a, label="Val Accuracy", color="#10b981", marker="s", lw=2)

    plt.title("Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()
    logger.info("Saved Accuracy Curve -> %s", output_dir / "accuracy_curve.png")

    # 3. F1 Curve
    plt.figure(figsize=(7, 5))
    eval_f1 = history.get("eval_f1", [])
    if eval_f1:
        x_f = [entry["epoch"] for entry in eval_f1]
        y_f = [entry["f1"] for entry in eval_f1]
        plt.plot(x_f, y_f, label="Val F1-Score", color="#8b5cf6", marker="^", lw=2)

    plt.title("Validation F1-Score", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "f1_curve.png", dpi=150)
    plt.close()
    logger.info("Saved F1 Curve -> %s", output_dir / "f1_curve.png")


def plot_confusion_matrix(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray], output_path: Path) -> None:
    """
    Generate and save a Seaborn heatmap representing the confusion matrix.

    Args:
        y_true      : True target labels.
        y_pred      : Predicted class labels.
        output_path : Target file path (.png).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Human Written", "AI Generated"],
        yticklabels=["Human Written", "AI Generated"],
        annot_kws={"size": 10, "weight": "bold"}
    )

    plt.title("Confusion Matrix Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Labels", labelpad=8)
    plt.ylabel("True Labels", labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved Confusion Matrix -> %s", output_path)


def plot_roc_curve(y_true: Union[list, np.ndarray], y_probs: Union[list, np.ndarray], output_path: Path) -> None:
    """
    Generate and save ROC curve.

    Args:
        y_true      : True target labels.
        y_probs     : Softmax probability scores of the positive class (AI Generated).
        output_path : Target file path (.png).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#f59e0b", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("ROC Curve Analysis", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", labelpad=8)
    plt.ylabel("True Positive Rate", labelpad=8)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved ROC Curve -> %s", output_path)


def plot_precision_recall_curve(y_true: Union[list, np.ndarray], y_probs: Union[list, np.ndarray], output_path: Path) -> None:
    """
    Generate and save Precision-Recall curve.

    Args:
        y_true      : True target labels.
        y_probs     : Softmax probability scores of the positive class (AI Generated).
        output_path : Target file path (.png).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#10b981", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.plot([0, 1], [0.5, 0.5], color="#9ca3af", linestyle="--", lw=1.5)  # assuming balanced classes for reference

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Precision-Recall Curve Analysis", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Recall", labelpad=8)
    plt.ylabel("Precision", labelpad=8)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved Precision-Recall Curve -> %s", output_path)
