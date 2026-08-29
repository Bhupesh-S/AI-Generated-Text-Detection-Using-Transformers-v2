"""
visualize.py
============
All plotting routines for the RoBERTa training pipeline.

Produces publication-quality figures saved as PNG:
  - Training / Validation Loss Curve
  - Accuracy Curve
  - F1 Score Curve
  - ROC Curve
  - Precision-Recall Curve
  - Confusion Matrix Heatmap

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for server / subprocess use
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

PALETTE = {
    "train":    "#4A90D9",
    "val":      "#E57373",
    "positive": "#66BB6A",
    "negative": "#EF5350",
    "accent":   "#AB47BC",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.8,
    }
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_path: PathLike, title: str) -> None:
    """Save figure to *output_path* and close it."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved {title} → {path}")


# ---------------------------------------------------------------------------
# Loss & Metric Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: Dict[str, list],
    output_path: PathLike,
) -> None:
    """
    Plot training loss (per-step) and validation loss (per-epoch).

    Args:
        history     : Output of :func:`utils.extract_training_history`.
        output_path : Where to save the PNG.
    """
    train_entries = history.get("train_loss", [])
    eval_entries  = history.get("eval_loss", [])

    if not train_entries and not eval_entries:
        logger.warning("No loss data available — skipping loss curve plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    if train_entries:
        steps  = [e["step"]  for e in train_entries]
        losses = [e["loss"]  for e in train_entries]
        ax.plot(steps, losses, color=PALETTE["train"], lw=1.4, alpha=0.8, label="Train Loss")

    if eval_entries:
        # Map epoch → step (approximate; multiply by avg steps/epoch)
        epochs = [e["epoch"] for e in eval_entries]
        losses = [e["loss"]  for e in eval_entries]
        ax.plot(epochs, losses, color=PALETTE["val"], lw=2, marker="o",
                markersize=6, label="Validation Loss")
        ax.set_xlabel("Epoch / Step")
    else:
        ax.set_xlabel("Step")

    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, output_path, "Loss Curves")


def plot_accuracy_curve(
    history: Dict[str, list],
    output_path: PathLike,
) -> None:
    """
    Plot validation accuracy over epochs.

    Args:
        history     : Output of :func:`utils.extract_training_history`.
        output_path : Where to save the PNG.
    """
    entries = history.get("eval_accuracy", [])
    if not entries:
        logger.warning("No accuracy data — skipping accuracy curve.")
        return

    epochs = [e["epoch"]   for e in entries]
    accs   = [e["accuracy"] for e in entries]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, accs, color=PALETTE["positive"], lw=2, marker="s",
            markersize=6, label="Val Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, output_path, "Accuracy Curve")


def plot_f1_curve(
    history: Dict[str, list],
    output_path: PathLike,
) -> None:
    """
    Plot validation F1 score over epochs.

    Args:
        history     : Output of :func:`utils.extract_training_history`.
        output_path : Where to save the PNG.
    """
    entries = history.get("eval_f1", [])
    if not entries:
        logger.warning("No F1 data — skipping F1 curve.")
        return

    epochs = [e["epoch"] for e in entries]
    f1s    = [e["f1"]    for e in entries]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, f1s, color=PALETTE["accent"], lw=2, marker="D",
            markersize=6, label="Val F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Validation F1 Score", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, output_path, "F1 Curve")


def plot_combined_curves(
    history: Dict[str, list],
    output_dir: PathLike,
) -> None:
    """
    Convenience: generate all training-history plots and save them
    individually inside *output_dir*.

    Produces:
      - ``loss_curves.png``
      - ``accuracy_curve.png``
      - ``f1_curve.png``

    Args:
        history    : Output of :func:`utils.extract_training_history`.
        output_dir : Directory where PNGs are saved.
    """
    d = Path(output_dir)
    plot_loss_curves(history,    d / "loss_curves.png")
    plot_accuracy_curve(history, d / "accuracy_curve.png")
    plot_f1_curve(history,       d / "f1_curve.png")


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: PathLike,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot and save a colour-coded confusion matrix heatmap.

    Args:
        labels      : Ground-truth integer labels.
        preds       : Predicted integer labels.
        output_path : Where to save the PNG.
        class_names : Optional list of class names.
                      Defaults to ``["Human Written", "AI Generated"]``.
    """
    if class_names is None:
        class_names = ["Human Written", "AI Generated"]

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    _save(fig, output_path, "Confusion Matrix")


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    labels: np.ndarray,
    probs_positive: np.ndarray,
    output_path: PathLike,
) -> None:
    """
    Plot ROC curve with AUC annotation.

    Args:
        labels         : Ground-truth integer labels.
        probs_positive : Predicted probability for the positive class (AI=1).
        output_path    : Where to save the PNG.
    """
    fpr, tpr, _ = roc_curve(labels, probs_positive)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["train"], lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    _save(fig, output_path, "ROC Curve")


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    labels: np.ndarray,
    probs_positive: np.ndarray,
    output_path: PathLike,
) -> None:
    """
    Plot Precision-Recall curve with average precision annotation.

    Args:
        labels         : Ground-truth integer labels.
        probs_positive : Predicted probability for the positive class (AI=1).
        output_path    : Where to save the PNG.
    """
    precision, recall, _ = precision_recall_curve(labels, probs_positive)
    ap = average_precision_score(labels, probs_positive)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["accent"], lw=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    _save(fig, output_path, "Precision-Recall Curve")
