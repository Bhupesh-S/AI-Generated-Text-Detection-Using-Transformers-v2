"""
evaluate.py
===========
Standalone evaluation script for a fine-tuned DistilBERT checkpoint.

Usage:
    python models/distilbert/evaluate.py --checkpoint outputs/distilbert/checkpoints/checkpoint-best

If no --checkpoint is given, loads the best model saved by train.py at
``outputs/distilbert/tokenizer`` (the final save_pretrained location).

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Allow ``python models/distilbert/evaluate.py`` to find sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from dataset import load_dataset_splits, tokenize_datasets
from model import load_model
from trainer import build_trainer, full_evaluate
from visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from utils import get_logger, set_seed, save_json

logger = get_logger(__name__, log_file=CFG.logs_dir / "evaluate.log")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned DistilBERT checkpoint on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CFG.tokenizer_dir),
        help="Path to model checkpoint directory (default: outputs/distilbert/tokenizer).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CFG.batch_size,
        help="Evaluation batch size.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run evaluation on the held-out test set and save metrics + plots."""
    args = parse_args()
    set_seed(CFG.seed)
    CFG.create_dirs()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load Data ────────────────────────────────────────────────────────────
    logger.info("Loading dataset splits …")
    raw_datasets = load_dataset_splits()

    # ── Load Tokenizer & Tokenize ─────────────────────────────────────────
    logger.info(f"Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenized = tokenize_datasets(raw_datasets, tokenizer)

    # ── Load Model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from: {checkpoint_path}")
    model = load_model(model_name_or_path=str(checkpoint_path))
    model.to(device)

    # ── Build Trainer ─────────────────────────────────────────────────────
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    t0 = time.time()
    metrics, preds, probs, labels = full_evaluate(trainer, tokenized["test"])
    elapsed = time.time() - t0
    metrics["inference_time_sec"] = round(elapsed, 3)
    metrics["samples_per_sec"] = round(len(tokenized["test"]) / elapsed, 1)

    logger.info(f"Test Metrics: {metrics}")

    # ── Save Metrics ──────────────────────────────────────────────────────
    metrics_path = CFG.reports_dir / "test_metrics.json"
    save_json(metrics, metrics_path)
    logger.info(f"Metrics saved → {metrics_path}")

    # ── Visualisations ────────────────────────────────────────────────────
    plot_confusion_matrix(
        labels, preds,
        output_path=CFG.figures_dir / "confusion_matrix.png",
    )
    plot_roc_curve(
        labels, probs[:, 1],
        output_path=CFG.figures_dir / "roc_curve.png",
    )
    plot_precision_recall_curve(
        labels, probs[:, 1],
        output_path=CFG.figures_dir / "precision_recall_curve.png",
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
