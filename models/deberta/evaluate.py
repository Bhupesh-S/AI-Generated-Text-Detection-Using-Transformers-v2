"""
evaluate.py
===========
Standalone evaluation CLI for any fine-tuned DeBERTa model checkpoint.

Run with:
    python models/deberta/evaluate.py --checkpoint <path_to_checkpoint_dir>

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Path bootstrap
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from dataset import load_dataset_splits, tokenize_datasets
from model import load_model
from trainer import build_trainer, full_evaluate
from utils import get_logger, save_json, print_section

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path: Path) -> None:
    """Load model checkpoint and evaluate performance on the test set."""
    print_section(f"Evaluating Checkpoint: {checkpoint_path.name}")

    # 1. Load data
    logger.info("Loading test dataset splits...")
    raw_datasets = load_dataset_splits()

    # 2. Load tokenizer and tokenize
    logger.info(f"Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    # 3. Load model from checkpoint
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    model = load_model(model_name_or_path=str(checkpoint_path), num_labels=CFG.num_labels)

    # 4. Build Trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )

    # 5. Evaluate
    t_start = time.time()
    test_metrics, preds, probs, labels = full_evaluate(
        trainer,
        tokenized_datasets["test"],
        prefix="eval"
    )
    elapsed = time.time() - t_start
    samples_per_sec = round(len(labels) / elapsed, 1)

    test_metrics["eval_inference_time_sec"] = round(elapsed, 3)
    test_metrics["eval_samples_per_sec"] = samples_per_sec

    print_section("Evaluation Metrics Results")
    for k, v in test_metrics.items():
        logger.info(f"  {k:<25}: {v}")

    # Save output metrics to checkpoint directory
    out_json = checkpoint_path / "eval_test_metrics.json"
    save_json(test_metrics, out_json)
    logger.info(f"Results saved successfully -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone DeBERTa Checkpoint Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned DeBERTa model directory. Defaults to CFG.tokenizer_dir."
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else CFG.tokenizer_dir

    if not ckpt.exists():
        logger.error(f"Target checkpoint path does not exist: {ckpt}")
        sys.exit(1)

    try:
        evaluate_checkpoint(ckpt)
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        sys.exit(1)
