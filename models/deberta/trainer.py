"""
trainer.py
==========
Custom Hugging Face Trainer wrapper for DeBERTa fine-tuning.
Optimised for Transformers v5.14.1 + RTX 3050 Laptop (4 GB VRAM).

Key optimizations:
  - BF16 mixed precision training (bf16=True) to prevent DeBERTa FP16 gradient crashes.
  - DataCollatorWithPadding(pad_to_multiple_of=8) for tensor core alignment.
  - Fused AdamW optimizer (single CUDA kernel per param update).
  - dataloader_num_workers=0 to bypass Windows spawn deadlocks.
  - pin_memory=True for fast direct host-to-device transfers.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1, and ROC-AUC from Trainer output.

    Args:
        eval_pred : Named tuple ``(predictions, label_ids)`` from the Trainer.
                    ``predictions`` are raw logits of shape ``(N, num_labels)``.

    Returns:
        Dictionary of metric name -> float value.
    """
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
    }


# ---------------------------------------------------------------------------
# TrainingArguments builder
# ---------------------------------------------------------------------------

def build_training_args(use_bf16: bool = True, warmup_steps: int = 0) -> TrainingArguments:
    """
    Build an optimised :class:`transformers.TrainingArguments` object.

    Args:
        use_bf16     : Request BF16 mixed precision (effective only on CUDA).
        warmup_steps : Number of warmup steps (computed from ratio dynamically).

    Returns:
        :class:`TrainingArguments` instance.
    """
    bf16 = use_bf16 and torch.cuda.is_available()
    if bf16:
        logger.info("Mixed precision (BF16) ENABLED.")
    else:
        logger.info("Mixed precision (BF16) DISABLED (CPU or no CUDA).")

    # Set logging dir via env var (kwarg deprecated)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(CFG.logs_dir))
    logger.info(f"Log dir: {CFG.logs_dir}")

    # Validate fused optimizer is available before passing to TrainingArguments
    optim = CFG.optim
    if optim == "adamw_torch_fused":
        if not (torch.cuda.is_available() and hasattr(torch.optim, "AdamW")):
            logger.warning(
                "adamw_torch_fused not available on this platform; "
                "falling back to adamw_torch."
            )
            optim = "adamw_torch"
        else:
            logger.info("Fused AdamW optimizer ENABLED (single CUDA kernel per step).")

    training_args = TrainingArguments(
        # ── Paths ─────────────────────────────────────────────────────────────
        output_dir=str(CFG.checkpoint_dir),

        # ── Batch & Accumulation ──────────────────────────────────────────────
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,

        # ── Optimizer & Scheduler ─────────────────────────────────────────────
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_steps=warmup_steps,            # v5 compatibility: replaced warmup_ratio
        max_grad_norm=CFG.max_grad_norm,
        lr_scheduler_type="linear",

        # ── Optimizer: Fused AdamW ────────────────────────────────────────────
        optim=optim,

        # ── Epochs ────────────────────────────────────────────────────────────
        num_train_epochs=CFG.epochs,

        # ── Mixed Precision ───────────────────────────────────────────────────
        bf16=bf16,
        fp16=False,                           # Explicitly disabled for DeBERTa
        gradient_checkpointing=getattr(CFG, 'gradient_checkpointing', False), # VRAM optimization
        label_smoothing_factor=getattr(CFG, 'label_smoothing_factor', 0.0),   # Calibration

        # ── Evaluation & Saving ───────────────────────────────────────────────
        eval_strategy=CFG.eval_strategy,
        save_strategy=CFG.save_strategy,
        load_best_model_at_end=CFG.load_best_model_at_end,
        metric_for_best_model=CFG.metric_for_best_model,
        greater_is_better=CFG.greater_is_better,
        save_total_limit=CFG.save_total_limit,

        # ── Logging ───────────────────────────────────────────────────────────
        logging_strategy="steps",
        logging_steps=CFG.logging_steps,
        report_to=CFG.report_to,

        # ── DataLoader Pipeline ───────────────────────────────────────────────
        dataloader_num_workers=CFG.dataloader_num_workers,
        dataloader_pin_memory=CFG.dataloader_pin_memory,
        dataloader_persistent_workers=CFG.dataloader_persistent_workers,
        dataloader_prefetch_factor=CFG.dataloader_prefetch_factor,

        # ── Seed ─────────────────────────────────────────────────────────────
        seed=CFG.seed,
    )
    return training_args


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_steps: Optional[int] = None,
) -> Trainer:
    """
    Instantiate a HuggingFace :class:`Trainer` with all optimizations applied.

    Args:
        model         : Fresh or loaded DeBERTa model.
        tokenizer     : Tokenizer (passed as ``processing_class``).
        train_dataset : Tokenized training split.
        eval_dataset  : Tokenized validation split.
        max_steps     : Optional override of max steps (useful for dry runs).

    Returns:
        Fully configured :class:`Trainer`.
    """
    # Compute steps per epoch
    steps_per_epoch = len(train_dataset) // (CFG.batch_size * CFG.gradient_accumulation_steps)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    total_steps = steps_per_epoch * CFG.epochs

    if max_steps is not None:
        total_steps = max_steps

    warmup_steps = int(CFG.warmup_ratio * total_steps)
    logger.info(f"Total steps: {total_steps} | Calculated warmup_steps: {warmup_steps}")

    training_args = build_training_args(use_bf16=CFG.bf16, warmup_steps=warmup_steps)
    if max_steps is not None:
        training_args.max_steps = max_steps
        training_args.num_train_epochs = 1.0  # safety clamp for short runs

    # Align padded length to NVIDIA tensor-core boundaries (8)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=CFG.pad_to_multiple_of,
    )
    logger.info(f"DataCollatorWithPadding initialized (pad_to_multiple_of={CFG.pad_to_multiple_of}).")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,     # v5: replaces deprecated `tokenizer=`
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    logger.info(
        "Trainer initialized - EarlyStopping(patience=%d, metric='%s').",
        CFG.early_stopping_patience,
        CFG.metric_for_best_model,
    )
    return trainer


# ---------------------------------------------------------------------------
# Full evaluation helper
# ---------------------------------------------------------------------------

def full_evaluate(
    trainer: Trainer,
    test_dataset: Dataset,
    prefix: str = "test",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on *test_dataset* and return metrics, predictions, probabilities, and labels.

    Args:
        trainer      : Configured Trainer.
        test_dataset : Tokenized test dataset.
        prefix       : Metric key prefix.

    Returns:
        Tuple of (metrics, preds, probs, labels).
    """
    logger.info("Running prediction on test set...")
    prediction_output = trainer.predict(test_dataset, metric_key_prefix=prefix)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        f"{prefix}_accuracy":  round(float(acc), 6),
        f"{prefix}_precision": round(float(precision), 6),
        f"{prefix}_recall":    round(float(recall), 6),
        f"{prefix}_f1":        round(float(f1), 6),
        f"{prefix}_roc_auc":   round(float(roc_auc), 6),
    }

    report_str = classification_report(
        labels, preds, target_names=["Human Written", "AI Generated"]
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    return metrics, preds, probs, labels
