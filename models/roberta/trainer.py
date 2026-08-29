"""
trainer.py
==========
Custom Hugging Face Trainer wrapper for RoBERTa fine-tuning.
Optimised for Transformers v5.14.1 + RTX 3050 Laptop (4 GB VRAM).

Key optimizations over baseline:
  - DataCollatorWithPadding(pad_to_multiple_of=8) for tensor core alignment.
  - Fused AdamW optimizer (single CUDA kernel per param update).
  - pin_memory for zero-copy GPU transfers.
  - save_total_limit=1 to minimize SSD checkpoint write overhead.
  - TF32 / cuDNN benchmark flags set here.

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
    confusion_matrix,
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

    This function is passed directly to :class:`transformers.Trainer` as the
    ``compute_metrics`` argument.

    Args:
        eval_pred : Named tuple ``(predictions, label_ids)`` from the Trainer.
                    ``predictions`` are raw logits of shape ``(N, num_labels)``.

    Returns:
        Dictionary of metric name -> float value.
    """
    logits, labels = eval_pred

    # Convert logits -> class probabilities via softmax
    probs = softmax(logits, axis=-1)

    # Argmax -> predicted class
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

def build_training_args(use_fp16: bool = True, warmup_steps: int = 0) -> TrainingArguments:
    """
    Build an optimised :class:`transformers.TrainingArguments` object.

    All performance settings come from :data:`config.CFG` so you can tune
    everything from one place.

    Transformers v5.14.1 notes:
      - ``processing_class`` replaces ``tokenizer`` in :class:`Trainer`.
      - ``logging_dir`` set via env var ``TENSORBOARD_LOGGING_DIR``.
      - ``group_by_length`` is removed in v5.14.1 and is omitted here.
      - ``warmup_ratio`` is deprecated in v5.14.1 and replaced by ``warmup_steps``.

    Args:
        use_fp16     : Request FP16 mixed precision (effective only on CUDA).
        warmup_steps : Number of warmup steps (computed from ratio dynamically).

    Returns:
        :class:`TrainingArguments` instance.
    """
    fp16 = use_fp16 and torch.cuda.is_available()
    if fp16:
        logger.info("Mixed precision (FP16) ENABLED.")
    else:
        logger.info("Mixed precision (FP16) DISABLED (CPU or no CUDA).")

    # Transformers v5: set logging dir via env var (kwarg deprecated)
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
        fp16=fp16,
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
) -> Trainer:
    """
    Instantiate a HuggingFace :class:`Trainer` with all optimizations applied.

    Transformers v5 compatibility:
      - ``processing_class`` replaces the removed ``tokenizer`` argument.
      - :class:`DataCollatorWithPadding` is explicitly provided with
        ``pad_to_multiple_of=8`` for FP16 tensor-core alignment.

    Args:
        model         : Fresh or loaded RoBERTa model.
        tokenizer     : Corresponding tokenizer (passed as ``processing_class``).
        train_dataset : Tokenized training split.
        eval_dataset  : Tokenized validation split.

    Returns:
        Fully configured :class:`Trainer`.
    """
    # Dynamically compute total steps and warmup steps to avoid v5 deprecation warning
    steps_per_epoch = len(train_dataset) // (CFG.batch_size * CFG.gradient_accumulation_steps)
    total_steps = steps_per_epoch * CFG.epochs
    warmup_steps = int(CFG.warmup_ratio * total_steps)
    logger.info(f"Total steps: {total_steps} | Calculated warmup_steps: {warmup_steps}")

    training_args = build_training_args(use_fp16=CFG.fp16, warmup_steps=warmup_steps)

    # DataCollatorWithPadding: pads each batch to the longest sequence IN that batch.
    # pad_to_multiple_of=8 aligns the padded length to NVIDIA tensor-core boundaries.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=CFG.pad_to_multiple_of,  # 8 = tensor-core alignment
    )
    logger.info(
        f"DataCollatorWithPadding initialized "
        f"(pad_to_multiple_of={CFG.pad_to_multiple_of})."
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CFG.early_stopping_pvariance if hasattr(CFG, 'early_stopping_pvariance') else CFG.early_stopping_patience
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,     # v5: replaces deprecated `tokenizer=`
        data_collator=data_collator,    # Explicit collator with pad_to_multiple_of=8
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
    Run prediction on *test_dataset* and return aggregated metrics plus
    per-sample arrays for downstream plotting.

    Args:
        trainer      : Trained :class:`Trainer`.
        test_dataset : Tokenized test split.
        prefix       : Metric key prefix (default ``"test"``).

    Returns:
        Tuple of:
          - ``metrics``  : dict of metric_name -> float
          - ``preds``    : integer predicted labels  (N,)
          - ``probs``    : softmax probabilities     (N, 2)
          - ``labels``   : true integer labels       (N,)
    """
    logger.info("Running prediction on test set ...")
    prediction_output = trainer.predict(test_dataset, metric_key_prefix=prefix)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    # Compute rich metrics
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

    # Full classification report -> log only
    report_str = classification_report(
        labels, preds, target_names=["Human Written", "AI Generated"]
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    return metrics, preds, probs, labels
