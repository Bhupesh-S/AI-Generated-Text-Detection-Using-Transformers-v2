"""
model.py
========
RoBERTa model loading and configuration for binary text classification.

Responsibilities:
  - Load ``roberta-base`` via ``AutoModelForSequenceClassification``.
  - Attach label ↔ id mappings.
  - Provide a lightweight parameter-count summary.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel

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
# Model Factory
# ---------------------------------------------------------------------------

def load_model(
    model_name_or_path: Optional[str] = None,
    num_labels: int = 2,
) -> PreTrainedModel:
    """
    Load (or fine-tuned checkpoint of) ``roberta-base`` for binary
    sequence classification.

    Args:
        model_name_or_path : Hugging Face model ID or local checkpoint path.
                             Defaults to ``CFG.model_name`` (``"roberta-base"``).
        num_labels         : Number of output classes (default 2).

    Returns:
        :class:`transformers.RobertaForSequenceClassification` on the best
        available device.
    """
    if model_name_or_path is None:
        model_name_or_path = CFG.model_name

    logger.info(f"Loading model: {model_name_or_path}  (num_labels={num_labels})")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=CFG.id2label,
        label2id=CFG.label2id,
        ignore_mismatched_sizes=True,   # Safe when loading from a checkpoint
    )

    _log_model_summary(model)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_model_summary(model: PreTrainedModel) -> None:
    """Log the total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Model loaded -> "
        f"Total params: {total_params:,}  |  "
        f"Trainable params: {trainable_params:,}  "
        f"({100 * trainable_params / total_params:.1f}%)"
    )


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Estimate model size in megabytes (based on float32 parameter count).

    Args:
        model : Loaded PyTorch model.

    Returns:
        Approximate size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)   # 4 bytes per float32
    return size_mb
