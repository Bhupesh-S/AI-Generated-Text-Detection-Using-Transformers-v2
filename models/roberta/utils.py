"""
utils.py
========
Shared utility helpers for the RoBERTa fine-tuning pipeline.

Responsibilities:
  - Logger setup
  - Reproducibility seeding
  - JSON / pickle save/load helpers
  - Training-time metrics helpers

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger that writes to both stdout and,
    optionally, a rotating log file.

    Args:
        name     : Logger name (usually __name__ of the calling module).
        log_file : Optional path to a .log file.
        level    : Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed : Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available torch device (CUDA GPU > CPU).

    Returns:
        :class:`torch.device`
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU detected: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU detected - using CPU (training will be slow).")
    return device


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Serialize *data* to JSON at *path*, creating parent dirs as needed.

    Args:
        data : Dictionary to serialize.
        path : Destination file path (.json).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=_json_serializer)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        path : JSON file path.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-serializable types (numpy, Path, etc.)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------

def extract_training_history(log_history: list) -> Dict[str, list]:
    """
    Parse the Hugging Face Trainer's ``log_history`` list into separate
    sequences for easier plotting.

    Args:
        log_history : ``trainer.state.log_history`` list of dicts.

    Returns:
        Dictionary with keys:
          - ``train_loss``   : list of (step, loss) tuples
          - ``eval_loss``    : list of (epoch, loss) tuples
          - ``eval_accuracy``: list of (epoch, accuracy) tuples
          - ``eval_f1``      : list of (epoch, f1) tuples
          - ``eval_roc_auc`` : list of (epoch, roc_auc) tuples
          - ``learning_rate``: list of (step, lr) tuples
    """
    history: Dict[str, list] = {
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_f1": [],
        "eval_roc_auc": [],
        "learning_rate": [],
    }

    for entry in log_history:
        epoch = entry.get("epoch")
        step = entry.get("step")

        # Training step entry
        if "loss" in entry and "eval_loss" not in entry:
            history["train_loss"].append({"step": step, "epoch": epoch, "loss": entry["loss"]})
        if "learning_rate" in entry:
            history["learning_rate"].append({"step": step, "epoch": epoch, "lr": entry["learning_rate"]})

        # Evaluation entry
        if "eval_loss" in entry:
            history["eval_loss"].append({"epoch": epoch, "loss": entry["eval_loss"]})
            if "eval_accuracy" in entry:
                history["eval_accuracy"].append({"epoch": epoch, "accuracy": entry["eval_accuracy"]})
            if "eval_f1" in entry:
                history["eval_f1"].append({"epoch": epoch, "f1": entry["eval_f1"]})
            if "eval_roc_auc" in entry:
                history["eval_roc_auc"].append({"epoch": epoch, "roc_auc": entry["eval_roc_auc"]})

    return history


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """
    Convert elapsed seconds to a human-readable string (H:MM:SS).

    Args:
        seconds : Float number of seconds.

    Returns:
        Formatted string like ``"0:04:33"``.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def print_section(title: str, width: int = 70) -> None:
    """Print a visually separated section header to stdout."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
