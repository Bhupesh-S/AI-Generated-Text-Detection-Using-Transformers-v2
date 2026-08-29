"""
config.py
=========
Central configuration for the DeBERTa Fine-Tuning Pipeline.
Optimised for Transformers v5.14.1 + PyTorch CUDA + Windows 11.

Optimized Settings for RTX 3050 Laptop (4 GB VRAM):
  - batch_size set to 4 with gradient_accumulation_steps set to 4
    (effective batch size = 16, matching baseline and convergence).
    This keeps peak PyTorch VRAM at ~2.8 GB, avoiding Windows graphics driver paging.
  - bf16 set to True to resolve the DeBERTa-v3 relative positional embedding
    gradient scaling runtime crash ('unscale FP16 gradients' error).
  - use_safetensors set to True to bypass vulnerable torch.load version restrictions.
  - dataloader_num_workers set to 0 to prevent spawn-context CUDA deadlocks on Windows.
  - TF32 enabled for faster matrix multiplication on Ampere GPU.
  - Fused AdamW optimizer enabled for single CUDA kernel execution per update.
  - save_total_limit set to 1 to reduce SSD write overhead.
  - logging_steps set to 200 to minimize CPU-GPU synchronizations.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base project root (two levels up from models/deberta/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DeBERTaConfig:
    """All configuration parameters for the DeBERTa training pipeline."""

    # -- Model ----------------------------------------------------------------
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 2
    label2id: dict = field(default_factory=lambda: {"Human Written": 0, "AI Generated": 1})
    id2label: dict = field(default_factory=lambda: {0: "Human Written", 1: "AI Generated"})
    use_safetensors: bool = True           # Safetensors bypasses torch.load CVE guard

    # -- Data -----------------------------------------------------------------
    data_path: Path = PROJECT_ROOT / "Datasets" / "merged" / "final_dataset.csv"
    text_column: str = "text"
    label_column: str = "label"

    # -- Splits ---------------------------------------------------------------
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42

    # -- Tokenization ---------------------------------------------------------
    max_length: int = 512
    padding: bool = False
    truncation: bool = True
    pad_to_multiple_of: int = 8   # Aligns tensors to GPU tensor-core width.

    # -- Training -------------------------------------------------------------
    epochs: int = 3
    # OPTIMIZATION: batch_size = 4, gradient_accumulation_steps = 4
    # Fits within the 4 GB VRAM limit safely at ~2.8 GB with gradient checkpointing.
    batch_size: int = 4
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4   # Micro-batch size 4 * 4 = 16 effective batch size
    max_grad_norm: float = 1.0

    # DeBERTa-v3 relative embedding bug workaround:
    # Use Ampere-native BF16 instead of FP16 to resolve gradient unscaling failure.
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True    # VRAM optimization
    label_smoothing_factor: float = 0.1    # Calibration improvement

    # -- DataLoader Pipeline Optimizations -------------------------------------
    dataloader_num_workers: int = 0         # Prevents Windows subprocess context locks
    dataloader_pin_memory: bool = True      # DMA transfer acceleration
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None

    # -- Optimizer Optimization ------------------------------------------------
    optim: str = "adamw_torch_fused"

    # -- CUDA Backend Optimizations --------------------------------------------
    tf32: bool = True
    cudnn_benchmark: bool = False

    # -- Evaluation & Saving ---------------------------------------------------
    save_total_limit: int = 1
    logging_steps: int = 200

    # -- Early Stopping -------------------------------------------------------
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True

    # -- Logging & Evaluation -------------------------------------------------
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    report_to: str = "none"

    # -- Output directories ---------------------------------------------------
    output_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "final"
    tokenizer_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "final"
    predictions_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "predictions"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "reports"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "figures"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "logs"

    def create_dirs(self) -> None:
        """Create all output directories if they don't already exist."""
        dirs = [
            self.output_dir,
            self.checkpoint_dir,
            self.tokenizer_dir,
            self.predictions_dir,
            self.reports_dir,
            self.figures_dir,
            self.logs_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary (for JSON/report saving)."""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "data_path": str(self.data_path),
            "text_column": self.text_column,
            "label_column": self.label_column,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "optim": self.optim,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_persistent_workers": self.dataloader_persistent_workers,
            "tf32": self.tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "save_total_limit": self.save_total_limit,
            "early_stopping_patience": self.early_stopping_patience,
            "metric_for_best_model": self.metric_for_best_model,
            "output_dir": str(self.output_dir),
        }


# ---------------------------------------------------------------------------
# Singleton instance - import this everywhere
# ---------------------------------------------------------------------------
CFG = DeBERTaConfig()
