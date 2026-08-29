"""
train_distilbert_v2_1.py
========================
Fine-tunes distilbert-base-uncased on Dataset V2.1 splits (train_v2_1.csv / val_v2_1.csv / test_v2_1.csv).

Optimized for Google Colab & Local Environments:
- max_length: 512 with dynamic batch padding
- micro_batch_size: 4 (configurable via --batch_size)
- gradient_accumulation_steps: 4 (effective batch size: 16)
- precision: BF16 (if supported) / FP16 (CUDA fallback) / FP32 (CPU)
- warmup_ratio: 0.10, weight_decay: 0.01, learning_rate: 1.5e-5
- early stopping: metric_for_best_model="f1", greater_is_better=True, patience=2
- active GPU training time tracking
- robust step-based checkpointing with auto-resume (--auto_resume) and safe cleanup
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
import platform
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
BASE_OUT_DIR = ROOT / "outputs" / "retraining_v2_1" / "distilbert"
CKPT_DIR = BASE_OUT_DIR / "checkpoints"
LOGS_DIR = BASE_OUT_DIR / "logs"
TOKENIZER_DIR = BASE_OUT_DIR / "tokenizer"
METRICS_DIR = BASE_OUT_DIR / "metrics"
FINAL_DIR = BASE_OUT_DIR / "final"

for d in [CKPT_DIR, LOGS_DIR, TOKENIZER_DIR, METRICS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def validate_checkpoint(ckpt_path: Path) -> bool:
    """Validates if a checkpoint directory contains required state files."""
    if not ckpt_path.is_dir():
        return False
    trainer_state_file = ckpt_path / "trainer_state.json"
    if not trainer_state_file.exists():
        return False
    # Check for model weights (safetensors or bin)
    has_weights = (ckpt_path / "model.safetensors").exists() or \
                  (ckpt_path / "pytorch_model.bin").exists() or \
                  (ckpt_path / "model.safetensors.index.json").exists()
    return has_weights


def find_latest_checkpoint(ckpt_dir: Path):
    """Detects all valid checkpoint-* directories in ckpt_dir and returns the one with the highest training step."""
    if not ckpt_dir.exists():
        return None
    valid_ckpts = []
    for item in ckpt_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            if validate_checkpoint(item):
                try:
                    step_num = int(item.name.split("-")[-1])
                    valid_ckpts.append((step_num, item))
                except ValueError:
                    pass
    if not valid_ckpts:
        return None
    valid_ckpts.sort(key=lambda x: x[0])
    return valid_ckpts[-1][1]


def print_hardware_summary(batch_size, grad_accum, lr, warmup_ratio, epochs):
    print("=" * 60)
    print("MODEL: DistilBERT-base-uncased")
    print("DATASET: final_dataset_v2_1.csv")
    print("=" * 60)

    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    torch_ver = torch.__version__
    tf_ver = transformers.__version__
    py_ver = platform.python_version()
    bf16_ok = torch.cuda.is_bf16_supported() if cuda_avail else False

    sys_ram_str = "Unknown"
    if psutil is not None:
        sys_ram_str = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"

    gpu_vram_str = "N/A"
    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        gpu_vram_str = f"{props.total_memory / (1024**3):.2f} GB"

    print(f"Python version      : {py_ver}")
    print(f"PyTorch version     : {torch_ver}")
    print(f"Transformers        : {tf_ver}")
    print(f"CUDA version        : {cuda_ver}")
    print(f"GPU                 : {device_name}")
    print(f"GPU VRAM            : {gpu_vram_str}")
    print(f"System RAM          : {sys_ram_str}")
    print(f"BF16 supported      : {bf16_ok}")
    print("=" * 60 + "\n")


def print_checkpoint_configuration(save_steps, save_total_limit, auto_resume, explicit_ckpt):
    auto_str = "ENABLED" if auto_resume else "DISABLED"
    explicit_str = explicit_ckpt if explicit_ckpt else "NONE"
    print("=" * 60)
    print("CHECKPOINT CONFIGURATION")
    print("=" * 60)
    print(f"Save strategy       : steps")
    print(f"Save every          : {save_steps} steps")
    print(f"Maximum checkpoints : {save_total_limit}")
    print(f"Auto resume         : {auto_str}")
    print(f"Explicit checkpoint : {explicit_str}")
    print("=" * 60 + "\n")


class CheckpointLoggingCallback(TrainerCallback):
    """Prints clear checkpoint save notifications in the console."""
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        print("\n" + "=" * 60)
        print("CHECKPOINT SAVED")
        print("=" * 60)
        print(f"Model        : DistilBERT")
        print(f"Training step: {state.global_step}")
        print(f"Checkpoint   : {ckpt_dir.as_posix()}")
        print("=" * 60 + "\n", flush=True)


class ActiveTimeProgressCallback(TrainerCallback):
    """Tracks active GPU step time excluding sleep/hibernation pauses."""
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.active_train_time = 0.0
        self.last_step_time = None
        self.epoch_start_active_time = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.time()
        if self.last_step_time is not None:
            delta = now - self.last_step_time
            if delta < 30.0:
                self.active_train_time += delta
        self.last_step_time = now

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_active_time = self.active_train_time
        current_epoch = int(state.epoch) + 1 if state.epoch else 1
        print(f"\n------------------------------------------------------------")
        print(f"Epoch {current_epoch} / {self.num_epochs}")
        print(f"------------------------------------------------------------")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch_active = self.active_train_time - self.epoch_start_active_time
        loss = metrics.get('eval_loss', metrics.get('test_loss', 0.0))
        acc = metrics.get('eval_accuracy', metrics.get('test_accuracy', 0.0))
        prec = metrics.get('eval_precision', metrics.get('test_precision', 0.0))
        rec = metrics.get('eval_recall', metrics.get('test_recall', 0.0))
        f1 = metrics.get('eval_f1', metrics.get('test_f1', 0.0))
        roc_auc = metrics.get('eval_roc_auc', metrics.get('test_roc_auc', 0.0))
        prefix = "Test" if any(k.startswith("test_") for k in metrics.keys()) else "Validation"
        print(f"{prefix} Loss:        {loss:.4f}")
        print(f"{prefix} Accuracy:    {acc:.4f}")
        print(f"{prefix} Precision:   {prec:.4f}")
        print(f"{prefix} Recall:      {rec:.4f}")
        print(f"{prefix} F1:          {f1:.4f}")
        print(f"{prefix} ROC-AUC:     {roc_auc:.4f}")
        if state.best_metric is not None:
            print(f"Best Validation F1: {state.best_metric:.4f}")
        print(f"Active Epoch Time:   {epoch_active:.1f}s | Total Active GPU Time: {self.active_train_time/60:.1f}m")
        print(f"------------------------------------------------------------")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try:
        roc_auc = float(roc_auc_score(labels, probs[:, 1]))
    except Exception:
        roc_auc = 0.5
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT on V2.1 Dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.10, help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to retain")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest valid checkpoint")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint path if interrupted")
    args = parser.parse_args()

    set_seed(42)
    print_hardware_summary(args.batch_size, args.gradient_accumulation_steps, args.lr, args.warmup_ratio, args.epochs)
    print_checkpoint_configuration(args.save_steps, args.save_total_limit, args.auto_resume, args.resume_from_checkpoint)

    # Determine resume path with explicit precedence
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if args.auto_resume:
            print("============================================================")
            print("EXPLICIT CHECKPOINT SUPPLIED")
            print("Ignoring automatic checkpoint selection.")
            print(f"Explicit checkpoint provided: {resume_path}")
            print("============================================================\n")
        else:
            print("============================================================")
            print("EXPLICIT CHECKPOINT RESUME")
            print(f"Resuming from: {resume_path}")
            print("============================================================\n")
    elif args.auto_resume:
        latest_ckpt = find_latest_checkpoint(CKPT_DIR)
        print("============================================================")
        print("AUTO-RESUME ENABLED")
        print("============================================================")
        if latest_ckpt:
            resume_path = str(latest_ckpt)
            print(f"\nLatest valid checkpoint:\n{latest_ckpt.as_posix()}\n")
            print(f"Resuming DistilBERT training from {latest_ckpt.name}")
            print("============================================================\n")
        else:
            print("\nNo valid checkpoint found.\n")
            print("Starting fresh DistilBERT training.")
            print("============================================================\n")

    train_csv = SPLIT_DIR / "train_v2_1.csv"
    val_csv   = SPLIT_DIR / "val_v2_1.csv"
    test_csv  = SPLIT_DIR / "test_v2_1.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Splits not found under {SPLIT_DIR}! Please run create_v2_1_splits.py first.")

    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df   = pd.read_csv(val_csv, low_memory=False)
    test_df  = pd.read_csv(test_csv, low_memory=False)

    print(f"Train samples       : {len(train_df):,} (Human: {(train_df['label']==0).sum():,}, AI: {(train_df['label']==1).sum():,})")
    print(f"Validation samples  : {len(val_df):,} (Human: {(val_df['label']==0).sum():,}, AI: {(val_df['label']==1).sum():,})")
    print(f"Test samples        : {len(test_df):,} (Human: {(test_df['label']==0).sum():,}, AI: {(test_df['label']==1).sum():,} - LOCKED)")
    print()

    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=512)

    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })

    print("Tokenizing train/validation/test datasets with dynamic padding...")
    tokenized = raw_datasets.map(tokenize, batched=True, num_proc=4)

    print(f"Loading pretrained model '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused_optim = "adamw_torch_fused" if device == "cuda" else "adamw_torch"
    is_bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not is_bf16_ok

    # Dynamically resolve TrainingArguments compatibility across Transformers versions
    import inspect
    sig_params = inspect.signature(TrainingArguments.__init__).parameters

    training_args_kwargs = dict(
        output_dir=str(CKPT_DIR),
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=100,
        fp16=use_fp16,
        bf16=is_bf16_ok,
        gradient_checkpointing=False,
        optim=fused_optim,
        label_smoothing_factor=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        report_to="none",
        disable_tqdm=False
    )

    # Handle eval_strategy vs evaluation_strategy compatibility
    if "eval_strategy" in sig_params:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = args.save_steps
    elif "evaluation_strategy" in sig_params:
        training_args_kwargs["evaluation_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = args.save_steps

    # Handle warmup_ratio vs warmup_steps compatibility (Transformers v5 / v4)
    if "warmup_ratio" in sig_params:
        training_args_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        total_steps = (len(tokenized["train"]) // max(1, effective_batch_size)) * args.epochs
        training_args_kwargs["warmup_steps"] = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(**training_args_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer_sig = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ActiveTimeProgressCallback(num_epochs=args.epochs),
            CheckpointLoggingCallback()
        ]
    )
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    try:
        print("\nStarting DistilBERT V2.1 Fine-Tuning...\n", flush=True)
        trainer.train(resume_from_checkpoint=resume_path)
    except KeyboardInterrupt:
        latest_ckpt = find_latest_checkpoint(CKPT_DIR)
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED")
        print("=" * 60)
        print("\nTraining was manually interrupted.\n")
        if latest_ckpt:
            print("Latest valid checkpoint:")
            print(f"{latest_ckpt.as_posix()}\n")
            print("The checkpoint has been preserved on Google Drive.\n")
            print("To resume automatically:")
            print("python train_distilbert_v2_1.py --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --lr 1.5e-5 --warmup_ratio 0.10 --auto_resume")
        else:
            print("No checkpoint is currently available.")
            print("Training progress since startup cannot be resumed.")
        print("=" * 60 + "\n", flush=True)
        sys.exit(0)

    # Finalization Steps
    model_saved = False
    tokenizer_saved = False
    val_completed = False
    test_completed = False
    metrics_saved = False

    try:
        print("\nSaving final best DistilBERT model to:", FINAL_DIR)
        trainer.save_model(str(FINAL_DIR))
        model_saved = True

        print("Saving tokenizer to:", TOKENIZER_DIR)
        tokenizer.save_pretrained(str(TOKENIZER_DIR))
        tokenizer_saved = True

        print("\nEvaluating best model on Validation Set...")
        val_results = trainer.evaluate(eval_dataset=tokenized["validation"])
        print("Validation Set Results:", val_results)
        val_completed = True

        print("\nEvaluating best model on Locked Test Set...")
        test_results = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
        print("Test Set Results:", test_results)
        test_completed = True

        metrics_file = METRICS_DIR / "distilbert_v2_1_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({'val': val_results, 'test': test_results}, f, indent=2)
        metrics_saved = True
    except Exception as e:
        print(f"\nERROR during finalization / evaluation pipeline: {e}")
        print("Temporary checkpoints will be PRESERVED for recovery.")
        raise e

    # Safe Final Checkpoint Cleanup
    cleanup_success = False
    if model_saved and tokenizer_saved and val_completed and test_completed and metrics_saved:
        try:
            print("\nAll pipeline steps succeeded. Removing temporary checkpoints directory...")
            if CKPT_DIR.exists():
                shutil.rmtree(CKPT_DIR)
            cleanup_success = True
        except Exception as cleanup_err:
            print(f"\nWARNING: Training completed successfully, but temporary checkpoint cleanup failed: {cleanup_err}")
            print("The checkpoints have been preserved.")
            print("Final model and metrics are valid.")

    print("\n" + "=" * 60)
    print("DISTILBERT RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Final model       : {'✓' if model_saved else '✗'}")
    print(f"Tokenizer         : {'✓' if tokenizer_saved else '✗'}")
    print(f"Validation        : {'✓' if val_completed else '✗'}")
    print(f"Locked test eval  : {'✓' if test_completed else '✗'}")
    print(f"Metrics JSON      : {'✓' if metrics_saved else '✗'}")
    print(f"Checkpoint cleanup: {'✓' if cleanup_success else '⚠ WARNING'}")
    print()
    if cleanup_success:
        print("Temporary checkpoints removed.")
    print(f"Final outputs preserved in {BASE_OUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

