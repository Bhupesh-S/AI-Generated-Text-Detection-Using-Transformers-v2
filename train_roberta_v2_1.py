"""
train_roberta_v2_1.py
=====================
Fine-tunes roberta-base on Dataset V2.1 splits (train_v2_1.csv / val_v2_1.csv / test_v2_1.csv).

Optimized for RTX 3050 (4GB VRAM):
- max_length: 512
- micro_batch_size: 4
- gradient_accumulation_steps: 4 (effective batch size: 16)
- FP16 automatic mixed precision
- Gradient checkpointing enabled
- Live terminal progress logging via tqdm & TrainerCallback
- Checkpoint saving & resume support for V2.1 runs
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
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

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
OUTPUT_DIR = ROOT / "outputs" / "retraining_v2_1" / "roberta"
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# GPU Monitoring Helper
def print_gpu_info():
    print("=" * 65)
    print("  GPU MONITORING & SYSTEM HARDWARE INFO")
    print("=" * 65)
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available     : {cuda_avail}")
    if cuda_avail:
        print(f"  Device Name        : {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch Version    : {torch.__version__}")
        print(f"  CUDA Version       : {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        alloc_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        res_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"  Total VRAM         : {total_vram:.2f} GB")
        print(f"  Allocated VRAM     : {alloc_vram:.2f} GB")
        print(f"  Reserved VRAM      : {res_vram:.2f} GB")
    else:
        print("  WARNING: CUDA not detected! Running on CPU.")
    print("=" * 65 + "\n")

class LiveProgressCallback(TrainerCallback):
    """Callback to print clear per-epoch progress directly to stdout."""
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.epoch_start_time = None
        self.start_time = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        current_epoch = int(state.epoch) + 1 if state.epoch else 1
        print(f"\n------------------------------------------------------------")
        print(f"Epoch {current_epoch} / {self.num_epochs}")
        print(f"------------------------------------------------------------")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        elapsed = time.time() - (self.epoch_start_time or time.time())
        total_elapsed = time.time() - self.start_time
        print(f"  Training Loss      : {metrics.get('eval_loss', 0.0):.4f}")
        print(f"  Validation Loss    : {metrics.get('eval_loss', 0.0):.4f}")
        print(f"  Validation Acc     : {metrics.get('eval_accuracy', 0.0):.4f}")
        print(f"  Validation Prec    : {metrics.get('eval_precision', 0.0):.4f}")
        print(f"  Validation Rec     : {metrics.get('eval_recall', 0.0):.4f}")
        print(f"  Validation F1      : {metrics.get('eval_f1', 0.0):.4f}")
        print(f"  Validation ROC-AUC : {metrics.get('eval_roc_auc', 0.0):.4f}")
        print(f"  Epoch Duration     : {elapsed:.1f}s | Total Elapsed: {total_elapsed/60:.1f}m")

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
    parser = argparse.ArgumentParser(description="Train RoBERTa on V2.1 Dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume checkpoint path if interrupted")
    args = parser.parse_args()

    set_seed(42)
    print_gpu_info()

    train_csv = SPLIT_DIR / "train_v2_1.csv"
    val_csv = SPLIT_DIR / "val_v2_1.csv"
    test_csv = SPLIT_DIR / "test_v2_1.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Splits not found! Please run python create_v2_1_splits.py first.")

    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)

    print("============================================================")
    print("MODEL: RoBERTa-base")
    print("DATASET: final_dataset_v2_1.csv (Unified 80/10/10 Splits)")
    print("============================================================")
    print(f"Train samples       : {len(train_df):,} (Human: {(train_df['label']==0).sum():,}, AI: {(train_df['label']==1).sum():,})")
    print(f"Validation samples  : {len(val_df):,} (Human: {(val_df['label']==0).sum():,}, AI: {(val_df['label']==1).sum():,})")
    print(f"Test samples        : {len(test_df):,} (Human: {(test_df['label']==0).sum():,}, AI: {(test_df['label']==1).sum():,})")
    print()
    print(f"Max length          : 512")
    print(f"Micro batch size    : {args.batch_size}")
    print(f"Gradient accum      : 4 (Effective batch size: {args.batch_size * 4})")
    print(f"Learning rate       : {args.lr}")
    print(f"Precision           : FP16 (Automatic Mixed Precision)")
    print(f"Epochs              : {args.epochs}")
    print("============================================================\n")

    # Load Tokenizer
    model_name = "roberta-base"
    print(f"Loading tokenizer & pretrained model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=512)

    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })

    print("Tokenizing train/validation/test datasets...")
    tokenized = raw_datasets.map(tokenize, batched=True, num_proc=4)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused_optim = "adamw_torch_fused" if device == "cuda" else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        fp16=(device == "cuda"),
        gradient_checkpointing=True,
        optim=fused_optim,
        label_smoothing_factor=0.1,
        dataloader_pin_memory=True,
        report_to="none",
        disable_tqdm=False
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            LiveProgressCallback(num_epochs=args.epochs)
        ]
    )

    print("\nStarting RoBERTa V2.1 Fine-Tuning...\n", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\nSaving best RoBERTa model checkpoint to:", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nEvaluating best model on Validation Set...")
    val_results = trainer.evaluate(eval_dataset=tokenized["validation"])
    print("Validation Set Results:", val_results)

    print("\nEvaluating best model on Locked Test Set...")
    test_results = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
    print("Test Set Results:", test_results)

    # Save metrics JSON
    with open(REPORTS_DIR / "roberta_v2_1_metrics.json", "w") as f:
        json.dump({'val': val_results, 'test': test_results}, f, indent=2)

    print(f"\nRoBERTa Training Complete! Model and metrics saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
