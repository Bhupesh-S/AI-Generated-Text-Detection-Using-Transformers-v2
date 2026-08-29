"""
================================================================================
Kaggle Notebook: DistilBERT V2.1 Multi-GPU PyTorch DDP Fine-Tuning
================================================================================
Target Accelerator: Kaggle NVIDIA T4 x2 (Dual GPU DistributedDataParallel FP16)
Input Dataset Path: /kaggle/input/... (Auto-detected or configurable)
Output Directory  : /kaggle/working/outputs/retraining_v2_1/distilbert
================================================================================
"""

import os
import sys
import time
import json
import gc
import logging
import platform
import inspect
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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

# ==============================================================================
# SECTION 1: GLOBAL CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "dataset_input_dir": "/kaggle/input",
    "dataset_folder_name": "v2-1-splits",
    "output_root": "/kaggle/working/outputs/retraining_v2_1/distilbert",
    "max_length": 512,
    "per_gpu_batch_size": 16,            # Micro-batch per GPU (Total global batch size = 32 on T4 x2)
    "gradient_accumulation_steps": 1,
    "learning_rate": 2.0e-5,
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "epochs": 3,
    "save_steps": 500,
    "save_total_limit": 3,
    "early_stopping_patience": 2,
    "seed": 42
}

# Path Setup
BASE_OUT_DIR = Path(CONFIG["output_root"])
CKPT_DIR = BASE_OUT_DIR / "checkpoints"
LOGS_DIR = BASE_OUT_DIR / "logs"
TOKENIZER_DIR = BASE_OUT_DIR / "tokenizer"
METRICS_DIR = BASE_OUT_DIR / "metrics"
FINAL_DIR = BASE_OUT_DIR / "final"


# ==============================================================================
# SECTION 2: DATASET LOCATOR & VERIFICATION
# ==============================================================================
def locate_dataset():
    input_root = Path(CONFIG["dataset_input_dir"])
    possible_paths = [
        input_root / CONFIG["dataset_folder_name"],
        input_root / "ai-text-detection-v2-1",
        input_root / "dataset-v2-1",
        Path("/kaggle/working/outputs/v2_1_splits"),
        Path("./outputs/v2_1_splits")
    ]

    found_dir = None
    for p in possible_paths:
        if (p / "train_v2_1.csv").exists():
            found_dir = p
            break

    if not found_dir and input_root.exists():
        for item in input_root.glob("**/train_v2_1.csv"):
            found_dir = item.parent
            break

    if not found_dir or not (found_dir / "train_v2_1.csv").exists():
        raise FileNotFoundError("Could not locate 'train_v2_1.csv'! Please check Kaggle 'Add Input' dataset attached to notebook.")

    return found_dir


# ==============================================================================
# SECTION 3: METRIC COMPUTATION & CALLBACKS
# ==============================================================================
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

class CheckpointLoggingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            print(f"\n[RANK 0 CHECKPOINT SAVED] Step {state.global_step} -> {ckpt_dir.as_posix()}", flush=True)

class ActiveTimeProgressCallback(TrainerCallback):
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.active_train_time = 0.0
        self.last_step_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.time()
        if self.last_step_time is not None:
            delta = now - self.last_step_time
            if delta < 30.0:
                self.active_train_time += delta
        self.last_step_time = now

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_world_process_zero:
            loss = metrics.get('eval_loss', metrics.get('test_loss', 0.0))
            acc = metrics.get('eval_accuracy', metrics.get('test_accuracy', 0.0))
            f1 = metrics.get('eval_f1', metrics.get('test_f1', 0.0))
            roc_auc = metrics.get('eval_roc_auc', metrics.get('test_roc_auc', 0.0))
            prefix = "Test" if any(k.startswith("test_") for k in metrics.keys()) else "Validation"
            print(f"[{prefix} Metrics] Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            if state.best_metric is not None:
                print(f"Best Validation ROC-AUC: {state.best_metric:.4f}")
            print(f"Total Active Training Time: {self.active_train_time / 60:.1f} minutes\n")


def find_latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    valid_ckpts = []
    for item in ckpt_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            if (item / "trainer_state.json").exists():
                try:
                    step_num = int(item.name.split("-")[-1])
                    valid_ckpts.append((step_num, item))
                except ValueError:
                    pass
    if not valid_ckpts:
        return None
    valid_ckpts.sort(key=lambda x: x[0])
    return valid_ckpts[-1][1]


# ==============================================================================
# SECTION 4: DDP WORKER PROCESS INITIALIZATION & TRAINING
# ==============================================================================
def ddp_worker(rank, world_size, split_dir_path):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    is_rank_zero = (rank == 0)
    set_seed(CONFIG["seed"] + rank)

    if is_rank_zero:
        print("=" * 65)
        print("  DISTRIBUTED DATA PARALLEL (DDP) TRAINING INITIALIZED")
        print("=" * 65)
        print(f"  GPU Count              : {world_size}")
        for i in range(world_size):
            print(f"  GPU {i}                  : {torch.cuda.get_device_name(i)}")
        print(f"  Distributed Training   : ENABLED (Backend: NCCL)")
        print(f"  Per-GPU Batch Size     : {CONFIG['per_gpu_batch_size']}")
        print(f"  Global Effective Batch : {CONFIG['per_gpu_batch_size'] * world_size * CONFIG['gradient_accumulation_steps']}")
        print(f"  DistilBERT Precision   : FP16 (Automatic Mixed Precision enabled)")
        print("=" * 65 + "\n")

    split_dir = Path(split_dir_path)
    train_df = pd.read_csv(split_dir / "train_v2_1.csv", low_memory=False)
    val_df   = pd.read_csv(split_dir / "val_v2_1.csv", low_memory=False)
    test_df  = pd.read_csv(split_dir / "test_v2_1.csv", low_memory=False)

    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df[['text', 'label']]),
        'validation': Dataset.from_pandas(val_df[['text', 'label']]),
        'test': Dataset.from_pandas(test_df[['text', 'label']])
    })
    del train_df, val_df, test_df
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def tokenize_function(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=CONFIG["max_length"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=max(1, (os.cpu_count() or 2) // max(1, world_size)),
        remove_columns=['text']
    )
    del raw_datasets
    gc.collect()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=2
    )

    resume_path = find_latest_checkpoint(CKPT_DIR)
    if is_rank_zero:
        if resume_path:
            print(f"RESUME DETECTED: Found latest checkpoint at {resume_path}")
        else:
            print("No previous checkpoint detected. Starting fresh fine-tuning run.")

    is_bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not is_bf16_ok

    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs = {
        "output_dir": str(CKPT_DIR),
        "save_strategy": "steps",
        "save_steps": CONFIG["save_steps"],
        "save_total_limit": CONFIG["save_total_limit"],
        "learning_rate": CONFIG["learning_rate"],
        "per_device_train_batch_size": CONFIG["per_gpu_batch_size"],
        "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
        "per_device_eval_batch_size": CONFIG["per_gpu_batch_size"] * 2,
        "num_train_epochs": CONFIG["epochs"],
        "weight_decay": CONFIG["weight_decay"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "max_grad_norm": 1.0,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "fp16": use_fp16,
        "bf16": is_bf16_ok,
        "gradient_checkpointing": False,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "dataloader_pin_memory": True,
        "dataloader_num_workers": max(1, (os.cpu_count() or 2) // max(1, world_size)),
        "report_to": "none",
        "disable_tqdm": not is_rank_zero
    }

    if "eval_strategy" in sig_params:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = CONFIG["save_steps"]
    elif "evaluation_strategy" in sig_params:
        training_args_kwargs["evaluation_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = CONFIG["save_steps"]

    if "warmup_ratio" in sig_params:
        training_args_kwargs["warmup_ratio"] = CONFIG["warmup_ratio"]
    elif "warmup_steps" in sig_params:
        eff_bs = CONFIG["per_gpu_batch_size"] * CONFIG["gradient_accumulation_steps"] * max(1, world_size)
        total_steps = (len(tokenized_datasets["train"]) // max(1, eff_bs)) * CONFIG["epochs"]
        training_args_kwargs["warmup_steps"] = int(total_steps * CONFIG["warmup_ratio"])

    if "ddp_find_unused_parameters" in sig_params:
        training_args_kwargs["ddp_find_unused_parameters"] = False
    if "local_rank" in sig_params and world_size > 1:
        training_args_kwargs["local_rank"] = rank

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    if not has_var_kw:
        training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in sig_params}

    training_args = TrainingArguments(**training_args_kwargs)
    if world_size > 1:
        training_args.local_rank = rank
        training_args._n_gpu = 1

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"]),
            ActiveTimeProgressCallback(num_epochs=CONFIG["epochs"]),
            CheckpointLoggingCallback()
        ]
    )

    try:
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    finally:
        if is_rank_zero:
            print(f"\nSaving Best DistilBERT Model to {FINAL_DIR}...")
            trainer.save_model(str(FINAL_DIR))
            tokenizer.save_pretrained(str(TOKENIZER_DIR))
            print("\nEvaluating Best Model on Validation Set...")

        val_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])

        if is_rank_zero:
            print("Validation Results:", val_results)
            print("\nEvaluating Best Model on Locked Test Set...")

        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")

        if is_rank_zero:
            print("Test Results:", test_results)
            with open(METRICS_DIR / "distilbert_v2_1_metrics.json", "w") as f:
                json.dump({'val': val_results, 'test': test_results}, f, indent=2)

            print(f"\n[COMPLETE] DistilBERT Multi-GPU Fine-Tuning Finished! Saved under {BASE_OUT_DIR}")

        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


# ==============================================================================
# SECTION 5: MASTER ENTRYPOINT FOR KAGGLE NOTEBOOK
# ==============================================================================
def is_notebook_environment():
    """Detect if executing inside an interactive Jupyter/Kaggle notebook kernel."""
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    return False

def save_notebook_cell_to_file(target_path):
    """Save the currently executing Jupyter cell code to target_path on disk."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            cell_code = ip.user_ns.get('_i', None)
            if not cell_code and hasattr(ip, 'history_manager') and ip.history_manager.input_hist_raw:
                cell_code = ip.history_manager.input_hist_raw[-1]
            if cell_code and len(cell_code.strip()) > 0:
                Path(target_path).write_text(cell_code, encoding="utf-8")
                print(f"Auto-saved notebook cell code to {target_path}")
                return True
    except Exception as e:
        print(f"Warning: Could not auto-save notebook cell code: {e}")
    return False

def main():
    for d in [CKPT_DIR, LOGS_DIR, TOKENIZER_DIR, METRICS_DIR, FINAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if world_size > 1 and is_notebook_environment():
        script_name = "kaggle_ddp_distilbert_v2_1.py"
        target_file = Path(script_name)

        # Always prioritize saving the latest notebook cell code to disk
        if save_notebook_cell_to_file(target_file):
            script_path = target_file
        else:
            candidates = [
                Path(script_name),
                Path(f"outputs/kaggle_notebooks/{script_name}"),
                Path(f"/kaggle/working/{script_name}")
            ]
            script_path = None
            for cand in candidates:
                if cand.exists():
                    script_path = cand
                    break
            if script_path is None:
                raise FileNotFoundError(
                    f"Could not locate or auto-create '{script_name}' on disk. "
                    f"Please save the code to '{script_name}' in your Kaggle working directory."
                )

        print(f"Notebook kernel detected. Delegating PyTorch DDP execution to standalone script ({script_path})...")
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"DDP Multi-GPU Training process failed with exit code {result.returncode}")
        return

    split_dir = locate_dataset()

    if world_size > 1:
        print(f"Launching PyTorch DDP across {world_size} GPUs...")
        mp.spawn(ddp_worker, args=(world_size, str(split_dir)), nprocs=world_size, join=True)
    else:
        print("Running in Single-GPU / CPU mode...")
        ddp_worker(0, 1, str(split_dir))

if __name__ == "__main__":
    main()
