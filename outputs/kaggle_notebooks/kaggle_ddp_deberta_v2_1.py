"""
================================================================================
DeBERTa-v3-base V2.1
KAGGLE 2x TESLA T4 DDP TRAINING SUITE (ROBUST LAUNCHER)
================================================================================
Instructions: Paste this ENTIRE block into a single Kaggle Notebook cell and run.
"""

import os
import sys
import subprocess
import py_compile
import torch
from pathlib import Path

# =============================================================================
# EXPLICIT TRAINING SCRIPT DEFINITION
# This guarantees we NEVER execute an older cell from IPython history.
# =============================================================================
TRAINING_SCRIPT = r'''
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import gc
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers

from datasets import Dataset, DatasetDict
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    set_seed,
)

CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "dataset_input_dir": "/kaggle/input",
    "dataset_folder_name": "v2-1-splits",
    "output_root": "/kaggle/working/outputs/retraining_v2_1/deberta",

    # CORE EXPERIMENT PARAMETERS
    "max_length": 512,
    "per_gpu_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "per_gpu_eval_batch_size": 2,
    "learning_rate": 1.5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "max_grad_norm": 1.0,
    "epochs": 3,
    "seed": 42,

    # FRESH RUN CHECKPOINTING
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "save_total_limit": 3,
}

BASE_OUT_DIR = Path(CONFIG["output_root"])
CKPT_DIR = BASE_OUT_DIR / "checkpoints"
LOGS_DIR = BASE_OUT_DIR / "logs"
TOKENIZER_DIR = BASE_OUT_DIR / "tokenizer"
METRICS_DIR = BASE_OUT_DIR / "metrics"
FINAL_DIR = BASE_OUT_DIR / "final"

def assert_model_fp32(model, location):
    bad = [(name, str(p.dtype)) for name, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"\n[FATAL DTYPE ERROR] at {location}.\nExpected torch.float32, found:\n{bad[:10]}")

def assert_model_finite(model, location):
    bad = [name for name, p in model.named_parameters() if p.requires_grad and torch.is_floating_point(p) and not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"\n[FATAL PARAMETER ERROR] at {location}.\nNon-finite parameters in:\n{bad[:10]}")

def inspect_optimizer_state(optimizer):
    bad_states = []
    for _, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all():
                bad_states.append(key)
    return bad_states

class NumericalHealthCallback(TrainerCallback):
    def __init__(self, diagnostic_steps=2000, log_every=100):
        self.diagnostic_steps = diagnostic_steps
        self.log_every = log_every

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        unwrapped = self._unwrap(model)
        assert_model_fp32(unwrapped, "at training start")
        assert_model_finite(unwrapped, "at training start")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        unwrapped = self._unwrap(model)
        assert_model_fp32(unwrapped, f"after optimizer step {state.global_step}")
        assert_model_finite(unwrapped, f"after optimizer step {state.global_step}")

        if optimizer is not None:
            bad_states = inspect_optimizer_state(optimizer)
            if bad_states:
                raise RuntimeError(f"Optimizer state became non-finite at step {state.global_step}: {bad_states[:10]}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    logits, labels = np.asarray(logits), np.asarray(labels)

    if not np.isfinite(logits).all():
        raise RuntimeError("Evaluation logits contain NaN/Inf.")

    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception as e:
        raise RuntimeError(f"Failed to compute ROC-AUC: {e}. Logits or labels may be invalid/single-class.")

    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc)}

def locate_dataset():
    input_root = Path(CONFIG["dataset_input_dir"])
    candidates = [
        input_root / CONFIG["dataset_folder_name"],
        input_root / "ai-text-detection-v2-1",
        input_root / "dataset-v2-1",
        input_root / "dataset-v2-1-splits",
    ]
    for cand in candidates:
        if (cand / "train_v2_1.csv").exists():
            return cand
    for train_file in input_root.glob("**/train_v2_1.csv"):
        return train_file.parent
    raise FileNotFoundError("Could not locate V2.1 dataset. Make sure the dataset is attached in Kaggle.")

def ddp_worker(rank, world_size, dataset_path):
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        rank = int(os.environ.get("RANK", rank))
        world_size = int(os.environ.get("WORLD_SIZE", world_size))
        is_rank_zero = (rank == 0)

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if world_size > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

        set_seed(CONFIG["seed"])

        if is_rank_zero:
            print("\n============================================================")
            print("ENVIRONMENT & INITIALIZATION")
            print("============================================================")
            print(f"Transformers : {transformers.__version__}")
            print(f"PyTorch      : {torch.__version__}")
            print(f"CUDA         : {torch.version.cuda}")
            print(f"Rank         : {rank} (Local: {local_rank})")
            print(f"World Size   : {world_size}")
            print(f"Model Name   : {CONFIG['model_name']}")
            print(f"Max Length   : {CONFIG['max_length']}")
            print(f"Train Batch  : {CONFIG['per_gpu_batch_size']} per GPU")
            print(f"Eval Batch   : {CONFIG['per_gpu_eval_batch_size']} per GPU")
            print(f"Accumulation : {CONFIG['gradient_accumulation_steps']}")
            print(f"Global Batch : {CONFIG['per_gpu_batch_size'] * world_size * CONFIG['gradient_accumulation_steps']}")
            print(f"Learn Rate   : {CONFIG['learning_rate']}")
            print(f"Weight Decay : {CONFIG['weight_decay']}")
            print(f"Optimizer    : adamw_torch")
            print(f"FP16         : False")
            print(f"BF16         : False")
            if torch.cuda.is_available():
                print(f"GPU Count    : {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}      : {torch.cuda.get_device_name(i)}")
            print("============================================================\n", flush=True)

        dataset_path = Path(dataset_path)
        train_df = pd.read_csv(dataset_path / "train_v2_1.csv", low_memory=False)
        val_df = pd.read_csv(dataset_path / "val_v2_1.csv", low_memory=False)
        test_df = pd.read_csv(dataset_path / "test_v2_1.csv", low_memory=False)

        for df in [train_df, val_df, test_df]:
            df["label"] = df["label"].astype(int)

        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
            "validation": Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False),
            "test": Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False),
        })
        del train_df, val_df, test_df
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        def tokenize_func(batch):
            return tokenizer(batch["text"], truncation=True, padding=False, max_length=CONFIG["max_length"])

        # Tokenization is isolated entirely within the local worker ensuring 100% correctness.
        tokenized = datasets.map(tokenize_func, batched=True, num_proc=1, remove_columns=["text"], desc="Tokenizing")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=2,
            use_safetensors=True,
            dtype=torch.float32
        )
        assert_model_fp32(model, "1. after model load")

        model = model.float()
        assert_model_fp32(model, "2. after model.float()")

        model = model.to(device=device, dtype=torch.float32)
        assert_model_fp32(model, "3. after model.to(device)")

        if is_rank_zero:
            print("\n============================================================")
            print("VOCABULARY PRESERVATION")
            print("============================================================")
            print(f"Tokenizer Length       : {len(tokenizer)}")
            print(f"Model Vocab Size       : {model.config.vocab_size}")
            print(f"Embedding Row Count    : {model.get_input_embeddings().num_embeddings}")
            print(f"Note: Model embeddings are NOT shrunken.")
            print("============================================================\n", flush=True)

        sanity_batch = [tokenized["train"][i] for i in range(2)]
        collated = data_collator(sanity_batch)
        collated = {k: v.to(device) for k, v in collated.items()}

        model.train()
        outputs = model(**collated)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError("Sanity check loss is not finite!")
        if loss.dtype != torch.float32:
            raise RuntimeError(f"Sanity check loss dtype is {loss.dtype}, expected torch.float32!")

        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Sanity check gradient for {name} is not finite!")

        if is_rank_zero:
            grad_norm_sq = sum((p.grad ** 2).sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
            max_grad = max((p.grad.abs().max().item() for p in model.parameters() if p.requires_grad and p.grad is not None), default=0.0)
            dtype_set = {str(p.dtype) for p in model.parameters() if p.requires_grad}

            print("\n============================================================")
            print("SANITY CHECK RESULTS")
            print("============================================================")
            print(f"Batch Size        : {collated['input_ids'].shape[0]}")
            print(f"Logits Shape      : {outputs.logits.shape}")
            print(f"Loss              : {loss.item():.4f}")
            print(f"Loss Dtype        : {loss.dtype}")
            print(f"Grad Norm         : {math.sqrt(grad_norm_sq):.4f}")
            print(f"Max Abs Grad      : {max_grad:.4e}")
            print(f"Parameter Dtypes  : {dtype_set}")
            print(f"Device            : {device}")
            print(f"ACTUAL MODEL PARAMETER DTYPE: {list(dtype_set)[0]}")
            print("============================================================\n", flush=True)

        assert_model_finite(model, "sanity check after backward")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        training_args = TrainingArguments(
            output_dir=str(CKPT_DIR),
            per_device_train_batch_size=CONFIG["per_gpu_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            per_device_eval_batch_size=CONFIG["per_gpu_eval_batch_size"],
            learning_rate=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
            warmup_ratio=CONFIG["warmup_ratio"],
            max_grad_norm=CONFIG["max_grad_norm"],
            num_train_epochs=CONFIG["epochs"],
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            seed=CONFIG["seed"],
            save_strategy="steps",
            save_steps=CONFIG["save_steps"],
            eval_strategy="steps",
            eval_steps=CONFIG["eval_steps"],
            logging_strategy="steps",
            logging_steps=CONFIG["logging_steps"],
            save_total_limit=CONFIG["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            ddp_find_unused_parameters=False,
            remove_unused_columns=True,
            disable_tqdm=not is_rank_zero
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            callbacks=[NumericalHealthCallback(diagnostic_steps=2000, log_every=100)]
        )

        assert_model_fp32(trainer.model, "4. after Trainer creation")

        if is_rank_zero:
            print("\nStarting fresh training (resume_from_checkpoint=None)...", flush=True)

        trainer.train(resume_from_checkpoint=None)

        expected_steps_approx = int(np.ceil(len(tokenized["train"]) / (CONFIG["per_gpu_batch_size"] * world_size * CONFIG["gradient_accumulation_steps"])) * CONFIG["epochs"])
        if trainer.state.global_step < (expected_steps_approx * 0.95):
            raise RuntimeError(f"Training stopped early! Completed {trainer.state.global_step} steps. Expected approx ~{expected_steps_approx}.")

        if is_rank_zero:
            print("\nTraining completed successfully! Evaluating Validation Set...", flush=True)

        val_results = trainer.evaluate(eval_dataset=tokenized["validation"])

        if is_rank_zero:
            print(f"Validation Results: {val_results}", flush=True)
            print("Evaluating Locked Test Set...", flush=True)

        test_results = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")

        assert_model_fp32(trainer.model, "7. before final model save")

        if is_rank_zero:
            print(f"Test Results: {test_results}", flush=True)
            print("\nSaving final model, tokenizer, and metrics...", flush=True)

            trainer.save_model(str(FINAL_DIR))
            tokenizer.save_pretrained(str(TOKENIZER_DIR))

            with open(METRICS_DIR / "final_metrics.json", "w") as f:
                json.dump({"val": val_results, "test": test_results}, f, indent=2)

            print("\n[SUCCESS] DeBERTa-v3 Multi-GPU Fine-Tuning Complete!", flush=True)

    except Exception as e:
        r = os.environ.get("RANK", rank)
        lr = os.environ.get("LOCAL_RANK", rank)
        print(f"\n[FATAL ERROR] Rank {r} (Local {lr}) encountered an exception:\n{str(e)}")
        traceback.print_exc()
        raise

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

def main():
    for d in [CKPT_DIR, LOGS_DIR, TOKENIZER_DIR, METRICS_DIR, FINAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    dataset_path = locate_dataset()

    if "LOCAL_RANK" in os.environ:
        ddp_worker(int(os.environ["RANK"]), world_size, str(dataset_path))
    elif world_size > 1:
        mp.spawn(ddp_worker, args=(world_size, str(dataset_path)), nprocs=world_size, join=True)
    else:
        ddp_worker(0, 1, str(dataset_path))

if __name__ == "__main__":
    main()
'''


# =============================================================================
# KAGGLE NOTEBOOK LAUNCHER
# =============================================================================
def launch():
    print("=" * 80)
    print("KAGGLE NOTEBOOK DDP LAUNCHER")
    print("=" * 80)

    # 1. Write the explicit string to disk
    script_path = Path("/kaggle/working/kaggle_ddp_deberta_v2_1.py")
    script_path.write_text(TRAINING_SCRIPT, encoding="utf-8")
    print(f"[OK] Wrote explicitly defined script to {script_path}")

    # 2. Syntax check
    try:
        py_compile.compile(str(script_path), doraise=True)
        print("[OK] Syntax check passed.")
    except py_compile.PyCompileError as e:
        raise RuntimeError(f"Syntax error in generated script:\n{e}")

    # 3. Verify markers are actually in the generated string
    content = script_path.read_text(encoding="utf-8")
    markers = [
        "microsoft/deberta-v3-base",
        "trainer.train",
        "NumericalHealthCallback",
        "ACCELERATE_MIXED_PRECISION",
        "adamw_torch",
        '"per_gpu_batch_size": 2',
        '"gradient_accumulation_steps": 8',
        '"max_length": 512'
    ]
    missing = [m for m in markers if m not in content]
    if missing:
        raise RuntimeError(f"[FAIL] Explicit script is missing critical markers: {missing}. Aborting.")
    print("[OK] Configuration markers verified.")

    # 4. Launch torchrun
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Create logs directory
    log_dir = Path("/kaggle/working/outputs/retraining_v2_1/deberta/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "torchrun_full_output.log"

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(script_path)
    ]
    print(f"\n[LAUNCH] Command: {' '.join(cmd)}")
    print(f"[LAUNCH] Full logs streaming to: {log_file_path}\n")

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()
        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"torchrun failed with exit code {process.returncode}")

    # 5. Report Checkpoints
    ckpt_dir = Path("/kaggle/working/outputs/retraining_v2_1/deberta/checkpoints")
    if not ckpt_dir.exists():
        raise RuntimeError("[FAIL] Checkpoint directory was not created. Checkpointing was accidentally disabled!")

    checkpoints = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")])
    print(f"\n============================================================")
    print("CHECKPOINT REPORT")
    print("============================================================")
    print(f"Total checkpoints retained: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f" - {ckpt.name}")
    print("============================================================\n")

if __name__ == "__main__":
    launch()
