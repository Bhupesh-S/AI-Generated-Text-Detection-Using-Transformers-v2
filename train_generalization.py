"""
train_generalization.py
=======================
Trains models for Phase 1 Cross-Source and Cross-Generator Generalization experiments.

Supported Models:
1. XGBoost (27 Handcrafted Stylometric Features)
2. DistilBERT-base (distilbert-base-uncased)
3. DeBERTa-v3-base (microsoft/deberta-v3-base)
4. RoBERTa-base (roberta-base)

Strict Scientific Standards:
- Fresh model initialization per fold from base pretrained weights (no weight reuse across folds).
- Strict feature isolation: Scalers/normalizers fit ONLY on train.csv.
- Validation split used exclusively for early stopping, model selection, calibration fitting, and threshold selection.
- Test split is NEVER accessed during training, calibration, or thresholding.
- Supports configurable sample limits (--max-train-samples, --max-val-samples) for CPU execution.
- Deterministic seed=42.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, log_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Repository paths
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.utils import extract_features

# Output directories
PHASE1_ROOT = REPO_ROOT / "outputs" / "generalization_phase1"
SPLITS_DIR = PHASE1_ROOT / "splits"
CKPT_DIR = PHASE1_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phase1_Trainer")

SEED = 42

def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Calibration Utilities (Fitted ONLY on Validation split)
# ---------------------------------------------------------------------------
class TemperatureScaler:
    """Calibrates logits by scaling by temperature T, fit on validation data."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_val: np.ndarray):
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        def loss_fn(T):
            scaled = logits / max(T[0], 1e-4)
            exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            probs_clip = np.clip(probs[:, 1], 1e-15, 1.0 - 1e-15)
            return log_loss(y_val, probs_clip)

        res = minimize(loss_fn, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        scaled = logits / max(self.temperature, 1e-4)
        exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)


class PlattScaler:
    """Logistic calibration for tree/probabilistic models, fit on validation data."""
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)

    def fit(self, probs: np.ndarray, y_val: np.ndarray):
        eps = 1e-15
        p_clipped = np.clip(probs[:, 1] if probs.ndim == 2 else probs, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        self.lr.fit(logits, y_val)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p_clipped = np.clip(probs[:, 1] if probs.ndim == 2 else probs, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        return self.lr.predict_proba(logits)


def optimize_threshold_on_val(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Searches for optimal decision threshold on VALIDATION predictions.
    Maximizes F1 while penalizing FPR > 2%.
    """
    best_t = 0.50
    best_score = -999.0
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true), 0, 0, 0)
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

        if fpr <= 0.01:
            score = f1 + 2.0
        elif fpr <= 0.02:
            score = f1 + 1.0
        else:
            score = f1 - (fpr * 5.0)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return float(best_t)


# ---------------------------------------------------------------------------
# Text Dataset for Transformers
# ---------------------------------------------------------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Model Trainer: XGBoost (27 Stylometric Features)
# ---------------------------------------------------------------------------
def train_xgboost_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_dir: Path
) -> Dict:
    """Trains XGBoost on 27 stylometric features with strict feature isolation."""
    logger.info("  [XGBoost] Extracting 27 stylometric features for Train and Val...")
    
    t0 = time.time()
    X_train_raw = [extract_features(t) for t in train_df["text"]]
    X_val_raw = [extract_features(t) for t in val_df["text"]]

    X_tr = pd.DataFrame(X_train_raw)
    X_va = pd.DataFrame(X_val_raw)
    y_tr = train_df["label"].values
    y_va = val_df["label"].values

    # Strict isolation: scaler fit ONLY on X_tr
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    logger.info("  [XGBoost] Training XGBClassifier (150 trees, max_depth=4, hist)...")
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_tr_scaled, y_tr, eval_set=[(X_va_scaled, y_va)], verbose=False)
    train_time = time.time() - t0

    # Validation evaluation & calibration fitting
    val_probs_raw = clf.predict_proba(X_va_scaled)
    platt = PlattScaler()
    platt.fit(val_probs_raw, y_va)
    val_probs_cal = platt.predict_proba(val_probs_raw)

    best_threshold = optimize_threshold_on_val(y_va, val_probs_cal[:, 1])
    val_f1 = f1_score(y_va, (val_probs_cal[:, 1] >= best_threshold).astype(int), zero_division=0)
    val_auc = roc_auc_score(y_va, val_probs_cal[:, 1])

    # Save artifacts
    save_dir = fold_dir / "xgboost"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_dir / "model.joblib")
    joblib.dump(scaler, save_dir / "scaler.joblib")
    joblib.dump(platt, save_dir / "calibrator.joblib")

    meta = {
        "model_type": "xgboost",
        "n_estimators": int(clf.n_estimators),
        "best_iteration": int(getattr(clf, "best_iteration", 149)),
        "train_time_sec": round(train_time, 2),
        "val_f1": round(float(val_f1), 4),
        "val_roc_auc": round(float(val_auc), 4),
        "best_threshold": round(best_threshold, 4),
        "feature_count": 27
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  [XGBoost] Trained in {train_time:.1f}s | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Threshold: {best_threshold:.2f}")
    return meta


# ---------------------------------------------------------------------------
# Model Trainer: Transformers (DistilBERT / DeBERTa / RoBERTa)
# ---------------------------------------------------------------------------
def train_transformer_fold(
    model_name: str,
    hf_model_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 256
) -> Dict:
    """Trains a fresh Transformer backbone initialized from base pretrained weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  [{model_name}] Loading fresh pretrained '{hf_model_id}' on {device}...")
    
    t0 = time.time()
    kwargs = {}
    if "deberta-v3" in hf_model_id:
        kwargs["revision"] = "refs/pr/14"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id, num_labels=2, **kwargs)
    model.to(device)

    # Tokenize datasets
    train_ds = SimpleTextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=max_length)
    val_ds = SimpleTextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(int(total_steps * 0.10), 1), num_training_steps=total_steps)

    logger.info(f"  [{model_name}] Starting training: {epochs} epochs, {len(train_loader)} batches/epoch...")
    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        valid_steps = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            valid_steps += 1

        # Validation pass
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_logits_list.append(outputs.logits.cpu().numpy())

        val_logits = np.concatenate(val_logits_list, axis=0)
        val_logits = np.nan_to_num(val_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        exp_l = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        val_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        y_va = val_df["label"].values

        val_f1 = f1_score(y_va, (val_probs[:, 1] >= 0.50).astype(int), zero_division=0)
        avg_loss = train_loss / max(valid_steps, 1)
        logger.info(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1 (0.50): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    train_time = time.time() - t0

    # Final validation pass for calibration & threshold optimization
    model.eval()
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_logits_list.append(outputs.logits.cpu().numpy())
    val_logits = np.concatenate(val_logits_list, axis=0)
    val_logits = np.nan_to_num(val_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    y_va = val_df["label"].values

    temp_scaler = TemperatureScaler()
    temp_scaler.fit(val_logits, y_va)
    val_probs_cal = temp_scaler.predict_proba(val_logits)

    best_threshold = optimize_threshold_on_val(y_va, val_probs_cal[:, 1])
    val_auc = roc_auc_score(y_va, val_probs_cal[:, 1])

    # Save artifacts
    save_dir = fold_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    joblib.dump(temp_scaler, save_dir / "temperature_scaler.joblib")

    meta = {
        "model_type": model_name,
        "hf_model_id": hf_model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "train_time_sec": round(train_time, 2),
        "val_f1": round(float(best_val_f1), 4),
        "val_roc_auc": round(float(val_auc), 4),
        "temperature": round(float(temp_scaler.temperature), 4),
        "best_threshold": round(best_threshold, 4)
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  [{model_name}] Saved -> {save_dir.relative_to(REPO_ROOT)} | Val F1: {best_val_f1:.4f} | Temp: {temp_scaler.temperature:.2f}")
    return meta


# ---------------------------------------------------------------------------
# Fold Training Orchestration
# ---------------------------------------------------------------------------
def train_fold(
    experiment: str,
    fold: str,
    models_to_run: List[str],
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    epochs: int = 2
) -> Dict:
    """Executes fresh training of all requested models for a single generalization fold."""
    logger.info(f"\n============================================================")
    logger.info(f"  TRAINING FOLD: [{experiment.upper()}] -> {fold}")
    logger.info(f"============================================================")

    split_dir = SPLITS_DIR / experiment / fold
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")

    if max_train_samples and len(train_df) > max_train_samples:
        logger.info(f"  Subsampling train from {len(train_df):,} to {max_train_samples:,} for fast CPU execution...")
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_train_samples // 2), random_state=SEED)
        ).reset_index(drop=True)

    if max_val_samples and len(val_df) > max_val_samples:
        logger.info(f"  Subsampling val from {len(val_df):,} to {max_val_samples:,} for fast CPU execution...")
        val_df = val_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_val_samples // 2), random_state=SEED)
        ).reset_index(drop=True)

    logger.info(f"  Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    fold_ckpt_dir = CKPT_DIR / experiment / fold
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. XGBoost Baseline
    if "xgboost" in models_to_run or "all" in models_to_run:
        results["xgboost"] = train_xgboost_fold(train_df, val_df, fold_ckpt_dir)

    # 2. DistilBERT
    if "distilbert" in models_to_run or "all" in models_to_run:
        results["distilbert"] = train_transformer_fold(
            model_name="distilbert",
            hf_model_id="distilbert-base-uncased",
            train_df=train_df,
            val_df=val_df,
            fold_dir=fold_ckpt_dir,
            epochs=epochs,
            batch_size=8,
            lr=2e-5,
            max_length=256
        )

    # 3. DeBERTa-v3
    if "deberta" in models_to_run or "all" in models_to_run:
        results["deberta"] = train_transformer_fold(
            model_name="deberta",
            hf_model_id="microsoft/deberta-v3-base",
            train_df=train_df,
            val_df=val_df,
            fold_dir=fold_ckpt_dir,
            epochs=epochs,
            batch_size=4,
            lr=1e-5,
            max_length=256
        )

    # 4. RoBERTa
    if "roberta" in models_to_run:
        results["roberta"] = train_transformer_fold(
            model_name="roberta",
            hf_model_id="roberta-base",
            train_df=train_df,
            val_df=val_df,
            fold_dir=fold_ckpt_dir,
            epochs=epochs,
            batch_size=8,
            lr=2e-5,
            max_length=256
        )

    summary_file = fold_ckpt_dir / "fold_training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Generalization Model Trainer")
    parser.add_argument("--experiment", type=str, choices=["source", "generator", "all"], default="all", help="Experiment type to train")
    parser.add_argument("--fold", type=str, default="all", help="Specific fold name or 'all'")
    parser.add_argument("--model", type=str, default="all", choices=["xgboost", "distilbert", "deberta", "roberta", "all"], help="Model to train")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples for fast execution")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap validation samples")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    source_folds = ["holdout_artnitolog", "holdout_hc3", "holdout_baseline"]
    gen_folds = ["holdout_llama3", "holdout_claude3", "holdout_gpt4o"]

    models_to_run = [args.model] if args.model != "all" else ["xgboost", "distilbert", "deberta"]

    # Run source folds
    if args.experiment in ["source", "all"]:
        folds = [args.fold] if args.fold != "all" and args.fold in source_folds else source_folds
        for f in folds:
            train_fold("cross_source", f, models_to_run, args.max_train_samples, args.max_val_samples, args.epochs)

    # Run generator folds
    if args.experiment in ["generator", "all"]:
        folds = [args.fold] if args.fold != "all" and args.fold in gen_folds else gen_folds
        for f in folds:
            train_fold("cross_generator", f, models_to_run, args.max_train_samples, args.max_val_samples, args.epochs)

if __name__ == "__main__":
    main()
