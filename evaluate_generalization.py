"""
evaluate_generalization.py
==========================
Evaluates trained Phase 1 models on held-out test splits.

Metrics Computed:
1. Core: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
2. Error Rates: TP, TN, FP, FN, Specificity, Sensitivity, FPR, FNR
3. Calibration: Brier Score, ECE, MCE
4. Distribution Stats: N, Human count, AI count, AI prevalence, mean/median word length
5. 95% Bootstrap Confidence Intervals (1,000 resamples, seed=42)
6. Dual Threshold Evaluation:
   - Validation-calibrated optimal threshold (tau)
   - Fixed threshold (tau = 0.50)
7. Visualizations: Confusion matrix plot, ROC curve, PR curve
8. Artifact Persistence: Per-sample predictions CSV, fold metrics JSON
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix,
    brier_score_loss
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Repository paths
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.utils import extract_features
from train_generalization import SimpleTextDataset, TemperatureScaler, PlattScaler

# Output directories
PHASE1_ROOT = REPO_ROOT / "outputs" / "generalization_phase1"
SPLITS_DIR = PHASE1_ROOT / "splits"
CKPT_DIR = PHASE1_ROOT / "checkpoints"
PREDS_DIR = PHASE1_ROOT / "predictions"
METRICS_DIR = PHASE1_ROOT / "metrics"
CM_DIR = PHASE1_ROOT / "confusion_matrices"
PLOTS_DIR = PHASE1_ROOT / "plots"

for d in [PREDS_DIR, METRICS_DIR, CM_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phase1_Evaluator")

SEED = 42

# ---------------------------------------------------------------------------
# Calibration Metrics Helper
# ---------------------------------------------------------------------------
def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Computes ECE, MCE, and Brier Score."""
    brier = float(brier_score_loss(y_true, y_prob))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if b == n_bins - 1:
            mask = mask | (y_prob == 1.0)
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / n) * gap
            mce = max(mce, gap)

    return {
        "brier_score": round(brier, 6),
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6)
    }

# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals (95% CI)
# ---------------------------------------------------------------------------
def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    preds: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict:
    """Computes empirical 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_acc = []
    boot_f1 = []
    boot_auc = []
    boot_prauc = []
    boot_fpr = []
    boot_rec = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        pred_b = preds[idx]

        if len(np.unique(y_b)) < 2:
            continue

        boot_acc.append(accuracy_score(y_b, pred_b))
        boot_f1.append(f1_score(y_b, pred_b, zero_division=0))
        boot_auc.append(roc_auc_score(y_b, p_b))

        prec_arr, rec_arr, _ = precision_recall_curve(y_b, p_b)
        boot_prauc.append(auc(rec_arr, prec_arr))

        cm = confusion_matrix(y_b, pred_b)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_b), 0, 0, 0)
        boot_fpr.append(fp / (tn + fp) if (tn + fp) > 0 else 0.0)
        boot_rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    def ci(arr):
        if not arr: return [0.0, 0.0]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "accuracy_95ci": ci(boot_acc),
        "f1_95ci": ci(boot_f1),
        "roc_auc_95ci": ci(boot_auc),
        "pr_auc_95ci": ci(boot_prauc),
        "fpr_95ci": ci(boot_fpr),
        "recall_95ci": ci(boot_rec)
    }

# ---------------------------------------------------------------------------
# Evaluator: XGBoost
# ---------------------------------------------------------------------------
def evaluate_xgboost(
    test_df: pd.DataFrame,
    ckpt_dir: Path
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generates predictions and probabilities for XGBoost on test_df."""
    model_path = ckpt_dir / "xgboost" / "model.joblib"
    scaler_path = ckpt_dir / "xgboost" / "scaler.joblib"
    calibrator_path = ckpt_dir / "xgboost" / "calibrator.joblib"
    meta_path = ckpt_dir / "xgboost" / "training_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    opt_threshold = meta.get("best_threshold", 0.50)

    # Feature extraction (per document)
    logger.info("    [XGBoost] Extracting features for test samples...")
    t0 = time.time()
    feats = [extract_features(t) for t in test_df["text"]]
    X_test_scaled = scaler.transform(pd.DataFrame(feats))

    # Predict probabilities
    raw_probs = clf.predict_proba(X_test_scaled)
    if calibrator is not None:
        probs = calibrator.predict_proba(raw_probs)
    else:
        probs = raw_probs

    y_prob = probs[:, 1]
    infer_time = time.time() - t0

    eval_info = {
        "inference_time_sec": round(infer_time, 2),
        "opt_threshold": opt_threshold
    }
    return y_prob, opt_threshold, eval_info

# ---------------------------------------------------------------------------
# Evaluator: Transformers
# ---------------------------------------------------------------------------
def evaluate_transformer(
    model_name: str,
    test_df: pd.DataFrame,
    ckpt_dir: Path,
    batch_size: int = 16,
    max_length: int = 256
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generates predictions and probabilities for a fine-tuned Transformer on test_df."""
    model_dir = ckpt_dir / model_name
    meta_path = model_dir / "training_meta.json"
    temp_path = model_dir / "temperature_scaler.joblib"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    opt_threshold = meta.get("best_threshold", 0.50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"    [{model_name}] Loading checkpoint from {model_dir.relative_to(REPO_ROOT)} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    test_ds = SimpleTextDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length=max_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_list.append(outputs.logits.cpu().numpy())

    infer_time = time.time() - t0
    logits = np.concatenate(logits_list, axis=0)

    # Temperature calibration
    if temp_path.exists():
        temp_scaler = joblib.load(temp_path)
        probs = temp_scaler.predict_proba(logits)
    else:
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    y_prob = probs[:, 1]
    eval_info = {
        "inference_time_sec": round(infer_time, 2),
        "opt_threshold": opt_threshold
    }
    return y_prob, opt_threshold, eval_info

# ---------------------------------------------------------------------------
# Metric Calculation & Plot Generation
# ---------------------------------------------------------------------------
def compute_metrics_and_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    opt_threshold: float,
    test_df: pd.DataFrame,
    experiment: str,
    fold: str,
    model_name: str,
    eval_info: Dict
) -> Dict:
    """Computes comprehensive metrics, plots confusion matrix, and saves predictions."""
    n = len(y_true)
    h_cnt = int((y_true == 0).sum())
    ai_cnt = int((y_true == 1).sum())

    words = test_df["text"].apply(lambda t: len(str(t).split())).values
    mean_len = float(np.mean(words))
    median_len = float(np.median(words))

    # ROC & PR curves
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(auc(rec_arr, prec_arr))

    # 1. Optimal Validation Threshold metrics
    preds_opt = (y_prob >= opt_threshold).astype(int)
    cm_opt = confusion_matrix(y_true, preds_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel() if cm_opt.size == 4 else (h_cnt, 0, 0, 0)
    fpr_opt = fp_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0
    fnr_opt = fn_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0

    # 2. Fixed Threshold (0.50) metrics
    preds_50 = (y_prob >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_true, preds_50)
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel() if cm_50.size == 4 else (h_cnt, 0, 0, 0)
    fpr_50 = fp_50 / (tn_50 + fp_50) if (tn_50 + fp_50) > 0 else 0.0
    fnr_50 = fn_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(y_true, y_prob)

    # Bootstrap 95% CIs
    boot_cis = compute_bootstrap_cis(y_true, y_prob, preds_opt, n_bootstraps=500, seed=SEED)

    full_metrics = {
        "experiment": experiment,
        "fold": fold,
        "model": model_name,
        "dataset_stats": {
            "total_samples": n,
            "human_samples": h_cnt,
            "ai_samples": ai_cnt,
            "ai_prevalence": round(ai_cnt / n, 4) if n > 0 else 0.0,
            "mean_word_length": round(mean_len, 1),
            "median_word_length": round(median_len, 1)
        },
        "threshold_independent": {
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6)
        },
        "optimal_threshold_metrics": {
            "threshold": round(opt_threshold, 4),
            "accuracy": round(float(accuracy_score(y_true, preds_opt)), 6),
            "precision": round(float(precision_score(y_true, preds_opt, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, preds_opt, zero_division=0)), 6),
            "f1_score": round(float(f1_score(y_true, preds_opt, zero_division=0)), 6),
            "specificity": round(float(tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0), 6),
            "sensitivity": round(float(tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0), 6),
            "fpr": round(float(fpr_opt), 6),
            "fnr": round(float(fnr_opt), 6),
            "confusion_matrix": {
                "tn": int(tn_opt), "fp": int(fp_opt),
                "fn": int(fn_opt), "tp": int(tp_opt)
            }
        },
        "fixed_threshold_50_metrics": {
            "threshold": 0.50,
            "accuracy": round(float(accuracy_score(y_true, preds_50)), 6),
            "precision": round(float(precision_score(y_true, preds_50, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, preds_50, zero_division=0)), 6),
            "f1_score": round(float(f1_score(y_true, preds_50, zero_division=0)), 6),
            "fpr": round(float(fpr_50), 6),
            "fnr": round(float(fnr_50), 6),
            "confusion_matrix": {
                "tn": int(tn_50), "fp": int(fp_50),
                "fn": int(fn_50), "tp": int(tp_50)
            }
        },
        "calibration": cal_metrics,
        "bootstrap_95ci": boot_cis,
        "runtime": {
            "inference_time_sec": eval_info.get("inference_time_sec", 0.0),
            "samples_per_sec": round(n / max(eval_info.get("inference_time_sec", 1.0), 1e-4), 1)
        }
    }

    # Save metrics JSON
    out_metric_file = METRICS_DIR / f"{experiment}_{fold}_{model_name}_metrics.json"
    with open(out_metric_file, "w") as f:
        json.dump(full_metrics, f, indent=2)

    # Save Confusion Matrix Visual Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_opt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Human", "Pred AI"],
        yticklabels=["True Human", "True AI"]
    )
    plt.title(f"Confusion Matrix: {model_name} ({fold})\nThreshold={opt_threshold:.2f} | Acc={full_metrics['optimal_threshold_metrics']['accuracy']*100:.1f}%")
    plt.tight_layout()
    cm_plot_file = CM_DIR / f"{experiment}_{fold}_{model_name}_cm.png"
    plt.savefig(cm_plot_file, dpi=300)
    plt.close()

    # Save ROC and PR plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_arr, tpr_arr, label=f"ROC-AUC = {roc_auc:.4f}", color="#2b5c8f", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(rec_arr, prec_arr, label=f"PR-AUC = {pr_auc:.4f}", color="#2b8f5c", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {model_name}")
    plt.legend(loc="lower left")

    plt.tight_layout()
    curve_plot_file = PLOTS_DIR / f"{experiment}_{fold}_{model_name}_curves.png"
    plt.savefig(curve_plot_file, dpi=300)
    plt.close()

    # Save per-sample predictions CSV (lightweight schema)
    preds_df = pd.DataFrame({
        "sample_id": test_df.get("sample_id", np.arange(n)),
        "sha256": test_df.get("sha256", [""] * n),
        "true_label": y_true,
        "source": test_df.get("source", ["unknown"] * n),
        "generator": test_df.get("generator", ["unknown"] * n),
        "domain": test_df.get("domain", ["unknown"] * n),
        "text_length": test_df.get("text_length", words),
        "prob_ai": np.round(y_prob, 5),
        "predicted_label": preds_opt,
        "fold": fold,
        "model": model_name
    })
    preds_csv = PREDS_DIR / f"{experiment}_{fold}_{model_name}_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)

    logger.info(f"    Saved metrics -> {out_metric_file.relative_to(REPO_ROOT)}")
    logger.info(f"    Saved CM plot -> {cm_plot_file.relative_to(REPO_ROOT)}")
    logger.info(f"    Saved predictions -> {preds_csv.relative_to(REPO_ROOT)}")

    return full_metrics

# ---------------------------------------------------------------------------
# Fold Evaluator Orchestration
# ---------------------------------------------------------------------------
def evaluate_fold(
    experiment: str,
    fold: str,
    models_to_eval: List[str],
    max_test_samples: Optional[int] = None
) -> Dict:
    """Evaluates all requested models on the held-out test split of a single fold."""
    logger.info(f"\n============================================================")
    logger.info(f"  EVALUATING FOLD: [{experiment.upper()}] -> {fold}")
    logger.info(f"============================================================")

    split_dir = SPLITS_DIR / experiment / fold
    ckpt_dir = CKPT_DIR / experiment / fold

    test_csv = split_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    test_df = pd.read_csv(test_csv)
    if max_test_samples and len(test_df) > max_test_samples:
        logger.info(f"  Subsampling test set from {len(test_df):,} to {max_test_samples:,} for fast evaluation...")
        test_df = test_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_test_samples // 2), random_state=SEED)
        ).reset_index(drop=True)

    logger.info(f"  Test samples: {len(test_df):,} | Human: {(test_df['label']==0).sum():,} | AI: {(test_df['label']==1).sum():,}")
    y_true = test_df["label"].values

    fold_results = {}

    # 1. XGBoost
    if "xgboost" in models_to_eval or "all" in models_to_eval:
        if (ckpt_dir / "xgboost" / "model.joblib").exists():
            logger.info("  Evaluating XGBoost...")
            y_prob, opt_t, info = evaluate_xgboost(test_df, ckpt_dir)
            fold_results["xgboost"] = compute_metrics_and_plots(
                y_true, y_prob, opt_t, test_df, experiment, fold, "xgboost", info
            )

    # 2. DistilBERT
    if "distilbert" in models_to_eval or "all" in models_to_eval:
        if (ckpt_dir / "distilbert" / "config.json").exists():
            logger.info("  Evaluating DistilBERT...")
            y_prob, opt_t, info = evaluate_transformer("distilbert", test_df, ckpt_dir)
            fold_results["distilbert"] = compute_metrics_and_plots(
                y_true, y_prob, opt_t, test_df, experiment, fold, "distilbert", info
            )

    # 3. DeBERTa
    if "deberta" in models_to_eval or "all" in models_to_eval:
        if (ckpt_dir / "deberta" / "config.json").exists():
            logger.info("  Evaluating DeBERTa-v3...")
            y_prob, opt_t, info = evaluate_transformer("deberta", test_df, ckpt_dir)
            fold_results["deberta"] = compute_metrics_and_plots(
                y_true, y_prob, opt_t, test_df, experiment, fold, "deberta", info
            )

    # 4. RoBERTa
    if "roberta" in models_to_eval:
        if (ckpt_dir / "roberta" / "config.json").exists():
            logger.info("  Evaluating RoBERTa...")
            y_prob, opt_t, info = evaluate_transformer("roberta", test_df, ckpt_dir)
            fold_results["roberta"] = compute_metrics_and_plots(
                y_true, y_prob, opt_t, test_df, experiment, fold, "roberta", info
            )

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Generalization Evaluator")
    parser.add_argument("--experiment", type=str, choices=["source", "generator", "all"], default="all")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--model", type=str, default="all", choices=["xgboost", "distilbert", "deberta", "roberta", "all"])
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    source_folds = ["holdout_artnitolog", "holdout_hc3", "holdout_baseline"]
    gen_folds = ["holdout_llama3", "holdout_claude3", "holdout_gpt4o"]

    models_to_eval = [args.model] if args.model != "all" else ["xgboost", "distilbert", "deberta"]

    if args.experiment in ["source", "all"]:
        folds = [args.fold] if args.fold != "all" and args.fold in source_folds else source_folds
        for f in folds:
            evaluate_fold("cross_source", f, models_to_eval, args.max_test_samples)

    if args.experiment in ["generator", "all"]:
        folds = [args.fold] if args.fold != "all" and args.fold in gen_folds else gen_folds
        for f in folds:
            evaluate_fold("cross_generator", f, models_to_eval, args.max_test_samples)

if __name__ == "__main__":
    main()
