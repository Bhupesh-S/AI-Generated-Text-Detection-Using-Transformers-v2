"""
run_experiment_v2.py
====================
Controlled Model-Improvement Experiment V2 for AI-vs-Human Text Detection.

Retrains:
  1. DeBERTa-v3-base
  2. DistilBERT-base-uncased

On the full expanded dataset of 55,829 samples without rebalancing or discarding data.
Outputs are strictly saved under: outputs/retraining_v2/
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)

from data_utils import get_unified_splits

# ---------------------------------------------------------------------------
# Global Settings & Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Datasets" / "merged" / "final_dataset.csv.bak"
OUTPUT_V2_DIR = BASE_DIR / "outputs" / "retraining_v2"

DEBERTA_OUT = OUTPUT_V2_DIR / "deberta"
DISTILBERT_OUT = OUTPUT_V2_DIR / "distilbert"

SEED = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ExperimentV2")


def setup_directories():
    """Ensure output subdirectories exist."""
    for root in [DEBERTA_OUT, DISTILBERT_OUT]:
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "tokenizer").mkdir(parents=True, exist_ok=True)
        (root / "predictions").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
    OUTPUT_V2_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics_eval(eval_pred) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer evaluation."""
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
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
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    }


def full_test_evaluation(model, tokenizer, test_dataset, batch_size=16) -> Dict:
    """Detailed evaluation on test set returning metrics, probabilities, and breakdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collator
    )

    all_logits = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels)

    infer_time = time.time() - start_time
    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)

    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(labels, probs[:, 1])

    prec_array, rec_array, _ = precision_recall_curve(labels, probs[:, 1])
    pr_auc = auc(rec_array, prec_array)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    human_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    ai_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "total_samples": len(labels),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "human_fpr": human_fpr,
        "ai_fnr": ai_fnr,
        "inference_time_sec": float(infer_time),
        "samples_per_sec": float(len(labels) / infer_time),
        "preds": preds,
        "probs": probs,
        "labels": labels
    }


def save_plots(eval_res: Dict, history_log: list, output_fig_dir: Path, model_title: str):
    """Generate and save Loss curves, Confusion Matrix, ROC, and PR curves."""
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    labels = eval_res["labels"]
    preds = eval_res["preds"]
    probs = eval_res["probs"][:, 1]

    # 1. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human (0)', 'AI (1)'],
                yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_title}')
    plt.tight_layout()
    plt.savefig(output_fig_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Training/Val Loss & Metrics curves
    epochs_train, train_loss = [], []
    epochs_eval, eval_loss, eval_acc, eval_f1 = [], [], [], []

    for entry in history_log:
        if "loss" in entry and "epoch" in entry:
            epochs_train.append(entry["epoch"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            epochs_eval.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_accuracy", 0))
            eval_f1.append(entry.get("eval_f1", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if train_loss:
        ax1.plot(epochs_train, train_loss, label='Train Loss', color='blue', alpha=0.7)
    if eval_loss:
        ax1.plot(epochs_eval, eval_loss, label='Val Loss', color='red', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss ({model_title})')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    if eval_acc:
        ax2.plot(epochs_eval, eval_acc, label='Val Accuracy', color='green', marker='o')
    if eval_f1:
        ax2.plot(epochs_eval, eval_f1, label='Val F1', color='purple', marker='s')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Validation Metrics ({model_title})')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig_dir / "learning_curves.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline Execution Functions
# ---------------------------------------------------------------------------
def run_dataset_verification() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 55,829 dataset and create/verify 80/10/10 stratified split."""
    logger.info("=== STEP 1: Dataset Verification & Splitting ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded raw dataset from: {DATA_PATH}")
    logger.info(f"Total raw rows: {len(df):,}")

    label_counts = df['label'].value_counts().to_dict()
    logger.info(f"Raw Label Counts -> 0 (Human): {label_counts.get(0, 0):,}, 1 (AI): {label_counts.get(1, 0):,}")

    assert len(df) == 55829, f"Expected 55,829 samples, but found {len(df)}"
    assert label_counts[0] == 32168, f"Expected 32,168 Human samples, got {label_counts[0]}"
    assert label_counts[1] == 23661, f"Expected 23,661 AI samples, got {label_counts[1]}"

    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

    train_df, val_df, test_df = get_unified_splits(df, test_size=0.10, val_size=0.10, seed=SEED)

    logger.info(f"Splits Created -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    logger.info(f"Sum of splits: {len(train_df) + len(val_df) + len(test_df):,} == 55,829")

    # Data Leakage Verification
    logger.info("=== STEP 2: Data Leakage Check ===")
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])

    tv_overlap = len(train_texts.intersection(val_texts))
    tt_overlap = len(train_texts.intersection(test_texts))
    vt_overlap = len(val_texts.intersection(test_texts))

    logger.info(f"Train vs Val text overlap: {tv_overlap}")
    logger.info(f"Train vs Test text overlap: {tt_overlap}")
    logger.info(f"Val vs Test text overlap: {vt_overlap}")

    leakage_info = {
        "total_samples": len(df),
        "ai_samples": label_counts[1],
        "human_samples": label_counts[0],
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_val_overlap": tv_overlap,
        "train_test_overlap": tt_overlap,
        "val_test_overlap": vt_overlap,
        "verified_no_leakage": (tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0)
    }

    with open(OUTPUT_V2_DIR / "data_leakage_check.json", "w") as f:
        json.dump(leakage_info, f, indent=4)

    return train_df, val_df, test_df


def train_and_evaluate_model(
    model_name: str,
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    epochs: int = 3,
    use_fp16: bool = True
) -> Dict:
    """Generic retraining & evaluation function for transformer components."""
    model_title = model_name.split("/")[-1]
    logger.info(f"==================================================")
    logger.info(f"Starting Training Pipeline for: {model_title}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Hyperparameters: LR={learning_rate}, Epochs={epochs}, Batch={batch_size}, GradAccum={grad_accum}")
    logger.info(f"==================================================")

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_train = Dataset.from_pandas(train_df[['text', 'label']])
    ds_val = Dataset.from_pandas(val_df[['text', 'label']])
    ds_test = Dataset.from_pandas(test_df[['text', 'label']])

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=512)

    logger.info(f"Tokenizing datasets with max_length=512...")
    tok_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['text'], desc="Tok Train")
    tok_val = ds_val.map(tokenize_fn, batched=True, remove_columns=['text'], desc="Tok Val")
    tok_test = ds_test.map(tokenize_fn, batched=True, remove_columns=['text'], desc="Tok Test")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human Written", 1: "AI Generated"},
        label2id={"Human Written": 0, "AI Generated": 1},
        use_safetensors=True
    )

    chk_dir = output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(chk_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        fp16=use_fp16 and torch.cuda.is_available(),
        bf16=False,                             # Disabled BF16 to avoid relative positional gradient issues
        label_smoothing_factor=0.0,             # Disabled label smoothing to ensure standard CrossEntropy loss
        save_total_limit=1,
        report_to="none",
        seed=SEED
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info(f"Training {model_title} on {len(tok_train):,} samples...")
    train_start = time.time()
    train_res = trainer.train()
    train_time_sec = time.time() - train_start
    logger.info(f"Training completed in {train_time_sec/60:.2f} minutes.")

    # Save best model & tokenizer
    save_tok_dir = output_dir / "tokenizer"
    trainer.save_model(str(save_tok_dir))
    tokenizer.save_pretrained(str(save_tok_dir))
    logger.info(f"Saved best model and tokenizer to: {save_tok_dir}")

    # Evaluate on held-out test set
    logger.info(f"Evaluating best {model_title} model on held-out Test set ({len(tok_test):,} samples)...")
    best_model = AutoModelForSequenceClassification.from_pretrained(save_tok_dir, use_safetensors=True)
    eval_res = full_test_evaluation(best_model, tokenizer, tok_test, batch_size=batch_size)

    eval_res["train_time_sec"] = train_time_sec
    eval_res["epochs_trained"] = train_res.metrics.get("epoch", epochs)
    eval_res["training_loss"] = train_res.training_loss

    # Save predictions CSV
    preds_df = pd.DataFrame({
        "true_label": eval_res["labels"],
        "predicted_label": eval_res["preds"],
        "prob_human": eval_res["probs"][:, 0],
        "prob_ai": eval_res["probs"][:, 1]
    })
    preds_df.to_csv(output_dir / "predictions" / "test_predictions.csv", index=False)

    # Save figures
    save_plots(eval_res, trainer.state.log_history, output_dir / "figures", model_title)

    # Save metrics JSON (without numpy arrays)
    metrics_to_save = {k: v for k, v in eval_res.items() if k not in ["preds", "probs", "labels"]}
    with open(output_dir / "reports" / "test_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    # Save training history JSON
    with open(output_dir / "reports" / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    logger.info(f"--- {model_title} Evaluation Results ---")
    logger.info(f"Accuracy : {eval_res['accuracy']*100:.2f}%")
    logger.info(f"Precision: {eval_res['precision']*100:.2f}%")
    logger.info(f"Recall   : {eval_res['recall']*100:.2f}%")
    logger.info(f"F1-Score : {eval_res['f1']*100:.2f}%")
    logger.info(f"ROC-AUC  : {eval_res['roc_auc']:.4f}")
    logger.info(f"PR-AUC   : {eval_res['pr_auc']:.4f}")
    logger.info(f"Human FPR: {eval_res['human_fpr']*100:.2f}%")
    logger.info(f"AI FNR   : {eval_res['ai_fnr']*100:.2f}%")
    logger.info(f"Confusion Matrix: TN={eval_res['confusion_matrix']['tn']}, FP={eval_res['confusion_matrix']['fp']}, FN={eval_res['confusion_matrix']['fn']}, TP={eval_res['confusion_matrix']['tp']}")

    return metrics_to_save


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    setup_directories()

    # Step 1 & 2: Dataset Verification & Leakage Checks
    train_df, val_df, test_df = run_dataset_verification()

    # Step 3: DeBERTa-v3 Retraining
    deberta_metrics = train_and_evaluate_model(
        model_name="microsoft/deberta-v3-base",
        output_dir=DEBERTA_OUT,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=4,
        grad_accum=4,          # Effective batch size = 16
        learning_rate=2e-5,
        epochs=3,
        use_fp16=False
    )

    # Step 4: DistilBERT Retraining
    distilbert_metrics = train_and_evaluate_model(
        model_name="distilbert-base-uncased",
        output_dir=DISTILBERT_OUT,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=8,
        grad_accum=2,          # Effective batch size = 16
        learning_rate=3e-5,
        epochs=3,
        use_fp16=True
    )

    # Step 5: Summary Report
    summary = {
        "dataset_size": 55829,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "deberta_v3": deberta_metrics,
        "distilbert": distilbert_metrics
    }

    with open(OUTPUT_V2_DIR / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("==================================================")
    logger.info("ALL RETRAINING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
