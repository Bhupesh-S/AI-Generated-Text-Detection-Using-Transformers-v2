"""
train.py
========
Main entry point for the RoBERTa fine-tuning pipeline.

Run with:
    python models/roberta/train.py

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, set_seed as hf_set_seed

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Make sibling modules importable when called as ``python models/roberta/train.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from dataset import load_dataset_splits, tokenize_datasets
from model import load_model, get_model_size_mb
from trainer import build_trainer, full_evaluate
from visualize import plot_combined_curves, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from report import generate_report
from utils import (
    get_logger,
    set_seed,
    save_json,
    extract_training_history,
    format_time,
    print_section,
)

# ---------------------------------------------------------------------------
# Logger (writes to console + file)
# ---------------------------------------------------------------------------
CFG.create_dirs()   # Ensure dirs exist before opening log file
logger = get_logger(
    __name__,
    log_file=CFG.logs_dir / "train.log",
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute the complete RoBERTa fine-tuning pipeline end-to-end."""

    pipeline_start = time.time()

    # -- 0. Seed & device -----------------------------------------------------
    print_section("Step 0 - Seeding & Device")
    set_seed(CFG.seed)
    hf_set_seed(CFG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # -- CUDA Backend Optimizations (Ampere-specific) ------------------
        # TF32: RTX 3050 = GA107 (Ampere). TF32 uses 10-bit mantissa for matmul
        # (vs 23-bit FP32), giving near-FP32 accuracy at 2-8x higher throughput.
        # Set BEFORE model creation so all weight initializations use TF32.
        if CFG.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 matmul ENABLED (Ampere tensor cores).")

        # cuDNN benchmark: tests several convolution algorithms at first batch
        # and picks the fastest. Best when input shape is fixed (batch=32, seq=256).
        # Adds ~5-10 sec warm-up but saves time on every subsequent step.
        if CFG.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode ENABLED (finds fastest conv algorithm).")
    else:
        logger.warning(
            "No CUDA GPU detected.  Training on CPU will be extremely slow "
            "for this dataset.  Consider using a GPU runtime."
        )

    # -- 1. Load Data ----------------------------------------------------------
    print_section("Step 1 - Loading Dataset")
    raw_datasets = load_dataset_splits()

    n_train = len(raw_datasets["train"])
    n_val   = len(raw_datasets["validation"])
    n_test  = len(raw_datasets["test"])
    n_total = n_train + n_val + n_test

    # Collect label distribution from training set for the report
    train_labels = raw_datasets["train"]["label"]
    label_counts = {
        int(lbl): int(train_labels.count(lbl))
        for lbl in set(train_labels)
    }

    dataset_info = {
        "total": n_total,
        "train": n_train,
        "val":   n_val,
        "test":  n_test,
        "label_counts": label_counts,
    }
    logger.info(f"Dataset info: {dataset_info}")

    # -- 2. Tokenize -----------------------------------------------------------
    print_section("Step 2 - Tokenization")
    logger.info(f"Loading tokenizer: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    # -- 3. Load Model ---------------------------------------------------------
    print_section("Step 3 - Loading Model")
    model = load_model(model_name_or_path=CFG.model_name, num_labels=CFG.num_labels)
    model_size_mb = get_model_size_mb(model)
    logger.info(f"Model size (float32 estimate): {model_size_mb:.0f} MB")

    # -- 4. Train --------------------------------------------------------------
    print_section("Step 4 - Training")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    train_start = time.time()
    logger.info("Starting training ...")
    train_result = trainer.train()
    training_time_sec = time.time() - train_start
    training_time_str = format_time(training_time_sec)
    logger.info(f"Training completed in {training_time_str}")
    logger.info(f"Training metrics: {train_result.metrics}")

    # -- 5. Save best model & tokenizer ---------------------------------------
    print_section("Step 5 - Saving Model & Tokenizer")
    logger.info(f"Saving best model to: {CFG.tokenizer_dir}")
    trainer.save_model(str(CFG.tokenizer_dir))       # saves model + tokenizer
    tokenizer.save_pretrained(str(CFG.tokenizer_dir))
    logger.info("Model and tokenizer saved via save_pretrained().")

    # Save training config as JSON
    config_path = CFG.reports_dir / "training_config.json"
    config_dict = CFG.to_dict()
    config_dict["training_time_str"] = training_time_str
    config_dict["train_samples"]     = n_train
    config_dict["val_samples"]       = n_val
    config_dict["test_samples"]      = n_test
    save_json(config_dict, config_path)
    logger.info(f"Training config saved -> {config_path}")

    # -- 6. Evaluate on Test Set -----------------------------------------------
    print_section("Step 6 - Test Set Evaluation")
    t_infer_start = time.time()
    test_metrics, preds, probs, labels = full_evaluate(
        trainer, tokenized_datasets["test"]
    )
    inference_time_sec = time.time() - t_infer_start
    samples_per_sec = round(n_test / inference_time_sec, 1)
    inference_time_str = f"{samples_per_sec} samples/sec ({format_time(inference_time_sec)} total)"

    test_metrics["inference_time_sec"] = round(inference_time_sec, 3)
    test_metrics["samples_per_sec"]    = samples_per_sec

    logger.info("-- Test Set Results --")
    for k, v in test_metrics.items():
        logger.info(f"  {k:<25}: {v}")

    # Save metrics JSON
    metrics_path = CFG.reports_dir / "test_metrics.json"
    save_json(test_metrics, metrics_path)
    logger.info(f"Test metrics saved -> {metrics_path}")

    # -- 7. Save Predictions CSV ----------------------------------------------
    print_section("Step 7 - Saving Predictions")
    preds_df = pd.DataFrame(
        {
            "true_label":      labels,
            "predicted_label": preds,
            "prob_human":      probs[:, 0],
            "prob_ai":         probs[:, 1],
        }
    )
    preds_path = CFG.predictions_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Predictions CSV saved -> {preds_path}")

    # -- 8. Training History ---------------------------------------------------
    print_section("Step 8 - Extracting Training History")
    history = extract_training_history(trainer.state.log_history)
    history_path = CFG.reports_dir / "training_history.json"
    save_json(history, history_path)
    logger.info(f"Training history saved -> {history_path}")

    # -- 9. Visualisations -----------------------------------------------------
    print_section("Step 9 - Generating Figures")

    # Training curves (loss / accuracy / f1)
    plot_combined_curves(history, output_dir=CFG.figures_dir)

    # Confusion matrix
    plot_confusion_matrix(
        labels, preds,
        output_path=CFG.figures_dir / "confusion_matrix.png",
    )

    # ROC curve
    plot_roc_curve(
        labels, probs[:, 1],
        output_path=CFG.figures_dir / "roc_curve.png",
    )

    # Precision-Recall curve
    plot_precision_recall_curve(
        labels, probs[:, 1],
        output_path=CFG.figures_dir / "precision_recall_curve.png",
    )

    logger.info("All figures saved to: %s", CFG.figures_dir)

    # -- 10. Generate Markdown Report ------------------------------------------
    print_section("Step 10 - Generating Markdown Report")
    report_path = CFG.reports_dir / "RoBERTa_Report.md"
    generate_report(
        metrics=test_metrics,
        config_dict=config_dict,
        history_summary=history,
        dataset_info=dataset_info,
        figures_dir=CFG.figures_dir,
        output_path=report_path,
        training_time_str=training_time_str,
        inference_time_str=inference_time_str,
    )

    # -- Summary ---------------------------------------------------------------
    total_time = format_time(time.time() - pipeline_start)
    print_section("Pipeline Complete")
    logger.info(f"Total pipeline time: {total_time}")
    logger.info("")
    logger.info("Outputs saved to:")
    logger.info(f"  * Model + Tokenizer : {CFG.tokenizer_dir}")
    logger.info(f"  * Checkpoints       : {CFG.checkpoint_dir}")
    logger.info(f"  * Metrics JSON      : {metrics_path}")
    logger.info(f"  * Predictions CSV   : {preds_path}")
    logger.info(f"  * Figures           : {CFG.figures_dir}")
    logger.info(f"  * Report            : {report_path}")
    logger.info(f"  * Logs              : {CFG.logs_dir / 'train.log'}")
    logger.info("")
    logger.info("Run inference with:")
    logger.info("  python models/roberta/inference.py --interactive")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Pipeline failed with error: {exc}")
        logger.error(traceback.format_exc())
        sys.exit(1)
