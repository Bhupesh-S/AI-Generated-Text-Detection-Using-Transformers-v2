"""
report.py
=========
Automatic Markdown report generator for the RoBERTa training pipeline.

Generates ``RoBERTa_Report.md`` with:
  - Dataset information
  - Training configuration & hyperparameters
  - Model architecture summary
  - Evaluation metrics table
  - Embedded visualisation figures
  - Research discussion (Strengths & Limitations)

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

import datetime
from pathlib import Path
from typing import Dict, Optional

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics: Dict[str, float],
    config_dict: Dict,
    history_summary: Dict[str, list],
    dataset_info: Dict,
    figures_dir: Path,
    output_path: Path,
    training_time_str: str = "N/A",
    inference_time_str: str = "N/A",
) -> None:
    """
    Write a comprehensive Markdown report to *output_path*.

    Args:
        metrics          : Test-set evaluation metrics dict.
        config_dict      : Serialised :class:`config.RoBERTaConfig` dict.
        history_summary  : Training history dict from
                           :func:`utils.extract_training_history`.
        dataset_info     : Dict with keys ``total``, ``train``, ``val``,
                           ``test``, ``label_counts``.
        figures_dir      : Directory where figure PNGs were saved.
        output_path      : Destination ``.md`` file path.
        training_time_str: Human-readable total training duration.
        inference_time_str: Inference speed string.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Metric helpers ──────────────────────────────────────────────────────
    def fmt(key: str, pct: bool = True) -> str:
        v = metrics.get(key, metrics.get(f"test_{key}", float("nan")))
        if pct:
            return f"{v * 100:.2f}%" if v == v else "N/A"
        return f"{v:.4f}" if v == v else "N/A"

    acc_str       = fmt("accuracy")
    prec_str      = fmt("precision")
    recall_str    = fmt("recall")
    f1_str        = fmt("f1")
    roc_auc_str   = fmt("roc_auc")

    # ── Figure paths (relative for portability) ─────────────────────────────
    fig = lambda name: str(figures_dir / name)

    # ── Epoch log table ─────────────────────────────────────────────────────
    eval_rows = ""
    eval_losses = history_summary.get("eval_loss", [])
    eval_f1s    = {e["epoch"]: e["f1"] for e in history_summary.get("eval_f1", [])}
    eval_accs   = {e["epoch"]: e["accuracy"] for e in history_summary.get("eval_accuracy", [])}
    for e in eval_losses:
        ep = e["epoch"]
        ev_loss = f"{e['loss']:.4f}"
        ev_f1   = f"{eval_f1s.get(ep, float('nan')):.4f}"
        ev_acc  = f"{eval_accs.get(ep, float('nan')) * 100:.2f}%"
        eval_rows += f"| {ep:.0f} | {ev_loss} | {ev_acc} | {ev_f1} |\n"

    epoch_table = ""
    if eval_rows:
        epoch_table = (
            "| Epoch | Val Loss | Val Accuracy | Val F1 |\n"
            "|:-----:|:--------:|:------------:|:------:|\n"
            + eval_rows
        )

    # ── Assemble report ─────────────────────────────────────────────────────
    report = f"""# RoBERTa Fine-Tuning Report
## Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text

> **Generated**: {now}

---

## 1. Project Overview

This report documents the fine-tuning of **RoBERTa-base** (`roberta-base`) on a
large-scale academic text corpus for binary classification of human-written vs.
AI-generated content.  The model is part of a broader **Hybrid Explainable
Transformer Framework** that combines transformer-based representation learning
with ensemble methods and explainability techniques.

---

## 2. Dataset Information

| Property | Value |
|:---------|:------|
| **Source** | `{config_dict.get("data_path", "N/A")}` |
| **Total Samples** | {dataset_info.get("total", "N/A"):,} |
| **Training Samples** | {dataset_info.get("train", "N/A"):,} |
| **Validation Samples** | {dataset_info.get("val", "N/A"):,} |
| **Test Samples** | {dataset_info.get("test", "N/A"):,} |
| **Split Ratio** | 80% / 10% / 10% (stratified) |
| **Human Written (0)** | {dataset_info.get("label_counts", {}).get(0, "N/A"):,} |
| **AI Generated (1)** | {dataset_info.get("label_counts", {}).get(1, "N/A"):,} |

---

## 3. Model Architecture

| Property | Value |
|:---------|:------|
| **Base Model** | `{config_dict.get("model_name", "roberta-base")}` |
| **Task** | Binary Sequence Classification |
| **Number of Labels** | `{config_dict.get("num_labels", 2)}` |
| **Max Sequence Length** | `{config_dict.get("max_length", 256)} tokens` |
| **Padding Strategy** | `{config_dict.get("padding", "max_length")}` |
| **Truncation** | `{config_dict.get("truncation", True)}` |

RoBERTa (Robustly Optimised BERT Pretraining Approach) is a transformer-based
language model that improves upon BERT with dynamic masking, larger batches,
and training on more data.  The classification head is a linear layer on top of
the `[CLS]` token representation.

---

## 4. Training Configuration

| Hyperparameter | Value |
|:---------------|:------|
| **Epochs** | `{config_dict.get("epochs", 3)}` |
| **Batch Size** | `{config_dict.get("batch_size", 16)}` |
| **Learning Rate** | `{config_dict.get("learning_rate", 2e-5)}` |
| **Weight Decay** | `{config_dict.get("weight_decay", 0.01)}` |
| **Warmup Ratio** | `{config_dict.get("warmup_ratio", 0.1)}` |
| **Optimizer** | AdamW |
| **LR Scheduler** | Linear with warmup |
| **Gradient Clipping** | `{config_dict.get("max_grad_norm", 1.0)}` |
| **Grad. Accumulation** | `{config_dict.get("gradient_accumulation_steps", 1)} step(s)` |
| **Mixed Precision** | FP16 (if CUDA available) |
| **Early Stopping** | Patience = {config_dict.get("early_stopping_patience", 2)} epochs (metric: F1) |
| **Seed** | `{config_dict.get("seed", 42)}` |
| **Total Training Time** | {training_time_str} |

---

## 5. Evaluation Metrics (Test Set)

| Metric | Score |
|:-------|:-----:|
| **Accuracy** | {acc_str} |
| **Precision** | {prec_str} |
| **Recall** | {recall_str} |
| **F1 Score** | {f1_str} |
| **ROC-AUC** | {roc_auc_str} |
| **Inference Speed** | {inference_time_str} |

---

## 6. Training History

{epoch_table if epoch_table else "_Training history not available._"}

---

## 7. Visualisations

### 7.1 Training & Validation Loss

![Loss Curves]({fig("loss_curves.png")})

### 7.2 Validation Accuracy

![Accuracy Curve]({fig("accuracy_curve.png")})

### 7.3 Validation F1 Score

![F1 Curve]({fig("f1_curve.png")})

### 7.4 Confusion Matrix

![Confusion Matrix]({fig("confusion_matrix.png")})

### 7.5 ROC Curve

![ROC Curve]({fig("roc_curve.png")})

### 7.6 Precision-Recall Curve

![PR Curve]({fig("precision_recall_curve.png")})

---

## 8. Research Discussion

### 8.1 Strengths

1. **State-of-the-Art Backbone**: RoBERTa-base provides robust contextual
   representations pre-trained on 160 GB of text, enabling strong transfer
   learning for academic text.

2. **Stratified Splitting**: The 80/10/10 stratified split ensures consistent
   class distributions across all splits, reducing evaluation bias.

3. **Mixed Precision Training**: FP16 training approximately halves memory
   consumption and speeds up GPU training by ~1.5–2×.

4. **Early Stopping**: Monitoring validation F1 prevents over-fitting and
   saves training compute when performance plateaus.

5. **Production-Ready Modular Code**: The pipeline is split into `config`,
   `dataset`, `model`, `trainer`, `visualize`, `evaluate`, and `inference`
   modules for maintainability and reusability.

6. **Comprehensive Evaluation**: All standard NLP classification metrics
   (Accuracy, Precision, Recall, F1, ROC-AUC) plus visual diagnostics
   (confusion matrix, ROC, PR curves) are reported.

### 8.2 Limitations

1. **Sequence Length Constraint**: RoBERTa can process at most 512 tokens;
   with `max_length=256` very long documents are truncated, potentially losing
   information in the tail of the text.

2. **Domain Shift**: The model is trained on the specific academic dataset
   provided.  Performance may degrade on out-of-domain text (e.g. social
   media, code, multilingual content).

3. **Computational Cost**: Fine-tuning a 125 M parameter model requires
   significant GPU memory (~6–8 GB).  CPU-only training is extremely slow
   for a 552 k-sample dataset.

4. **Static Tokenisation**: The tokenizer was pre-trained on general English
   text; specialised academic jargon, mathematical notation, or LaTeX
   commands may be split into unusual sub-word tokens.

5. **Binary Classification Only**: The current setup distinguishes only
   *human-written* vs *AI-generated* without identifying the specific LLM
   (GPT-4, Claude, Gemini, etc.) responsible for generating the text.

---

## 9. Conclusion

The fine-tuned RoBERTa-base model achieves strong performance on the
binary AI-generated text detection task.  The results validate the
suitability of transformer-based representation learning for this problem,
forming a solid foundation for the downstream Ensemble Learning and
Explainability modules of the Hybrid Explainable Transformer Framework.

---

*Report auto-generated by `models/roberta/report.py`*
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved → {output_path}")
