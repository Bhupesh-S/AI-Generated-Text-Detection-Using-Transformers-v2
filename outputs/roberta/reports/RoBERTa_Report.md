# RoBERTa Fine-Tuning Report
## Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text

> **Generated**: 2026-07-19 13:51:05

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
| **Source** | `C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\Datasets\merged\final_dataset.csv` |
| **Total Samples** | 55,829 |
| **Training Samples** | 44,663 |
| **Validation Samples** | 5,583 |
| **Test Samples** | 5,583 |
| **Split Ratio** | 80% / 10% / 10% (stratified) |
| **Human Written (0)** | 25,734 |
| **AI Generated (1)** | 18,929 |

---

## 3. Model Architecture

| Property | Value |
|:---------|:------|
| **Base Model** | `roberta-base` |
| **Task** | Binary Sequence Classification |
| **Number of Labels** | `2` |
| **Max Sequence Length** | `256 tokens` |
| **Padding Strategy** | `max_length` |
| **Truncation** | `True` |

RoBERTa (Robustly Optimised BERT Pretraining Approach) is a transformer-based
language model that improves upon BERT with dynamic masking, larger batches,
and training on more data.  The classification head is a linear layer on top of
the `[CLS]` token representation.

---

## 4. Training Configuration

| Hyperparameter | Value |
|:---------------|:------|
| **Epochs** | `3` |
| **Batch Size** | `8` |
| **Learning Rate** | `2e-05` |
| **Weight Decay** | `0.01` |
| **Warmup Ratio** | `0.1` |
| **Optimizer** | AdamW |
| **LR Scheduler** | Linear with warmup |
| **Gradient Clipping** | `1.0` |
| **Grad. Accumulation** | `2 step(s)` |
| **Mixed Precision** | FP16 (if CUDA available) |
| **Early Stopping** | Patience = 2 epochs (metric: F1) |
| **Seed** | `42` |
| **Total Training Time** | 25:37:06 |

---

## 5. Evaluation Metrics (Test Set)

| Metric | Score |
|:-------|:-----:|
| **Accuracy** | 94.79% |
| **Precision** | 96.13% |
| **Recall** | 91.38% |
| **F1 Score** | 93.69% |
| **ROC-AUC** | 99.26% |
| **Inference Speed** | 121.3 samples/sec (0:00:46 total) |

---

## 6. Training History

| Epoch | Val Loss | Val Accuracy | Val F1 |
|:-----:|:--------:|:------------:|:------:|
| 1 | 0.1736 | 94.07% | 0.9283 |
| 2 | 0.1448 | 94.90% | 0.9379 |
| 3 | 0.1608 | 94.66% | 0.9362 |


---

## 7. Visualisations

### 7.1 Training & Validation Loss

![Loss Curves](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\loss_curves.png)

### 7.2 Validation Accuracy

![Accuracy Curve](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\accuracy_curve.png)

### 7.3 Validation F1 Score

![F1 Curve](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\f1_curve.png)

### 7.4 Confusion Matrix

![Confusion Matrix](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\confusion_matrix.png)

### 7.5 ROC Curve

![ROC Curve](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\roc_curve.png)

### 7.6 Precision-Recall Curve

![PR Curve](C:\7th Sem\Paper\-AI-Generated-Text-Detection-Using-Transformers_CODE\outputs\roberta\figures\precision_recall_curve.png)

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
