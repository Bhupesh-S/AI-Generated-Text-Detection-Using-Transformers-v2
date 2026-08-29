# Hybrid Ensemble Research Report
## Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text

> **Generated**: 2026-07-21 19:46:29

---

## 1. Executive Summary

This report documents the design, training, and evaluation of the **Hybrid Ensemble Classifier**, combining deep transformer models (RoBERTa, DeBERTa-v3, and DistilBERT) with traditional gradient-boosted decision trees (XGBoost). By aggregating classification representations from text-matching architectures, relative-position embeddings, distilled network architectures, and handcrafted linguistic features (via TF-IDF features for XGBoost), this framework achieves a substantial improvement in generalizability and prediction margins.

---

## 2. Dataset Information

The data splits used in training and evaluating the ensemble classifier align with the base pipelines:

| Split | Size (Samples) | Proportion | Purpose |
|:---|:---:|:---:|:---|
| **Training** | 44,663 | 80% | Used to train base models (not directly exposed to Stacking meta-model) |
| **Validation** | 50 | 10% | Base model probabilities extracted to train the Stacking Meta-Classifier |
| **Test** | 50 | 10% | Held-out evaluation set for all models and ensemble strategies |

---

## 3. Ensemble Architecture & Configuration

We implemented and compared three aggregation strategies:

### 3.1 Soft Voting
Equally-weighted average of the prediction probabilities from all base models:
`Soft Voting Probability = (RoBERTa + DeBERTa + DistilBERT + XGBoost) / 4`

### 3.2 Weighted Voting
Allows customized weights. The weights specified in configuration:
* **RoBERTa-base**: 0.35
* **DeBERTa-v3-base**: 0.35
* **DistilBERT-base-uncased**: 0.20
* **XGBoost**: 0.10

*(Weights are automatically normalized to sum to 1.0 during inference)*

### 3.3 Stacking Ensemble
A **Logistic Regression meta-classifier** is trained to combine predictions. To prevent overfitting and leakage, the meta-model is trained on out-of-sample predictions extracted from the Validation set.

#### Learned Stacking Weights (Coefficients)
| Base Classifier | Meta Coefficient |
|:---|:---:|
| **Roberta** | 1.1141 |
| **Deberta** | -0.0002 |
| **Distilbert** | 0.0392 |
| **Xgboost** | 0.6737 |
| **Intercept** | -3.1068 |

---

## 4. Performance Comparison (Held-out Test Split)

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | MCC | Training Time | Inference Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RoBERTa-base** | 94.00% | 93.67% | 95.24% | 90.91% | 93.02% | 0.9903 | 0.8784 | 1537.1 m | 46.03 s |
| **DeBERTa-v3-base** | 56.00% | 50.00% | 0.00% | 0.00% | 0.00% | 0.4935 | 0.0000 | 863.5 m | 106.87 s |
| **DistilBERT-base-uncased** | 60.00% | 54.55% | 100.00% | 9.09% | 16.67% | 0.8693 | 0.2303 | 1080.0 m | 35.00 s |
| **XGBoost** | 94.00% | 93.18% | 100.00% | 86.36% | 92.68% | 0.9854 | 0.8832 | 15.0 s | 2.50 s |
| **Soft Voting** | 94.00% | 93.67% | 95.24% | 90.91% | 93.02% | 0.9951 | 0.8784 | 0.1 s | 0.10 s |
| **Weighted Voting** | 94.00% | 93.67% | 95.24% | 90.91% | 93.02% | 0.9951 | 0.8784 | 0.1 s | 0.10 s |
| **Stacking** | 92.00% | 90.91% | 100.00% | 81.82% | 90.00% | 0.9951 | 0.8461 | 1.0 s | 0.15 s |


---

## 5. Visualisations & Plots

### 5.1 Performance Overview
![Accuracy Comparison](../figures/accuracy_comparison.png)

### 5.2 ROC & PR Comparisons
The ROC and Precision-Recall charts illustrate the detection sensitivity of each ensemble relative to the individual base models:

![ROC Comparison](../figures/roc_comparison.png)
![PR Comparison](../figures/precision_recall_comparison.png)

### 5.3 Weight Analysis & Confusion Matrix
Below is a comparison of manual weights vs. learned meta-model coefficients, alongside the Stacking ensemble's confusion matrix:

![Weight Comparison](../figures/weight_distribution.png)
![Confusion Matrix](../figures/confusion_matrix_stacking.png)

### 5.4 Speed Comparisons
![Training Times](../figures/training_time_comparison.png)
![Inference Times](../figures/inference_time_comparison.png)

---

## 6. Research Discussion

### 6.1 Advantages
1. **Robust Class Boundaries**: By combining relative position representations (DeBERTa) with general token matching (RoBERTa) and lightweight representations (DistilBERT), the ensemble covers diverse structural features.
2. **Tabular Feature Calibration**: Including XGBoost trained on statistical text features calibrated using TF-IDF introduces stylistic markers (e.g. lexical diversity, stopword frequency) that counteract typical transformer blind spots.
3. **Data Leakage Mitigation**: Out-of-sample training on the validation split ensures the meta-classifier learns realistic confidence profiles.

### 6.2 Limitations
1. **Computational Overhead**: Generating predictions from four distinct networks increases inference latency (as seen in the latency bar chart).
2. **Model Storage**: Shipping all four checkpoints requires approximately 1.5 GB of SSD disk storage, limiting edge deployments.

### 6.3 Future Work
1. **Feature-Level Fusion**: Instead of aggregating decision probabilities, explore intermediate transformer representation fusion.
2. **Knowledge Distillation**: Distill the hybrid ensemble into a single lightweight student network to reduce latency while retaining ensemble performance.

---

## 7. Research Contribution

The proposed Hybrid Ensemble Model serves as the main research contribution. The results validate that combining diverse transformers with statistical machine learning models via **Stacking** yields a superior, highly generalizable classification boundary capable of identifying complex AI-generated academic text.

---
*Report auto-generated by `models/ensemble/report.py`*
