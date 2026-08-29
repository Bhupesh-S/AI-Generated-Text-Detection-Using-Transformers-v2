# Comprehensive Evaluation & Benchmark Report
Generated At: 2026-07-21 19:51:25

## 1. Executive Summary
This report documents the diagnostic auditing of the **Hybrid Ensemble Detector** on the hold-out test split.

## 2. Dataset Auditing Results
* **Total Samples evaluated**: 55829
* **Short Text ratio (< 250 words)**: 26.88%
* **Long Text ratio (> 750 words)**: 3.66%
* **Exact Duplicates in dataset**: 0
* **Estimated Near-Duplicates (Jaccard >= 0.85)**: 69

## 3. Overall Performance Metrics
| Metric | Value | Meaning |
| :--- | :--- | :--- |
| Accuracy | 92.50% | Proportion of correct predictions |
| Balanced Accuracy | 91.67% | Mean recall across both classes |
| Precision | 100.00% | AI Predicted class sharpness |
| Recall (Sensitivity) | 83.33% | Captured proportion of AI texts |
| Specificity | 100.00% | Captured proportion of Human texts |
| F1-score | 90.91% | Harmonized precision/recall score |
| MCC | 0.8563 | Matthews Correlation Coefficient |
| Cohen's Kappa | 0.8462 | Inter-annotator alignment |
| Brier Score | 0.0345 | Mean squared probability error |
| ECE (Calibration) | 0.0792 | Expected Calibration Error |
| False Positive Rate (FPR) | 0.00% | Human texts flagged as AI (lower is better) |

## 4. Failure Mode Diagnostics
### False Positives Count: 0
Most common FP failure triggers:

### False Negatives Count: 6
Most common FN failure triggers:
* **Short AI output outline**: 3 times
* **High entropy generator text shifting**: 3 times

## 5. Domain Generalization Slices
### Inferred Academic Disciplines
| Discipline | Accuracy | Precision | Recall | ECE |
| :--- | :--- | :--- | :--- | :--- |
| Medicine | 100.00% | 100.00% | 100.00% | 0.0586 |
| Biology/Chemistry | 100.00% | 100.00% | 100.00% | 0.0657 |
| Engineering/CS | 100.00% | 100.00% | 100.00% | 0.0395 |
| Law/Social Science | 100.00% | 100.00% | 100.00% | 0.0390 |
| General Academic | 87.50% | 100.00% | 75.00% | 0.0989 |

### Generator Robustness (AI Recall)
| Simulated Generator LLM | Recall Accuracy | Evaluation Counts |
| :--- | :--- | :--- |
| Llama | 82.35% | 34 |