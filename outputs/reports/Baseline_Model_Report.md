# Baseline Model Evaluation Report
*Generated: 2026-07-21 19:33:58*

---

## 1. Dataset Information

| Attribute | Value |
|-----------|-------|
| Dataset | `Datasets/final_dataset.csv` |
| Total Samples | ~552,000 |
| Classes | 2 — `0` Human Written · `1` AI Generated |
| Train / Valid / Test | 70 % / 10 % / 20 % (stratified) |

---

## 2. Feature Engineering

### 2.1 Text Preprocessing Steps

| Step | Operation |
|------|-----------|
| 1 | Unicode-safe re-encoding (ignore malformed bytes) |
| 2 | HTML entity decoding (`&amp;` → `&`, etc.) |
| 3 | HTML tag removal |
| 4 | URL removal |
| 5 | Lowercase conversion |
| 6 | Collapse horizontal whitespace |
| 7 | Strip leading / trailing whitespace |

### 2.2 Handcrafted Features (11 features)

| Feature | Description |
|---------|-------------|
| `word_count` | Number of whitespace-split tokens |
| `char_count` | Total characters (incl. spaces) |
| `sentence_count` | Sentences split by `[.!?]` |
| `avg_word_length` | Mean characters per token |
| `avg_sentence_length` | Mean words per sentence |
| `vocab_size` | Number of unique tokens |
| `lexical_diversity` | `vocab_size / word_count` |
| `stopword_ratio` | Stopwords / word_count |
| `punctuation_ratio` | Punctuation chars / char_count |
| `digit_ratio` | Digit chars / char_count |
| `uppercase_ratio` | Uppercase chars / char_count |

### 2.3 TF-IDF Configuration

| Parameter | Value |
|-----------|-------|
| `max_features` | 50,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `min_df` | 5 |
| `max_df` | 0.95 |
| `stop_words` | English |
| `sublinear_tf` | True (log-scaling) |
| `dtype` | float32 |

---

## 3. Model Comparison

### 3.1 Performance & Timing

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------|---------------|----------------|
| Logistic Regression | 0.9368 | 0.9267 | 0.9239 | 0.9253 | 0.9890 | 1.36s | 0.0042s |
| Multinomial Naive Bayes | 0.9295 | 0.9191 | 0.9142 | 0.9166 | 0.9854 | 0.04s | 0.0133s |
| Linear SVM ⭐ | 0.9416 | 0.9333 | 0.9286 | 0.9309 | 0.9902 | 3.19s | 0.0265s |
| Random Forest | 0.8983 | 0.9777 | 0.7777 | 0.8663 | 0.9800 | 1.42s | 0.1190s |
| XGBoost | 0.8833 | 0.8848 | 0.8331 | 0.8582 | 0.9578 | 0.55s | 0.0098s |

> ⭐ Best model by F1-Score

### 3.2 Observations

1. **Best Model**: **Linear SVM** — F1=0.9309 | Accuracy=0.9416 | ROC-AUC=0.9902
2. **Fastest Training**: Multinomial Naive Bayes (0.04s)
3. **Fastest Inference**: Logistic Regression (0.0042s)
4. **TF-IDF effectiveness**: Unigram + bigram TF-IDF with 50 k features provides a
   rich lexical signal that simple models can leverage effectively.
5. **Handcrafted features**: Linguistic statistics (lexical diversity, stopword ratio)
   serve as complementary signals — particularly useful for ensemble stacking later.

---

## 4. Research Insights

- **Baseline ceiling**: Linear models (Logistic Regression, Linear SVM) trained on
  TF-IDF often match or beat tree-based ensembles on high-dimensional sparse text
  representations.  This validates that feature-level signals are strong before
  turning to transformer fine-tuning.

- **AI writing patterns**: AI-generated text tends to exhibit higher lexical diversity
  and lower stopword ratios than human text — both features captured by the
  handcrafted set.

- **Towards transformers**: While these baselines establish a strong competitive
  floor, the hybrid transformer framework (DeBERTa / RoBERTa) is expected to
  substantially outperform them by leveraging contextual embeddings.

- **Ensemble potential**: Model disagreement among the five baselines suggests
  complementary strengths; stacking or voting ensembles should yield additional gains.

---

## 5. Output Files

### 5.1 Saved Models (`outputs/models/`)
- `logistic_regression.joblib`
- `multinomial_naive_bayes.joblib`
- `linear_svm.joblib`
- `random_forest.joblib`
- `xgboost.joblib`

### 5.2 Figures (`outputs/figures/`)
- `all_pr_curves.png`
- `all_roc_curves.png`
- `Linear SVM_confusion_matrix.png`
- `Linear SVM_pr_curve.png`
- `Linear SVM_roc_curve.png`
- `linear_svm_confusion_matrix.png`
- `linear_svm_pr_curve.png`
- `linear_svm_roc_curve.png`
- `Logistic Regression_confusion_matrix.png`
- `Logistic Regression_pr_curve.png`
- `Logistic Regression_roc_curve.png`
- `logistic_regression_confusion_matrix.png`
- `logistic_regression_pr_curve.png`
- `logistic_regression_roc_curve.png`
- `model_comparison_accuracy.png`
- `model_comparison_f1_score.png`
- `model_comparison_inference_time.png`
- `model_comparison_roc_auc.png`
- `model_comparison_training_time.png`
- `Multinomial Naive Bayes_confusion_matrix.png`
- `Multinomial Naive Bayes_pr_curve.png`
- `Multinomial Naive Bayes_roc_curve.png`
- `multinomial_naive_bayes_confusion_matrix.png`
- `multinomial_naive_bayes_pr_curve.png`
- `multinomial_naive_bayes_roc_curve.png`
- `Random Forest_confusion_matrix.png`
- `Random Forest_pr_curve.png`
- `Random Forest_roc_curve.png`
- `random_forest_confusion_matrix.png`
- `random_forest_pr_curve.png`
- `random_forest_roc_curve.png`
- `xgboost_confusion_matrix.png`
- `xgboost_pr_curve.png`
- `xgboost_roc_curve.png`

### 5.3 Reports (`outputs/reports/`)
- `baseline_results.csv` — aggregated metric table
- `*_classification_report.csv` — per-class precision / recall / F1 per model
- `interactive_model_comparison.html` — interactive Plotly chart
- `Baseline_Model_Report.md` — this document

---
*Report generated automatically by the Baseline Model Evaluation Pipeline*
