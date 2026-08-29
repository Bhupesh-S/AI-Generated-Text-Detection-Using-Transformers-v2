# Phase 1 — Dataset Inventory & Mapping Report

**Project:** Hybrid Explainable Transformer Framework for AI-Generated Text Detection  
**Audit Date:** August 30, 2026  
**Auditor:** Automated Audit Engine v2  

---

## 1. Executive Inventory

This inventory maps all primary, secondary, raw, and cleaned dataset artifacts currently existing in the workspace.

| File / Folder Path | Description | Total Rows | File Size | Human Samples | AI Samples | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `preprocessing_report.md` | Log report of initial master multi-source dataset cleaning | **552,307** | ~2 KB | 317,755 (57.53%) | 234,552 (42.47%) | Reference Master Summary |
| `Datasets/merged/final_dataset.csv.bak` | Expanded merged dataset (Pre-balancing backup) | **55,829** | 111.7 MB | 32,168 (57.62%) | 23,661 (42.38%) | Active Expanded Corpus |
| `Datasets/merged/final_dataset.csv` | Currently active 50/50 balanced dataset | **47,322** | 93.8 MB | 23,661 (50.00%) | 23,661 (50.00%) | Active Experiment Dataset |
| `Datasets/merged/final_dataset_balanced.csv` | Backup copy of active balanced dataset | **47,322** | 93.8 MB | 23,661 (50.00%) | 23,661 (50.00%) | Verified Mirror |

---

## 2. Raw Source Dataset Mapping

The initial master merged corpus (**552,307 clean rows** derived from **869,203 raw rows**) was assembled by `preprocessing/preprocess.py` from six heterogeneous public datasets:

| Raw Source Dataset | Original Raw Rows | Cleaned Rows | Null Removed | Short Text Removed (<20 words) | Duplicates Removed | Final Human (0) | Final AI (1) | Primary Domain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ai_human` | 487,235 | 465,107 | 4 | 32 | 22,092 | 285,352 | 179,755 | Student Essays (PERSUADE) & GPT-3.5/4 |
| `fast_text_rebirth_dataset_315k` | 315,000 | 77,137 | 0 | 0 | 237,863 | 31,359 | 45,778 | Web Articles & LLM Generations |
| `train_v2_drcat_02` | 44,868 | 44,860 | 0 | 2 | 6 | 27,365 | 17,495 | Kaggle DAIGT V2 Essays & Prompts |
| `advanced_hc3_3class_small` | 15,000 | 12,879 | 0 | 0 | 2,121 | 4,483 | 8,396 | Human-ChatGPT QA & Wikipedia |
| `essay` | 7,000 | 6,993 | 6 | 1 | 0 | 993 | 6,000 | Student Essays & GPT Outputs |
| `other_gptzero` | 100 | 100 | 0 | 0 | 0 | 50 | 50 | GPTZero Validation Benchmark |
| **MASTER TOTAL** | **869,203** | **607,076** | **10** | **35** | **262,082** | **349,602** | **257,474** | **Multi-Domain Mixture** |

---

## 3. Dataset Pipeline Relationships

```
[6 Raw External Data Sources (869,203 rows)]
                     │
                     ▼
       [preprocessing/preprocess.py]
  (Cleaned nulls, collapsed spaces, word_count >= 20, deduplicated)
                     │
                     ▼
    [Master Merged Corpus (552,307 rows)]
  (Human: 317,755 (57.5%), AI: 234,552 (42.5%))
                     │
                     ▼
  [Expanded Baseline Dataset: final_dataset.csv.bak (55,829 rows)]
  (Human: 32,168 (57.6%), AI: 23,661 (42.4%))
                     │
                     ▼
   [balance_dataset.py / run_balanced_experiment.py]
  (Downsampled Human samples to match 23,661 AI samples)
                     │
                     ▼
  [Active Experiment Dataset: final_dataset.csv (47,322 rows)]
  (Human: 23,661 (50.0%), AI: 23,661 (50.0%))
```

---

## 4. Pipeline Script Mapping

* **`preprocessing/preprocess.py`**: Discovers top-level CSV/JSONL/.txt files in `Datasets/`, normalizes labels (0: Human, 1: AI), applies `MIN_WORD_COUNT = 20`, drops nulls/empty lines, removes cross-dataset duplicates, and outputs `datasets/merged/final_dataset.csv`.
* **`clean_dataset.py`**: Standalone cleaner ensuring null removal, empty text removal, and 50/50 class balancing.
* **`balance_dataset.py`**: Backs up `final_dataset.csv` to `final_dataset.csv.bak` and samples equal Human and AI instances.
* **`data_utils.py`**: Provides `get_unified_splits(df, test_size=0.10, val_size=0.10, seed=42)` for deterministic 80/10/10 train/validation/test splitting.
