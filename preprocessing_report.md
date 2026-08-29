# Preprocessing Report

**Generated:** 2026-07-09 14:52:09  
**Project:** Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text  
**Minimum word count threshold:** 20 words  

---

## 1. Per-Dataset Summary

| Dataset | Original Rows | Final Rows | Null Removed | Whitespace Removed | Short Text Removed | Duplicates Removed | Human (0) | AI (1) |
|---------|:-------------:|:----------:|:------------:|:-----------------:|:-----------------:|:-----------------:|:---------:|:------:|
| advanced_hc3_3class_small | 15,000 | 12,879 | 0 | 0 | 0 | 2,121 | 4,483 | 8,396 |
| ai_human | 487,235 | 465,107 | 4 | 4 | 32 | 22,092 | 285,352 | 179,755 |
| fast_text_rebirth_dataset_315k | 315,000 | 77,137 | 0 | 0 | 0 | 237,863 | 31,359 | 45,778 |
| train_v2_drcat_02 | 44,868 | 44,860 | 0 | 0 | 2 | 6 | 27,365 | 17,495 |
| essay | 7,000 | 6,993 | 6 | 6 | 1 | 0 | 993 | 6,000 |
| other_gptzero | 100 | 100 | 0 | 0 | 0 | 0 | 50 | 50 |
| **TOTAL** | **869,203** | **607,076** | **10** | -- | **35** | **262,082** | **349,602** | **257,474** |

---

## 2. Merged Dataset Statistics

- **Total rows after merge & dedup:** 552,307
- **Human samples (label = 0):** 317,755
- **AI samples    (label = 1):** 234,552
- **Missing values:** 0
- **Duplicate rows:** 0
- **Average text length:** 361.2 words
- **Output file:** `datasets/merged/final_dataset.csv`

---

## 3. Final Class Distribution

| Label | Count | Percentage |
|-------|------:|----------:|
| Human (0) | 317,755 | 57.53% |
| AI Generated (1) | 234,552 | 42.47% |

---

## 4. Notes

- All text was cleaned: null bytes removed, newlines collapsed to spaces, extra whitespace trimmed.
- Unicode characters were preserved (no ASCII-only filtering).
- Rows with fewer than 20 words were discarded.
- Duplicate rows (identical text + label) were removed within each dataset and globally.
- The merged dataset was shuffled with `random_state=42` for reproducibility.
- Label mapping: 0 = Human, 1 = AI-Generated.

---

*Report generated automatically by `preprocessing/preprocess.py`*