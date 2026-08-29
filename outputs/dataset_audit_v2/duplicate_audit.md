# Phase 3 — Duplicate and Near-Duplicate Audit Report

**Project:** Hybrid AI-vs-Human Text Detection Framework
**Audit Date:** August 30, 2026
**Auditor:** Automated Audit Engine v2

---

## 1. Executive Summary

This audit evaluated exact, normalized, and near-duplicate text samples across both the **Active Experiment Dataset (`final_dataset.csv` - 47,322 rows)** and the **Baseline Expanded Corpus (`final_dataset.csv.bak` - 55,829 rows)**.

---

## 2. Duplicate Analysis Summary Table

| Dataset Analyzed | Total Rows | Exact Full Duplicates (Text + Label) | Exact Text Duplicates | Normalized Duplicates (Lowercase & Space Collapsed) | Cross-Label Normalized Conflicts | Extrapolated Near-Duplicate Groups (TF-IDF Cosine Similarity ≥ 0.88) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `final_dataset.csv` (Active 50/50) | **47,322** | **0** | **0** | **5** | **0** | **~8,800 pairs** |
| `final_dataset.csv.bak` (Baseline Expanded) | **55,829** | **0** | **0** | **10** | **0** | **~15,600 pairs** |

---

## 3. Near-Duplicate Analysis & Pattern Findings

Using TF-IDF (word n-gram range 1–2) cosine similarity on random sample slices (5,000 texts), we identified high-similarity pairs (similarity index ≥ 0.88):

### Identified Suspicious Near-Duplicate Patterns:
1. **Prompt Template Reiteration:** Multiple AI-generated texts (mostly from Kaggle DAIGT / PERSUADE subsets) share identical intro sentences, prompt instructions, or boilerplate concluding statements while varying only body sentences.
2. **Paraphrased / Multi-Prompt Generations:** Pairs where the same underlying topic (e.g., "The Choice on Electoral College", "Facial Technology in Cars") was generated twice by GPT using slightly different system prompts.
3. **Cross-Source Overlap:** Minor overlaps between Kaggle DAIGT V2 and HC3 Wikipedia QA where similar factual explanations share common structural phrasing.

### Data Quality Verification:
- **Zero Exact Duplicates:** Confirmed that prior cleaning (`clean_dataset.py` & `preprocess.py`) successfully eliminated all exact identical string matches.
- **Zero Cross-Label Conflicts:** No single text string is labeled as both Human (0) and AI (1).
