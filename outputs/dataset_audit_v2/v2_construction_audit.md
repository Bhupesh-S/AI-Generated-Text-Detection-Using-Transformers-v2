# Phase 1 Construction & Lineage Audit Report

**Project:** Hybrid AI-vs-Human Text Detection Framework  
**Report Date:** 2026-08-30  
**Target File:** `outputs/dataset_audit_v2/v2_construction_audit.md`  

---

## 1. Precise Construction Metrics for Dataset V2

- **Total Rows in V2:** **125,012**
- **Rows derived from old `final_dataset.csv` (47,322):** **47,322** (100% of old active set included in `final_dataset.csv.bak`)
- **Rows derived from `final_dataset.csv.bak` (55,829):** **55,829** (100% of pre-balancing expanded baseline included)
- **Rows derived from External Datasets:** **69,183** (artnitolog: 15,000 processed / 6,000 usable after dedup; HC3: 65,636 processed / 63,183 usable after dedup)

---

## 2. 552k Master Corpus Lineage & Availability

- **Is the full 552,307-row master corpus available?**  
  **YES.** Located at `C:\Users\bhupe\Downloads\final_dataset.txt` (1.16 GB).
- **Master Corpus Breakdown by Raw Source:**
  - `ai_human`: 465,107 rows (285,352 Human, 179,755 AI)
  - `fast_text_rebirth_dataset_315k`: 77,137 rows (31,359 Human, 45,778 AI)
  - `train_v2_drcat_02`: 44,860 rows (27,365 Human, 17,495 AI)
  - `advanced_hc3_3class_small`: 12,879 rows (4,483 Human, 8,396 AI)
  - `essay`: 6,993 rows (993 Human, 6,000 AI)
  - `other_gptzero`: 100 rows (50 Human, 50 AI)

---

## 3. Why Master Corpus Samples Were Excluded from V2

- **Rows from Master Corpus included in V2:** **55,829**
- **Rows from Master Corpus currently excluded from V2:** **496,478**
- **Primary Reasons for Exclusion:**
  1. **Previous Artificial Balancing:** `balance_dataset.py` downsampled Human text from 317,755 samples down to 23,661 to force a 50/50 balance with AI text (23,661), throwing away 294,094 clean human samples.
  2. **V2 Baseline Selection:** V2 construction used `final_dataset.csv.bak` (55,829) as its baseline foundation rather than ingesting all un-sampled rows from the 552k master corpus.
  3. **Data Quality & Duplicate Removal:** 262,082 raw rows were pruned during initial preprocessing due to duplicates across source datasets.
