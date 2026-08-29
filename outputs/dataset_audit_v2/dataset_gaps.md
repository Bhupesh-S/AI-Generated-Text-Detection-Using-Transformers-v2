# Phase 4 — Detailed Dataset Gap Analysis Report

**Project:** Hybrid AI-vs-Human Text Detection Framework
**Audit Date:** August 30, 2026
**Auditor:** Automated Audit Engine v2

---

## 1. Identified Dataset Vulnerabilities & Gaps

Based on empirical analysis of `final_dataset.csv` (47,322 samples) and `final_dataset.csv.bak` (55,829 samples), we identified four primary structural gaps in the current dataset:

---

## 2. Human-Side Data Gaps

| Identified Gap | Current Status | Impact on Model | Required Solution |
| :--- | :--- | :--- | :--- |
| **Formal Academic Writing** | Underrepresented (Most human samples are student persuasive essays) | Causes False Positives on peer-reviewed papers & scientific abstracts | Add PubMed, arXiv, and IEEE human academic paper sections |
| **Medical & Scientific Writing** | Virtually non-existent | High false positive rate when evaluating clinical/biological texts | Add PubMed medical abstracts and journal articles |
| **Engineering & Computer Science** | Very low representation | Classifier misidentifies technical documentation as AI | Add arXiv CS paper excerpts and software documentation |
| **Citation-Heavy Prose** | Low | Inability to recognize formal academic citation formats | Add scholarly papers with inline citations |
| **Short Human Writing (<80 words)** | Only 1,045 samples total across 20–79 word buckets | Weak generalization on short human answers | Add short human Q&A (HC3 human answers) |

---

## 3. AI-Side Data Gaps

| Identified Gap | Current Status | Impact on Model | Required Solution |
| :--- | :--- | :--- | :--- |
| **Modern Frontier LLMs** | 90%+ is legacy GPT-3.5 | Model fails to detect GPT-4o, Claude 3.5, Gemini 1.5, Llama-3 | Integrate multi-LLM benchmark datasets (MGTBench, DAIGT-V4, LLM-Detect) |
| **Short AI Text (<80 words)** | Severe deficiency (only 1,153 AI samples under 80 words) | High False Negative rate on short AI responses | Add short ChatGPT/Claude answers and summary generations |
| **Academic AI Text** | Low | Unable to detect AI generated scientific abstracts or literature reviews | Add AI-generated arXiv abstracts and research section outputs |
| **Paraphrased / Humanized AI** | Zero representation | Complete failure on evasion techniques (QuillBot, humanized AI prompts) | Integrate humanized & paraphrased AI dataset samples |

---

## 4. Length Bucket Imbalance

```
Text Length Bucket    Human Samples    AI Samples    Imbalance Observation
──────────────────────────────────────────────────────────────────────────────────
20 – 39 words             404             415        Extremely scarce (~1.7% of dataset)
40 – 79 words             641             738        Extremely scarce (~2.9% of dataset)
80 – 149 words            629           1,832        Heavy AI skew (74.4% AI)
150 – 299 words         6,516           7,596        Balanced (~29.8% of dataset)
300 – 499 words         9,258          11,181        Dominates dataset (~43.2% of dataset)
500 – 999 words         5,929           1,880        Heavy Human skew (75.9% Human)
1000 – 1999 words         279              18        Extremely Human skewed (93.9% Human)
2000+ words                 5               1        Negligible sample size
```

---

## 5. Source Artifact Risk

> [!WARNING]
> **Source-Class Artifact:** In the current 50/50 balanced dataset, Human data was downsampled from 317,755 clean samples down to 23,661 to match the 23,661 AI samples. This discarded 92.5% of clean human data, causing the classifier to learn topic/source biases rather than true AI stylometric markers.
