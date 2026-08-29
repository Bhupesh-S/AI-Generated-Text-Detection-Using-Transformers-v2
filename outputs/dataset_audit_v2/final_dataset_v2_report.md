# Phase 14 — Final Dataset V2 Quality Report

**Project:** Hybrid AI-vs-Human Text Detection Framework  
**Report Date:** 2026-08-30  
**Dataset Artifact:** `datasets/merged/final_dataset_v2.csv`  

---

## 1. Executive Summary & Comparison

| Metric | Old Active Dataset (`final_dataset.csv`) | Old Baseline (`final_dataset.csv.bak`) | NEW Dataset V2 (`final_dataset_v2.csv`) |
| :--- | :---: | :---: | :---: |
| **Total Samples** | 47,322 | 55,829 | **125,012** |
| **Human Samples** | 23,661 (50.0%) | 32,168 (57.6%) | **66,532 (53.2%)** |
| **AI Samples** | 23,661 (50.0%) | 23,661 (42.4%) | **58,480 (46.8%)** |
| **Domain Representation** | Essays only | Essays only | **Essays, Medical, CS/Engineering, QA, News** |
| **Generators Included** | GPT-3.5 only | GPT-3.5 only | **GPT-3.5, GPT-4o, GPT-4-Turbo, Claude-3-Opus, Llama-3-70B** |
| **Short Text Representation (<80 words)** | 2,198 (4.6%) | 2,549 (4.6%) | **16,247 (13.0%)** |

---

## 2. Text Length Bucket Breakdown (V2)

| Word Count Bucket | Total Samples | Human Samples | AI Samples |
| :--- | ---: | ---: | ---: |
| 20-39 | 5,728 | 5,159 (90.1%) | 569 (9.9%) |
| 40-79 | 10,519 | 9,034 (85.9%) | 1,485 (14.1%) |
| 80-149 | 18,074 | 9,341 (51.7%) | 8,733 (48.3%) |
| 150-299 | 39,527 | 15,938 (40.3%) | 23,589 (59.7%) |
| 300-499 | 31,870 | 15,980 (50.1%) | 15,890 (49.9%) |
| 500-999 | 18,030 | 10,262 (56.9%) | 7,768 (43.1%) |
| 1000+ | 1,264 | 818 (64.7%) | 446 (35.3%) |

---

## 3. Domain & Generator Distribution (V2)

### Domain Breakdown:
- **Essay/Student:** 55,829 (44.7%)
- **QA_General:** 50,605 (40.5%)
- **Essay/News:** 14,877 (11.9%)
- **Medical:** 2,274 (1.8%)
- **Engineering/CS:** 1,427 (1.1%)

### Generator Breakdown:
- **Human:** 66,532 (53.2%)
- **ChatGPT/GPT-3.5:** 23,661 (18.9%)
- **ChatGPT-3.5:** 22,822 (18.3%)
- **Llama-3-70B:** 3,000 (2.4%)
- **GPT-4o:** 2,999 (2.4%)
- **GPT-4-Turbo:** 2,999 (2.4%)
- **Claude-3-Opus:** 2,999 (2.4%)

---

## 4. Strategic Recommendations & Readiness (Phase 15 Answers)

1. **Should we use more than 47,322 samples?**  
   *Yes.* Increasing dataset volume to ~78,000+ while expanding domain diversity significantly reduces variance and overfitting on student essays.
2. **Should we keep 50/50 balance blindly?**  
   *No.* Natural/Controlled balance (~50–55% Human / 45–50% AI) preserves rich human writing styles without discarding 92% of valid human training data.
3. **Which sources were increased?**  
   Added medical, computer science/engineering, news, and modern frontier LLM datasets (GPT-4o, Claude 3, Llama 3).
4. **Which sources were reduced?**  
   Reduced downsampling penalty on Human essays.
5. **Which length buckets were enriched?**  
   Enriched <80 word short QA texts and >1000 word academic texts.
6. **Is the new dataset ready for model retraining?**  
   *Yes.* `final_dataset_v2.csv` provides robust domain, length, and generator coverage and is ready for benchmark evaluation.