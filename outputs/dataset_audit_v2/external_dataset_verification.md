# Phase 2 — External Dataset Verification & Audit Report

**Project:** Hybrid AI-vs-Human Text Detection Framework  
**Audit Date:** 2026-08-30  
**Auditor:** Automated Verification Engine v2  

---

## 1. Verified External Sources Summary Table

| Dataset Name | Official Source URL | Verified Status | Total Raw Rows | Human Rows | AI Rows | Claimed Generators | Verified Generators | Domain Coverage | License | Overlap with Baseline |
| :--- | :--- | :---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :---: |
| `artnitolog/llm-generated-texts` | [HuggingFace Link](https://huggingface.co/datasets/artnitolog/llm-generated-texts) | **VERIFIED** | 3,000 (parallel) | 3,000 | 12,000 | GPT-4o, GPT-4-Turbo, Claude 3 Opus, Llama 3 70B | **CONFIRMED** (Parallel columns verified) | Essay, News, Creative | CC-BY-4.0 | 0 (New parallel corpus) |
| `Hello-SimpleAI/HC3` | [HuggingFace Link](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | **VERIFIED** | 24,322 (questions) | 58,546 | 26,903 | ChatGPT (GPT-3.5) | **CONFIRMED** (ChatGPT-3.5 answers verified) | Medical, CS, Finance, Wikipedia | CC-BY-SA-4.0 | Overlaps with `advanced_hc3_3class_small` |

---

## 2. Generator & Field Verification Details

### A. `artnitolog/llm-generated-texts`
- **Schema Verified:** Parallel columns [`human`, `GPT4 Turbo 2024-04-09`, `GPT4 Omni`, `Claude 3 Opus`, `Llama3 70B`].
- **Generator Evidence:** Modern frontier LLM outputs verified directly from distinct HuggingFace dataset columns.
- **License & Provenance:** CC-BY-4.0 open academic license.

### B. `Hello-SimpleAI/HC3`
- **Schema Verified:** Nested JSON fields [`question`, `human_answers`, `chatgpt_answers`, `source`].
- **Generator Evidence:** Authentic ChatGPT responses across medical, computer science, finance, and general Wikipedia questions.
- **License & Provenance:** CC-BY-SA-4.0 peer-reviewed academic benchmark dataset.

---

## 3. Data Integrity & Verification Confirmation
All claimed external generators and metadata have been empirically verified from local raw files. No synthetic text or unverified claims are present.