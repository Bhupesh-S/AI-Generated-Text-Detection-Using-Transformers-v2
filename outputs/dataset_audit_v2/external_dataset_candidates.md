# Phase 6 — External Dataset Candidates Report

**Project:** Hybrid AI-vs-Human Text Detection Framework
**Audit Date:** August 30, 2026
**Auditor:** Automated Audit Engine v2

---

## 1. Candidate Dataset Evaluation Matrix

The following candidate datasets were identified on Hugging Face, Kaggle, and public research repositories specifically to address the vulnerabilities identified in Phase 4 (Dataset Gap Analysis).

| Dataset | Source | Human Samples | AI Samples | Primary Domain | Generators Represented | License | Why Needed (Gap Addressed) |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| `IntelLabs/AI-Peer-Review-Detection-Benchmark` | Hugging Face / Intel Labs | ~395,000 | ~395,000 | Academic Peer Reviews & Papers (ICLR/NeurIPS) | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Qwen 2.5, Llama 3.1 | MIT / Open Research | Fills **Formal Academic Human** & **Frontier LLM AI** gaps |
| `artnitolog/llm-generated-texts` | Hugging Face | 10,500 | 10,500 | Academic Essays, News, Creative Writing | GPT-4o, GPT-4 Turbo, Claude 3 Opus, Mistral | CC-BY-4.0 | Fills **Modern Frontier LLMs** & **Short-to-Medium Essay** gaps |
| `HanxiGuo/BiScope_Data` | Hugging Face / NeurIPS 2024 | 15,000 | 15,000 | ArXiv Abstracts, Essays, QA, Code | GPT-4 Turbo, Claude 3 Opus/Sonnet, Paraphrased AI | Apache-2.0 | Fills **Academic ArXiv**, **Short QA**, & **Paraphrased AI** gaps |
| `Hello-SimpleAI/HC3` | Hugging Face / HC3 | 24,000 | 27,000 | Medicine, CS, Finance, Wikipedia QA | ChatGPT (GPT-3.5) | CC-BY-SA-4.0 | Fills **Medical**, **Engineering/CS**, & **Short Human/AI QA** gaps |
| `Rajarshi-Roy-research/Defactify_Text_Dataset` | Hugging Face | 36,000 | 37,000 | Journalistic News & Formal Articles | GPT-4o, Gemma-2, LLaMA-8B | Open / Academic | Fills **Journalistic/News Prose** & **Gemma/LLaMA** gaps |

---

## 2. Selection Rationale & Targeted Downsampling Strategy

To comply with project principles ("Do NOT blindly download large datasets", "Download only relevant data required to fill gaps"), we will **NOT** import all 790k+ rows.

Instead, we will sample targeted subsets (approx. 2,000 to 8,000 rows per candidate dataset) focusing strictly on:
1. **Academic Peer Reviews & Papers** (from Intel Labs & BiScope) -> ~6,000 human + 6,000 AI samples.
2. **Medical & Engineering/CS QA** (from HC3) -> ~4,000 human + 4,000 AI samples.
3. **Frontier LLM Outputs** (GPT-4o, Claude 3.5, Gemini 1.5, Llama 3 from artnitolog & Defactify) -> ~5,000 AI samples.
4. **Short Text Samples (<150 words)** to equalize length bucket representations.
