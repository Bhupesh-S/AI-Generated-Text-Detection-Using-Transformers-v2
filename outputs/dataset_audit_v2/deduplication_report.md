# Phase 9 — Cross-Dataset Deduplication Report

**Project:** Hybrid AI-vs-Human Text Detection Framework  
**Date:** 2026-08-30  

---

## 1. Summary of External Data Ingestion & Deduplication

- **Raw external samples downloaded:** 80,636
- **Removed invalid / short rows (<20 words):** 1,340
- **Removed exact & normalized duplicates against baseline dataset:** 6,590
- **Removed internal external duplicates:** 3,523
- **Final unique usable external rows:** **69,183**

---

## 2. Usable External Data Distribution

- **Human Samples added:** 34,364
- **AI Samples added:** 34,819

### Generator Breakdown:
- **Human:** 34,364
- **ChatGPT-3.5:** 22,822
- **Llama-3-70B:** 3,000
- **GPT-4o:** 2,999
- **GPT-4-Turbo:** 2,999
- **Claude-3-Opus:** 2,999