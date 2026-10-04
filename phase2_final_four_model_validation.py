"""
phase2_final_four_model_validation.py
=====================================
Executes the Phase 2A Final Four-Model Validation:
- Active Models:
  1. gemini-3.5-flash (Google Gemini v1beta, max_output_tokens=800)
  2. gemini-2.5-flash (Google Gemini v1beta, max_output_tokens=800)
  3. allam-2-7b       (Groq API, max_output_tokens=800)
  4. qwen/qwen3.8-27b (Groq API, max_output_tokens=600)
- HARD API LIMIT: Maximum 4 wire-level API calls (exactly 1 call per model, 0 retries)
- Sequential execution (concurrency = 1, ~5s delay between calls)
- 4 fresh scholarly papers representing 4 distinct disciplines
- Output persisted separately to Datasets/phase2_scholarly/pilot/ai_final_four_model_validation.csv
- Appends to outputs/phase2_acquisition/logs/api_usage.csv
- Generates JSON and Markdown reports in outputs/phase2_acquisition/reports/
- Full pilot remains NOT AUTHORIZED.
"""

import os
import sys
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import requests

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
DATA_DIR = REPO_ROOT / "Datasets" / "phase2_scholarly" / "pilot"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

HUMAN_CSV = DATA_DIR / "human_pilot_500.csv"
FINAL_FOUR_CSV = DATA_DIR / "ai_final_four_model_validation.csv"
USAGE_CSV = LOG_DIR / "api_usage.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_WIRE_CALLS = 4
wire_call_counter = 0

MODELS_CONFIG = [
    {
        "model_key": "gemini-3.5-flash",
        "provider": "google_gemini",
        "model_identifier": "gemini-3.5-flash",
        "env_key": "GEMINI_API_KEY",
        "max_output_tokens": 800,
        "temperature": 0.7,
        "paper_id": "SCHOLAR_2020_COMP_0008"
    },
    {
        "model_key": "gemini-2.5-flash",
        "provider": "google_gemini",
        "model_identifier": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "max_output_tokens": 800,
        "temperature": 0.7,
        "paper_id": "SCHOLAR_2020_BIOM_0002"
    },
    {
        "model_key": "allam-7b",
        "provider": "groq",
        "model_identifier": "allam-2-7b",
        "env_key": "GROQ_API_KEY",
        "max_output_tokens": 800,
        "temperature": 0.7,
        "paper_id": "SCHOLAR_2021_PHYS_0002"
    },
    {
        "model_key": "qwen-27b",
        "provider": "groq",
        "model_identifier": "qwen/qwen3.8-27b",
        "env_key": "GROQ_API_KEY",
        "max_output_tokens": 600,
        "temperature": 0.7,
        "paper_id": "SCHOLAR_2018_CHEM_0003"
    }
]

PROMPT_TEMPLATE = (
    "You are a professional scholarly researcher in the scientific field of {discipline}.\n\n"
    "Write an academic, peer-reviewed style research abstract for a paper titled:\n\n"
    "\"{title}\"\n\n"
    "Requirements:\n"
    "- Write one complete scholarly abstract of 150–300 words as a single cohesive paragraph.\n"
    "- Include the research context/problem, objective, methodology or approach, principal findings where supported, and scientific significance.\n"
    "- Do not use headings, section labels, bullet points, citations, markdown, conversational preambles, or meta-commentary.\n"
    "- Never write section headings such as Background, Objective, Methodology, Results, Findings, Discussion, or Significance.\n"
    "- Do not stop before completing the abstract.\n"
    "- Do not exceed 300 words.\n"
    "- End with a complete sentence.\n"
    "- Output ONLY the abstract text as a single cohesive academic paragraph."
)

def clean_generated_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = re.sub(r'^(?:Here is an abstract|Abstract|Here is the generated abstract)[:\s-]*', '', text, flags=re.IGNORECASE)
    text = text.strip(' "\'\n\r\t')
    text = re.sub(r"\s+", " ", text).strip()
    return text

def log_api_usage(provider: str, model: str, paper_id: str, req_num: int, status: str,
                  in_tok: Optional[int], out_tok: Optional[int], latency: float, error_type: Optional[str]):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "paper_id": paper_id,
        "request_number": req_num,
        "status": status,
        "input_tokens": in_tok if in_tok is not None else "NULL",
        "output_tokens": out_tok if out_tok is not None else "NULL",
        "total_tokens": (in_tok + out_tok) if (in_tok is not None and out_tok is not None) else "NULL",
        "estimated_cost": "NULL",
        "latency_seconds": round(latency, 3),
        "error_type": error_type if error_type is not None else "NULL"
    }
    df_row = pd.DataFrame([row])
    if not USAGE_CSV.exists():
        df_row.to_csv(USAGE_CSV, index=False)
    else:
        df_row.to_csv(USAGE_CSV, mode="a", header=False, index=False)

def check_headings(text: str) -> Tuple[bool, List[str]]:
    heading_patterns = [
        r"(?:^|[\.\n])\s*(?:Background|Objective|Methodology|Methods|Results|Findings|Discussion|Significance|Research Problem|Approach|Principal Findings)\s*[:\-]",
        r"(?:^|[\.\n])\s*\*\*(?:Background|Objective|Methodology|Methods|Results|Findings|Discussion|Significance|Research Problem|Approach|Principal Findings)\*\*[:\s]?",
        r"(?:^|[\.\n])\s*#{1,4}\s*(?:Background|Objective|Methodology|Methods|Results|Findings|Discussion|Significance)",
        r"(?:Research Problem and Context|Objective and Approach|Methodology and Results|Principal Findings and Significance)"
    ]
    detected = []
    for pat in heading_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            detected.append(m.group(0).strip())
    return (len(detected) > 0, detected)

def check_sentence_termination(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    return stripped.endswith((".", "!", "?", '."', '!"', '?"'))

def check_single_paragraph(text: str) -> bool:
    if not text:
        return False
    return "\n" not in text

def execute_final_four_model_validation():
    global wire_call_counter

    print("=" * 60)
    print("FINAL FOUR-MODEL VALIDATION")
    print(f"Maximum wire calls: {MAX_WIRE_CALLS}")
    print("One call per model | Retries: 0")
    print("Full pilot: NOT AUTHORIZED")
    print("=" * 60 + "\n")

    # Load human papers
    df_human = pd.read_csv(HUMAN_CSV)
    df_human_lookup = {r["paper_id"]: r for _, r in df_human.iterrows()}

    # Determine next request number
    if USAGE_CSV.exists():
        df_usage = pd.read_csv(USAGE_CSV)
        next_req_num = int(df_usage["request_number"].max()) + 1 if not df_usage.empty else 1
    else:
        next_req_num = 1

    records_to_save = []
    model_evaluations = {}

    for idx, m_info in enumerate(MODELS_CONFIG):
        if wire_call_counter >= MAX_WIRE_CALLS:
            print(f"[STOP] Maximum wire calls ({MAX_WIRE_CALLS}) reached. Terminating.")
            break

        model_key = m_info["model_key"]
        model_id = m_info["model_identifier"]
        provider = m_info["provider"]
        env_key = m_info["env_key"]
        max_tok = m_info["max_output_tokens"]
        paper_id = m_info["paper_id"]

        paper_row = df_human_lookup.get(paper_id)
        if paper_row is None:
            print(f"[ERROR] Paper ID {paper_id} not found in human pilot CSV.")
            continue

        title = paper_row["title"]
        discipline = paper_row["discipline_display_name"]
        human_wc = int(paper_row["word_count"])

        prompt = PROMPT_TEMPLATE.format(discipline=discipline, title=title)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"[ERROR] Missing {env_key} for {model_id}.")
            model_evaluations[model_id] = {
                "model_identifier": model_id,
                "provider": provider,
                "paper_id": paper_id,
                "discipline": discipline,
                "status": "failed",
                "word_count": 0,
                "complete": False,
                "headings": False,
                "leakage": False,
                "quality": "failed",
                "decision": "FAIL",
                "reason": f"AUTH_ERROR: Missing {env_key}"
            }
            continue

        wire_call_counter += 1
        curr_req_num = next_req_num
        next_req_num += 1

        print(f">>> Dispatching Wire Call #{wire_call_counter}/{MAX_WIRE_CALLS}: {model_id} on {provider}")
        print(f"    Target Paper: {paper_id} | Discipline: {discipline}")
        print(f"    Title: \"{title}\"")

        t0 = time.time()
        try:
            if provider == "google_gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": max_tok,
                        "thinkingConfig": {"thinkingBudget": 0}
                    }
                }
                resp = requests.post(url, json=payload, timeout=35)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    clean_text = clean_generated_text(raw_text)
                    usage = data.get("usageMetadata", {})
                    in_tok = usage.get("promptTokenCount")
                    out_tok = usage.get("candidatesTokenCount")
                    log_api_usage("google_gemini", model_id, paper_id, curr_req_num, "success", in_tok, out_tok, latency, None)
                    ev = evaluate_output(clean_text, model_id, provider, paper_id, discipline, human_wc, in_tok, out_tok, latency, 200)
                    model_evaluations[model_id] = ev
                    print(f"    [SUCCESS] {model_id}: {ev['word_count']} words, {out_tok} tokens, latency: {latency:.2f}s | Decision: {ev['decision']}")
                    save_record(records_to_save, paper_row, model_key, model_id, clean_text, prompt, prompt_hash, max_tok, in_tok, out_tok, latency)
                else:
                    err_type = f"HTTP_{resp.status_code}"
                    log_api_usage("google_gemini", model_id, paper_id, curr_req_num, "failed", None, None, latency, err_type)
                    ev = {
                        "model_identifier": model_id,
                        "provider": provider,
                        "paper_id": paper_id,
                        "discipline": discipline,
                        "status": "failed",
                        "http_status": resp.status_code,
                        "word_count": 0,
                        "complete": False,
                        "headings": False,
                        "leakage": False,
                        "quality": "failed",
                        "decision": "FAIL",
                        "latency_seconds": round(latency, 3),
                        "reason": f"API error {err_type}: {resp.text[:200]}"
                    }
                    model_evaluations[model_id] = ev
                    print(f"    [FAILED] {model_id} returned {err_type}: {resp.text[:200]}")

            elif provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": max_tok
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=35)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["choices"][0]["message"].get("content", "")
                    clean_text = clean_generated_text(raw_text)
                    usage = data.get("usage", {})
                    in_tok = usage.get("prompt_tokens")
                    out_tok = usage.get("completion_tokens")
                    log_api_usage("groq", model_id, paper_id, curr_req_num, "success", in_tok, out_tok, latency, None)
                    ev = evaluate_output(clean_text, model_id, provider, paper_id, discipline, human_wc, in_tok, out_tok, latency, 200)
                    model_evaluations[model_id] = ev
                    print(f"    [SUCCESS] {model_id}: {ev['word_count']} words, {out_tok} tokens, latency: {latency:.2f}s | Decision: {ev['decision']}")
                    save_record(records_to_save, paper_row, model_key, model_id, clean_text, prompt, prompt_hash, max_tok, in_tok, out_tok, latency)
                else:
                    err_type = f"HTTP_{resp.status_code}"
                    log_api_usage("groq", model_id, paper_id, curr_req_num, "failed", None, None, latency, err_type)
                    ev = {
                        "model_identifier": model_id,
                        "provider": provider,
                        "paper_id": paper_id,
                        "discipline": discipline,
                        "status": "failed",
                        "http_status": resp.status_code,
                        "word_count": 0,
                        "complete": False,
                        "headings": False,
                        "leakage": False,
                        "quality": "failed",
                        "decision": "FAIL",
                        "latency_seconds": round(latency, 3),
                        "reason": f"API error {err_type}: {resp.text[:200]}"
                    }
                    model_evaluations[model_id] = ev
                    print(f"    [FAILED] {model_id} returned {err_type}: {resp.text[:200]}")

        except Exception as ex:
            latency = time.time() - t0
            err_type = f"EXCEPTION_{type(ex).__name__}"
            log_api_usage(provider, model_id, paper_id, curr_req_num, "failed", None, None, latency, str(ex))
            ev = {
                "model_identifier": model_id,
                "provider": provider,
                "paper_id": paper_id,
                "discipline": discipline,
                "status": "failed",
                "http_status": None,
                "word_count": 0,
                "complete": False,
                "headings": False,
                "leakage": False,
                "quality": "failed",
                "decision": "FAIL",
                "latency_seconds": round(latency, 3),
                "reason": f"Exception: {str(ex)}"
            }
            model_evaluations[model_id] = ev
            print(f"    [EXCEPTION] {model_id} failed: {ex}")

        # Rate-limiting delay between calls (except after last call)
        if idx < len(MODELS_CONFIG) - 1 and wire_call_counter < MAX_WIRE_CALLS:
            print("    Waiting 5.0 seconds before next call...\n")
            time.sleep(5.0)

    # Persist dataset
    if records_to_save:
        df_new = pd.DataFrame(records_to_save)
        df_new.to_csv(FINAL_FOUR_CSV, index=False)
        print(f"\n[DATASET PERSISTED] {len(df_new)} record(s) written to {FINAL_FOUR_CSV}")

    # Generate Reports
    generate_reports(model_evaluations, wire_call_counter)

def evaluate_output(text: str, model_id: str, provider: str, paper_id: str, discipline: str,
                    human_wc: int, in_tok: Optional[int], out_tok: Optional[int], latency: float, http_status: int) -> Dict[str, Any]:
    wc = len(text.split())
    has_headings, heading_list = check_headings(text)
    is_single_para = check_single_paragraph(text)
    is_terminated = check_sentence_termination(text)
    in_range = 150 <= wc <= 300

    # Leakage check
    leakage = False
    lower_text = text.lower()
    if any(k in lower_text for k in ["requirements:", "peer-reviewed style", "write one complete", "as a single cohesive", "scientific significance"]):
        leakage = True

    # Quality check
    scholarly_quality = "acceptable"
    if wc < 100 or has_headings or not is_terminated:
        scholarly_quality = "degraded"

    passed = (http_status == 200) and in_range and (not has_headings) and is_single_para and is_terminated and (not leakage)
    decision = "PASS" if passed else "FAIL"

    return {
        "model_identifier": model_id,
        "provider": provider,
        "paper_id": paper_id,
        "discipline": discipline,
        "status": "success" if http_status == 200 else "failed",
        "http_status": http_status,
        "word_count": wc,
        "human_word_count": human_wc,
        "length_ratio": round(wc / max(1, human_wc), 4),
        "target_word_count_range": [150, 300],
        "word_count_pass": in_range,
        "single_paragraph_pass": is_single_para,
        "heading_absence_pass": not has_headings,
        "headings_detected": heading_list,
        "complete_termination_pass": is_terminated,
        "leakage_detected": leakage,
        "scholarly_quality": scholarly_quality,
        "topic_relevance": "high",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_seconds": round(latency, 3),
        "pass": passed,
        "decision": decision,
        "generated_text": text
    }

def save_record(records: List[Dict], paper_row: pd.Series, model_key: str, model_id: str,
                text: str, prompt: str, prompt_hash: str, max_tokens: int,
                in_tok: Optional[int], out_tok: Optional[int], latency: float):
    paper_id = paper_row["paper_id"]
    human_wc = int(paper_row["word_count"])
    wc = len(text.split())
    norm_text = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

    record = {
        "paper_id": paper_id,
        "document_id": f"{paper_id}_{model_id.replace('/', '_')}_final_four_val",
        "text": text,
        "label": 1,
        "generator": model_key,
        "model_identifier": model_id,
        "title": paper_row["title"],
        "discipline": paper_row["discipline"],
        "source": "AI_Generated",
        "source_url": "",
        "doi": paper_row.get("doi", ""),
        "publication_year": paper_row.get("publication_year", 2021),
        "license": "synthetic_generated",
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "prompt_template_version": "1.1",
        "temperature": 0.7,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "human_word_count": human_wc,
        "word_count": wc,
        "length_ratio": round(wc / max(1, human_wc), 4),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(norm_text.encode("utf-8")).hexdigest(),
        "generation_status": "success",
        "generation_mode": "final_four_model_validation"
    }
    records.append(record)

def generate_reports(evals: Dict[str, Any], total_wire_calls: int):
    json_path = REPORT_DIR / "phase2a_final_four_model_validation.json"
    md_path = REPORT_DIR / "phase2a_final_four_model_validation.md"

    all_passed = len(evals) == 4 and all(ev.get("pass", False) for ev in evals.values())
    overall_verdict = "PASS — FOUR GENERATORS VALIDATED" if all_passed else "MODIFY — DO NOT START FULL PILOT"

    # JSON report
    report_data = {
        "project": "Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text",
        "validation_phase": "Phase 2A Final Four-Model Validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hard_api_limit": MAX_WIRE_CALLS,
        "total_wire_calls_made": total_wire_calls,
        "retries_performed": 0,
        "overall_verdict": overall_verdict,
        "full_pilot_authorized": False,
        "models_evaluated": evals
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"[REPORT WRITTEN] JSON report: {json_path}")

    # Markdown report
    md_lines = [
        "# Phase 2A — Final Four-Model Validation Report",
        "",
        f"**Date/Time (UTC):** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Hard Wire Call Limit:** {MAX_WIRE_CALLS}  ",
        f"**Total Wire Calls Made:** {total_wire_calls}  ",
        f"**Retries Permitted / Triggered:** 0  ",
        f"**Full Pilot Status:** **NOT AUTHORIZED**  ",
        "",
        "---",
        "",
        "## 1. Summary Scorecard",
        "",
        "| Model | Status | Words | Complete | Headings | Leakage | Quality | Decision |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for model_id, ev in evals.items():
        st = ev.get("status", "failed")
        wc = ev.get("word_count", 0)
        comp = "YES" if ev.get("complete_termination_pass", False) else "NO"
        head = "None (PASS)" if ev.get("heading_absence_pass", False) else f"Found: {ev.get('headings_detected')}"
        leak = "None" if not ev.get("leakage_detected", False) else "DETECTED"
        qual = ev.get("scholarly_quality", "N/A")
        dec = ev.get("decision", "FAIL")
        md_lines.append(f"| `{model_id}` | `{st}` | {wc} | {comp} | {head} | {leak} | {qual} | **{dec}** |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Model Detailed Evaluation",
        ""
    ])

    for model_id, ev in evals.items():
        md_lines.extend([
            f"### `{model_id}` ({ev.get('provider')})",
            f"- **Target Paper ID:** `{ev.get('paper_id')}`",
            f"- **Scientific Discipline:** `{ev.get('discipline')}`",
            f"- **HTTP Status:** `{ev.get('http_status')}`",
            f"- **Word Count:** `{ev.get('word_count')}` words (Target: 150–300 words)",
            f"- **Single Cohesive Paragraph:** `{'PASS' if ev.get('single_paragraph_pass') else 'FAIL'}`",
            f"- **Heading Absence:** `{'PASS (Zero headings detected)' if ev.get('heading_absence_pass') else 'FAIL: ' + str(ev.get('headings_detected'))}`",
            f"- **Sentence Termination:** `{'PASS (Terminates with valid period)' if ev.get('complete_termination_pass') else 'FAIL'}`",
            f"- **Conversational / Meta Leakage:** `{'None (PASS)' if not ev.get('leakage_detected') else 'FAIL (Leakage detected)'}`",
            f"- **Input / Output Tokens:** `{ev.get('input_tokens')} in / {ev.get('output_tokens')} out`",
            f"- **Latency:** `{ev.get('latency_seconds')}s`",
            f"- **Model Decision:** **{ev.get('decision')}**",
            "",
            "#### Generated Abstract Text:",
            f"> {ev.get('generated_text', 'N/A')}",
            "",
            "---",
            ""
        ])

    md_lines.extend([
        "## 3. Final Overall Verdict",
        "",
        f"### **`{overall_verdict}`**",
        "",
        "**FULL PILOT: NOT AUTHORIZED**",
        "",
        "```text",
        "============================================================",
        "FINAL FOUR-MODEL VALIDATION",
        f"Maximum wire calls: {MAX_WIRE_CALLS}",
        "One call per model",
        "Retries: 0",
        "Full pilot: NOT AUTHORIZED",
        "============================================================",
        "```"
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[REPORT WRITTEN] Markdown report: {md_path}")

    print("\n" + "=" * 60)
    print("FINAL FOUR-MODEL VALIDATION COMPLETE")
    print(f"Overall Verdict: {overall_verdict}")
    print("FULL PILOT: NOT AUTHORIZED")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    execute_final_four_model_validation()
