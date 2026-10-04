"""
phase2_generate_ai.py
=====================
Generates matched AI scholarly abstracts for human papers across 4 generator families:
1. GPT-4o (OpenAI API)
2. Claude-3.5-Sonnet (Anthropic API)
3. Llama-3-70B-Instruct (Groq / Together / OpenAI-compatible API)
4. Gemini-1.5-Pro (Google Generative AI API)

Supports:
- Mode A (Topic-conditioned generation, default)
- Mode B (Abstract-conditioned rewriting)
- Full provenance and prompt hashing
- Exponential backoff and rate-limit recovery
- Persistent status tracking and resumability
- Clean handling of unavailable models without fabricating data

Usage:
    python phase2_generate_ai.py --dry-run
    python phase2_generate_ai.py --input Datasets/phase2_scholarly/pilot/human_pilot_500.csv --generators gpt-4o,claude-3-5-sonnet,llama-3-70b-instruct,gemini-1-5-pro
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "Datasets" / "phase2_scholarly"
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
CONFIG_FILE = OUTPUT_DIR / "config" / "generation_config.json"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
LOG_DIR = OUTPUT_DIR / "logs"
STATUS_FILE = MANIFEST_DIR / "generation_status.csv"
AI_OUT_DIR = DATA_DIR / "ai"

for d in [AI_OUT_DIR, DATA_DIR / "pilot", MANIFEST_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "phase2_generation.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("Phase2_Generator")

def load_config() -> Dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_hashes(text: str) -> Tuple[str, str]:
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    norm_text = re.sub(r"\s+", " ", text.lower().strip())
    norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
    return raw_hash, norm_hash

def clean_generated_text(text: str) -> str:
    """Strip extraneous quotes or markdown framing while preserving prose."""
    if not text:
        return ""
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = re.sub(r'^(?:Here is an abstract|Abstract|Here is the generated abstract)[:\s-]*', '', text, flags=re.IGNORECASE)
    text = text.strip(' "\'\n\r\t')
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── API Client Implementations ──────────────────────────────────────────────

class LLMClient:
    def __init__(self, config: Dict):
        self.config = config
        self.generators = config.get("generators", {})

    def check_availability(self, gen_key: str) -> Tuple[bool, str]:
        gen_cfg = self.generators.get(gen_key)
        if not gen_cfg:
            return False, f"Unknown generator '{gen_key}'"
        env_key = gen_cfg.get("env_api_key")
        if env_key and os.environ.get(env_key):
            return True, "API key configured"
        # Check fallbacks
        for fb in gen_cfg.get("fallback_env_keys", []):
            if os.environ.get(fb):
                return True, f"Fallback API key '{fb}' configured"
        return False, f"Missing API key '{env_key}' in environment"

    def generate(self, gen_key: str, prompt: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Returns: (generated_text, status, error_type)
        Statuses: 'success', 'failed', 'unavailable'
        """
        is_avail, reason = self.check_availability(gen_key)
        if not is_avail:
            return None, "unavailable", f"auth_missing: {reason}"

        gen_cfg = self.generators[gen_key]
        family = gen_cfg["family"]
        model_id = gen_cfg["model_identifier"]
        temp = gen_cfg.get("temperature", 0.7)
        max_tokens = gen_cfg.get("max_output_tokens", 600)
        retries = gen_cfg.get("max_retries", 3)

        for attempt in range(retries):
            try:
                if family == "openai":
                    import openai
                    client = openai.OpenAI(api_key=os.environ.get(gen_cfg["env_api_key"]))
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=max_tokens,
                        timeout=gen_cfg.get("timeout_seconds", 30)
                    )
                    content = resp.choices[0].message.content
                    return clean_generated_text(content), "success", None

                elif family == "anthropic":
                    import anthropic
                    client = anthropic.Anthropic(api_key=os.environ.get(gen_cfg["env_api_key"]))
                    resp = client.messages.create(
                        model=model_id,
                        max_tokens=max_tokens,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=gen_cfg.get("timeout_seconds", 30)
                    )
                    content = resp.content[0].text
                    return clean_generated_text(content), "success", None

                elif family == "meta_llama":
                    import openai
                    # Uses Groq or OpenAI-compatible endpoint
                    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("TOGETHER_API_KEY")
                    base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None
                    client = openai.OpenAI(api_key=api_key, base_url=base_url)
                    resp = client.chat.completions.create(
                        model="llama3-70b-8192" if base_url else model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=max_tokens,
                        timeout=gen_cfg.get("timeout_seconds", 30)
                    )
                    content = resp.choices[0].message.content
                    return clean_generated_text(content), "success", None

                elif family == "google_gemini":
                    import google.generativeai as genai
                    genai.configure(api_key=os.environ.get(gen_cfg["env_api_key"]))
                    model = genai.GenerativeModel(model_id)
                    resp = model.generate_content(
                        prompt,
                        generation_config={"temperature": temp, "max_output_tokens": max_tokens}
                    )
                    content = resp.text
                    return clean_generated_text(content), "success", None

            except Exception as e:
                err_str = str(e).lower()
                logger.warning("Attempt %d for %s failed: %s", attempt + 1, gen_key, e)
                if "rate" in err_str:
                    time.sleep(5 * (attempt + 1))
                elif "auth" in err_str or "key" in err_str:
                    return None, "failed", f"authentication_failure: {e}"
                else:
                    time.sleep(2 * (attempt + 1))

        return None, "failed", "transient_max_retries_exceeded"

# ── Status Table Helpers ───────────────────────────────────────────────────

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def init_status_table(df_human: pd.DataFrame, generators: List[str]) -> pd.DataFrame:
    """Initialize or load persistent status tracking table."""
    if STATUS_FILE.exists():
        status_df = pd.read_csv(STATUS_FILE)
        return status_df

    rows = []
    for paper_id in df_human["paper_id"]:
        for gen in generators:
            rows.append({
                "paper_id": paper_id,
                "generator": gen,
                "status": "pending",
                "attempts": 0,
                "timestamp": "",
                "error_type": ""
            })
    status_df = pd.DataFrame(rows)
    status_df.to_csv(STATUS_FILE, index=False)
    return status_df

# ── Generation Pipeline ────────────────────────────────────────────────────

def format_prompt(template: str, row: pd.Series, target_wc: int) -> Tuple[str, str]:
    prompt = template.format(
        title=row["title"],
        discipline=row.get("discipline_display_name", row["discipline"]),
        target_word_count=target_wc,
        min_word_count=max(50, target_wc - 40),
        max_word_count=target_wc + 40,
        original_abstract=row.get("abstract", "")
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return prompt, prompt_hash

def run_dry_run(generators: List[str]):
    config = load_config()
    client = LLMClient(config)
    print("\n" + "=" * 60)
    print("PHASE 2A AI GENERATION — DRY RUN MODE")
    print("=" * 60)
    print(f"Protocol Version : {config.get('generation_protocol_version', '1.0')}")
    print(f"Primary Mode     : {config.get('primary_mode', 'Mode_A_topic_conditioned')}")
    print("\nGenerator Availability Check:")
    all_ready = True
    for g in generators:
        avail, reason = client.check_availability(g)
        status_sym = "[READY]" if avail else "[UNAVAILABLE]"
        print(f"  - {g:22s}: {status_sym} ({reason})")
        if not avail:
            all_ready = False

    print("\nGeneration Status Manifest:")
    print(f"  - Status file path: {STATUS_FILE}")
    print("=" * 60)
    if not all_ready:
        print("NOTICE: One or more requested generators lack API credentials.")
        print("Unavailable generators will be recorded as 'status=unavailable' in the audit.")
    print("DRY RUN COMPLETED\n")
    return True

def generate_ai_corpus(
    human_csv_path: Path,
    generators: List[str],
    mode: str = "Mode_A_topic_conditioned"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    client = LLMClient(config)
    template = config["prompt_templates"][mode]

    df_human = pd.read_csv(human_csv_path)
    status_df = init_status_table(df_human, generators)
    status_map = {}
    for _, r in status_df.iterrows():
        status_map[(r["paper_id"], r["generator"])] = {
            "status": str(r["status"]),
            "attempts": int(r.get("attempts", 0) if pd.notna(r.get("attempts")) else 0),
            "timestamp": str(r.get("timestamp", "") if pd.notna(r.get("timestamp")) else ""),
            "error_type": str(r.get("error_type", "") if pd.notna(r.get("error_type")) else "")
        }

    ai_records = []
    failures = []

    logger.info("Starting AI generation for %d human papers across %d generators...", len(df_human), len(generators))

    for idx, row in df_human.iterrows():
        p_id = row["paper_id"]
        target_wc = int(row.get("word_count", 200))

        for gen_key in generators:
            prev_status = status_map.get((p_id, gen_key), {})
            if prev_status.get("status") == "success":
                continue

            prompt, p_hash = format_prompt(template, row, target_wc)
            gen_cfg = config["generators"].get(gen_key, {})
            model_id = gen_cfg.get("model_identifier", gen_key)

            text_out, status, err = client.generate(gen_key, prompt)

            status_map[(p_id, gen_key)] = {
                "status": status,
                "attempts": prev_status.get("attempts", 0) + 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error_type": err or ""
            }

            if status == "success" and text_out:
                words = text_out.split()
                raw_h, norm_h = compute_hashes(text_out)
                rec = {
                    "paper_id": p_id,
                    "document_id": f"{p_id}_{gen_key}",
                    "text": text_out,
                    "label": 1,
                    "generator": gen_key,
                    "model_identifier": model_id,
                    "title": row["title"],
                    "discipline": row["discipline"],
                    "source": "AI_Generated",
                    "source_url": "",
                    "doi": row.get("doi", ""),
                    "publication_year": row.get("publication_year", 2021),
                    "license": "synthetic_generated",
                    "prompt": prompt,
                    "prompt_hash": p_hash,
                    "prompt_template_version": config.get("generation_protocol_version", "1.0"),
                    "temperature": gen_cfg.get("temperature", 0.7),
                    "generation_timestamp": datetime.utcnow().isoformat() + "Z",
                    "human_word_count": target_wc,
                    "word_count": len(words),
                    "length_ratio": round(len(words) / target_wc, 4) if target_wc > 0 else 1.0,
                    "text_sha256": raw_h,
                    "normalized_text_sha256": norm_h,
                    "generation_status": "success",
                    "generation_mode": mode
                }
                ai_records.append(rec)
            else:
                failures.append({
                    "paper_id": p_id,
                    "generator": gen_key,
                    "status": status,
                    "error_type": err or "unknown_error",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

    # Save in-memory status table to disk
    updated_rows = []
    for (p_id, gen), info in status_map.items():
        updated_rows.append({"paper_id": p_id, "generator": gen, **info})
    pd.DataFrame(updated_rows).to_csv(STATUS_FILE, index=False)

    ai_df = pd.DataFrame(ai_records)
    fail_df = pd.DataFrame(failures)

    # Save outputs
    if not ai_df.empty:
        ai_out_csv = DATA_DIR / "pilot" / "ai_pilot_generations.csv"
        ai_df.to_csv(ai_out_csv, index=False)
        logger.info("Saved %d AI generations to %s", len(ai_df), ai_out_csv)

    fail_out_csv = OUTPUT_DIR / "audits" / "generation_failures.csv"
    fail_df.to_csv(fail_out_csv, index=False)
    logger.info("Saved generation status audit to %s (%d non-success instances)", fail_out_csv, len(fail_df))

    return ai_df, fail_df

def main():
    parser = argparse.ArgumentParser(description="Phase 2A AI Generation Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run without calling APIs")
    parser.add_argument("--input", type=str, default="Datasets/phase2_scholarly/pilot/human_pilot_500.csv", help="Input human CSV")
    parser.add_argument("--generators", type=str, default="gpt-4o,claude-3-5-sonnet,llama-3-70b-instruct,gemini-1-5-pro", help="Comma-separated generator keys")
    parser.add_argument("--mode", type=str, default="Mode_A_topic_conditioned", help="Generation mode")
    args = parser.parse_args()

    gen_list = [g.strip() for g in args.generators.split(",") if g.strip()]

    if args.dry_run:
        run_dry_run(gen_list)
    else:
        generate_ai_corpus(Path(args.input), gen_list, mode=args.mode)

if __name__ == "__main__":
    main()
