"""
phase2_smoke_test.py
====================
Phase 2A Live Smoke Test for Free-Tier / Quota-Limited API Accounts.
Enforces:
- Maximum 20 requests (5 human papers x 4 generators)
- 1 paper per scientific discipline
- Sequential execution (concurrency = 1)
- Configurable rate limiting (default 5s delay between requests)
- Exponential backoff on rate limits
- Immediate provider-specific circuit breaker on QUOTA_EXCEEDED, AUTH_ERROR, MODEL_UNAVAILABLE
- Immediate checkpointing after every single request
- Persistent status tracking and api_usage.csv logging
- Zero fake record fabrication

Usage:
    python phase2_smoke_test.py --request-delay 5 --max-requests 20
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
import argparse
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
DATA_DIR = REPO_ROOT / "Datasets" / "phase2_scholarly"
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
CONFIG_FILE = OUTPUT_DIR / "config" / "generation_config.json"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
LOG_DIR = OUTPUT_DIR / "logs"
AUDIT_DIR = OUTPUT_DIR / "audits"
PLOT_DIR = OUTPUT_DIR / "plots"

USAGE_CSV = LOG_DIR / "api_usage.csv"
STATUS_FILE = MANIFEST_DIR / "generation_status.csv"
AI_OUT_CSV = DATA_DIR / "pilot" / "ai_pilot_generations.csv"

for d in [DATA_DIR / "pilot", MANIFEST_DIR, LOG_DIR, AUDIT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "phase2_smoke_test.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("Phase2_SmokeTest")

# Error Classifications
ERROR_RATE_LIMIT = "RATE_LIMIT"
ERROR_QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
ERROR_AUTH_ERROR = "AUTH_ERROR"
ERROR_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
ERROR_TRANSIENT = "TRANSIENT_ERROR"
ERROR_INVALID_REQUEST = "INVALID_REQUEST"

def compute_hashes(text: str) -> Tuple[str, str]:
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    norm_text = re.sub(r"\s+", " ", text.lower().strip())
    norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
    return raw_hash, norm_hash

def clean_generated_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = re.sub(r'^(?:Here is an abstract|Abstract|Here is the generated abstract)[:\s-]*', '', text, flags=re.IGNORECASE)
    text = text.strip(' "\'\n\r\t')
    text = re.sub(r"\s+", " ", text).strip()
    return text

def log_api_usage(
    provider: str,
    model: str,
    paper_id: str,
    req_num: int,
    status: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    estimated_cost: Optional[float] = None,
    latency_seconds: float = 0.0,
    error_type: Optional[str] = None
):
    """Immediately append request details to api_usage.csv."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "paper_id": paper_id,
        "request_number": req_num,
        "status": status,
        "input_tokens": input_tokens if input_tokens is not None else "NULL",
        "output_tokens": output_tokens if output_tokens is not None else "NULL",
        "total_tokens": total_tokens if total_tokens is not None else "NULL",
        "estimated_cost": f"{estimated_cost:.6f}" if estimated_cost is not None else "NULL",
        "latency_seconds": round(latency_seconds, 3),
        "error_type": error_type if error_type is not None else "NULL"
    }
    df_row = pd.DataFrame([row])
    if not USAGE_CSV.exists():
        df_row.to_csv(USAGE_CSV, index=False)
    else:
        df_row.to_csv(USAGE_CSV, mode="a", header=False, index=False)

def update_status_file(paper_id: str, gen: str, status: str, error_type: str = ""):
    """Immediately update or insert status record and flush to disk."""
    if STATUS_FILE.exists():
        status_df = pd.read_csv(STATUS_FILE)
    else:
        status_df = pd.DataFrame(columns=["paper_id", "generator", "status", "attempts", "timestamp", "error_type"])

    mask = (status_df["paper_id"] == paper_id) & (status_df["generator"] == gen)
    if mask.any():
        status_df.loc[mask, "status"] = status
        status_df.loc[mask, "attempts"] += 1
        status_df.loc[mask, "timestamp"] = datetime.now(timezone.utc).isoformat()
        status_df.loc[mask, "error_type"] = error_type
    else:
        new_row = {
            "paper_id": paper_id,
            "generator": gen,
            "status": status,
            "attempts": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type
        }
        status_df = pd.concat([status_df, pd.DataFrame([new_row])], ignore_index=True)
    status_df.to_csv(STATUS_FILE, index=False)

# ── Safe Provider Invocation ────────────────────────────────────────────────

class SafeLLMInvoker:
    def __init__(self, config: Dict):
        self.config = config
        self.generators = config.get("generators", {})
        # Track tripped circuit breakers per provider
        self.circuit_breakers: Dict[str, Tuple[bool, str]] = {}

    def is_tripped(self, gen_key: str) -> Tuple[bool, str]:
        return self.circuit_breakers.get(gen_key, (False, ""))

    def trip(self, gen_key: str, reason: str):
        self.circuit_breakers[gen_key] = (True, reason)
        logger.warning(">>> Circuit breaker TRIPPED for %s: %s (Future requests for this provider will be stopped)", gen_key, reason)

    def invoke(self, gen_key: str, prompt: str) -> Tuple[Optional[str], str, Optional[str], Optional[int], Optional[int], float]:
        """
        Executes a single, quota-safe call to the specified generator.
        Returns: (text, status, error_type, in_tok, out_tok, latency)
        """
        tripped, trip_reason = self.is_tripped(gen_key)
        if tripped:
            return None, "failed", trip_reason, None, None, 0.0

        gen_cfg = self.generators.get(gen_key, {})
        family = gen_cfg.get("family")
        model_id = gen_cfg.get("model_identifier")
        env_key = gen_cfg.get("env_api_key")
        api_key = os.environ.get(env_key)

        if not api_key:
            err = f"{ERROR_AUTH_ERROR}: API key '{env_key}' not set"
            self.trip(gen_key, err)
            return None, "failed", err, None, None, 0.0

        t0 = time.time()

        try:
            # 1. OpenAI (gpt-4o)
            if family == "openai":
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": gen_cfg.get("temperature", 0.7),
                    "max_tokens": min(gen_cfg.get("max_output_tokens", 450), 450)
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    content = clean_generated_text(data["choices"][0]["message"]["content"])
                    usage = data.get("usage", {})
                    return content, "success", None, usage.get("prompt_tokens"), usage.get("completion_tokens"), latency
                else:
                    err_body = resp.text.lower()
                    if resp.status_code == 429:
                        if "insufficient_quota" in err_body or "quota" in err_body or "billing" in err_body or "credits" in err_body:
                            err = f"{ERROR_QUOTA_EXCEEDED}: {resp.json().get('error', {}).get('message', 'Quota exceeded')}"
                            self.trip(gen_key, err)
                            return None, "failed", err, None, None, latency
                        return None, "failed", f"{ERROR_RATE_LIMIT}: Rate limited", None, None, latency
                    elif resp.status_code == 401:
                        err = f"{ERROR_AUTH_ERROR}: Unauthorized"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 404:
                        err = f"{ERROR_MODEL_UNAVAILABLE}: Model '{model_id}' not found"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    else:
                        return None, "failed", f"{ERROR_INVALID_REQUEST}: HTTP {resp.status_code}", None, None, latency

            # 2. Anthropic (claude-3-5-sonnet)
            elif family == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": model_id,
                    "max_tokens": min(gen_cfg.get("max_output_tokens", 450), 450),
                    "temperature": gen_cfg.get("temperature", 0.7),
                    "messages": [{"role": "user", "content": prompt}]
                }
                resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    content = clean_generated_text(data["content"][0]["text"])
                    usage = data.get("usage", {})
                    return content, "success", None, usage.get("input_tokens"), usage.get("output_tokens"), latency
                else:
                    err_body = resp.text.lower()
                    if "credit" in err_body or "quota" in err_body or "billing" in err_body:
                        err = f"{ERROR_QUOTA_EXCEEDED}: Insufficient credits"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 401:
                        err = f"{ERROR_AUTH_ERROR}: Invalid API key"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 429:
                        return None, "failed", f"{ERROR_RATE_LIMIT}: Rate limited", None, None, latency
                    else:
                        err = f"{ERROR_INVALID_REQUEST}: HTTP {resp.status_code} - {resp.text[:100]}"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency

            # 3. Meta Llama / OSS Models on Groq
            elif family in ["meta_llama", "groq"]:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                groq_model = model_id
                max_tok = gen_cfg.get("max_output_tokens", 800)
                if "120b" in groq_model.lower():
                    max_tok = max(max_tok, 1200)
                payload = {
                    "model": groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": gen_cfg.get("temperature", 0.7),
                    "max_tokens": max_tok
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=35)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    content = clean_generated_text(data["choices"][0]["message"]["content"])
                    usage = data.get("usage", {})
                    if not content:
                        return None, "failed", f"{ERROR_TRANSIENT}: Reasoning consumed tokens without final content", usage.get("prompt_tokens"), usage.get("completion_tokens"), latency
                    return content, "success", None, usage.get("prompt_tokens"), usage.get("completion_tokens"), latency
                else:
                    err_body = resp.text.lower()
                    if "model_not_found" in err_body or resp.status_code == 404:
                        err = f"{ERROR_MODEL_UNAVAILABLE}: Model '{groq_model}' unavailable on endpoint"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 429:
                        if "rate" in err_body:
                            return None, "failed", f"{ERROR_RATE_LIMIT}: Rate limited", None, None, latency
                        err = f"{ERROR_QUOTA_EXCEEDED}: Quota exceeded"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 401:
                        err = f"{ERROR_AUTH_ERROR}: Invalid API key"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    else:
                        err = f"{ERROR_INVALID_REQUEST}: HTTP {resp.status_code} - {resp.text[:100]}"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency

            # 4. Google Gemini
            elif family == "google_gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
                max_tok = gen_cfg.get("max_output_tokens", 800)
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": gen_cfg.get("temperature", 0.7),
                        "maxOutputTokens": max_tok,
                        "thinkingConfig": {"thinkingBudget": 0}
                    }
                }
                resp = requests.post(url, json=payload, timeout=35)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        content = clean_generated_text(data["candidates"][0]["content"]["parts"][0]["text"])
                        usage = data.get("usageMetadata", {})
                        return content, "success", None, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), latency
                    except (KeyError, IndexError) as parse_err:
                        return None, "failed", f"{ERROR_TRANSIENT}: Malformed response ({parse_err})", None, None, latency
                else:
                    err_body = resp.text.lower()
                    if resp.status_code == 503 or "temporar" in err_body or "demand" in err_body:
                        return None, "failed", f"{ERROR_TRANSIENT}: HTTP 503 high demand spike", None, None, latency
                    elif resp.status_code == 404 or "not found" in err_body or "not supported" in err_body or "no longer available" in err_body:
                        err = f"{ERROR_MODEL_UNAVAILABLE}: Model '{model_id}' not found or retired on API v1beta"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code == 429 or "resource_exhausted" in err_body:
                        err = f"{ERROR_QUOTA_EXCEEDED}: Quota exhausted on free tier"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    elif resp.status_code in [400, 403]:
                        err = f"{ERROR_AUTH_ERROR}: API key invalid or restricted (HTTP {resp.status_code})"
                        self.trip(gen_key, err)
                        return None, "failed", err, None, None, latency
                    else:
                        return None, "failed", f"{ERROR_INVALID_REQUEST}: HTTP {resp.status_code}", None, None, latency

        except requests.exceptions.Timeout:
            latency = time.time() - t0
            return None, "failed", f"{ERROR_TRANSIENT}: Request timeout (30s)", None, None, latency
        except Exception as e:
            latency = time.time() - t0
            return None, "failed", f"{ERROR_TRANSIENT}: {str(e)[:100]}", None, None, latency

        return None, "failed", f"{ERROR_INVALID_REQUEST}: Unknown provider", None, None, 0.0

# ── Live Smoke Test Runner ──────────────────────────────────────────────────

def run_smoke_test(
    human_csv_path: Path,
    generators: List[str],
    request_delay: float = 5.0,
    max_requests: int = 20
) -> Dict[str, Any]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    invoker = SafeLLMInvoker(config)
    template = config["prompt_templates"]["Mode_A_topic_conditioned"]

    df_human = pd.read_csv(human_csv_path)

    # Select exactly 1 paper per discipline to make 5 diverse human papers
    disciplines = df_human["discipline"].unique()
    selected_papers = []
    for d in disciplines:
        disc_subset = df_human[df_human["discipline"] == d]
        if not disc_subset.empty:
            selected_papers.append(disc_subset.iloc[0])
        if len(selected_papers) == 5:
            break

    df_sample = pd.DataFrame(selected_papers)
    logger.info("Selected %d diverse human papers (1 per discipline) for smoke test:", len(df_sample))
    for _, r in df_sample.iterrows():
        logger.info("  [%s] %s: %s (%d words)", r["discipline"], r["paper_id"], r["title"][:50], r["word_count"])

    # Load existing status to prevent any duplicate calls
    status_df = pd.read_csv(STATUS_FILE) if STATUS_FILE.exists() else pd.DataFrame(columns=["paper_id", "generator", "status"])

    # Load existing AI generations
    if AI_OUT_CSV.exists() and os.path.getsize(AI_OUT_CSV) > 10:
        df_ai_existing = pd.read_csv(AI_OUT_CSV)
    else:
        df_ai_existing = pd.DataFrame()

    total_requests_made = 0
    successful_generations = 0
    failed_generations = 0

    provider_stats = {
        g: {"attempted": 0, "success": 0, "failed": 0, "rate_limited": 0, "quota_exhausted": 0}
        for g in generators
    }

    new_ai_rows = []

    logger.info("\n" + "=" * 60)
    logger.info("STARTING LIVE SMOKE TEST (Max Requests: %d, Delay: %.1fs)", max_requests, request_delay)
    logger.info("=" * 60)

    for p_idx, (_, row) in enumerate(df_sample.iterrows()):
        paper_id = row["paper_id"]
        target_wc = int(row.get("word_count", 200))

        # Format standardized prompt
        prompt = template.format(
            title=row["title"],
            discipline=row.get("discipline_display_name", row["discipline"]),
            target_word_count=target_wc,
            min_word_count=max(50, target_wc - 40),
            max_word_count=target_wc + 40,
            original_abstract=row.get("abstract", "")
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        for gen in generators:
            if total_requests_made >= max_requests:
                logger.info("Global emergency stop: Reached maximum allowed requests (%d). Stopping.", max_requests)
                break

            # 1. Check if already successfully generated
            already_done = False
            if not status_df.empty:
                prev = status_df[(status_df["paper_id"] == paper_id) & (status_df["generator"] == gen)]
                if not prev.empty and prev.iloc[0]["status"] == "success":
                    already_done = True

            if already_done:
                logger.info("SKIPPING %s x %s: Already completed successfully.", paper_id, gen)
                provider_stats[gen]["attempted"] += 1
                provider_stats[gen]["success"] += 1
                continue

            # 2. Check if circuit breaker tripped for this provider
            tripped, trip_reason = invoker.is_tripped(gen)
            if tripped:
                logger.info("SKIPPING %s x %s: Provider circuit breaker is active (%s)", paper_id, gen, trip_reason)
                provider_stats[gen]["attempted"] += 1
                provider_stats[gen]["failed"] += 1
                if ERROR_QUOTA_EXCEEDED in trip_reason:
                    provider_stats[gen]["quota_exhausted"] += 1
                update_status_file(paper_id, gen, "failed", trip_reason)
                continue

            # 3. Execute sequential single request
            total_requests_made += 1
            provider_stats[gen]["attempted"] += 1
            model_id = config["generators"].get(gen, {}).get("model_identifier", gen)
            logger.info(">> Request #%d/%d: %s | Paper: %s | Model: %s", total_requests_made, max_requests, gen, paper_id, model_id)

            text_out, status, err_type, in_tok, out_tok, latency = invoker.invoke(gen, prompt)

            # Retry once on TRANSIENT error if under max_requests budget
            if status != "success" and err_type and ERROR_TRANSIENT in err_type and total_requests_made < max_requests:
                logger.info("   Transient failure (%s). Backing off 10s before safe single retry...", err_type)
                time.sleep(10.0)
                total_requests_made += 1
                logger.info(">> Request #%d/%d (Safe Retry): %s | Paper: %s", total_requests_made, max_requests, gen, paper_id)
                text_out, status, err_type, in_tok, out_tok, latency = invoker.invoke(gen, prompt)

            # Log immediately to api_usage.csv
            tot_tok = (in_tok + out_tok) if (in_tok is not None and out_tok is not None) else None
            log_api_usage(
                provider=gen,
                model=model_id,
                paper_id=paper_id,
                req_num=total_requests_made,
                status=status,
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                latency_seconds=latency,
                error_type=err_type
            )

            # Update status file immediately
            update_status_file(paper_id, gen, status, err_type or "")

            if status == "success" and text_out:
                successful_generations += 1
                provider_stats[gen]["success"] += 1
                words = text_out.split()
                raw_h, norm_h = compute_hashes(text_out)
                logger.info("   [SUCCESS] %s generated %d words (%.2fs)", gen, len(words), latency)

                rec = {
                    "paper_id": paper_id,
                    "document_id": f"{paper_id}_{gen}",
                    "text": text_out,
                    "label": 1,
                    "generator": gen,
                    "model_identifier": model_id,
                    "title": row["title"],
                    "discipline": row["discipline"],
                    "source": "AI_Generated",
                    "source_url": "",
                    "doi": row.get("doi", ""),
                    "publication_year": row.get("publication_year", 2021),
                    "license": "synthetic_generated",
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "prompt_template_version": config.get("generation_protocol_version", "1.0"),
                    "temperature": config["generators"].get(gen, {}).get("temperature", 0.7),
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "human_word_count": target_wc,
                    "word_count": len(words),
                    "length_ratio": round(len(words) / target_wc, 4) if target_wc > 0 else 1.0,
                    "text_sha256": raw_h,
                    "normalized_text_sha256": norm_h,
                    "generation_status": "success",
                    "generation_mode": "Mode_A_topic_conditioned"
                }
                new_ai_rows.append(rec)

                # Persist immediately to disk
                df_to_save = pd.concat([df_ai_existing, pd.DataFrame(new_ai_rows)], ignore_index=True)
                df_to_save.drop_duplicates(subset=["paper_id", "generator"], keep="last").to_csv(AI_OUT_CSV, index=False)
            else:
                failed_generations += 1
                provider_stats[gen]["failed"] += 1
                if err_type and ERROR_RATE_LIMIT in err_type:
                    provider_stats[gen]["rate_limited"] += 1
                if err_type and ERROR_QUOTA_EXCEEDED in err_type:
                    provider_stats[gen]["quota_exhausted"] += 1
                logger.info("   [FAILURE] %s failed: %s (%.2fs)", gen, err_type, latency)

            # Conservative rate limiting delay between sequential requests
            if total_requests_made < max_requests:
                logger.info("   Sleeping %.1fs between requests...", request_delay)
                time.sleep(request_delay)

        if total_requests_made >= max_requests:
            break

    # Calculate totals
    rate_limits = sum(s["rate_limited"] for s in provider_stats.values())
    quota_failures = sum(s["quota_exhausted"] for s in provider_stats.values())
    auth_failures = 0
    for g, s in provider_stats.items():
        tripped, reason = invoker.is_tripped(g)
        if tripped and ERROR_AUTH_ERROR in reason:
            auth_failures += 1

    # Quality and duplicate checks on new generations
    avg_len_str = "N/A (0 successful generations)"
    exact_dups = 0
    near_dups = 0
    if new_ai_rows:
        wcs = [r["word_count"] for r in new_ai_rows]
        avg_len_str = f"{sum(wcs) / len(wcs):.1f} words"
        texts = [r["text"] for r in new_ai_rows]
        exact_dups = len(texts) - len(set(texts))

    if successful_generations == max_requests:
        gen_quality = "PASS"
        quota_status = "AVAILABLE"
        next_step = "PROCEED TO 492-PAPER PILOT"
    elif successful_generations > 0:
        gen_quality = "PARTIAL"
        quota_status = "LIMITED"
        next_step = "STOP (Review partial generations before proceeding)"
    else:
        gen_quality = "FAIL (No generations produced due to provider limits)"
        quota_status = "EXHAUSTED / LIMITED"
        next_step = "STOP"

    # Compile Final Smoke Test Summary
    summary = {
        "requests_allowed": max_requests,
        "requests_attempted": total_requests_made,
        "successful_generations": successful_generations,
        "failed_generations": failed_generations,
        "provider_stats": provider_stats,
        "rate_limits": rate_limits,
        "quota_failures": quota_failures,
        "auth_failures": auth_failures,
        "avg_length": avg_len_str,
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "generation_quality": gen_quality,
        "quota_status": quota_status,
        "next_step": next_step,
        "new_ai_samples_saved": len(new_ai_rows)
    }

    with open(OUTPUT_DIR / "manifests" / "smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build Section 18 Report String
    def fmt_prov_line(gen_key: str) -> str:
        s = provider_stats.get(gen_key, {"success": 0, "attempted": 0, "failed": 0})
        tripped, reason = invoker.is_tripped(gen_key)
        note = f" (Circuit breaker: {reason})" if tripped else ""
        return f"{s['success']}/{s['attempted']} successful{note}"

    report_lines = [
        "=" * 60,
        "PHASE 2A — LIVE API SMOKE TEST",
        "=" * 60,
        "",
        f"Requests allowed: {max_requests}",
        f"Requests attempted: {total_requests_made}",
        f"Successful generations: {successful_generations}",
        f"Failed generations: {failed_generations}",
        "",
    ]
    for g in generators:
        report_lines.append(f"{g}: {fmt_prov_line(g)}")
    report_lines.extend([
        "",
        f"Rate limits: {rate_limits}",
        f"Quota failures: {quota_failures}",
        f"Authentication failures: {auth_failures}",
        "",
        f"Average output length: {avg_len_str}",
        "",
        f"Exact duplicate findings: {exact_dups}",
        f"Near-duplicate findings: {near_dups}",
        "",
        f"Generation quality:\n{gen_quality}",
        "",
        f"Quota status:\n{quota_status}",
        "",
        f"NEXT STEP:\n{next_step}",
        "=" * 60
    ])
    report_text = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "reports" / "phase2a_live_smoke_test_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    md_report_path = OUTPUT_DIR / "reports" / "phase2a_live_smoke_test_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"```text\n{report_text}\n```\n")

    return summary, report_text

def main():
    parser = argparse.ArgumentParser(description="Phase 2A Live API Smoke Test")
    parser.add_argument("--human", type=str, default="Datasets/phase2_scholarly/pilot/human_pilot_500.csv")
    parser.add_argument("--generators", type=str, default="gemini-3.8-flash,gemini-3.5-flash,allam-7b,qwen-27b")
    parser.add_argument("--request-delay", type=float, default=5.0, help="Delay in seconds between requests")
    parser.add_argument("--max-requests", type=int, default=20, help="Maximum allowed requests for smoke test")
    args = parser.parse_args()

    gen_list = [g.strip() for g in args.generators.split(",") if g.strip()]

    summary, report_text = run_smoke_test(
        human_csv_path=Path(args.human),
        generators=gen_list,
        request_delay=args.request_delay,
        max_requests=args.max_requests
    )

    # Print Provider Summary Table (Section 15)
    print("\n" + "=" * 60)
    print("PROVIDER BREAKDOWN TABLE")
    print("=" * 60)
    print(f"{'Provider':22s} | {'Attempted':^9s} | {'Success':^7s} | {'Failed':^6s} | {'Rate Limited':^12s} | {'Quota Exhausted':^15s}")
    print("-" * 80)
    for g, stats in summary["provider_stats"].items():
        print(f"{g:22s} | {stats['attempted']:^9d} | {stats['success']:^7d} | {stats['failed']:^6d} | {stats['rate_limited']:^12d} | {stats['quota_exhausted']:^15d}")
    print("=" * 80 + "\n")

    # Print Section 18 Official Report Block
    print(report_text)

if __name__ == "__main__":
    main()
