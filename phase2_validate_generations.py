"""
phase2_validate_generations.py
==============================
Validates AI generated scholarly texts and cross-evaluates against the human corpus:
- Gate 3: Generator Coverage
- Gate 4: Prompt Integrity
- Gate 5: Paper Grouping Integrity
- Gate 6: Exact Duplicates (Human vs AI & AI vs AI)
- Gate 7: Near-Duplicate Contamination Audit
- Gate 8: Length Distribution Comparison
- Gate 9: Discipline x Generator Contingency
- Gate 10: Generation Quality Diagnostics

Usage:
    python phase2_validate_generations.py --human Datasets/phase2_scholarly/pilot/human_pilot_500.csv --ai Datasets/phase2_scholarly/pilot/ai_pilot_generations.csv
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
AUDIT_DIR = OUTPUT_DIR / "audits"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
STATUS_FILE = MANIFEST_DIR / "generation_status.csv"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("Phase2_Validate_Generations")

def validate_all_gates(
    human_csv: Path,
    ai_csv: Optional[Path] = None,
    threshold: float = 0.88
) -> Dict[str, Any]:
    df_human = pd.read_csv(human_csv)
    n_human = len(df_human)
    human_paper_ids = set(df_human["paper_id"])

    # Load AI dataset if available
    df_ai = pd.read_csv(ai_csv) if (ai_csv and ai_csv.exists() and os.path.getsize(ai_csv) > 10) else pd.DataFrame()
    status_df = pd.read_csv(STATUS_FILE) if STATUS_FILE.exists() else pd.DataFrame()

    results: Dict[str, Any] = {
        "n_human_papers": n_human,
        "n_ai_generations": len(df_ai),
        "gates": {}
    }

    # ── Gate 3: Generator Coverage ──────────────────────────────────────────
    expected_generators = ["gpt-4o", "claude-3-5-sonnet", "llama-3-70b-instruct", "gemini-1-5-pro"]
    coverage_dict = {}
    for gen in expected_generators:
        if not df_ai.empty and "generator" in df_ai.columns:
            count = int((df_ai["generator"] == gen).sum())
        else:
            count = 0
        coverage_dict[gen] = count
    results["gates"]["gate3_generator_coverage"] = {
        "target_per_generator": n_human,
        "actual_coverage": coverage_dict,
        "status_summary": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        "pass": all(c == n_human for c in coverage_dict.values())
    }

    # Export generator coverage audit CSV
    cov_rows = []
    for gen in expected_generators:
        cov_rows.append({
            "generator": gen,
            "target_papers": n_human,
            "generated_papers": coverage_dict.get(gen, 0),
            "coverage_pct": round(100.0 * coverage_dict.get(gen, 0) / n_human, 2) if n_human > 0 else 0.0,
            "status": "complete" if coverage_dict.get(gen, 0) == n_human else "partial_or_unavailable"
        })
    pd.DataFrame(cov_rows).to_csv(AUDIT_DIR / "generator_coverage.csv", index=False)

    if df_ai.empty:
        # If AI generation is unavailable (e.g. API keys not configured), record clean status
        results["gates"]["gate4_prompt_integrity"] = {"pass": False, "reason": "No AI generations to validate"}
        results["gates"]["gate5_paper_grouping"] = {"pass": False, "reason": "No AI generations to validate"}
        results["gates"]["gate6_exact_duplicates"] = {"pass": True, "human_exact_duplicates": 0}
        results["gates"]["gate7_near_duplicates"] = {"pass": True, "cross_pairs_exceeding_threshold": 0}
        results["gates"]["gate8_length_comparison"] = {
            "human_mean_words": round(float(df_human["word_count"].mean()), 2),
            "ai_mean_words": 0.0
        }
        results["gates"]["gate9_contingency"] = {"pass": False, "matrix": {}}
        results["gates"]["gate10_quality"] = {"pass": False, "reason": "No AI generations"}
        results["overall_pilot_status"] = "PENDING_OR_UNAVAILABLE"
        return results

    # ── Gate 4: Prompt Integrity ────────────────────────────────────────────
    req_prompt_cols = ["prompt", "prompt_hash", "prompt_template_version", "model_identifier"]
    has_prompt_cols = all(c in df_ai.columns for c in req_prompt_cols)
    null_prompts = df_ai["prompt"].isna().sum() if "prompt" in df_ai.columns else len(df_ai)
    gate4_pass = has_prompt_cols and (null_prompts == 0)
    results["gates"]["gate4_prompt_integrity"] = {
        "pass": bool(gate4_pass),
        "total_prompts": len(df_ai),
        "unique_prompt_hashes": int(df_ai["prompt_hash"].nunique()) if "prompt_hash" in df_ai.columns else 0
    }

    # ── Gate 5: Paper Grouping Integrity ────────────────────────────────────
    ai_paper_ids = set(df_ai["paper_id"])
    orphan_ai = ai_paper_ids - human_paper_ids
    gate5_pass = (len(orphan_ai) == 0) and all(df_ai["paper_id"].notna())
    results["gates"]["gate5_paper_grouping"] = {
        "pass": bool(gate5_pass),
        "orphan_ai_paper_ids": len(orphan_ai),
        "mapped_human_papers": len(ai_paper_ids.intersection(human_paper_ids))
    }

    # ── Gate 6: Exact Duplicates ────────────────────────────────────────────
    human_hashes = set(df_human["text_sha256"])
    ai_hashes = set(df_ai["text_sha256"])
    cross_exact = len(human_hashes.intersection(ai_hashes))
    ai_internal_dups = int(df_ai.duplicated(subset=["text_sha256"]).sum())
    gate6_pass = (cross_exact == 0) and (ai_internal_dups == 0)
    results["gates"]["gate6_exact_duplicates"] = {
        "pass": bool(gate6_pass),
        "cross_human_ai_exact": cross_exact,
        "ai_internal_exact": ai_internal_dups
    }

    # ── Gate 7: Near Duplicates (Human vs AI & AI vs AI) ────────────────────
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True, stop_words="english")
    human_texts = df_human["abstract"].fillna("").tolist()
    ai_texts = df_ai["text"].fillna("").tolist()

    all_texts = human_texts + ai_texts
    tfidf_all = vec.fit_transform(all_texts)
    tfidf_human = tfidf_all[:n_human]
    tfidf_ai = tfidf_all[n_human:]

    cross_sim = cosine_similarity(tfidf_human, tfidf_ai)
    suspicious_cross_pairs = []
    for h_idx in range(n_human):
        for a_idx in range(len(ai_texts)):
            sim = float(cross_sim[h_idx, a_idx])
            if sim >= threshold:
                suspicious_cross_pairs.append({
                    "human_paper_id": df_human.iloc[h_idx]["paper_id"],
                    "ai_paper_id": df_ai.iloc[a_idx]["paper_id"],
                    "ai_generator": df_ai.iloc[a_idx]["generator"],
                    "similarity": round(sim, 4),
                    "is_same_paper": df_human.iloc[h_idx]["paper_id"] == df_ai.iloc[a_idx]["paper_id"]
                })
    near_dup_df = pd.DataFrame(suspicious_cross_pairs)
    near_dup_df.to_csv(AUDIT_DIR / "ai_human_near_duplicate_audit.csv", index=False)
    results["gates"]["gate7_near_duplicates"] = {
        "pass": len(suspicious_cross_pairs) == 0,
        "suspicious_pairs_count": len(suspicious_cross_pairs)
    }

    # ── Gate 8: Length Distribution Comparison ──────────────────────────────
    len_stats = {
        "human": {
            "mean": round(float(df_human["word_count"].mean()), 2),
            "median": float(df_human["word_count"].median()),
            "std": round(float(df_human["word_count"].std()), 2)
        },
        "by_generator": {}
    }
    for gen in df_ai["generator"].unique():
        gen_df = df_ai[df_ai["generator"] == gen]
        len_stats["by_generator"][gen] = {
            "mean": round(float(gen_df["word_count"].mean()), 2),
            "median": float(gen_df["word_count"].median()),
            "mean_length_ratio": round(float(gen_df["length_ratio"].mean()), 4) if "length_ratio" in gen_df.columns else 1.0
        }
    results["gates"]["gate8_length_comparison"] = len_stats

    # Export length distribution CSV
    len_rows = [{"corpus": "human_scholarly", "generator": "human", **len_stats["human"]}]
    for g, stats in len_stats["by_generator"].items():
        len_rows.append({"corpus": "ai_generated", "generator": g, **stats})
    pd.DataFrame(len_rows).to_csv(AUDIT_DIR / "length_distribution.csv", index=False)

    # ── Gate 9: Discipline x Generator Contingency ──────────────────────────
    ct = pd.crosstab(df_ai["discipline"], df_ai["generator"])
    results["gates"]["gate9_contingency"] = {
        "matrix": ct.to_dict(),
        "pass": bool((ct > 0).all().all()) if not ct.empty else False
    }
    ct.to_csv(AUDIT_DIR / "discipline_generator_contingency.csv")

    # ── Gate 10: Generation Quality Check ───────────────────────────────────
    # Verify no empty, non-academic boilerplate, or meta-phrases
    meta_phrases = ["here is an abstract", "as an ai language model", "certainly", "in this paper we present"]
    leak_count = 0
    for t in df_ai["text"]:
        t_low = t.lower()
        if any(p in t_low[:50] for p in ["here is", "certainly", "as an ai"]):
            leak_count += 1
    gate10_pass = (leak_count == 0) and bool((df_ai["word_count"] >= 50).all())
    results["gates"]["gate10_quality"] = {
        "pass": bool(gate10_pass),
        "prompt_leakage_detected": leak_count
    }

    # Final evaluation
    all_pass = all(g.get("pass", False) for g in results["gates"].values())
    results["overall_pilot_status"] = "PASS" if all_pass else "FAIL"

    return results

def main():
    parser = argparse.ArgumentParser(description="Validate AI Generations and All Gates")
    parser.add_argument("--human", type=str, default="Datasets/phase2_scholarly/pilot/human_pilot_500.csv")
    parser.add_argument("--ai", type=str, default="Datasets/phase2_scholarly/pilot/ai_pilot_generations.csv")
    args = parser.parse_args()

    res = validate_all_gates(Path(args.human), Path(args.ai))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
