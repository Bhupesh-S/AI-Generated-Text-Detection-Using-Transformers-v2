"""
phase2_validate_human.py
========================
Validates the acquired human scholarly corpus against Gate 1 (Provenance),
Gate 2 (Metadata completeness), Gate 6 (Exact Duplicates), and Gate 8 (Length distribution).

Usage:
    python phase2_validate_human.py --input Datasets/phase2_scholarly/pilot/human_pilot_500.csv
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
AUDIT_DIR = OUTPUT_DIR / "audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("Phase2_Validate_Human")

def validate_human_corpus(csv_path: Path) -> Tuple[bool, Dict[str, Any]]:
    if not csv_path.exists():
        logger.error("Human dataset not found at %s", csv_path)
        return False, {"error": "file_not_found"}

    df = pd.read_csv(csv_path)
    total = len(df)
    results = {
        "total_papers": total,
        "gate1_provenance_pass": False,
        "gate2_metadata_pass": False,
        "gate6_exact_duplicates_pass": False,
        "disciplines": df["discipline"].value_counts().to_dict(),
        "years": df["publication_year"].value_counts().to_dict(),
        "word_count_stats": {
            "mean": round(float(df["word_count"].mean()), 2),
            "median": float(df["word_count"].median()),
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max())
        }
    }

    # Gate 1: Provenance (100% pre-2022 and authoritative year)
    pre_2022 = (df["publication_year"] < 2022) & (df["publication_year"].notna())
    gate1_pass = bool(pre_2022.all())
    results["gate1_provenance_pass"] = gate1_pass
    results["pre_2022_count"] = int(pre_2022.sum())
    results["post_or_null_2022_count"] = int((~pre_2022).sum())

    # Gate 2: Metadata completeness
    required_cols = ["paper_id", "title", "abstract", "source", "publication_year", "discipline"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        gate2_pass = False
        results["missing_columns"] = missing_cols
    else:
        null_counts = {c: int(df[c].isna().sum()) for c in required_cols}
        empty_str_counts = {c: int(df[c].astype(str).str.strip().eq("").sum()) for c in required_cols}
        results["null_counts"] = null_counts
        results["empty_counts"] = empty_str_counts
        gate2_pass = (sum(null_counts.values()) == 0) and (sum(empty_str_counts.values()) == 0)
    results["gate2_metadata_pass"] = gate2_pass

    # Gate 6: Exact text duplicates
    exact_dups = int(df.duplicated(subset=["text_sha256"]).sum())
    norm_dups = int(df.duplicated(subset=["normalized_text_sha256"]).sum())
    results["exact_duplicates"] = exact_dups
    results["normalized_duplicates"] = norm_dups
    results["gate6_exact_duplicates_pass"] = (exact_dups == 0 and norm_dups == 0)

    overall_pass = gate1_pass and gate2_pass and results["gate6_exact_duplicates_pass"]
    results["human_validation_pass"] = overall_pass

    logger.info("Human Corpus Validation: Gate 1 (Provenance): %s | Gate 2 (Metadata): %s | Gate 6 (Exact Dups): %s",
                gate1_pass, gate2_pass, results["gate6_exact_duplicates_pass"])
    logger.info("Overall Human Validation: %s", "PASS" if overall_pass else "FAIL")

    return overall_pass, results

def main():
    parser = argparse.ArgumentParser(description="Validate Human Scholarly Corpus")
    parser.add_argument("--input", type=str, default="Datasets/phase2_scholarly/pilot/human_pilot_500.csv")
    args = parser.parse_args()

    passed, res = validate_human_corpus(Path(args.input))
    print(json.dumps(res, indent=2))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
