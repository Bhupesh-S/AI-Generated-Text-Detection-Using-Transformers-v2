"""
phase2_deduplicate.py
=====================
Performs exact and near-duplicate audits on human scholarly papers and AI generations.
Detects identical texts, normalized duplicates, DOI collisions, and high-similarity
abstract pairs using TF-IDF cosine similarity.

Usage:
    python phase2_deduplicate.py --input Datasets/phase2_scholarly/pilot/human_pilot_500.csv
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2_acquisition"
AUDIT_DIR = OUTPUT_DIR / "audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("Phase2_Deduplicate")

def run_exact_deduplication(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit and remove exact duplicates based on SHA-256, normalized hash, and DOI."""
    total_initial = len(df)
    
    # 1. Exact text SHA-256 duplicates
    text_dup_mask = df.duplicated(subset=["text_sha256"], keep="first")
    
    # 2. Normalized text SHA-256 duplicates
    norm_dup_mask = df.duplicated(subset=["normalized_text_sha256"], keep="first")
    
    # 3. DOI duplicates (where DOI is present and non-empty)
    doi_dup_mask = df["doi"].str.strip().ne("") & df.duplicated(subset=["doi"], keep="first")
    
    any_dup_mask = text_dup_mask | norm_dup_mask | doi_dup_mask
    duplicates_df = df[any_dup_mask].copy()
    duplicates_df["dup_reason"] = "unknown"
    duplicates_df.loc[text_dup_mask, "dup_reason"] = "exact_text_hash"
    duplicates_df.loc[norm_dup_mask & ~text_dup_mask, "dup_reason"] = "normalized_text_hash"
    duplicates_df.loc[doi_dup_mask & ~norm_dup_mask & ~text_dup_mask, "dup_reason"] = "doi_collision"
    
    clean_df = df[~any_dup_mask].copy()
    
    logger.info("Exact Dedup: %d initial, %d duplicates found, %d clean retained.",
                total_initial, len(duplicates_df), len(clean_df))
    return clean_df, duplicates_df

def run_near_duplicate_audit(
    df: pd.DataFrame,
    threshold: float = 0.88,
    text_col: str = "abstract"
) -> pd.DataFrame:
    """Audit near duplicates using TF-IDF (1-2 ngrams) cosine similarity."""
    if len(df) < 2:
        return pd.DataFrame()
        
    logger.info("Running near-duplicate audit on %d texts (threshold >= %.2f)...", len(df), threshold)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_mat = vectorizer.fit_transform(df[text_col].fillna(""))
    sim_matrix = cosine_similarity(tfidf_mat, tfidf_mat)
    
    near_dup_pairs = []
    n = len(df)
    paper_ids = df["paper_id"].values if "paper_id" in df.columns else df.index.values
    titles = df["title"].values if "title" in df.columns else [""] * n
    disciplines = df["discipline"].values if "discipline" in df.columns else [""] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                near_dup_pairs.append({
                    "idx_a": i,
                    "idx_b": j,
                    "paper_id_a": paper_ids[i],
                    "paper_id_b": paper_ids[j],
                    "discipline_a": disciplines[i],
                    "discipline_b": disciplines[j],
                    "similarity": round(sim, 4),
                    "title_a": titles[i],
                    "title_b": titles[j],
                    "category": "identical_topic" if titles[i] == titles[j] else "high_stylometric_similarity"
                })
                
    near_dup_df = pd.DataFrame(near_dup_pairs)
    logger.info("Found %d near-duplicate pairs (sim >= %.2f).", len(near_dup_df), threshold)
    return near_dup_df

def main():
    parser = argparse.ArgumentParser(description="Phase 2A Deduplication and Near-Duplicate Audit")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path to audit")
    parser.add_argument("--threshold", type=float, default=0.88, help="Cosine similarity threshold")
    parser.add_argument("--clean-output", type=str, default="", help="Path to save deduplicated CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    clean_df, exact_dups_df = run_exact_deduplication(df)
    
    # Save exact duplicate audit
    dup_audit_csv = AUDIT_DIR / "human_duplicate_audit.csv"
    exact_dups_df.to_csv(dup_audit_csv, index=False)
    
    # Run near-duplicate audit
    near_dups_df = run_near_duplicate_audit(clean_df, threshold=args.threshold)
    near_audit_csv = AUDIT_DIR / "near_duplicate_audit.csv"
    near_dups_df.to_csv(near_audit_csv, index=False)
    
    if args.clean_output:
        clean_path = Path(args.clean_output)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(clean_path, index=False)
        logger.info("Saved clean deduplicated dataset to %s", clean_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
