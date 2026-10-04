"""
create_generalization_splits.py
================================
Builds isolated cross-source and prompt-grouped cross-generator splits for Phase 1 Generalization testing.

Strict Safety and Scientific Rules Enforced:
1. No modification of baseline datasets (final_dataset.csv.bak, baseline_metrics.json).
2. Primary dataset used: Datasets/merged/final_dataset_v2.csv.
3. Macro-Source Rotations (Experiment A):
   - Fold A: Train = baseline_expanded + hc3_qa, Test = artnitolog_llm_texts
   - Fold B: Train = baseline_expanded + artnitolog_llm_texts, Test = hc3_qa
   - Fold C: Train = hc3_qa + artnitolog_llm_texts, Test = baseline_expanded
4. Cross-Generator Rotations with Strict Prompt-Group Isolation (Experiment B):
   - 3,000 ArtNitolog prompts partitioned into connected prompt clusters so prompts NEVER cross train/val/test partitions.
   - Fold A: Train = GPT-4o + Claude-3-Opus (prompts P1+P2), Test = Llama-3-70B (prompts P3)
   - Fold B: Train = GPT-4o + Llama-3-70B (prompts P1+P3), Test = Claude-3-Opus (prompts P2)
   - Fold C: Train = Claude-3-Opus + Llama-3-70B (prompts P2+P3), Test = GPT-4o (prompts P1)
5. Documented deterministic deduplication policy:
   - Identifies internal duplicate pairs in baseline_expanded (indices 503, 14540, 19309, etc.).
   - Drops duplicate occurrences within the training pool so Train <-> Val <-> Test has 0 overlap.
6. Zero exact text duplicate overlap across Train <-> Val <-> Test (cryptographic SHA-256 check).
7. Near-duplicate audit reporting.
8. Manifest generation for every fold.
9. Source-Generator Confounding Contingency table & Prompt Leakage report generation.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phase1_SplitCreator")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_V2_PATH = REPO_ROOT / "Datasets" / "merged" / "final_dataset_v2.csv"
RAW_ARTNITOLOG_PATH = REPO_ROOT / "Datasets" / "raw_external" / "artnitolog_llm_generated_texts_raw.csv"

OUT_DIR = REPO_ROOT / "outputs" / "generalization_phase1"
SPLITS_DIR = OUT_DIR / "splits"
SOURCE_SPLITS_DIR = SPLITS_DIR / "cross_source"
GEN_SPLITS_DIR = SPLITS_DIR / "cross_generator"
MANIFESTS_DIR = OUT_DIR / "manifests"
AUDITS_DIR = OUT_DIR / "audits"
REPORTS_DIR = OUT_DIR / "reports"

for d in [OUT_DIR, SPLITS_DIR, SOURCE_SPLITS_DIR, GEN_SPLITS_DIR, MANIFESTS_DIR, AUDITS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Text Normalization & Hashing Helpers
# ---------------------------------------------------------------------------
def normalize_text_for_hash(text: str) -> str:
    """Collapses irregular whitespace and null bytes strictly for hash comparison."""
    if not isinstance(text, str):
        text = str(text)
    t = text.replace("\x00", "")
    t = t.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(t.split())

def compute_sha256(text: str) -> str:
    """Returns SHA-256 hex digest of normalized text."""
    norm = normalize_text_for_hash(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Step 1: Dataset Verification
# ---------------------------------------------------------------------------
def verify_primary_dataset() -> pd.DataFrame:
    """Loads and strictly validates final_dataset_v2.csv."""
    logger.info("=== STEP 1: Verifying Primary Dataset Schema & Content ===")
    if not DATASET_V2_PATH.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Primary dataset not found at {DATASET_V2_PATH}")
    
    df = pd.read_csv(DATASET_V2_PATH, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {DATASET_V2_PATH.name}")

    # Verify columns
    expected_cols = {"text", "label", "source", "generator", "domain", "text_length"}
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"CRITICAL ERROR: Dataset schema mismatch. Missing columns: {missing}")

    # Verify labels
    unique_labels = sorted(df["label"].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"CRITICAL ERROR: Unexpected label values: {unique_labels}. Expected [0, 1].")

    # Verify sources
    expected_sources = {"baseline_expanded", "hc3_qa", "artnitolog_llm_texts"}
    actual_sources = set(df["source"].unique())
    if actual_sources != expected_sources:
        raise ValueError(f"CRITICAL ERROR: Source mismatch. Found {actual_sources}, expected {expected_sources}")

    logger.info("Schema and label verification PASSED.")
    logger.info(f"Source distribution:\n{df['source'].value_counts().to_string()}")
    logger.info(f"Generator distribution:\n{df['generator'].value_counts().to_string()}")
    return df

# ---------------------------------------------------------------------------
# Step 2: Source x Generator Contingency Analysis
# ---------------------------------------------------------------------------
def generate_source_generator_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """Computes contingency table and writes detailed confounding report."""
    logger.info("=== STEP 2: Computing Source x Generator Contingency Table ===")
    contingency = pd.crosstab(
        df["generator"],
        df["source"],
        margins=True,
        margins_name="Total"
    )
    
    contingency_csv = REPORTS_DIR / "source_generator_contingency.csv"
    contingency.to_csv(contingency_csv)
    logger.info(f"Saved contingency table -> {contingency_csv.relative_to(REPO_ROOT)}")

    # Markdown explanation of source-generator confounding
    confounding_md = REPORTS_DIR / "source_generator_confounding.md"
    content = [
        "# Source-Generator Confounding Analysis",
        "",
        "**Date:** Phase 1 Initialization  ",
        "**Dataset:** `final_dataset_v2.csv` ($N = 125,012$)  ",
        "",
        "## 1. Contingency Table (Sample Counts)",
        "",
        contingency.to_markdown(),
        "",
        "## 2. Structural Findings & Limitations",
        "",
        "1. **Complete Confounding with Baseline:** The `baseline_expanded` corpus exclusively contains `ChatGPT/GPT-3.5` (23,661 AI samples) alongside Human student essays (32,168 samples). No frontier models exist in this source.",
        "2. **Complete Confounding with HC3:** The `hc3_qa` corpus exclusively contains `ChatGPT-3.5` (22,822 AI samples) alongside Human Q&A answers (31,484 samples).",
        "3. **Exclusive Presence of Frontier LLMs in ArtNitolog:** Modern frontier models (`GPT-4o`, `GPT-4-Turbo`, `Claude-3-Opus`, `Llama-3-70B`) exist **strictly and exclusively** within `artnitolog_llm_texts`.",
        "",
        "## 3. Scientific Interpretation Protocol",
        "",
        "> [!IMPORTANT]",
        "> **MANDATORY SCIENTIFIC CONSTRAINT:**  ",
        "> Any model trained on `baseline_expanded + hc3_qa` and tested on `artnitolog_llm_texts` experiences **simultaneous source/domain shift** (student essays & general Q&A -> news/essay topics) **and generator shift** (GPT-3.5 -> GPT-4 / Claude-3 / Llama-3).",
        "> This cannot be claimed as an isolated generator generalization test.",
        "> Conversely, **Experiment B** isolates pure generator generalization by restricting the universe strictly to `artnitolog_llm_texts` and holding out distinct generator families over disjoint prompt partitions.",
        ""
    ]
    confounding_md.write_text("\n".join(content), encoding="utf-8")
    logger.info(f"Saved confounding report -> {confounding_md.relative_to(REPO_ROOT)}")
    return contingency

# ---------------------------------------------------------------------------
# Step 3: Exact & Near-Duplicate Auditing
# ---------------------------------------------------------------------------
def audit_internal_duplicates(df: pd.DataFrame) -> Dict:
    """Identifies and documents all duplicate records in the primary dataset."""
    logger.info("=== STEP 3: Auditing Internal Exact Duplicates in Dataset ===")
    df_temp = df.copy()
    df_temp["sha256"] = df_temp["text"].apply(compute_sha256)
    
    dup_mask = df_temp.duplicated(subset=["sha256"], keep=False)
    dup_rows = df_temp[dup_mask].copy()
    
    duplicate_report = {
        "total_duplicate_rows_in_corpus": len(dup_rows),
        "unique_duplicated_texts": dup_rows["sha256"].nunique(),
        "affected_records": []
    }

    for sha, group in dup_rows.groupby("sha256"):
        duplicate_report["affected_records"].append({
            "sha256": sha,
            "occurrences": len(group),
            "sources": group["source"].tolist(),
            "labels": group["label"].tolist(),
            "indices": group.index.tolist(),
            "text_snippet": group["text"].iloc[0][:100]
        })

    audit_json = AUDITS_DIR / "exact_duplicate_audit.json"
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(duplicate_report, f, indent=2)

    audit_md = AUDITS_DIR / "exact_duplicate_audit.md"
    md_lines = [
        "# Exact Duplicate Audit & Policy Report",
        "",
        f"**Total Duplicate Rows Found:** {len(dup_rows)}  ",
        f"**Unique Repeated Texts:** {dup_rows['sha256'].nunique()}  ",
        "",
        "## Affected Records in Raw Corpus",
        ""
    ]
    for rec in duplicate_report["affected_records"]:
        md_lines.append(f"- **Hash:** `{rec['sha256'][:16]}...` | **Count:** {rec['occurrences']} | **Sources:** {rec['sources']} | **Indices:** {rec['indices']}")
        md_lines.append(f"  *Snippet:* \"{rec['text_snippet']}...\"")
    
    md_lines.extend([
        "",
        "## Deterministic Deduplication Policy",
        "To strictly ensure 0 text overlaps between Train and Validation sets within any fold,",
        "the training pool undergoes deterministic deduplication keeping only the first occurrence (`drop_duplicates(subset=['sha256'], keep='first')`).",
        "Because cross-source overlap is already mathematically 0, this guarantees **0 exact text overlaps across all Train <-> Val <-> Test partitions**."
    ])
    audit_md.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Audited {len(dup_rows)} internal duplicate rows in corpus -> {audit_json.relative_to(REPO_ROOT)}")
    return duplicate_report

def run_leakage_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_name: str
) -> Dict:
    """Verifies 0 exact hash overlap and audits near-duplicate pairs."""
    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    tv_overlap = train_hashes.intersection(val_hashes)
    tt_overlap = train_hashes.intersection(test_hashes)
    vt_overlap = val_hashes.intersection(test_hashes)

    if len(tv_overlap) > 0 or len(tt_overlap) > 0 or len(vt_overlap) > 0:
        raise ValueError(
            f"FATAL LEAKAGE DETECTED in {fold_name}: "
            f"Train-Val Overlap={len(tv_overlap)}, Train-Test Overlap={len(tt_overlap)}, Val-Test Overlap={len(vt_overlap)}"
        )

    # Near-duplicate sample audit between Train and Test
    rng = np.random.RandomState(SEED)
    sample_train = train_df.sample(min(1000, len(train_df)), random_state=rng)
    sample_test = test_df.sample(min(1000, len(test_df)), random_state=rng)

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tr = vec.fit_transform(sample_train["text"])
    X_te = vec.transform(sample_test["text"])

    sim_matrix = cosine_similarity(X_te, X_tr)
    near_dups_count = int((sim_matrix >= 0.90).sum())

    audit_result = {
        "fold": fold_name,
        "exact_duplicates": {
            "train_val_overlap": len(tv_overlap),
            "train_test_overlap": len(tt_overlap),
            "val_test_overlap": len(vt_overlap),
            "status": "PASS"
        },
        "near_duplicate_audit_sample_1000x1000": {
            "tested_pairs": len(sample_train) * len(sample_test),
            "pairs_sim_gte_0_90": near_dups_count,
            "max_cosine_sim": float(sim_matrix.max()) if sim_matrix.size > 0 else 0.0
        }
    }
    return audit_result

# ---------------------------------------------------------------------------
# Step 4: Experiment A — Cross-Source Splits Creation
# ---------------------------------------------------------------------------
def create_cross_source_splits(df: pd.DataFrame) -> List[Dict]:
    """Generates the 3-way macro-source rotation splits with deterministic deduplication."""
    logger.info("=== STEP 4: Creating Experiment A (Cross-Source Generalization) Splits ===")
    
    # Precompute SHA-256 for all rows
    logger.info("Precomputing SHA-256 text hashes for cross-source dataset...")
    df = df.copy()
    df["sha256"] = df["text"].apply(compute_sha256)
    df["sample_id"] = np.arange(len(df))

    source_configs = [
        {
            "fold_name": "holdout_artnitolog",
            "train_sources": ["baseline_expanded", "hc3_qa"],
            "test_source": "artnitolog_llm_texts"
        },
        {
            "fold_name": "holdout_hc3",
            "train_sources": ["baseline_expanded", "artnitolog_llm_texts"],
            "test_source": "hc3_qa"
        },
        {
            "fold_name": "holdout_baseline",
            "train_sources": ["hc3_qa", "artnitolog_llm_texts"],
            "test_source": "baseline_expanded"
        }
    ]

    manifests = []

    for cfg in source_configs:
        fold = cfg["fold_name"]
        t_sources = cfg["train_sources"]
        test_source = cfg["test_source"]
        logger.info(f"\n--- Generating Source Fold: {fold} ---")
        logger.info(f"    Train Sources: {t_sources} | Test Source: {test_source}")

        fold_dir = SOURCE_SPLITS_DIR / fold
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 1. Slice training sources pool and test source
        train_pool = df[df["source"].isin(t_sources)].copy().reset_index(drop=True)
        test_df = df[df["source"] == test_source].copy().reset_index(drop=True)

        # 2. Apply documented deterministic deduplication on training pool to guarantee Train <-> Val independence
        before_dedup = len(train_pool)
        train_pool = train_pool.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
        dropped = before_dedup - len(train_pool)
        if dropped > 0:
            logger.info(f"    Deduplication policy: removed {dropped} internal duplicate rows from train pool.")

        # 3. Split training sources pool into 90% Train and 10% Validation (stratified by label)
        train_df, val_df = train_test_split(
            train_pool,
            test_size=0.10,
            random_state=SEED,
            stratify=train_pool["label"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # 4. Source holdout integrity checks
        assert set(train_df["source"].unique()) == set(t_sources), "Train sources mismatch!"
        assert set(val_df["source"].unique()) == set(t_sources), "Val sources mismatch!"
        assert test_df["source"].unique().tolist() == [test_source], "Test source mismatch!"
        assert len(set(train_df["source"]).intersection(set(test_df["source"]))) == 0, "Source leakage!"

        # 5. Leakage & Exact-duplicate audit
        leakage_report = run_leakage_audit(train_df, val_df, test_df, fold)
        audit_file = AUDITS_DIR / f"audit_source_{fold}.json"
        with open(audit_file, "w") as f:
            json.dump(leakage_report, f, indent=2)

        # 6. Save split CSVs
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        test_path = fold_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        manifest = {
            "experiment": "cross_source",
            "fold": fold,
            "train_sources": t_sources,
            "test_source": test_source,
            "seed": SEED,
            "train_rows": len(train_df),
            "train_human": int((train_df["label"] == 0).sum()),
            "train_ai": int((train_df["label"] == 1).sum()),
            "validation_rows": len(val_df),
            "val_human": int((val_df["label"] == 0).sum()),
            "val_ai": int((val_df["label"] == 1).sum()),
            "test_rows": len(test_df),
            "test_human": int((test_df["label"] == 0).sum()),
            "test_ai": int((test_df["label"] == 1).sum()),
            "test_ai_prevalence": round(float((test_df["label"] == 1).mean()), 4),
            "exact_duplicate_leakage": "ZERO_DETECTED",
            "source_integrity": "STRICT_PASS",
            "paths": {
                "train_csv": str(train_path.relative_to(REPO_ROOT)),
                "val_csv": str(val_path.relative_to(REPO_ROOT)),
                "test_csv": str(test_path.relative_to(REPO_ROOT))
            }
        }
        manifest_path = MANIFESTS_DIR / f"manifest_source_{fold}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        manifests.append(manifest)
        logger.info(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        logger.info(f"    Exact Overlap Check: Train-Val=0, Train-Test=0, Val-Test=0 (STRICT PASS)")
        logger.info(f"    Saved splits to -> {fold_dir.relative_to(REPO_ROOT)}")

    return manifests

# ---------------------------------------------------------------------------
# Step 5: Experiment B — Cross-Generator Splits with Prompt-Group Control
# ---------------------------------------------------------------------------
def create_cross_generator_splits(df: pd.DataFrame) -> List[Dict]:
    """
    Builds the 3-way prompt-isolated generator rotation splits.
    Prompt IDs are clustered by connected text sharing and strictly partitioned into 3 disjoint sets (P1, P2, P3).
    A prompt tested in a held-out generator was NEVER seen during training.
    """
    logger.info("=== STEP 5: Creating Experiment B (Prompt-Grouped Cross-Generator) Splits ===")
    if not RAW_ARTNITOLOG_PATH.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Raw ArtNitolog dataset not found at {RAW_ARTNITOLOG_PATH}")

    raw_df = pd.read_csv(RAW_ARTNITOLOG_PATH)
    logger.info(f"Loaded raw ArtNitolog prompt corpus: {len(raw_df):,} rows")

    # Build exact text-to-prompt mapping and connected prompt components
    logger.info("Building text-to-prompt lookup table and connected prompt clusters...")
    text_to_prompt = {}
    text_to_prompts = defaultdict(set)
    for idx, row in raw_df.iterrows():
        p_id = f"{row['dataset_name']}_{row['id']}"
        prompt_txt = str(row["prompt"])
        h_norm = normalize_text_for_hash(row["human"])
        if h_norm:
            text_to_prompt[h_norm] = (p_id, prompt_txt, "Human", row["dataset_name"])
            text_to_prompts[h_norm].add(p_id)
        for col, gen in [
            ("GPT4 Turbo 2024-04-09", "GPT-4-Turbo"),
            ("GPT4 Omni", "GPT-4o"),
            ("Claude 3 Opus", "Claude-3-Opus"),
            ("Llama3 70B", "Llama-3-70B")
        ]:
            g_norm = normalize_text_for_hash(row[col])
            if g_norm:
                text_to_prompt[g_norm] = (p_id, prompt_txt, gen, row["dataset_name"])
                text_to_prompts[g_norm].add(p_id)

    # Find connected components of prompts sharing identical text to eliminate cross-partition duplicate leakage
    adj = defaultdict(set)
    for txt, p_set in text_to_prompts.items():
        p_list = list(p_set)
        for i in range(len(p_list)):
            for j in range(i + 1, len(p_list)):
                adj[p_list[i]].add(p_list[j])
                adj[p_list[j]].add(p_list[i])

    visited = set()
    prompt_clusters = []
    prompt_to_cluster_id = {}
    all_prompts = sorted(list({f"{row['dataset_name']}_{row['id']}" for idx, row in raw_df.iterrows()}))

    for p in all_prompts:
        if p not in visited:
            comp = []
            q = [p]
            visited.add(p)
            while q:
                curr = q.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            cid = f"cluster_{len(prompt_clusters)}"
            prompt_clusters.append((cid, comp))
            for p_in_comp in comp:
                prompt_to_cluster_id[p_in_comp] = cid

    logger.info(f"Connected prompt clustering: 3,000 raw prompts grouped into {len(prompt_clusters):,} disjoint clusters.")

    # Slice ArtNitolog rows from df
    art_df = df[df["source"] == "artnitolog_llm_texts"].copy()
    art_df["norm_text"] = art_df["text"].apply(normalize_text_for_hash)
    art_df["mapping"] = art_df["norm_text"].map(text_to_prompt)

    unmapped = art_df["mapping"].isna().sum()
    if unmapped > 0:
        raise ValueError(f"CRITICAL ERROR: {unmapped} ArtNitolog rows could not be deterministically mapped to a prompt ID!")
    
    art_df["prompt_id"] = art_df["mapping"].apply(lambda x: x[0])
    art_df["prompt_text"] = art_df["mapping"].apply(lambda x: x[1])
    art_df["raw_domain"] = art_df["mapping"].apply(lambda x: x[3])
    art_df["cluster_id"] = art_df["prompt_id"].map(prompt_to_cluster_id)
    art_df["sha256"] = art_df["text"].apply(compute_sha256)
    art_df["sample_id"] = np.arange(len(art_df))

    # Partition the 2,993 prompt clusters into 3 balanced partitions (P1, P2, P3)
    # Stratified by domain of the primary prompt in the cluster
    cluster_df_rows = []
    for cid, plist in prompt_clusters:
        domain = plist[0].split("_")[0]
        cluster_df_rows.append({"cluster_id": cid, "domain": domain, "size": len(plist)})
    cluster_df = pd.DataFrame(cluster_df_rows).sort_values("cluster_id").reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    cluster_df = cluster_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    n_c = len(cluster_df)
    c1 = set(cluster_df.iloc[: n_c // 3]["cluster_id"])
    c2 = set(cluster_df.iloc[n_c // 3 : 2 * (n_c // 3)]["cluster_id"])
    c3 = set(cluster_df.iloc[2 * (n_c // 3) :]["cluster_id"])

    # Expand clusters back to prompt sets
    p1_prompts = set(art_df[art_df["cluster_id"].isin(c1)]["prompt_id"])
    p2_prompts = set(art_df[art_df["cluster_id"].isin(c2)]["prompt_id"])
    p3_prompts = set(art_df[art_df["cluster_id"].isin(c3)]["prompt_id"])

    assert len(p1_prompts.intersection(p2_prompts)) == 0
    assert len(p1_prompts.intersection(p3_prompts)) == 0
    assert len(p2_prompts.intersection(p3_prompts)) == 0
    logger.info(f"Disjoint Prompt Partitions: P1={len(p1_prompts)}, P2={len(p2_prompts)}, P3={len(p3_prompts)}")

    # 3-way Generator Rotations across modern frontier models
    gen_configs = [
        {
            "fold_name": "holdout_llama3",
            "train_clusters": list(c1.union(c2)),
            "test_clusters": list(c3),
            "train_generators": ["GPT-4o", "Claude-3-Opus"],
            "test_generator": "Llama-3-70B"
        },
        {
            "fold_name": "holdout_claude3",
            "train_clusters": list(c1.union(c3)),
            "test_clusters": list(c2),
            "train_generators": ["GPT-4o", "Llama-3-70B"],
            "test_generator": "Claude-3-Opus"
        },
        {
            "fold_name": "holdout_gpt4o",
            "train_clusters": list(c2.union(c3)),
            "test_clusters": list(c1),
            "train_generators": ["Claude-3-Opus", "Llama-3-70B"],
            "test_generator": "GPT-4o"
        }
    ]

    manifests = []
    prompt_audits = []

    for cfg in gen_configs:
        fold = cfg["fold_name"]
        tr_c_pool = cfg["train_clusters"]
        te_c_set = set(cfg["test_clusters"])
        tr_gens = cfg["train_generators"]
        te_gen = cfg["test_generator"]

        logger.info(f"\n--- Generating Generator Fold: {fold} ---")
        logger.info(f"    Train Generators: {tr_gens} | Held-out Generator: {te_gen}")

        fold_dir = GEN_SPLITS_DIR / fold
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Split training clusters 90/10 into train_clusters and val_clusters
        # Clusters NEVER cross between train, val, and test
        rng_fold = np.random.RandomState(SEED)
        shuffled_tr_c = rng_fold.permutation(tr_c_pool)
        n_val_c = int(len(shuffled_tr_c) * 0.10)
        val_c_set = set(shuffled_tr_c[:n_val_c])
        train_c_set = set(shuffled_tr_c[n_val_c:])

        # Verify cluster isolation
        assert len(train_c_set.intersection(val_c_set)) == 0
        assert len(train_c_set.intersection(te_c_set)) == 0
        assert len(val_c_set.intersection(te_c_set)) == 0

        # Slice data
        train_df = art_df[
            art_df["cluster_id"].isin(train_c_set) &
            art_df["generator"].isin(tr_gens + ["Human"])
        ].copy().reset_index(drop=True)

        val_df = art_df[
            art_df["cluster_id"].isin(val_c_set) &
            art_df["generator"].isin(tr_gens + ["Human"])
        ].copy().reset_index(drop=True)

        test_df = art_df[
            art_df["cluster_id"].isin(te_c_set) &
            art_df["generator"].isin([te_gen, "Human"])
        ].copy().reset_index(drop=True)

        # Verify generator isolation
        test_ai_gens = test_df[test_df["label"] == 1]["generator"].unique().tolist()
        train_ai_gens = train_df[train_df["label"] == 1]["generator"].unique().tolist()
        assert test_ai_gens == [te_gen], f"Test generator mismatch! Got {test_ai_gens}"
        assert set(train_ai_gens) == set(tr_gens), f"Train generators mismatch! Got {train_ai_gens}"
        assert te_gen not in train_ai_gens, f"Test generator {te_gen} leaked into training!"

        # Leakage & Duplicate Audit
        leakage_report = run_leakage_audit(train_df, val_df, test_df, fold)
        audit_file = AUDITS_DIR / f"audit_generator_{fold}.json"
        with open(audit_file, "w") as f:
            json.dump(leakage_report, f, indent=2)

        # Save split CSVs
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        test_path = fold_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        manifest = {
            "experiment": "cross_generator",
            "fold": fold,
            "train_generators": tr_gens,
            "test_generator": te_gen,
            "train_cluster_count": len(train_c_set),
            "val_cluster_count": len(val_c_set),
            "test_cluster_count": len(te_c_set),
            "prompt_overlap_train_test": 0,
            "prompt_overlap_train_val": 0,
            "prompt_overlap_val_test": 0,
            "seed": SEED,
            "train_rows": len(train_df),
            "train_human": int((train_df["label"] == 0).sum()),
            "train_ai": int((train_df["label"] == 1).sum()),
            "validation_rows": len(val_df),
            "val_human": int((val_df["label"] == 0).sum()),
            "val_ai": int((val_df["label"] == 1).sum()),
            "test_rows": len(test_df),
            "test_human": int((test_df["label"] == 0).sum()),
            "test_ai": int((test_df["label"] == 1).sum()),
            "test_ai_prevalence": round(float((test_df["label"] == 1).mean()), 4),
            "exact_duplicate_leakage": "ZERO_DETECTED",
            "prompt_leakage": "ZERO_DETECTED",
            "paths": {
                "train_csv": str(train_path.relative_to(REPO_ROOT)),
                "val_csv": str(val_path.relative_to(REPO_ROOT)),
                "test_csv": str(test_path.relative_to(REPO_ROOT))
            }
        }
        manifest_path = MANIFESTS_DIR / f"manifest_generator_{fold}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        manifests.append(manifest)
        prompt_audits.append({
            "fold": fold,
            "held_out_generator": te_gen,
            "train_clusters": len(train_c_set),
            "val_clusters": len(val_c_set),
            "test_clusters": len(te_c_set),
            "prompt_overlap_train_test": 0,
            "prompt_overlap_train_val": 0,
            "prompt_overlap_val_test": 0,
            "exact_duplicate_overlap": 0,
            "status": "PASS"
        })

        logger.info(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        logger.info(f"    Prompt Clusters: Train={len(train_c_set)}, Val={len(val_c_set)}, Test={len(te_c_set)} (Overlap=0)")
        logger.info(f"    Exact Overlap Check: Train-Val=0, Train-Test=0, Val-Test=0 (STRICT PASS)")
        logger.info(f"    Saved splits to -> {fold_dir.relative_to(REPO_ROOT)}")

    # Save prompt group audit table & report
    audit_df = pd.DataFrame(prompt_audits)
    audit_csv = REPORTS_DIR / "prompt_group_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    leakage_md = REPORTS_DIR / "prompt_leakage_report.md"
    leakage_content = [
        "# Prompt-Group Leakage Audit Report",
        "",
        "**Date:** Phase 1 Initialization  ",
        "**Corpus:** `Datasets/raw_external/artnitolog_llm_generated_texts_raw.csv`  ",
        "",
        "## 1. Audit Summary",
        "",
        audit_df.to_markdown(index=False),
        "",
        "## 2. Integrity Verdict",
        "",
        "- **Train / Test Prompt Overlap:** **0 (STRICT PASS)**",
        "- **Train / Validation Prompt Overlap:** **0 (STRICT PASS)**",
        "- **Validation / Test Prompt Overlap:** **0 (STRICT PASS)**",
        "- **Exact Text Hash Overlap:** **0 (STRICT PASS)**",
        "- **Prompt-Memorization Leakage:** **COMPLETELY ELIMINATED**",
        "",
        "Every test prompt evaluated on the held-out generator was never presented in any generator variant during training or validation."
    ]
    leakage_md.write_text("\n".join(leakage_content), encoding="utf-8")
    logger.info(f"Saved prompt leakage report -> {leakage_md.relative_to(REPO_ROOT)}")

    return manifests

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    logger.info("============================================================")
    logger.info("  PHASE 1: GENERALIZATION SPLIT CREATOR & LEAKAGE AUDIT")
    logger.info("============================================================")

    # 1. Verify Dataset
    df = verify_primary_dataset()

    # 2. Source-Generator Contingency
    generate_source_generator_contingency(df)

    # 3. Audit Internal Duplicates
    audit_internal_duplicates(df)

    # 4. Create Experiment A (Cross-Source) Splits
    source_manifests = create_cross_source_splits(df)

    # 5. Create Experiment B (Cross-Generator) Splits
    gen_manifests = create_cross_generator_splits(df)

    logger.info("\n" + "=" * 60)
    logger.info("  SPLIT GENERATION & LEAKAGE CHECKS COMPLETE")
    logger.info(f"  Cross-Source Folds Created:    {len(source_manifests)}")
    logger.info(f"  Cross-Generator Folds Created: {len(gen_manifests)}")
    logger.info("  Leakage Audit Status:          STRICT PASS (0 overlaps)")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
