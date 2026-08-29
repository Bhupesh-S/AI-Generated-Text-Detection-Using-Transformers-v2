"""
=============================================================================
 Hybrid Explainable Transformer Framework for Detecting AI-Generated Text
 Dataset Preprocessing Pipeline
=============================================================================
 Author  : Final Year Project
 Purpose : Convert, clean, and merge multiple heterogeneous datasets into a
           single unified CSV ready for Transformer model training
           (RoBERTa / DeBERTa-v3 / DistilBERT).

 Supported raw-data layouts (auto-detected):
   * Flat CSV / TSV files with text + label columns
   * JSONL / JSON files with text + label fields
   * Directory trees of .txt files (subfolder name = label or labels.txt)
   * The perturb dataset with a shared labels.txt file
   * The essay / reuter / wp / other datasets (paired human & AI txt dirs)
   * The wp subdatasets (numbered subdirectory trees)

 Output schema per row:
   text   -- cleaned document text
   label  -- 0 = Human, 1 = AI-Generated
   source -- dataset name derived from the file / directory name
=============================================================================
"""

import os
import glob
import json
import re
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "Datasets"
CLEANED_DIR  = BASE_DIR / "datasets" / "cleaned"
MERGED_DIR   = BASE_DIR / "datasets" / "merged"
REPORT_PATH  = BASE_DIR / "preprocessing_report.md"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

MIN_WORD_COUNT = 20
RANDOM_SEED    = 42

LABEL_MAP = {
    "human"    : 0,
    "0"        : 0,
    "false"    : 0,
    "ai"       : 1,
    "gpt"      : 1,
    "chatgpt"  : 1,
    "generated": 1,
    "1"        : 1,
    "true"     : 1,
    "reborn"   : 1,
}

AI_FOLDER_NAMES = {
    "gpt", "claude", "gpt_prompt1", "gpt_prompt2",
    "gpt_semantic", "gpt_writing",
}


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def detect_file_type(path):
    """Return canonical file-type string: csv | tsv | json | jsonl | txt."""
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix in {"csv", "tsv", "txt", "json", "jsonl"} else suffix


def normalize_label(raw_label):
    """Map any label representation to 0 (Human) or 1 (AI). Returns None if unknown."""
    if raw_label is None:
        return None
    try:
        if pd.isna(raw_label):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(raw_label, (int, float, np.integer, np.floating)):
        v = int(raw_label)
        return v if v in (0, 1) else None
    return LABEL_MAP.get(str(raw_label).strip().lower(), None)


def clean_text(text):
    """Clean whitespace, remove null bytes, preserve Unicode."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def word_count(text):
    """Return number of whitespace-delimited words."""
    return len(text.split())


# =============================================================================
# FILE LOADERS
# =============================================================================

def _pick_text_col(df, filename):
    candidates = ["text", "content", "essay", "document", "response",
                  "generated_text", "human_text", "article", "body"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        col = str_cols[0] if str_cols else (df.columns[0] if len(df.columns) else None)
        if col:
            logger.warning("  !  '%s' -> using '%s' as text column", filename, col)
    return col


def _pick_label_col(df, filename):
    candidates = ["label", "generated", "class", "category", "type",
                  "is_generated", "ai_generated", "target"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        logger.error("  X  No label column found in %s", filename)
    return col


def load_csv(path, source_name):
    """Load a CSV/TSV file and return a normalised DataFrame."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    except Exception as exc:
        logger.error("  X  Cannot read %s: %s", path.name, exc)
        return None

    logger.info("  -> %d rows from %s", len(df), path.name)
    text_col  = _pick_text_col(df, path.name)
    label_col = _pick_label_col(df, path.name)
    if text_col is None or label_col is None:
        return None

    return pd.DataFrame({
        "text"   : df[text_col].astype(str),
        "label"  : df[label_col],
        "source" : source_name,
    })


def load_json(path, source_name):
    """Load a JSON/JSONL file and return a normalised DataFrame."""
    records = []
    try:
        if path.suffix.lower() == ".jsonl":
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
            records = data if isinstance(data, list) else [data]
    except Exception as exc:
        logger.error("  X  Cannot parse %s: %s", path.name, exc)
        return None

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info("  -> %d records from %s", len(df), path.name)

    text_col  = _pick_text_col(df, path.name)
    label_col = _pick_label_col(df, path.name)
    if text_col is None or label_col is None:
        return None

    return pd.DataFrame({
        "text"   : df[text_col].astype(str),
        "label"  : df[label_col],
        "source" : source_name,
    })


def load_txt_directory(base_dir, source_name, human_subdir="human",
                       ai_subdirs=None, labels_file=None):
    """
    Load .txt-file-based datasets.

    Strategy A: subfolder name encodes label (human/ -> 0, gpt/ -> 1 ...).
    Strategy B: sibling labels.txt provides per-file integer labels.
    """
    if ai_subdirs is None:
        ai_subdirs = []
    records = []

    # Strategy B -- labels.txt
    if labels_file is not None and labels_file.exists():
        raw_labels = [
            ln.strip()
            for ln in labels_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        txt_files = []
        for sub in sorted(base_dir.iterdir()):
            if sub.is_dir() and sub.name not in {"logprobs", "logprobs_babbage", "prompts"}:
                txt_files.extend(sorted(sub.glob("*.txt")))

        n = min(len(txt_files), len(raw_labels))
        if len(txt_files) != len(raw_labels):
            logger.warning("  !  %s: %d files vs %d labels -- using %d",
                           source_name, len(txt_files), len(raw_labels), n)
        for fp, rl in zip(txt_files[:n], raw_labels[:n]):
            lbl = normalize_label(rl)
            if lbl is None:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                records.append({"text": text, "label": lbl, "source": source_name})
            except Exception:
                pass
        logger.info("  -> %d records (labels.txt) from %s", len(records), source_name)
        return pd.DataFrame(records) if records else None

    # Strategy A -- subfolder labels
    subdirs = [d for d in base_dir.iterdir()
                if d.is_dir() and d.name not in {"logprobs", "logprobs_babbage", "prompts"}]
    if not ai_subdirs:
        ai_subdirs = [d.name for d in subdirs if d.name != human_subdir]

    for sub in subdirs:
        if sub.name == human_subdir:
            label = 0
        elif sub.name in ai_subdirs:
            label = 1
        else:
            continue
        for fp in sorted(sub.glob("*.txt")):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                records.append({"text": text, "label": label, "source": source_name})
            except Exception:
                pass

    logger.info("  -> %d records (subfolder) from %s", len(records), source_name)
    return pd.DataFrame(records) if records else None


def load_wp_numbered_subdirs(base_dir, source_name):
    """
    Load the wp dataset which has numbered batch sub-directories (0/, 1/, 2/ ...)
    each containing human/ and AI-model subdirs of .txt files.
    """
    records = []
    for batch_dir in sorted(base_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        for sub in batch_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == "human":
                label = 0
            elif sub.name in AI_FOLDER_NAMES:
                label = 1
            else:
                continue
            for fp in sorted(sub.glob("*.txt")):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    records.append({"text": text, "label": label, "source": source_name})
                except Exception:
                    pass
    logger.info("  -> %d records from %s", len(records), source_name)
    return pd.DataFrame(records) if records else None


# =============================================================================
# COLUMN NORMALISATION
# =============================================================================

def normalize_columns(df):
    """Ensure the DataFrame has exactly columns: text, label, source."""
    if "source" not in df.columns:
        df = df.copy()
        df["source"] = "unknown"
    df = df[["text", "label", "source"]].copy()
    df["label"] = df["label"].apply(normalize_label)
    return df


# =============================================================================
# CLEANING PIPELINE
# =============================================================================

def clean_dataset(df, source_name):
    """
    Full cleaning pipeline returning (cleaned_df, stats_dict).

    Steps: drop nulls -> clean text -> drop whitespace-only -> drop unknown
    labels -> drop short texts -> drop duplicates.
    """
    stats = {
        "source"               : source_name,
        "original_rows"        : len(df),
        "null_rows_removed"    : 0,
        "whitespace_removed"   : 0,
        "short_text_removed"   : 0,
        "duplicates_removed"   : 0,
        "unknown_label_removed": 0,
        "final_rows"           : 0,
        "human_count"          : 0,
        "ai_count"             : 0,
    }

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    stats["null_rows_removed"] = before - len(df)

    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    stats["whitespace_removed"] = before - len(df)

    before = len(df)
    df = df[df["label"].notna() & df["label"].isin([0, 1])]
    stats["unknown_label_removed"] = before - len(df)

    df = df.copy()
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df[df["text"].apply(word_count) >= MIN_WORD_COUNT]
    stats["short_text_removed"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    stats["duplicates_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    stats["final_rows"]  = len(df)
    stats["human_count"] = int((df["label"] == 0).sum())
    stats["ai_count"]    = int((df["label"] == 1).sum())

    return df, stats


# =============================================================================
# DEDUPLICATION & MERGE
# =============================================================================

def remove_duplicates(df):
    """Remove cross-dataset duplicate rows (same text + label)."""
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info("  Removed %d cross-dataset duplicates", removed)
    return df


def merge_datasets(cleaned_frames, balance_classes=True):
    """Concatenate, deduplicate, optionlly balance, and shuffle all cleaned DataFrames."""
    merged = pd.concat(cleaned_frames, ignore_index=True)
    merged = remove_duplicates(merged)

    if balance_classes and "label" in merged.columns:
        human_df = merged[merged["label"] == 0]
        ai_df    = merged[merged["label"] == 1]
        if not human_df.empty and not ai_df.empty:
            min_cnt = min(len(human_df), len(ai_df))
            h_sampled = human_df.sample(n=min_cnt, random_state=RANDOM_SEED)
            a_sampled = ai_df.sample(n=min_cnt, random_state=RANDOM_SEED)
            merged = pd.concat([h_sampled, a_sampled], ignore_index=True)

    return merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# =============================================================================
# STATISTICS PRINTER
# =============================================================================

def print_statistics(df):
    """Print summary statistics of the merged dataset."""
    print("\n" + "=" * 62)
    print("           MERGED DATASET STATISTICS")
    print("=" * 62)
    print(f"  Total rows           : {len(df):,}")
    print(f"  Human samples  (0)   : {(df['label'] == 0).sum():,}")
    print(f"  AI samples     (1)   : {(df['label'] == 1).sum():,}")
    print(f"  Missing values       : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows       : {df.duplicated(subset=['text','label']).sum()}")
    print(f"  Avg text length      : {df['text'].apply(word_count).mean():,.1f} words")
    print("=" * 62 + "\n")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_report(all_stats, merged_df):
    """Write a Markdown preprocessing report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Preprocessing Report",
        f"\n**Generated:** {now}  ",
        "**Project:** Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text  ",
        f"**Minimum word count threshold:** {MIN_WORD_COUNT} words  \n",
        "---\n",
        "## 1. Per-Dataset Summary\n",
        "| Dataset | Original Rows | Final Rows | Null Removed | Whitespace Removed | Short Text Removed | Duplicates Removed | Human (0) | AI (1) |",
        "|---------|:-------------:|:----------:|:------------:|:-----------------:|:-----------------:|:-----------------:|:---------:|:------:|",
    ]

    totals = {"original_rows": 0, "final_rows": 0, "null": 0,
              "short_text_removed": 0, "duplicates_removed": 0,
              "human_count": 0, "ai_count": 0}

    for s in all_stats:
        null_tot = s["null_rows_removed"] + s["whitespace_removed"] + s["unknown_label_removed"]
        lines.append(
            f"| {s['source']} | {s['original_rows']:,} | {s['final_rows']:,} "
            f"| {null_tot:,} | {s['whitespace_removed']:,} "
            f"| {s['short_text_removed']:,} | {s['duplicates_removed']:,} "
            f"| {s['human_count']:,} | {s['ai_count']:,} |"
        )
        totals["original_rows"]     += s["original_rows"]
        totals["final_rows"]        += s["final_rows"]
        totals["null"]              += null_tot
        totals["short_text_removed"]+= s["short_text_removed"]
        totals["duplicates_removed"]+= s["duplicates_removed"]
        totals["human_count"]       += s["human_count"]
        totals["ai_count"]          += s["ai_count"]

    lines.append(
        f"| **TOTAL** | **{totals['original_rows']:,}** | **{totals['final_rows']:,}** "
        f"| **{totals['null']:,}** | -- | **{totals['short_text_removed']:,}** "
        f"| **{totals['duplicates_removed']:,}** | **{totals['human_count']:,}** "
        f"| **{totals['ai_count']:,}** |"
    )

    total = len(merged_df)
    avg_len = merged_df["text"].apply(word_count).mean()
    lines += [
        "\n---\n",
        "## 2. Merged Dataset Statistics\n",
        f"- **Total rows after merge & dedup:** {total:,}",
        f"- **Human samples (label = 0):** {(merged_df['label'] == 0).sum():,}",
        f"- **AI samples    (label = 1):** {(merged_df['label'] == 1).sum():,}",
        f"- **Missing values:** {merged_df.isnull().sum().sum()}",
        f"- **Duplicate rows:** {merged_df.duplicated(subset=['text','label']).sum()}",
        f"- **Average text length:** {avg_len:,.1f} words",
        "- **Output file:** `datasets/merged/final_dataset.csv`\n",
        "---\n",
        "## 3. Final Class Distribution\n",
        "| Label | Count | Percentage |",
        "|-------|------:|----------:|",
    ]
    for lbl, name in [(0, "Human (0)"), (1, "AI Generated (1)")]:
        n   = int((merged_df["label"] == lbl).sum())
        pct = 100.0 * n / total if total else 0.0
        lines.append(f"| {name} | {n:,} | {pct:.2f}% |")

    lines += [
        "\n---\n",
        "## 4. Notes\n",
        "- All text was cleaned: null bytes removed, newlines collapsed to spaces, extra whitespace trimmed.",
        "- Unicode characters were preserved (no ASCII-only filtering).",
        f"- Rows with fewer than {MIN_WORD_COUNT} words were discarded.",
        "- Duplicate rows (identical text + label) were removed within each dataset and globally.",
        "- The merged dataset was shuffled with `random_state=42` for reproducibility.",
        "- Label mapping: 0 = Human, 1 = AI-Generated.",
        "\n---\n",
        "*Report generated automatically by `preprocessing/preprocess.py`*",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved -> %s", REPORT_PATH)


# =============================================================================
# DATASET REGISTRY  (auto-discover all sources)
# =============================================================================

def discover_datasets():
    """
    Return list of (kind, path_str, source_name) tuples for every dataset.

    Automatically discovers:
      - CSV/JSONL/JSON files in Datasets/
      - essay/, reuter/ directories with human/ & gpt/ subdirs
      - wp/ directory with numbered batch sub-directories
      - perturb/, perturb_old/ directories with labels.txt
      - other/* sub-directories (each with human/ & gpt/ subdirs)
      - Any new CSV/JSONL/JSON placed in datasets/raw/ (extensible)
    """
    registry = []

    # Top-level CSV files
    for p in sorted(RAW_DIR.glob("*.csv")):
        registry.append(("csv", str(p), p.stem.lower().replace(" ", "_")))

    # Top-level JSONL files
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        registry.append(("jsonl", str(p), p.stem.lower().replace(" ", "_")))

    # Top-level JSON files
    for p in sorted(RAW_DIR.glob("*.json")):
        registry.append(("json", str(p), p.stem.lower().replace(" ", "_")))

    # essay dataset
    if (RAW_DIR / "essay").is_dir():
        registry.append(("essay_dir", str(RAW_DIR / "essay"), "essay"))

    # reuter dataset
    if (RAW_DIR / "reuter").is_dir():
        registry.append(("essay_dir", str(RAW_DIR / "reuter"), "reuter"))

    # wp dataset (numbered batch sub-dirs)
    if (RAW_DIR / "wp").is_dir():
        registry.append(("wp_batched", str(RAW_DIR / "wp"), "wp"))

    # perturb dataset
    if (RAW_DIR / "perturb").is_dir():
        registry.append(("perturb", str(RAW_DIR / "perturb"), "perturb"))

    # perturb_old dataset
    if (RAW_DIR / "perturb_old").is_dir():
        registry.append(("perturb", str(RAW_DIR / "perturb_old"), "perturb_old"))

    # other/* sub-datasets
    other_dir = RAW_DIR / "other"
    if other_dir.is_dir():
        for sub in sorted(other_dir.iterdir()):
            if sub.is_dir():
                registry.append(("essay_dir", str(sub), f"other_{sub.name}"))

    # Extensible: datasets/raw/ folder (drop any new CSVs/JSONLs here)
    raw_input_dir = BASE_DIR / "datasets" / "raw"
    if raw_input_dir.is_dir():
        for p in sorted(raw_input_dir.glob("**/*.csv")):
            registry.append(("csv", str(p), p.stem.lower().replace(" ", "_")))
        for p in sorted(raw_input_dir.glob("**/*.jsonl")):
            registry.append(("jsonl", str(p), p.stem.lower().replace(" ", "_")))
        for p in sorted(raw_input_dir.glob("**/*.json")):
            registry.append(("json", str(p), p.stem.lower().replace(" ", "_")))

    return registry


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    logger.info("=" * 62)
    logger.info("  AI Text Detection -- Dataset Preprocessing Pipeline")
    logger.info("=" * 62)
    logger.info("RAW_DIR    : %s", RAW_DIR)
    logger.info("CLEANED_DIR: %s", CLEANED_DIR)
    logger.info("MERGED_DIR : %s", MERGED_DIR)

    registry = discover_datasets()
    logger.info("\nDiscovered %d dataset sources:", len(registry))
    for kind, path, src in registry:
        logger.info("  [%-10s]  %-30s  %s", kind, src, Path(path).name)

    all_stats      = []
    cleaned_frames = []

    for kind, path_str, source_name in tqdm(registry, desc="Datasets", unit="ds"):
        path = Path(path_str)
        logger.info("\n-- Processing: %s --", source_name)
        raw_df = None

        if kind in {"csv", "tsv"}:
            raw_df = load_csv(path, source_name)
        elif kind in {"jsonl", "json"}:
            raw_df = load_json(path, source_name)
        elif kind == "essay_dir":
            raw_df = load_txt_directory(
                path, source_name, human_subdir="human",
                ai_subdirs=list(AI_FOLDER_NAMES)
            )
        elif kind == "perturb":
            raw_df = load_txt_directory(
                path, source_name,
                labels_file=path / "labels.txt"
            )
        elif kind == "wp_batched":
            raw_df = load_wp_numbered_subdirs(path, source_name)

        if raw_df is None or raw_df.empty:
            logger.warning("  !  No data loaded for %s -- skipping", source_name)
            continue

        try:
            norm_df = normalize_columns(raw_df)
        except Exception as exc:
            logger.error("  X  Column normalisation failed for %s: %s", source_name, exc)
            continue

        cleaned_df, stats = clean_dataset(norm_df, source_name)

        if cleaned_df.empty:
            logger.warning("  !  All rows removed after cleaning for %s", source_name)
            continue

        safe_name = re.sub(r"[^\w\-]", "_", source_name)
        out_path  = CLEANED_DIR / f"{safe_name}_clean.csv"
        cleaned_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("  OK  Saved %d rows -> %s", len(cleaned_df), out_path.name)

        all_stats.append(stats)
        cleaned_frames.append(cleaned_df)

    if not cleaned_frames:
        logger.error("No datasets were processed successfully. Exiting.")
        return

    logger.info("\n-- Merging %d cleaned datasets --", len(cleaned_frames))
    final_df = merge_datasets(cleaned_frames)

    merged_path = MERGED_DIR / "final_dataset.csv"
    final_df.to_csv(merged_path, index=False, encoding="utf-8")
    logger.info("OK  Merged dataset -> %s  (%d rows)", merged_path, len(final_df))

    print_statistics(final_df)
    generate_report(all_stats, final_df)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
