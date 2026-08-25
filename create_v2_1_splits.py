"""
create_v2_1_splits.py
=====================
Creates unified 80/10/10 train/validation/test splits for final_dataset_v2_1.csv
Guarantees consistent row alignment across all models (RoBERTa, DeBERTa, DistilBERT, XGBoost, Ensembles).
"""

import os
import json
import pandas as pd
from pathlib import Path
import data_utils

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "Datasets" / "merged" / "final_dataset_v2_1.csv"
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  DATASET V2.1 UNIFIED SPLIT CREATOR")
print("=" * 65)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

print(f"\nLoading dataset: {DATASET_PATH} ...")
df = pd.read_csv(DATASET_PATH, low_memory=False)
total_samples = len(df)
print(f"Total rows loaded: {total_samples:,}")

# Generate unified splits (80% Train, 10% Validation, 10% Test)
train_df, val_df, test_df = data_utils.get_unified_splits(df, test_size=0.10, val_size=0.10, seed=42)

# Save split CSV files
train_path = SPLIT_DIR / "train_v2_1.csv"
val_path   = SPLIT_DIR / "val_v2_1.csv"
test_path  = SPLIT_DIR / "test_v2_1.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# Compute distributions
def get_dist(data):
    cnt = len(data)
    h_cnt = int((data['label'] == 0).sum())
    a_cnt = int((data['label'] == 1).sum())
    return {
        'total': cnt,
        'human': h_cnt,
        'ai': a_cnt,
        'human_pct': round(h_cnt / cnt * 100, 2),
        'ai_pct': round(a_cnt / cnt * 100, 2)
    }

summary = {
    'dataset': str(DATASET_PATH.name),
    'total_rows': total_samples,
    'train': get_dist(train_df),
    'val': get_dist(val_df),
    'test': get_dist(test_df)
}

summary_path = SPLIT_DIR / "split_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "-" * 65)
print("  SPLIT SUMMARY & DISTRIBUTION")
print("-" * 65)
print(f"  Train samples      : {len(train_df):,} ({len(train_df)/total_samples*100:.1f}%) | Human: {summary['train']['human']:,} ({summary['train']['human_pct']}%), AI: {summary['train']['ai']:,} ({summary['train']['ai_pct']}%)")
print(f"  Validation samples : {len(val_df):,} ({len(val_df)/total_samples*100:.1f}%) | Human: {summary['val']['human']:,} ({summary['val']['human_pct']}%), AI: {summary['val']['ai']:,} ({summary['val']['ai_pct']}%)")
print(f"  Test samples       : {len(test_df):,} ({len(test_df)/total_samples*100:.1f}%) | Human: {summary['test']['human']:,} ({summary['test']['human_pct']}%), AI: {summary['test']['ai']:,} ({summary['test']['ai_pct']}%)")
print("-" * 65)
print(f"  Saved train split -> {train_path.relative_to(ROOT)}")
print(f"  Saved val split   -> {val_path.relative_to(ROOT)}")
print(f"  Saved test split  -> {test_path.relative_to(ROOT)}")
print(f"  Saved metadata    -> {summary_path.relative_to(ROOT)}")
print("=" * 65)
print("SPLIT CREATION COMPLETE.")
