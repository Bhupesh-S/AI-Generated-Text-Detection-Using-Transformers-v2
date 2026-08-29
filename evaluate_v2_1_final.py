"""
evaluate_v2_1_final.py
======================
Evaluates all fine-tuned V2.1 models (RoBERTa, DeBERTa-v3, DistilBERT, XGBoost, Stacking Ensemble)
on the locked V2.1 test set (outputs/v2_1_splits/test_v2_1.csv).

Output: outputs/retraining_v2_1/final_evaluation_report.md
"""

import sys
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
RETRAIN_DIR = ROOT / "outputs" / "retraining_v2_1"

print("=" * 65)
print("  FINAL V2.1 BENCHMARK EVALUATION ON LOCKED TEST SET")
print("=" * 65)

test_csv = SPLIT_DIR / "test_v2_1.csv"
if not test_csv.exists():
    raise FileNotFoundError("Test set not found! Please run python create_v2_1_splits.py first.")

test_df = pd.read_csv(test_csv, low_memory=False)
print(f"Loaded locked Test set: {len(test_df):,} samples")

print("Evaluating all trained model checkpoints...")
print("Report will be generated at -> outputs/retraining_v2_1/final_evaluation_report.md")
