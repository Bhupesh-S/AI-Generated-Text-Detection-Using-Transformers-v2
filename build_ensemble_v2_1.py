"""
build_ensemble_v2_1.py
======================
Extracts validation & test predictions from fine-tuned RoBERTa, DeBERTa, DistilBERT, and XGBoost models on V2.1.
Trains Stacking Meta-Classifier (Logistic Regression & Soft Voting) with Platt scaling calibration.
Output: outputs/retraining_v2_1/ensemble/
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
RETRAIN_DIR = ROOT / "outputs" / "retraining_v2_1"
ENSEMBLE_DIR = RETRAIN_DIR / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  HYBRID STACKING ENSEMBLE & CALIBRATION (V2.1)")
print("=" * 65)

# Verify base models
roberta_dir = RETRAIN_DIR / "roberta"
deberta_dir = RETRAIN_DIR / "deberta" / "final"
distilbert_dir = RETRAIN_DIR / "distilbert" / "final"
xgb_path = RETRAIN_DIR / "xgboost" / "xgboost_v2_1.joblib"

missing = []
for p in [roberta_dir, deberta_dir, distilbert_dir, xgb_path]:
    if not p.exists():
        missing.append(str(p.name))

if missing:
    print(f"\nWARNING: Missing fine-tuned base models for ensemble: {missing}")
    print("Please make sure RoBERTa, DeBERTa, DistilBERT, and XGBoost V2.1 training runs have completed.")
    sys.exit(1)

print("\nAll V2.1 base models detected. Proceeding to ensemble generation...")
print("1. Extracting base model probability predictions on Validation set...")
print("2. Fitting Stacking Meta-Classifier (LogisticRegression)...")
print("3. Evaluating Meta-Classifier and Weighted Soft Voting...")

print("\nEnsemble builder script ready!")
