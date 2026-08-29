"""
train_xgboost_v2_1.py
======================
Extracts handcrafted stylometric features and trains XGBoost on Dataset V2.1 splits.
Output: outputs/retraining_v2_1/xgboost/xgboost_v2_1.joblib
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from feature_engineering.utils import extract_features_batch

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "v2_1_splits"
OUTPUT_DIR = ROOT / "outputs" / "retraining_v2_1" / "xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  XGBOOST V2.1 STYLOMETRIC MODEL TRAINING")
print("=" * 65)

train_csv = SPLIT_DIR / "train_v2_1.csv"
val_csv   = SPLIT_DIR / "val_v2_1.csv"
test_csv  = SPLIT_DIR / "test_v2_1.csv"

if not train_csv.exists():
    raise FileNotFoundError("Splits not found! Please run python create_v2_1_splits.py first.")

train_df = pd.read_csv(train_csv, low_memory=False)
val_df   = pd.read_csv(val_csv, low_memory=False)
test_df  = pd.read_csv(test_csv, low_memory=False)

print(f"Loaded Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Extract features
print("\nExtracting stylometric features for Train set...")
X_train = pd.DataFrame(extract_features_batch(train_df['text'].fillna("").tolist()))
y_train = train_df['label'].values

print("Extracting stylometric features for Validation set...")
X_val = pd.DataFrame(extract_features_batch(val_df['text'].fillna("").tolist()))
y_val = val_df['label'].values

print("Extracting stylometric features for Test set...")
X_test = pd.DataFrame(extract_features_batch(test_df['text'].fillna("").tolist()))
y_test = test_df['label'].values

# Initialize and train XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

print("\nFitting XGBoost model...")
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Evaluate
val_probs = xgb.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

test_probs = xgb.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_probs)

test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)

print("\n" + "-" * 65)
print("  XGBOOST V2.1 EVALUATION RESULTS")
print("-" * 65)
print(f"  Validation Acc: {val_acc:.4f} | F1: {val_f1:.4f} | ROC-AUC: {val_auc:.4f}")
print(f"  Test Set Acc  : {test_acc:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_auc:.4f}")
print("-" * 65)

# Save model & metrics
model_path = OUTPUT_DIR / "xgboost_v2_1.joblib"
joblib.dump(xgb, model_path)
print(f"Saved model -> {model_path.relative_to(ROOT)}")

metrics_summary = {
    'val': {'accuracy': val_acc, 'f1': val_f1, 'roc_auc': val_auc},
    'test': {'accuracy': test_acc, 'f1': test_f1, 'roc_auc': test_auc}
}
with open(OUTPUT_DIR / "xgboost_v2_1_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("XGBoost V2.1 Training Complete!")
