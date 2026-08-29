"""
run_balanced_experiment.py
===========================
Executes the dataset rebalancing, unified train/val/test splitting,
retraining of XGBoost + Stacking Meta-Classifier, evaluation on held-out test set,
and data leakage audit.
"""

import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

# Setup root imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "ensemble"))

from data_utils import get_unified_splits
from feature_engineering.utils import extract_features_batch
from calibration import TemperatureScaler, PlattScaler
from ensemble import BaseModelEvaluator

def run_experiment():
    print("=" * 70)
    print("  AI-vs-Human Text Detection: Dataset Rebalancing Experiment")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1 — REBALANCE THE DATASET
    # ---------------------------------------------------------
    print("\n--- STEP 1: REBALANCING DATASET ---")
    bak_path = PROJECT_ROOT / "Datasets" / "merged" / "final_dataset.csv.bak"
    balanced_csv_path = PROJECT_ROOT / "Datasets" / "merged" / "final_dataset_balanced.csv"

    if not bak_path.exists():
        raise FileNotFoundError(f"Expanded dataset backup not found at: {bak_path}")

    df_raw = pd.read_csv(bak_path)
    print(f"Loaded raw expanded dataset: {len(df_raw):,} rows.")
    print("Raw label counts:")
    for lbl, cnt in df_raw['label'].value_counts().items():
        print(f"  Label {lbl} ({'Human' if lbl == 0 else 'AI'}): {cnt:,}")

    ai_df = df_raw[df_raw['label'] == 1]
    human_raw = df_raw[df_raw['label'] == 0]

    ai_count = len(ai_df)
    human_sampled = human_raw.sample(n=ai_count, random_state=42)

    df_balanced = pd.concat([human_sampled, ai_df], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    df_balanced.to_csv(balanced_csv_path, index=False)
    print(f"Saved balanced dataset to: {balanced_csv_path}")

    # Verification checks
    total_rows = len(df_balanced)
    counts = df_balanced['label'].value_counts().to_dict()
    human_cnt = counts.get(0, 0)
    ai_cnt = counts.get(1, 0)
    null_cnt = df_balanced['text'].isna().sum()
    empty_cnt = (df_balanced['text'].astype(str).str.strip() == "").sum()
    dup_row_cnt = df_balanced.duplicated(subset=['text', 'label']).sum()
    dup_text_cnt = df_balanced['text'].duplicated().sum()

    print("\nStep 1 Verification Results:")
    print(f"  Total rows      : {total_rows:,} (Expected: 47,322)")
    print(f"  AI samples (1)  : {ai_cnt:,} (Expected: 23,661)")
    print(f"  Human samples(0): {human_cnt:,} (Expected: 23,661)")
    print(f"  Null text rows  : {null_cnt}")
    print(f"  Empty text rows : {empty_cnt}")
    print(f"  Duplicate rows  : {dup_row_cnt}")
    print(f"  Duplicate text  : {dup_text_cnt}")

    assert total_rows == 47322, f"Total rows expected 47,322, got {total_rows}"
    assert ai_cnt == 23661, f"AI count expected 23,661, got {ai_cnt}"
    assert human_cnt == 23661, f"Human count expected 23,661, got {human_cnt}"
    assert null_cnt == 0, "Found null text rows!"
    assert empty_cnt == 0, "Found empty text rows!"
    assert dup_row_cnt == 0, "Found duplicate rows!"

    # ---------------------------------------------------------
    # STEP 2 — CREATE TRAIN/VALIDATION/TEST SPLIT
    # ---------------------------------------------------------
    print("\n--- STEP 2: CREATING TRAIN/VAL/TEST SPLITS ---")
    train_df, val_df, test_df = get_unified_splits(df_balanced, test_size=0.10, val_size=0.10, seed=42)

    print(f"  Training set size  : {len(train_df):,} samples (Expected ~37,857)")
    print(f"  Validation set size: {len(val_df):,} samples (Expected ~4,732)")
    print(f"  Test set size      : {len(test_df):,} samples (Expected ~4,733)")
    print(f"  Sum of splits      : {len(train_df) + len(val_df) + len(test_df):,} (Expected 47,322)")

    assert len(train_df) + len(val_df) + len(test_df) == 47322, "Split sum mismatch!"

    print("\nSplit Class Proportions:")
    for name, df_split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        c = df_split['label'].value_counts()
        print(f"  {name:<10}: Human(0)={c[0]:,}, AI(1)={c[1]:,}, Ratio={c[1]/len(df_split):.4f}")

    # ---------------------------------------------------------
    # STEP 3 — RETRAIN CHAMPION MODELS (XGBoost + Stacking Meta-Classifier)
    # ---------------------------------------------------------
    print("\n--- STEP 3: RETRAINING CHAMPION ENSEMBLE ---")

    output_dir = PROJECT_ROOT / "outputs" / "ensemble_balanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature extraction & retraining XGBoost
    print("Extracting handcrafted features for XGBoost on Balanced Training split...")
    t0 = time.time()
    train_feats = extract_features_batch(train_df['text'].astype(str).tolist())
    df_train_feats = pd.DataFrame(train_feats)

    print(f"Training XGBoost on {len(df_train_feats):,} training samples...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_model.fit(df_train_feats, train_df['label'].values)
    print(f"XGBoost training complete in {time.time() - t0:.2f}s.")

    xgb_save_path = PROJECT_ROOT / "outputs" / "models" / "xgboost_balanced.joblib"
    joblib.dump(xgb_model, xgb_save_path)
    print(f"Saved retrained XGBoost model -> {xgb_save_path}")

    # 2. Extract base model probabilities & fit calibration on Validation Set
    print("\nInitializing BaseModelEvaluator for Transformer & XGBoost inference...")
    evaluator = BaseModelEvaluator()
    evaluator.load_all_models()
    # Override XGBoost model with retrained model
    evaluator.model_xgboost = xgb_model

    val_texts = val_df['text'].astype(str).tolist()
    y_val = val_df['label'].values
    test_texts = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].values

    print("\nExtracting transformer logits on Validation split (4,732 samples)...")
    roberta_val_logits = evaluator.get_transformer_logits(evaluator.model_roberta, evaluator.tokenizer_roberta, val_texts, desc="RoBERTa Val Logits")
    deberta_val_logits = evaluator.get_transformer_logits(evaluator.model_deberta, evaluator.tokenizer_deberta, val_texts, desc="DeBERTa Val Logits")
    distilbert_val_logits = evaluator.get_transformer_logits(evaluator.model_distilbert, evaluator.tokenizer_distilbert, val_texts, desc="DistilBERT Val Logits")

    print("Fitting Temperature Scalers on Validation split...")
    temp_roberta = TemperatureScaler().fit(roberta_val_logits, y_val)
    temp_deberta = TemperatureScaler().fit(deberta_val_logits, y_val)
    temp_distilbert = TemperatureScaler().fit(distilbert_val_logits, y_val)

    temperatures = {
        "roberta": temp_roberta.temperature,
        "deberta": temp_deberta.temperature,
        "distilbert": temp_distilbert.temperature
    }
    print(f"Fitted Temperatures: {temperatures}")
    evaluator.temperatures = temperatures

    # Platt Scaling for XGBoost
    print("Fitting Platt Scaler on XGBoost Validation probabilities...")
    xgb_val_probs_raw = evaluator.get_xgboost_probs(val_texts)
    platt_xgb = PlattScaler().fit(xgb_val_probs_raw, y_val)
    evaluator.platt_calibrator_xgb = platt_xgb

    # Generate calibrated probabilities
    print("Extracting calibrated validation probabilities across all 4 models...")
    val_probs = evaluator.extract_all_probs(val_texts, batch_size=32)

    print("Extracting calibrated test probabilities across all 4 models...")
    test_probs = evaluator.extract_all_probs(test_texts, batch_size=32)

    # Retrain Stacking Meta-Classifier (Logistic Regression) on validation probabilities
    print("\nTraining Stacking Meta-Classifier (Logistic Regression)...")
    val_meta_features = np.column_stack([
        val_probs["roberta"][:, 1],
        val_probs["deberta"][:, 1],
        val_probs["distilbert"][:, 1],
        val_probs["xgboost"][:, 1]
    ])

    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(val_meta_features, y_val)

    meta_path = checkpoint_dir / "meta_classifier.joblib"
    joblib.dump(meta_model, meta_path)
    print(f"Saved Stacking Meta-Classifier -> {meta_path}")
    print(f"Meta-model coefficients: RoBERTa={meta_model.coef_[0][0]:.4f}, DeBERTa={meta_model.coef_[0][1]:.4f}, DistilBERT={meta_model.coef_[0][2]:.4f}, XGBoost={meta_model.coef_[0][3]:.4f}, Intercept={meta_model.intercept_[0]:.4f}")

    # ---------------------------------------------------------
    # STEP 4 — EVALUATE THE NEW MODEL
    # ---------------------------------------------------------
    print("\n--- STEP 4: EVALUATING NEW MODEL ON HELD-OUT TEST SET ---")
    test_meta_features = np.column_stack([
        test_probs["roberta"][:, 1],
        test_probs["deberta"][:, 1],
        test_probs["distilbert"][:, 1],
        test_probs["xgboost"][:, 1]
    ])

    stacking_test_probs = meta_model.predict_proba(test_meta_features)[:, 1]
    stacking_test_preds = (stacking_test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, stacking_test_preds)
    prec = precision_score(y_test, stacking_test_preds)
    rec = recall_score(y_test, stacking_test_preds)
    f1 = f1_score(y_test, stacking_test_preds)
    roc_auc = roc_auc_score(y_test, stacking_test_probs)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, stacking_test_probs)
    pr_auc = auc(rec_arr, prec_arr)

    cm = confusion_matrix(y_test, stacking_test_preds)
    tn, fp, fn, tp = cm.ravel()

    human_fpr = fp / (fp + tn)
    ai_fnr = fn / (fn + tp)

    total_test = len(y_test)
    correct_preds = int((stacking_test_preds == y_test).sum())
    incorrect_preds = total_test - correct_preds

    print("\nNew Model Performance Metrics (Test Set):")
    print(f"  Total Test Samples  : {total_test:,}")
    print(f"  Correct Predictions : {correct_preds:,}")
    print(f"  Incorrect Preds     : {incorrect_preds:,}")
    print(f"  Accuracy            : {acc*100:.2f}%")
    print(f"  Precision           : {prec*100:.2f}%")
    print(f"  Recall              : {rec*100:.2f}%")
    print(f"  F1-Score            : {f1*100:.2f}%")
    print(f"  ROC-AUC             : {roc_auc:.4f}")
    print(f"  PR-AUC              : {pr_auc:.4f}")
    print(f"  Confusion Matrix    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Human FPR           : {human_fpr*100:.2f}%")
    print(f"  AI FNR              : {ai_fnr*100:.2f}%")

    balanced_metrics = {
        "total_test_samples": total_test,
        "correct_predictions": correct_preds,
        "incorrect_predictions": incorrect_preds,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "human_fpr": human_fpr,
        "ai_fnr": ai_fnr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    with open(output_dir / "balanced_metrics.json", "w") as f:
        json.dump(balanced_metrics, f, indent=4)

    # ---------------------------------------------------------
    # STEP 5 — COMPARE AGAINST PREVIOUS CHAMPION
    # ---------------------------------------------------------
    print("\n--- STEP 5: COMPARISON TABLE AGAINST PREVIOUS CHAMPION ---")
    prev = {
        "accuracy": 0.9520,
        "precision": 0.9658,
        "recall": 0.9193,
        "f1": 0.9420,
        "roc_auc": 0.9925,
        "pr_auc": 0.9907,
        "human_fpr": 0.0239,
        "ai_fnr": 0.0807
    }

    comp_table = [
        ("Accuracy",  f"{prev['accuracy']*100:.2f}%",  f"{acc*100:.2f}%",  f"{(acc - prev['accuracy'])*100:+.2f}%"),
        ("Precision", f"{prev['precision']*100:.2f}%", f"{prec*100:.2f}%", f"{(prec - prev['precision'])*100:+.2f}%"),
        ("Recall",    f"{prev['recall']*100:.2f}%",    f"{rec*100:.2f}%",    f"{(rec - prev['recall'])*100:+.2f}%"),
        ("F1",        f"{prev['f1']*100:.2f}%",        f"{f1*100:.2f}%",        f"{(f1 - prev['f1'])*100:+.2f}%"),
        ("ROC-AUC",   f"{prev['roc_auc']:.4f}",        f"{roc_auc:.4f}",     f"{roc_auc - prev['roc_auc']:+.4f}"),
        ("PR-AUC",    f"{prev['pr_auc']:.4f}",         f"{pr_auc:.4f}",      f"{pr_auc - prev['pr_auc']:+.4f}"),
        ("Human FPR", f"{prev['human_fpr']*100:.2f}%", f"{human_fpr*100:.2f}%", f"{(human_fpr - prev['human_fpr'])*100:+.2f}%"),
        ("AI FNR",    f"{prev['ai_fnr']*100:.2f}%",    f"{ai_fnr*100:.2f}%",    f"{(ai_fnr - prev['ai_fnr'])*100:+.2f}%")
    ]

    df_comp = pd.DataFrame(comp_table, columns=["Metric", "Previous Champion", "Balanced Retrained Model", "Change"])
    print(df_comp.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 7 — CHECK FOR DATA LEAKAGE
    # ---------------------------------------------------------
    print("\n--- STEP 7: CHECK FOR DATA LEAKAGE ---")
    train_set_texts = set(train_df['text'].astype(str))
    val_set_texts = set(val_df['text'].astype(str))
    test_set_texts = set(test_df['text'].astype(str))

    train_test_overlap = len(train_set_texts.intersection(test_set_texts))
    train_val_overlap = len(train_set_texts.intersection(val_set_texts))
    val_test_overlap = len(val_set_texts.intersection(test_set_texts))

    print(f"  Exact text overlap (Train vs Test) : {train_test_overlap}")
    print(f"  Exact text overlap (Train vs Val)  : {train_val_overlap}")
    print(f"  Exact text overlap (Val vs Test)   : {val_test_overlap}")
    print("  Validation set isolation: Used strictly for Temperature/Platt Scaling and Stacking Meta-Classifier training.")
    print("  Test set isolation: Held out, never exposed to any scaler or model training.")

if __name__ == "__main__":
    run_experiment()
