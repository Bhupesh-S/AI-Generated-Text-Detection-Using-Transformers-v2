"""
Dataset Cleaner
================
Removes:
  1. Rows where 'text' is null / NaN
  2. Rows where 'text' is empty or whitespace-only
  3. Fully duplicate rows (same text + same label)
  4. Rows with duplicate text (keeps first occurrence)

Saves the cleaned dataset back to the same CSV path.

Run from repo root:
    python clean_dataset.py
"""
import pandas as pd
from pathlib import Path

CSV = Path("Datasets/merged/final_dataset.csv")

print("=" * 55)
print("  Dataset Cleaner — AI Text Detection Project")
print("=" * 55)

# ── Load (keep only the two useful columns) ──────────────────
print(f"\n[1/5] Loading {CSV} ...")
df = pd.read_csv(CSV, usecols=["text", "label"], low_memory=False)
print(f"      Raw rows loaded    : {len(df):,}")

# ── Step 1: Drop rows with null text ──────────────────────────
before = len(df)
df = df.dropna(subset=["text"])
removed_null = before - len(df)
print(f"\n[2/5] Null text rows removed  : {removed_null:,}")
print(f"      Rows remaining           : {len(df):,}")

# ── Step 2: Drop rows with empty / whitespace-only text ───────
before = len(df)
df = df[df["text"].astype(str).str.strip() != ""]
removed_empty = before - len(df)
print(f"\n[3/5] Empty text rows removed : {removed_empty:,}")
print(f"      Rows remaining           : {len(df):,}")

# ── Step 3: Cast label to int ──────────────────────────────────
df["label"] = df["label"].astype(int)

# ── Step 4: Drop fully duplicate rows (same text + same label) ─
before = len(df)
df = df.drop_duplicates(subset=["text", "label"])
removed_full_dup = before - len(df)
print(f"\n[4/5] Full duplicate rows removed : {removed_full_dup:,}")
print(f"      Rows remaining               : {len(df):,}")

# ── Step 5: Drop rows with duplicate text (cross-label) ────────
before = len(df)
df = df.drop_duplicates(subset=["text"], keep="first")
removed_text_dup = before - len(df)
print(f"\n[5/6] Cross-label duplicate text removed : {removed_text_dup:,}")
print(f"      Rows remaining                      : {len(df):,}")

# ── Step 6: Equalize Human (0) and AI (1) classes (50/50) ──────
human_df = df[df["label"] == 0]
ai_df    = df[df["label"] == 1]
min_count = min(len(human_df), len(ai_df))
df = pd.concat([
    human_df.sample(n=min_count, random_state=42),
    ai_df.sample(n=min_count, random_state=42)
], ignore_index=True).sample(frac=1, random_state=42)
print(f"\n[6/6] Equalized class samples to {min_count:,} per class (50/50)")
print(f"      Final dataset size                 : {len(df):,}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Summary")
print("=" * 55)
total_removed = removed_null + removed_empty + removed_full_dup + removed_text_dup
print(f"  Total non-balance rows removed : {total_removed:,}")
print(f"  Final dataset size             : {len(df):,}")
print(f"\n  Label distribution :")
vc = df["label"].value_counts().sort_index()
for lbl, cnt in vc.items():
    name = "Human Written" if lbl == 0 else "AI Generated"
    pct  = 100 * cnt / len(df)
    print(f"    {lbl} ({name}) : {cnt:,}  ({pct:.1f}%)")

# ── Save ─────────────────────────────────────────────────────
df = df.reset_index(drop=True)
df.to_csv(CSV, index=False)
print(f"\n  Saved cleaned dataset -> {CSV}")
print("=" * 55)
print("Done.")
