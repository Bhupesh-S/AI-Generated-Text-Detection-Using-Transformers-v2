"""
Dataset audit: check size, nulls, duplicates, and clean.
Run from repo root: python dataset_audit.py
"""
import pandas as pd

CSV = "Datasets/merged/final_dataset.csv"

print("Loading dataset ...")
df = pd.read_csv(CSV, usecols=["text", "label"])

print(f"\n--- Dataset Audit ---")
print(f"Total rows         : {len(df):,}")
print(f"Null text rows     : {df['text'].isna().sum():,}")
print(f"Empty text rows    : {(df['text'].astype(str).str.strip() == '').sum():,}")
print(f"Duplicate rows     : {df.duplicated().sum():,}")
print(f"Duplicate text col : {df['text'].duplicated().sum():,}")
print(f"\nLabel distribution :")
counts = df['label'].value_counts().sort_index()
for lbl, count in counts.items():
    name = "Human" if lbl == 0 else "AI"
    pct = 100.0 * count / len(df)
    print(f"  Label {lbl} ({name}): {count:,} ({pct:.2f}%)")

is_balanced = (len(counts) == 2 and counts.iloc[0] == counts.iloc[1])
print(f"\nClass balance status: {'BALANCED (50/50)' if is_balanced else 'IMBALANCED'}")

