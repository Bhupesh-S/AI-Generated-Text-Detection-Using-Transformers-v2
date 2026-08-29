"""
Dataset Cleaner — Save Step
Run from repo root:
    python save_clean_dataset.py
"""
import pandas as pd
import os
from pathlib import Path

CSV     = Path("Datasets/merged/final_dataset.csv")
CSV_TMP = Path("Datasets/merged/final_dataset_clean_tmp.csv")

print("Loading already-computed cleaned dataframe ...")

# Reload the original (now partially saved) data and re-clean
df = pd.read_csv(CSV, usecols=["text", "label"], low_memory=False)
df = df.dropna(subset=["text"])
df = df[df["text"].astype(str).str.strip() != ""]
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"], keep="first")
df = df.reset_index(drop=True)

print(f"Clean rows : {len(df):,}")
print(f"Labels     : {df['label'].value_counts().to_dict()}")

# Write to a temp file first, then atomically replace
print(f"Writing to temp file {CSV_TMP} ...")
df.to_csv(CSV_TMP, index=False)

# Now replace the original
print("Replacing original CSV ...")
if CSV.exists():
    os.chmod(CSV, 0o666)   # ensure write permission
    CSV.unlink()

CSV_TMP.rename(CSV)
print(f"Done. Saved -> {CSV}")
print(f"Final size : {len(df):,} rows x {len(df.columns)} columns")
