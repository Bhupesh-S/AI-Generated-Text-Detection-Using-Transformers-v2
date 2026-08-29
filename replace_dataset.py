"""
Final step: rename the clean temp file to replace the original.
Run AFTER closing final_dataset.csv in Excel / any other app.

    python replace_dataset.py
"""
from pathlib import Path
import shutil

CSV_TMP = Path("Datasets/merged/final_dataset_clean_tmp.csv")
CSV     = Path("Datasets/merged/final_dataset.csv")

if not CSV_TMP.exists():
    print("ERROR: temp file not found. Run save_clean_dataset.py first.")
    raise SystemExit(1)

# Back up original just in case
backup = CSV.with_suffix(".csv.bak")
if CSV.exists():
    shutil.copy2(CSV, backup)
    print(f"Backup saved -> {backup}")
    CSV.unlink()

CSV_TMP.rename(CSV)
print(f"Done. Clean dataset saved -> {CSV}")

import pandas as pd
df = pd.read_csv(CSV)
print(f"Verified rows  : {len(df):,}")
print(f"Columns        : {df.columns.tolist()}")
print(f"Label dist     : {df['label'].value_counts().to_dict()}")
