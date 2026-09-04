"""
Dataset Balancer
================
Equalizes Human (label=0) and AI (label=1) text samples in final_dataset.csv
to create a perfectly balanced 50/50 dataset.

Run from repo root:
    python balance_dataset.py
"""
import pandas as pd
from pathlib import Path

PATHS = [
    Path("Datasets/merged/final_dataset.csv"),
    Path("datasets/merged/final_dataset.csv"),
]

def balance_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"Skipping (not found): {csv_path}")
        return

    print("=" * 60)
    print(f"  Balancing Dataset: {csv_path}")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"  Loaded total rows  : {len(df):,}")

    counts = df["label"].value_counts()
    print("  Initial class counts:")
    for lbl, count in counts.items():
        name = "Human" if lbl == 0 else "AI"
        pct = 100.0 * count / len(df)
        print(f"    Label {lbl} ({name}): {count:,} ({pct:.2f}%)")

    human_df = df[df["label"] == 0]
    ai_df    = df[df["label"] == 1]

    min_count = min(len(human_df), len(ai_df))
    print(f"\n  Equalizing classes to {min_count:,} samples each...")

    sampled_human = human_df.sample(n=min_count, random_state=42)
    sampled_ai    = ai_df.sample(n=min_count, random_state=42)

    balanced_df = pd.concat([sampled_human, sampled_ai], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save backup
    backup_path = csv_path.with_suffix(".csv.bak")
    df.to_csv(backup_path, index=False)
    print(f"  Backup saved to    : {backup_path}")

    # Save balanced dataset
    balanced_df.to_csv(csv_path, index=False)
    print(f"  Balanced CSV saved : {csv_path}")
    print(f"  New total rows     : {len(balanced_df):,}")

    print("\n  Updated class distribution:")
    new_counts = balanced_df["label"].value_counts().sort_index()
    for lbl, count in new_counts.items():
        name = "Human" if lbl == 0 else "AI"
        pct = 100.0 * count / len(balanced_df)
        print(f"    Label {lbl} ({name}): {count:,} ({pct:.2f}%)")
    print("=" * 60)

def main():
    seen_resolved = set()
    for p in PATHS:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        balance_csv(p)

if __name__ == "__main__":
    main()
