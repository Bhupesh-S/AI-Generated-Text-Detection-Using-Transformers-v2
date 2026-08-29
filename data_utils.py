import pandas as pd
from sklearn.model_selection import train_test_split

def get_unified_splits(df, test_size=0.10, val_size=0.10, seed=42):
    """
    Generate stratified train, validation, and test splits with a standard seed.
    Guarantees exactly aligned row mappings across all training pipelines.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    # Calculate the relative size of val split from train_val split
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=seed, stratify=train_val_df['label']
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
