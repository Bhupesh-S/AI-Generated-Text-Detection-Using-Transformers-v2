"""
dataset.py
==========
Dataset loading, splitting, and tokenization for the DistilBERT pipeline.

Responsibilities:
  - Load CSV with validation checks (including Git LFS pointer detection).
  - Stratified 80/10/10 train/val/test split.
  - Wrap splits in HuggingFace ``datasets.Dataset`` objects.
  - Batch tokenization via ``AutoTokenizer``.

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_csv(path: Path) -> pd.DataFrame:
    """
    Load and validate the CSV dataset file.

    Checks:
      - File exists.
      - File is not a Git LFS pointer (< 10 KB with lfs header).
      - Required columns ``text`` and ``label`` are present.
      - Drops NaN rows in those columns.
      - Casts columns to correct dtypes.

    Args:
        path : Absolute or relative path to the CSV file.

    Returns:
        Cleaned :class:`pd.DataFrame`.

    Raises:
        FileNotFoundError : If the CSV does not exist.
        ValueError        : If the file is a Git LFS pointer or missing columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please ensure 'final_dataset.csv' exists under Datasets/merged/."
        )

    # Git LFS pointer guard
    if path.stat().st_size < 10_000:
        snippet = path.read_text(encoding="utf-8", errors="replace")[:200]
        if "git-lfs" in snippet:
            raise ValueError(
                f"'{path}' is a Git LFS pointer file.\n"
                "Run `git lfs pull` to download the actual data."
            )

    logger.info(f"Reading CSV from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw dataset shape: {df.shape}")

    required_cols = {CFG.text_column, CFG.label_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=[CFG.text_column, CFG.label_column])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped:,} rows with NaN in text/label columns.")

    df[CFG.text_column] = df[CFG.text_column].astype(str).str.strip()

    # Deduplicate raw dataset to prevent leakage and target inflation
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[CFG.text_column]).reset_index(drop=True)
    deduped = before_dedup - len(df)
    if deduped > 0:
        logger.warning(f"Dropped {deduped:,} duplicate rows based on text column.")

    df[CFG.label_column] = df[CFG.label_column].astype(int)

    label_counts = df[CFG.label_column].value_counts().to_dict()
    logger.info(f"Label distribution: {label_counts}")
    return df


def _split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train / validation / test split.

    Args:
        df          : Full cleaned DataFrame.
        train_ratio : Fraction for training set.
        val_ratio   : Fraction for validation set (rest goes to test).
        seed        : Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-6

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[CFG.label_column],
        random_state=seed,
    )

    # Second split: separate val from train_val
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[CFG.label_column],
        random_state=seed,
    )

    logger.info(
        f"Split sizes → Train: {len(train_df):,}  "
        f"Val: {len(val_df):,}  "
        f"Test: {len(test_df):,}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset_splits() -> DatasetDict:
    """
    Load the CSV, validate it, split it, and return a HuggingFace
    :class:`DatasetDict` with keys ``train``, ``validation``, and ``test``.

    Returns:
        :class:`datasets.DatasetDict`
    """
    df = _validate_csv(Path(CFG.data_path))

    train_df, val_df, test_df = _split_dataframe(
        df,
        train_ratio=CFG.train_ratio,
        val_ratio=CFG.val_ratio,
        seed=CFG.seed,
    )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[[CFG.text_column, CFG.label_column]]),
            "validation": Dataset.from_pandas(val_df[[CFG.text_column, CFG.label_column]]),
            "test": Dataset.from_pandas(test_df[[CFG.text_column, CFG.label_column]]),
        }
    )
    logger.info(f"DatasetDict created:\n{dataset_dict}")
    return dataset_dict


def tokenize_datasets(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """
    Apply batch tokenization to all splits in *dataset_dict*.

    Tokenizer settings are read from :data:`config.CFG`:
      - ``max_length`` = 256
      - ``padding``    = "max_length"
      - ``truncation`` = True

    Args:
        dataset_dict : HuggingFace DatasetDict with raw text.
        tokenizer    : Loaded :class:`AutoTokenizer` instance.

    Returns:
        Tokenized :class:`DatasetDict` with ``input_ids``,
        ``attention_mask``, and ``labels`` columns.
    """
    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[CFG.text_column],
            max_length=CFG.max_length,
            padding=CFG.padding,
            truncation=CFG.truncation,
        )

    logger.info(
        f"Tokenizing datasets (max_length={CFG.max_length}, "
        f"padding='{CFG.padding}', truncation={CFG.truncation}) …"
    )

    tokenized = dataset_dict.map(
        _tokenize,
        batched=True,
        batch_size=1000,                # Process 1000 samples per map batch
        num_proc=4 if os.cpu_count() and os.cpu_count() >= 4 else 1,
        remove_columns=[CFG.text_column],  # keep 'label' column
        desc="Tokenizing",
    )

    # Rename label → labels (HuggingFace Trainer expects 'labels' plural for loss)
    tokenized = tokenized.rename_column(CFG.label_column, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info("Tokenization complete.")
    return tokenized
