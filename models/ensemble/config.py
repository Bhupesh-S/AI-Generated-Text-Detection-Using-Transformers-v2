"""
config.py
=========
Central configuration for the Hybrid Ensemble Pipeline.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

from pathlib import Path
from dataclasses import dataclass, field

# Base project root (two levels up from models/ensemble/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EnsembleConfig:
    """All configuration parameters for the Ensemble model."""

    # ── Model Paths ──────────────────────────────────────────────────────────
    roberta_path: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "roberta"
    deberta_path: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "deberta" / "final"
    distilbert_path: Path = PROJECT_ROOT / "outputs" / "retraining_v2_1" / "distilbert" / "final"
    xgboost_path: Path = PROJECT_ROOT / "outputs" / "models" / "xgboost.joblib"
    vectorizer_path: Path = PROJECT_ROOT / "outputs" / "tfidf" / "tfidf_vectorizer.pkl"

    # ── Data ─────────────────────────────────────────────────────────────────
    data_path: Path = PROJECT_ROOT / "Datasets" / "merged" / "final_dataset.csv"
    text_column: str = "text"
    label_column: str = "label"

    # ── Splits ───────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42

    # ── Voting Configuration ─────────────────────────────────────────────────
    # Default model weights
    roberta_weight: float = 0.35
    deberta_weight: float = 0.35
    distilbert_weight: float = 0.20
    xgboost_weight: float = 0.10

    # ── Output directories ───────────────────────────────────────────────────
    output_dir: Path = PROJECT_ROOT / "outputs" / "ensemble"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "ensemble" / "checkpoints"
    predictions_dir: Path = PROJECT_ROOT / "outputs" / "ensemble" / "predictions"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "ensemble" / "reports"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "ensemble" / "figures"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "ensemble" / "logs"

    # ── Stacking Checkpoint ──────────────────────────────────────────────────
    meta_model_path: Path = PROJECT_ROOT / "outputs" / "ensemble" / "checkpoints" / "meta_classifier.joblib"
    voting_config_path: Path = PROJECT_ROOT / "outputs" / "ensemble" / "checkpoints" / "voting_config.json"

    def create_dirs(self) -> None:
        """Create all output directories if they don't already exist."""
        dirs = [
            self.output_dir,
            self.checkpoint_dir,
            self.predictions_dir,
            self.reports_dir,
            self.figures_dir,
            self.logs_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize configuration variables to a dictionary."""
        return {
            "roberta_path": str(self.roberta_path),
            "deberta_path": str(self.deberta_path),
            "distilbert_path": str(self.distilbert_path),
            "xgboost_path": str(self.xgboost_path),
            "vectorizer_path": str(self.vectorizer_path),
            "data_path": str(self.data_path),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "weights": {
                "roberta": self.roberta_weight,
                "deberta": self.deberta_weight,
                "distilbert": self.distilbert_weight,
                "xgboost": self.xgboost_weight,
            },
            "output_dir": str(self.output_dir),
        }


# Singleton config instance
CFG = EnsembleConfig()
