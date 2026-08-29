"""
Baseline Model Training Pipeline
==================================
Trains five baseline classifiers on TF-IDF features, evaluates each on the
held-out test split, saves models + artefacts, and generates a full report.

Models
------
1. Logistic Regression   (scikit-learn / saga solver)
2. Multinomial Naive Bayes
3. Linear SVM            (LinearSVC + CalibratedClassifierCV)
4. Random Forest
5. XGBoost               (GPU if CUDA available and VRAM sufficient)

Usage
-----
    # From repository root:
    python models/baseline/train.py

    # Feature engineering is triggered automatically if artefacts are absent.
    # To force re-running feature engineering:
    python models/baseline/train.py --rerun-fe

Options
-------
    --rerun-fe      Force feature-engineering even if artefacts exist
    --force-cpu     Disable GPU for XGBoost
    --n-jobs N      Number of parallel workers (-1 = all cores)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from tqdm import tqdm
import xgboost as xgb

# ---------------------------------------------------------------------------
# Path bootstrap — make repository root importable regardless of CWD
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT  = _THIS_FILE.parent.parent.parent   # .../CODE/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Now import from sibling packages
from feature_engineering.vectorizer import load_sparse_matrix, load_labels
from feature_engineering.feature_engineering import FeatureEngineeringPipeline

# evaluate lives in the same directory as this file
sys.path.insert(0, str(_THIS_FILE.parent))
from evaluate import ModelEvaluator

# ---------------------------------------------------------------------------
# Logging  (console + rotating file in logs/)
# ---------------------------------------------------------------------------
_LOG_DIR = _REPO_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler
if not any(isinstance(h, logging.StreamHandler) for h in _root_logger.handlers):
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt)
    _root_logger.addHandler(_ch)

# Rotating file handler  (10 MB × 5 backups)
from logging.handlers import RotatingFileHandler as _RFH  # noqa: E402
_fh = _RFH(
    _LOG_DIR / "baseline_training.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_fh.setFormatter(_fmt)
_root_logger.addHandler(_fh)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-wide paths (all relative to REPO_ROOT)
# ---------------------------------------------------------------------------
DATA_PATH   = _REPO_ROOT / "Datasets" / "merged" / "final_dataset.csv"
OUTPUT_DIR  = _REPO_ROOT / "outputs"
TFIDF_DIR   = OUTPUT_DIR / "tfidf"
LABELS_DIR  = OUTPUT_DIR / "labels"
MODELS_DIR  = OUTPUT_DIR / "models"


# ===========================================================================
# GPU helpers
# ===========================================================================

def _check_cuda() -> bool:
    """Return True if CUDA (nvidia-smi + XGBoost) is available."""
    # Method 1: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("CUDA detected via nvidia-smi.")
            return True
    except Exception:
        pass

    # Method 2: probe XGBoost CUDA with a tiny model
    try:
        probe = xgb.XGBClassifier(
            device="cuda", tree_method="hist",
            n_estimators=1, max_depth=1, verbosity=0,
        )
        rng = np.random.default_rng(0)
        probe.fit(rng.random((10, 4)), rng.integers(0, 2, 10), verbose=False)
        logger.info("CUDA detected via XGBoost probe.")
        return True
    except Exception:
        pass

    logger.info("CUDA not available — XGBoost will run on CPU.")
    return False


def _gpu_free_gb() -> Optional[float]:
    """Return free GPU VRAM in GB (None if unavailable)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0]) / 1024.0
    except Exception:
        pass
    return None


def _estimate_xgb_vram_gb(X_train, n_estimators: int = 100) -> float:
    """Rough upper-bound estimate of VRAM needed (GB) for XGBoost GPU training."""
    n, d = X_train.shape
    data_gb  = (n * d * 4) / (1024 ** 3)     # dense float32 copy on GPU
    grad_gb  = data_gb * 2.0                  # gradients + hessians
    tree_gb  = (n_estimators * d * 8) / (1024 ** 3)
    return (data_gb + grad_gb + tree_gb) * 1.25   # 25 % headroom


# ===========================================================================
# Trainer
# ===========================================================================

class BaselineModelTrainer:
    """
    Orchestrates training, evaluation, and persistence of all five baseline
    classifiers.

    Parameters
    ----------
    output_dir : str | Path
        Root output directory.
    n_jobs : int
        Parallel workers for scikit-learn models (-1 = all cores).
    force_cpu : bool
        Disable GPU for XGBoost even if CUDA is available.
    """

    def __init__(
        self,
        output_dir: str | Path = OUTPUT_DIR,
        n_jobs: int = -1,
        force_cpu: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_jobs     = n_jobs
        self.force_cpu  = force_cpu

        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = ModelEvaluator(str(self.output_dir))

        self._models: Dict = {}
        self._train_times: Dict[str, float] = {}

        # Determine XGBoost device (CPU / cuda)
        self._cuda = (not force_cpu) and _check_cuda()
        self._xgb_device = "cpu"

        n_cpu = os.cpu_count() or 1
        logger.info("System CPUs : %d", n_cpu)
        logger.info("n_jobs      : %d", n_jobs)
        logger.info("CUDA avail  : %s", self._cuda)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_dir: str | Path) -> Tuple:
        """
        Load pre-computed TF-IDF matrices, handcrafted features, and labels from *data_dir*.

        Parameters
        ----------
        data_dir : str | Path
            Directory containing ``tfidf/`` and ``labels/`` sub-folders.

        Returns
        -------
        X_train, X_valid, X_test, y_train, y_valid, y_test, (hc_train, hc_valid, hc_test)
        """
        d = Path(data_dir)
        logger.info("Loading feature-engineered data from %s …", d)

        X_train = load_sparse_matrix(d / "tfidf" / "X_train.npz")
        X_valid = load_sparse_matrix(d / "tfidf" / "X_valid.npz")
        X_test  = load_sparse_matrix(d / "tfidf" / "X_test.npz")

        y_train = load_labels(d / "labels" / "y_train.csv")
        y_valid = load_labels(d / "labels" / "y_valid.csv")
        y_test  = load_labels(d / "labels" / "y_test.csv")

        # Load handcrafted features
        hc_train = pd.read_csv(d / "handcrafted" / "train_features.csv")
        hc_valid = pd.read_csv(d / "handcrafted" / "valid_features.csv")
        hc_test  = pd.read_csv(d / "handcrafted" / "test_features.csv")

        logger.info(
            "Shapes — X_train %s | X_valid %s | X_test %s | Handcrafted Train %s",
            X_train.shape, X_valid.shape, X_test.shape, hc_train.shape
        )

        # Decide XGBoost device now that data shape is known
        self._select_xgb_device(X_train)

        return X_train, X_valid, X_test, y_train, y_valid, y_test, (hc_train, hc_valid, hc_test)

    def _select_xgb_device(self, X_train) -> None:
        """Pick GPU or CPU for XGBoost based on VRAM availability."""
        if not self._cuda:
            self._xgb_device = "cpu"
            return

        needed = _estimate_xgb_vram_gb(X_train)
        free   = _gpu_free_gb()

        if free is not None:
            logger.info(
                "XGBoost VRAM estimate: %.2f GB needed | %.2f GB free", needed, free
            )
            if free >= needed + 0.5:
                self._xgb_device = "cuda"
                logger.info("XGBoost → GPU (CUDA)")
                return
            else:
                logger.warning(
                    "Insufficient VRAM (%.2f GB free < %.2f GB needed+buffer); "
                    "falling back to CPU.", free, needed + 0.5
                )

        self._xgb_device = "cpu"
        logger.info("XGBoost → CPU")

    # ------------------------------------------------------------------
    # Individual model trainers
    # ------------------------------------------------------------------

    def train_logistic_regression(
        self, X_train, y_train,
        max_iter: int = 1000, C: float = 1.0,
    ) -> LogisticRegression:
        """
        Logistic Regression with 'saga' solver.

        'saga' handles sparse matrices efficiently and supports both L1/L2.
        """
        logger.info("▶  Training Logistic Regression …")
        model = LogisticRegression(
            C=C, max_iter=max_iter,
            solver="saga",
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        self._train_times["Logistic Regression"] = elapsed
        logger.info("Logistic Regression trained in %.2f s", elapsed)
        return model

    def train_naive_bayes(
        self, X_train, y_train, alpha: float = 0.1
    ) -> MultinomialNB:
        """
        Multinomial Naive Bayes.

        alpha=0.1 (Lidstone smoothing) tends to outperform alpha=1.0 on TF-IDF.
        """
        logger.info("▶  Training Multinomial Naive Bayes …")
        model = MultinomialNB(alpha=alpha)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        self._train_times["Multinomial Naive Bayes"] = elapsed
        logger.info("Naive Bayes trained in %.2f s", elapsed)
        return model

    def train_linear_svm(
        self, X_train, y_train,
        C: float = 1.0, max_iter: int = 2000,
    ) -> CalibratedClassifierCV:
        """
        LinearSVC wrapped in CalibratedClassifierCV for probability outputs.

        dual=False is faster when n_samples >> n_features.
        """
        logger.info("▶  Training Linear SVM …")
        base_svm = LinearSVC(
            C=C, max_iter=max_iter, dual=False, random_state=42, verbose=0,
        )
        # 3-fold sigmoid calibration: faster than 5-fold isotonic
        model = CalibratedClassifierCV(
            base_svm, cv=3, method="sigmoid", n_jobs=self.n_jobs,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        self._train_times["Linear SVM"] = elapsed
        logger.info("Linear SVM trained in %.2f s", elapsed)
        return model

    def train_random_forest(
        self, X_train, y_train,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
    ) -> RandomForestClassifier:
        """
        Random Forest with sqrt feature sampling (best for high-dimensional text).
        """
        logger.info("▶  Training Random Forest …")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=42,
            n_jobs=self.n_jobs,
            oob_score=False,
            verbose=0,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        self._train_times["Random Forest"] = elapsed
        logger.info("Random Forest trained in %.2f s", elapsed)
        return model

    def train_xgboost(
        self, X_train, y_train, X_valid=None, y_valid=None,
        n_estimators: int = 150,    # Let early stopping decide early termination
        max_depth: int = 4,         # Shallow trees prevent overfitting
        learning_rate: float = 0.05,
    ) -> xgb.XGBClassifier:
        """
        XGBoost with hist tree method, early stopping, and regularisation.

        Automatically uses GPU (CUDA) when VRAM is sufficient; falls back to CPU.
        """
        device_str = "GPU (CUDA)" if self._xgb_device == "cuda" else "CPU"
        logger.info("▶  Training XGBoost on %s …", device_str)

        n_jobs_xgb = 1 if self._xgb_device == "cuda" else self.n_jobs
        early_stopping_rounds = 10 if (X_valid is not None and y_valid is not None) else None

        model = xgb.XGBClassifier(
            # Architecture
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            # Tree algorithm
            tree_method="hist",
            max_bin=256,
            # Regularisation
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=1.0,           # L1 penalty
            reg_lambda=2.0,          # L2 penalty
            min_child_weight=3,      # Prevent overfitting to small leaves
            scale_pos_weight=1.0,
            # Objective
            objective="binary:logistic",
            eval_metric="logloss",
            booster="gbtree",
            # Device / parallelism
            device=self._xgb_device,
            n_jobs=n_jobs_xgb,
            # Early stopping
            early_stopping_rounds=early_stopping_rounds,
            # Misc
            random_state=42,
            verbosity=0,
            enable_categorical=False,
        )

        t0 = time.perf_counter()
        if X_valid is not None and y_valid is not None:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        elapsed = time.perf_counter() - t0
        self._train_times["XGBoost"] = elapsed

        # Log how many trees were actually used
        best_iteration = getattr(model, "best_iteration", n_estimators - 1)
        logger.info(
            "XGBoost trained in %.4fs (used %d trees, stopped at iteration %d) on %s",
            elapsed, best_iteration + 1, best_iteration + 1, device_str
        )
        return model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model, model_name: str) -> Path:
        """Serialise *model* to ``outputs/models/<name>.joblib``."""
        slug = model_name.strip().lower().replace(" ", "_")
        fp = self.models_dir / f"{slug}.joblib"
        joblib.dump(model, fp, compress=3)
        logger.info("Model saved → %s", fp)
        return fp

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train_xgboost_only(
        self,
        X_train, X_test,
        y_train, y_test,
        X_valid=None, y_valid=None,
    ) -> Dict:
        """
        Train **only** XGBoost, evaluate it, and save its artefact.

        Returns
        -------
        Dict of { 'XGBoost': fitted_model }
        """
        logger.info("=" * 65)
        logger.info("  XGBoost-Only Training")
        logger.info("=" * 65)

        model = self.train_xgboost(X_train, y_train, X_valid, y_valid)
        self._models["XGBoost"] = model
        self.save_model(model, "XGBoost")
        self.evaluator.evaluate_model(
            model, X_test, y_test, "XGBoost",
            self._train_times["XGBoost"],
        )

        logger.info("=" * 65)
        logger.info("  XGBoost training & evaluation complete.")
        logger.info("=" * 65)
        return self._models

    def train_all_models(
        self,
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        hc_splits=None,
    ) -> Dict:
        """
        Train all five models, evaluate each, and save artefacts.

        Returns
        -------
        Dict of { model_name: fitted_model }
        """
        logger.info("=" * 65)
        logger.info("  Baseline Training Pipeline — 5 Models")
        logger.info("=" * 65)

        steps = [
            ("Logistic Regression",   self.train_logistic_regression),
            ("Multinomial Naive Bayes", self.train_naive_bayes),
            ("Linear SVM",            self.train_linear_svm),
            ("Random Forest",         self.train_random_forest),
        ]

        for name, train_fn in tqdm(steps, desc="Training models", unit="model"):
            model = train_fn(X_train, y_train)
            self._models[name] = model
            self.save_model(model, name)
            self.evaluator.evaluate_model(
                model, X_test, y_test, name,
                self._train_times[name],
            )

        # XGBoost is trained on handcrafted features if available
        if hc_splits is not None:
            hc_train, hc_valid, hc_test = hc_splits
            logger.info("Training XGBoost on Handcrafted Stylometric Features...")
            xgb_model = self.train_xgboost(hc_train, y_train, hc_valid, y_valid)
            self._models["XGBoost"] = xgb_model
            self.save_model(xgb_model, "XGBoost")
            self.evaluator.evaluate_model(
                xgb_model, hc_test, y_test, "XGBoost",
                self._train_times["XGBoost"],
            )
        else:
            xgb_model = self.train_xgboost(X_train, y_train, X_valid, y_valid)
            self._models["XGBoost"] = xgb_model
            self.save_model(xgb_model, "XGBoost")
            self.evaluator.evaluate_model(
                xgb_model, X_test, y_test, "XGBoost",
                self._train_times["XGBoost"],
            )

        logger.info("=" * 65)
        logger.info("  All 5 models trained and evaluated.")
        logger.info("=" * 65)
        return self._models

    # ------------------------------------------------------------------
    # Plots + reporting
    # ------------------------------------------------------------------

    def generate_all_plots(self) -> None:
        """Generate all comparison visualisations."""
        logger.info("Generating aggregate comparison plots …")
        ev = self.evaluator

        for metric in ("accuracy", "f1_score", "roc_auc",
                       "training_time", "inference_time"):
            ev.plot_model_comparison(metric)

        ev.plot_all_roc_curves()
        ev.plot_all_pr_curves()
        ev.plot_interactive_comparison()

        logger.info("All plots generated.")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        data_dir: str | Path,
        run_feature_engineering: bool = False,
        xgboost_only: bool = False,
    ) -> None:
        """
        Execute the end-to-end training + evaluation pipeline.

        Parameters
        ----------
        data_dir : str | Path
            Directory containing ``tfidf/`` and ``labels/`` artefacts.
        run_feature_engineering : bool
            If True, runs Phase-1 feature engineering first.
        xgboost_only : bool
            If True, skip all other models and train only XGBoost.
        """
        data_dir = Path(data_dir)

        # ── Optional Phase 1 ─────────────────────────────────────────
        artefacts_exist = (
            (data_dir / "tfidf" / "X_train.npz").exists()
            and (data_dir / "labels" / "y_train.csv").exists()
        )

        if run_feature_engineering or not artefacts_exist:
            logger.info("Running Phase-1 Feature Engineering …")
            fe_pipeline = FeatureEngineeringPipeline(
                max_features=50_000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
                test_size=0.10,      # Standardized 80/10/10 split
                valid_size=0.10,     # Standardized 80/10/10 split
                random_state=42,
            )
            fe_pipeline.run_pipeline(
                data_path=str(DATA_PATH),
                output_dir=str(data_dir),
            )
        else:
            logger.info(
                "TF-IDF artefacts found — skipping feature engineering. "
                "(Use --rerun-fe to force.)"
            )

        # ── Phase 2 — Load data ────────────────────────────────────────
        X_train, X_valid, X_test, y_train, y_valid, y_test, hc_splits = self.load_data(data_dir)
        hc_train, hc_valid, hc_test = hc_splits

        # ── Train ─────────────────────────────────────────────────────
        if xgboost_only:
            self.train_xgboost_only(hc_train, hc_test, y_train, y_test, hc_valid, y_valid)
        else:
            self.train_all_models(X_train, X_valid, X_test, y_train, y_valid, y_test, hc_splits)

        # ── Plots ─────────────────────────────────────────────────────
        self.generate_all_plots()

        # ── Reports ───────────────────────────────────────────────────
        self.evaluator.save_results_table()
        self.evaluator.generate_report()

        logger.info("=" * 65)
        logger.info("  Pipeline complete.  All outputs → %s", self.output_dir)
        logger.info("=" * 65)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Explainable Transformer Framework — Baseline Training"
    )
    parser.add_argument(
        "--rerun-fe", action="store_true",
        help="Force re-running feature engineering even if artefacts exist.",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Disable GPU (CUDA) for XGBoost.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel workers for scikit-learn models (-1 = all cores).",
    )
    parser.add_argument(
        "--xgboost-only", action="store_true",
        help="Train and evaluate only the XGBoost model (skips all others).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point — called when running:

        python models/baseline/train.py [options]
    """
    args = _parse_args()

    logger.info("=" * 65)
    logger.info("  Hybrid Explainable Transformer Framework")
    logger.info("  Baseline Model Training Pipeline")
    logger.info("=" * 65)
    logger.info("Repository root : %s", _REPO_ROOT)
    logger.info("Dataset         : %s", DATA_PATH)
    logger.info("Output dir      : %s", OUTPUT_DIR)
    logger.info("n_jobs          : %d", args.n_jobs)
    logger.info("Force CPU       : %s", args.force_cpu)
    logger.info("Rerun FE        : %s", args.rerun_fe)
    logger.info("XGBoost Only    : %s", args.xgboost_only)

    trainer = BaselineModelTrainer(
        output_dir=OUTPUT_DIR,
        n_jobs=args.n_jobs,
        force_cpu=args.force_cpu,
    )

    try:
        trainer.run_pipeline(
            data_dir=OUTPUT_DIR,
            run_feature_engineering=args.rerun_fe,
            xgboost_only=args.xgboost_only,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
