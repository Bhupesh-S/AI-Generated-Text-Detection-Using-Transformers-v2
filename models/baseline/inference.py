"""
Inference Module
================
Loads a trained baseline model + TF-IDF vectorizer and makes predictions
on raw text (single or batch).

Usage
-----
    python models/baseline/inference.py

Or import:

    from models.baseline.inference import ModelInference
    inf = ModelInference(model_path=..., vectorizer_path=...)
    result = inf.predict_single("Some text to classify.")
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_engineering.preprocessing import TextPreprocessor
from feature_engineering.vectorizer import TfidfVectorizerWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_LABEL_MAP = {0: "Human Written", 1: "AI Generated"}


class ModelInference:
    """
    Prediction interface for a trained baseline classifier.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.joblib`` model file.
    vectorizer_path : str | Path
        Path to the fitted TF-IDF vectorizer (``.pkl``).
    model_name : str, optional
        Human-readable name used in logging.
    """

    def __init__(
        self,
        model_path: str | Path,
        vectorizer_path: str | Path,
        model_name: Optional[str] = None,
    ) -> None:
        self.model_path      = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model_name      = model_name or self.model_path.stem

        logger.info("Loading model '%s' from %s", self.model_name, self.model_path)
        self.model = joblib.load(self.model_path)

        logger.info("Loading TF-IDF vectorizer from %s", self.vectorizer_path)
        self.vectorizer = TfidfVectorizerWrapper()
        self.vectorizer.load(str(self.vectorizer_path))

        self.preprocessor = TextPreprocessor()
        logger.info("ModelInference ready — %s", self.model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        return self.preprocessor.clean_text(text)

    def _vectorise(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def _build_result(
        self,
        prediction: int,
        probabilities: Optional[np.ndarray],
    ) -> Dict:
        label = _LABEL_MAP.get(prediction, "Unknown")
        if probabilities is not None:
            confidence = float(np.max(probabilities))
            prob_human = float(probabilities[0])
            prob_ai    = float(probabilities[1])
        else:
            confidence = prob_human = prob_ai = None

        return {
            "prediction":       prediction,
            "label":            label,
            "confidence":       confidence,
            "prob_human_written": prob_human,
            "prob_ai_generated":  prob_ai,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, text: str) -> Dict:
        """
        Predict the origin of a single text.

        Args:
            text: Raw input string.

        Returns:
            Dict with keys: prediction, label, confidence,
            prob_human_written, prob_ai_generated.
        """
        cleaned = self._clean(text)
        if not cleaned:
            logger.warning("Empty text after preprocessing.")
            return self._build_result(0, None)

        t0 = time.perf_counter()
        vec = self._vectorise([cleaned])
        y_pred = int(self.model.predict(vec)[0])
        elapsed = time.perf_counter() - t0

        try:
            proba = self.model.predict_proba(vec)[0]
        except AttributeError:
            proba = None

        result = self._build_result(y_pred, proba)
        result["inference_time_s"] = round(elapsed, 6)
        logger.info(
            "Prediction: %s (confidence=%.4f) in %.4fs",
            result["label"], result["confidence"] or 0.0, elapsed,
        )
        return result

    def predict_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Predict origin for a list of texts.

        Args:
            texts:         List of raw input strings.
            show_progress: Show tqdm progress bar.

        Returns:
            List of result dicts (same order as *texts*).
        """
        if show_progress:
            texts = list(tqdm(texts, desc="Preprocessing", unit="doc"))

        cleaned = [self._clean(t) for t in texts]
        empty_mask = [c == "" for c in cleaned]
        valid_texts = [c for c in cleaned if c]

        if not valid_texts:
            logger.warning("No valid texts after preprocessing.")
            return [self._build_result(0, None)] * len(texts)

        vec = self._vectorise(valid_texts)
        preds = self.model.predict(vec)
        try:
            probas = self.model.predict_proba(vec)
        except AttributeError:
            probas = None

        results: List[Dict] = []
        valid_idx = 0
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                results.append(self._build_result(0, None))
            else:
                proba = probas[valid_idx] if probas is not None else None
                results.append(self._build_result(int(preds[valid_idx]), proba))
                valid_idx += 1

        return results

    def predict_from_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_column: str = "text",
    ) -> pd.DataFrame:
        """
        Run batch inference on a CSV file and save results.

        Args:
            input_csv:   Path to input CSV.
            output_csv:  Path to save predictions CSV.
            text_column: Name of the text column.

        Returns:
            DataFrame with original data + prediction columns appended.
        """
        df = pd.read_csv(input_csv)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {input_csv}.")

        texts   = df[text_column].fillna("").tolist()
        results = self.predict_batch(texts)

        df["prediction"]          = [r["prediction"]          for r in results]
        df["label"]               = [r["label"]               for r in results]
        df["confidence"]          = [r["confidence"]          for r in results]
        df["prob_human_written"]  = [r["prob_human_written"]  for r in results]
        df["prob_ai_generated"]   = [r["prob_ai_generated"]   for r in results]

        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Predictions saved → %s", out)
        return df


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR       = _REPO_ROOT / "outputs"
    MODEL_PATH       = OUTPUT_DIR / "models" / "logistic_regression.joblib"
    VECTORIZER_PATH  = OUTPUT_DIR / "tfidf"  / "tfidf_vectorizer.pkl"

    sample_texts = [
        "This essay was hastily written by me last night with lots of mistakes.",
        (
            "The proliferation of large language models has engendered significant "
            "discourse regarding the authenticity and attribution of textual content "
            "within academic and professional domains."
        ),
        "I think climate change is real and we should do something about it soon.",
    ]

    if not MODEL_PATH.exists():
        logger.error(
            "No trained model found at %s. Run train.py first.", MODEL_PATH
        )
        return

    inf = ModelInference(
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
        model_name="Logistic Regression",
    )

    logger.info("\n%s\nSingle-text predictions\n%s", "─" * 50, "─" * 50)
    for i, text in enumerate(sample_texts, 1):
        result = inf.predict_single(text)
        logger.info(
            "[%d] %-75s → %s (conf=%.4f)",
            i, text[:75], result["label"], result["confidence"] or 0.0,
        )

    logger.info("\nBatch prediction …")
    batch_results = inf.predict_batch(sample_texts, show_progress=True)
    for i, r in enumerate(batch_results, 1):
        logger.info("[%d] %s (conf=%.4f)", i, r["label"], r["confidence"] or 0.0)


if __name__ == "__main__":
    main()
