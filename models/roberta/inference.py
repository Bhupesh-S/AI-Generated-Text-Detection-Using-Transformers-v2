"""
inference.py
============
Inference module for the fine-tuned RoBERTa model.

Usage (CLI):
    python models/roberta/inference.py --text "Your academic text here"
    python models/roberta/inference.py --file path/to/texts.txt
    python models/roberta/inference.py --interactive

Programmatic usage:
    from models.roberta.inference import RoBERTaInference
    predictor = RoBERTaInference()
    result = predictor.predict("Your text here")
    print(result)

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np

# Allow ``python models/roberta/inference.py`` to find sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class RoBERTaInference:
    """
    High-level inference wrapper for the fine-tuned RoBERTa classifier.

    Example::

        predictor = RoBERTaInference()
        result = predictor.predict("Large language models have transformed NLP.")
        print(result)
        # {'text': '...', 'prediction': 'AI Generated',
        #  'confidence': 97.43, 'label_id': 1,
        #  'probabilities': {'Human Written': 2.57, 'AI Generated': 97.43}}
    """

    # Mapping from integer label → human-readable string
    ID2LABEL: Dict[int, str] = {0: "Human Written", 1: "AI Generated"}

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Load the tokenizer and model from *model_dir*.

        Args:
            model_dir : Directory containing the saved model and tokenizer
                        (output of ``save_pretrained``).  Defaults to
                        ``outputs/roberta/tokenizer``.
            device    : Torch device.  Defaults to CUDA if available.
        """
        if model_dir is None:
            model_dir = CFG.tokenizer_dir

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Run 'python models/roberta/train.py' first to train and save the model."
            )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")

        # ── Load tokenizer ───────────────────────────────────────────────
        logger.info(f"Loading tokenizer from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # ── Load model ───────────────────────────────────────────────────
        logger.info(f"Loading model from: {self.model_dir}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info("Model ready for inference.")

    # ── Core Prediction ──────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Classify a single text string.

        Args:
            text : Raw input text (will be truncated to ``max_length`` tokens).

        Returns:
            Dictionary with keys:
              - ``text``          : The (possibly truncated) input text.
              - ``prediction``    : Class label string (``"Human Written"`` or
                                    ``"AI Generated"``).
              - ``label_id``      : Integer label (0 or 1).
              - ``confidence``    : Confidence in % for the predicted class.
              - ``probabilities`` : Dict of class → probability (%).
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        t0 = time.time()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=CFG.max_length,
            padding="max_length",
            truncation=True,
        )
        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.cpu().numpy()
        probs  = softmax(logits, axis=-1)[0]           # shape: (2,)
        pred_id = int(np.argmax(probs))
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "text": text[:200] + " …" if len(text) > 200 else text,
            "prediction": self.ID2LABEL[pred_id],
            "label_id": pred_id,
            "confidence": round(float(probs[pred_id]) * 100, 2),
            "probabilities": {
                self.ID2LABEL[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            },
            "inference_time_ms": round(elapsed_ms, 1),
        }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Classify a list of texts in mini-batches for efficiency.

        Args:
            texts      : List of raw text strings.
            batch_size : Mini-batch size.

        Returns:
            List of result dicts (same structure as :meth:`predict`).
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=CFG.max_length,
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits.cpu().numpy()
            probs_batch = softmax(logits, axis=-1)

            for j, (text, probs) in enumerate(zip(batch, probs_batch)):
                pred_id = int(np.argmax(probs))
                results.append(
                    {
                        "text": text[:200] + " …" if len(text) > 200 else text,
                        "prediction": self.ID2LABEL[pred_id],
                        "label_id": pred_id,
                        "confidence": round(float(probs[pred_id]) * 100, 2),
                        "probabilities": {
                            self.ID2LABEL[k]: round(float(p) * 100, 2)
                            for k, p in enumerate(probs)
                        },
                    }
                )
        return results

    # ── Pretty print ─────────────────────────────────────────────────────────

    @staticmethod
    def format_result(result: Dict) -> str:
        """
        Format a prediction result dict into a human-readable string.

        Example output::

            ┌──────────────────────────────────────────────────┐
            │  Prediction  : AI Generated                      │
            │  Confidence  : 99.14%                            │
            │  Human Written : 0.86%                           │
            │  AI Generated  : 99.14%                          │
            │  Inference     : 23.4 ms                         │
            └──────────────────────────────────────────────────┘
        """
        lines = [
            "┌" + "─" * 52 + "┐",
            f"│  {'Prediction':<14}: {result['prediction']:<34}│",
            f"│  {'Confidence':<14}: {result['confidence']:.2f}%{'':<29}│",
        ]
        for cls, prob in result["probabilities"].items():
            lines.append(f"│  {cls:<14}: {prob:.2f}%{'':<29}│")
        if "inference_time_ms" in result:
            lines.append(f"│  {'Inference':<14}: {result['inference_time_ms']:.1f} ms{'':<26}│")
        lines.append("└" + "─" * 52 + "┘")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with the fine-tuned RoBERTa model."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--text",
        type=str,
        help="Single text string to classify.",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a .txt file; each line is treated as a separate sample.",
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive REPL loop.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(CFG.tokenizer_dir),
        help="Directory of the saved model + tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI usage."""
    args = parse_args()

    # Load model once
    predictor = RoBERTaInference(model_dir=args.model_dir)

    if args.text:
        result = predictor.predict(args.text)
        print(predictor.format_result(result))

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)

        texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        logger.info(f"Classifying {len(texts)} lines from {file_path} …")

        results = predictor.predict_batch(texts)
        for i, (text, result) in enumerate(zip(texts, results), start=1):
            print(f"\n[{i}] {text[:80]} …")
            print(predictor.format_result(result))

    else:
        # Interactive REPL
        print("\nRoBERTa AI-Generated Text Detector (type 'quit' to exit)\n")
        while True:
            try:
                text = input("Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if text.lower() in {"quit", "exit", "q"}:
                break
            if not text:
                print("Please enter some text.\n")
                continue

            result = predictor.predict(text)
            print(predictor.format_result(result))
            print()


if __name__ == "__main__":
    main()
