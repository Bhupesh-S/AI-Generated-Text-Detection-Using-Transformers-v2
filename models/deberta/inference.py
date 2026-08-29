"""
inference.py
============
Interactive CLI and programmatic API for DeBERTa sequence classification.

Features:
  - Single-text classification.
  - Text file batch processing (one text sample per line).
  - Interactive REPL console shell.

Run with:
    python models/deberta/inference.py --interactive
    python models/deberta/inference.py --text "Academic text to classify..."
    python models/deberta/inference.py --file texts.txt

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Path bootstrap
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core Inference Predictor
# ---------------------------------------------------------------------------

class DeBERTaInference:
    """Interface to run inference using a fine-tuned DeBERTa model checkpoint."""

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the predictor, loading the model and tokenizer to device.

        Args:
            checkpoint_path : Local directory path. Defaults to CFG.tokenizer_dir.
        """
        if checkpoint_path is None:
            checkpoint_path = CFG.tokenizer_dir

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Fine-tuned model directory not found at: {checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading tokenizer from: {checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

        logger.info(f"Loading model checkpoint from: {checkpoint_path} (use_safetensors={CFG.use_safetensors})")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            use_safetensors=CFG.use_safetensors
        ).to(self.device)
        self.model.eval()

        # Labels mapping
        self.id2label = CFG.id2label

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Classify a single input string and return the predicted class and confidence.

        Args:
            text : Raw academic text string.

        Returns:
            Dictionary containing:
              - ``prediction``  : class label string ('Human Written' or 'AI Generated')
              - ``confidence``  : float probability percentage of the prediction (0-100)
              - ``prob_human``  : float probability of Human Written class
              - ``prob_ai``     : float probability of AI Generated class
              - ``elapsed_ms``  : float prediction latency in milliseconds
        """
        t_start = time.perf_counter()

        # Tokenize and format tensor inputs
        inputs = self.tokenizer(
            text,
            max_length=CFG.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        logits = self.model(**inputs).logits.cpu().numpy()
        probs = softmax(logits, axis=-1)[0]

        pred_id = int(probs.argmax())
        prediction = self.id2label[pred_id]
        confidence = float(probs[pred_id]) * 100.0

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "prob_human": round(float(probs[0]), 5),
            "prob_ai": round(float(probs[1]), 5),
            "elapsed_ms": round(elapsed_ms, 2)
        }


# ---------------------------------------------------------------------------
# Interactive & Command Line Shells
# ---------------------------------------------------------------------------

def run_interactive_repl(predictor: DeBERTaInference) -> None:
    """Run an interactive REPL shell on the command console."""
    print("\n" + "=" * 70)
    print("  DeBERTa-v3 AI-Generated Text Detector Console REPL Shell")
    print("  Type 'exit' or 'quit' to close the interactive session.")
    print("=" * 70)

    while True:
        try:
            text = input("\nEnter text sample -> ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                print("Exiting interactive console shell. Goodbye!")
                break

            result = predictor.predict(text)

            print("-" * 50)
            print(f"Prediction : {result['prediction']}")
            print(f"Confidence : {result['confidence']}%")
            print(f"Probabilities:")
            print(f"  * Human Written : {result['prob_human'] * 100.0:.2f}%")
            print(f"  * AI Generated : {result['prob_ai'] * 100.0:.2f}%")
            print(f"Latency    : {result['elapsed_ms']:.1f} ms")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nExiting interactive console shell. Goodbye!")
            break
        except Exception as exc:
            logger.error(f"Inference error occurred: {exc}")


def process_file(predictor: DeBERTaInference, file_path: Path) -> None:
    """Read a text file line-by-line and run batch sequence classification."""
    logger.info(f"Reading input file: {file_path}")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info(f"Classifying {len(lines)} text samples...")
    t_start = time.time()

    for i, line in enumerate(lines, 1):
        # Snippet for output logging
        snippet = line[:60] + "..." if len(line) > 60 else line
        result = predictor.predict(line)
        print(
            f"[{i:03d}] "
            f"Predict: {result['prediction']} ({result['confidence']}%) | "
            f"Text: '{snippet}'"
        )

    elapsed = time.time() - t_start
    logger.info(f"Completed processing {len(lines)} rows in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBERTa AI-Generated Text Detector Inference CLI")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to local fine-tuned model checkpoint directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--interactive", action="store_true", help="Launch console REPL shell.")
    group.add_argument("--text", type=str, help="Single raw text sample to classify.")
    group.add_argument("--file", type=str, help="Text file path containing one sample per line.")

    args = parser.parse_args()

    target_dir = Path(args.model_dir) if args.model_dir else CFG.tokenizer_dir

    try:
        predictor = DeBERTaInference(checkpoint_path=target_dir)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        sys.exit(1)

    if args.interactive:
        run_interactive_repl(predictor)
    elif args.text:
        result = predictor.predict(args.text)
        print("\n" + "=" * 50)
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}%")
        print(f"Probabilities:")
        print(f"  * Human Written : {result['prob_human'] * 100.0:.2f}%")
        print(f"  * AI Generated : {result['prob_ai'] * 100.0:.2f}%")
        print(f"Latency    : {result['elapsed_ms']:.1f} ms")
        print("=" * 50 + "\n")
    elif args.file:
        process_file(predictor, Path(args.file))
