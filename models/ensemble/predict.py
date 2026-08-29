"""
predict.py
==========
Interactive CLI and programmatic inference engine for the Hybrid Ensemble framework.

Canonical Result Schema:
{
    "prediction": "Human Written",              # or "AI Generated"
    "ai_probability": 0.0501,                   # float in [0.0, 1.0]
    "human_probability": 0.9499,                # float in [0.0, 1.0]
    "confidence": 0.9499,                       # float in [0.0, 1.0]
    "strategy_used": "Stacking Meta-Classifier",
    "latency_ms": 783.0,
    "text": "Four score and seven years ago...",

    "model_breakdown": {
        "roberta": {
            "ai_probability": 0.1179,
            "human_probability": 0.8821,
            "prediction": "Human Written"
        },
        "deberta": {
            "ai_probability": 0.2323,
            "human_probability": 0.7677,
            "prediction": "Human Written"
        },
        "distilbert": {
            "ai_probability": 0.0593,
            "human_probability": 0.9407,
            "prediction": "Human Written"
        },
        "xgboost": {
            "ai_probability": 0.1118,
            "human_probability": 0.8882,
            "prediction": "Human Written"
        }
    },

    "consensus": {
        "ai_votes": 0,
        "human_votes": 4,
        "state": "UNANIMOUS CONSENSUS (4/4 Human)",
        "votes": {
            "roberta": "Human",
            "deberta": "Human",
            "distilbert": "Human",
            "xgboost": "Human"
        }
    },

    "meta_classifier": {
        "classes": [0, 1],
        "raw_probabilities": [0.9499, 0.0501],
        "ai_probability": 0.0501,
        "human_probability": 0.9499
    }
}

Author  : Bharanidharan K & Antigravity
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow finding sibling modules
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_FILE.parent))

from config import CFG
from utils import get_logger, load_json
from stacking import prepare_stacking_features

logger = get_logger(__name__)


def validate_result_schema(result: Dict) -> None:
    """Validate canonical result schema types, required keys, and ranges."""
    required_keys = [
        "prediction",
        "ai_probability",
        "human_probability",
        "confidence",
        "model_breakdown",
        "consensus",
        "meta_classifier",
        "strategy_used"
    ]
    for key in required_keys:
        if key not in result:
            raise ValueError(
                f"HybridInferencePipeline returned an invalid result schema: missing required field '{key}'"
            )

    ai_p = result["ai_probability"]
    hu_p = result["human_probability"]
    conf = result["confidence"]

    if not isinstance(ai_p, (float, np.floating)) or not (0.0 <= ai_p <= 1.0):
        raise ValueError(f"Invalid ai_probability: {ai_p}. Expected float in [0.0, 1.0].")
    if not isinstance(hu_p, (float, np.floating)) or not (0.0 <= hu_p <= 1.0):
        raise ValueError(f"Invalid human_probability: {hu_p}. Expected float in [0.0, 1.0].")
    if not isinstance(conf, (float, np.floating)) or not (0.0 <= conf <= 1.0):
        raise ValueError(f"Invalid confidence: {conf}. Expected float in [0.0, 1.0].")

    if abs((ai_p + hu_p) - 1.0) > 1e-3:
        raise ValueError(f"Probabilities do not sum to 1.0: ai_prob={ai_p}, human_prob={hu_p}")


class HybridInferencePipeline:
    """High-level wrapper to run inference using the Hybrid Ensemble Model."""

    ID2LABEL: Dict[int, str] = {0: "Human Written", 1: "AI Generated"}

    def __init__(self):
        logger.info("Initializing Hybrid Inference Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # ── 1. Validate & Load Base Models ────────────────────────────────────
        roberta_path = CFG.roberta_path
        if "models/roberta" in str(roberta_path).replace("\\", "/"):
            v2_1_roberta = _REPO_ROOT / "outputs" / "retraining_v2_1" / "roberta"
            tokenizer_roberta = _REPO_ROOT / "outputs" / "roberta" / "tokenizer"
            if v2_1_roberta.exists():
                roberta_path = v2_1_roberta
            elif tokenizer_roberta.exists():
                roberta_path = tokenizer_roberta
            else:
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Stale model path '{roberta_path}' rejected and no fine-tuned "
                    f"V2.1 model found at '{v2_1_roberta}' or '{tokenizer_roberta}'."
                )

        if not roberta_path.exists():
            raise FileNotFoundError(f"RoBERTa model path does not exist: {roberta_path}")
        if not CFG.deberta_path.exists():
            raise FileNotFoundError(f"DeBERTa model path does not exist: {CFG.deberta_path}")
        if not CFG.distilbert_path.exists():
            raise FileNotFoundError(f"DistilBERT model path does not exist: {CFG.distilbert_path}")
        if not CFG.xgboost_path.exists():
            raise FileNotFoundError(f"XGBoost model path does not exist: {CFG.xgboost_path}")

        # Load TF-IDF Vectorizer & XGBoost
        logger.info(f"Loading TF-IDF Vectorizer: {CFG.vectorizer_path}")
        self.vectorizer = joblib.load(str(CFG.vectorizer_path))
        logger.info(f"Loading XGBoost model: {CFG.xgboost_path}")
        self.model_xgboost = joblib.load(str(CFG.xgboost_path))

        if hasattr(self.model_xgboost, "classes_"):
            xgb_classes = list(self.model_xgboost.classes_)
            if xgb_classes != [0, 1]:
                raise ValueError(f"Incompatible XGBoost classes_: {xgb_classes}. Expected [0, 1].")

        # Load Transformers
        def load_tok(local_path, base_name):
            try:
                return AutoTokenizer.from_pretrained(str(local_path))
            except Exception:
                return AutoTokenizer.from_pretrained(base_name)

        # RoBERTa
        self.tokenizer_roberta = load_tok(roberta_path, "roberta-base")
        self.model_roberta = AutoModelForSequenceClassification.from_pretrained(str(roberta_path)).to(self.device)
        self.model_roberta.eval()
        self.roberta_path_actual = roberta_path

        # DeBERTa
        self.tokenizer_deberta = load_tok(CFG.deberta_path, "microsoft/deberta-v3-base")
        self.model_deberta = AutoModelForSequenceClassification.from_pretrained(str(CFG.deberta_path)).to(self.device).float()
        self.model_deberta.eval()

        # DistilBERT
        self.tokenizer_distilbert = load_tok(CFG.distilbert_path, "distilbert-base-uncased")
        self.model_distilbert = AutoModelForSequenceClassification.from_pretrained(str(CFG.distilbert_path)).to(self.device)
        self.model_distilbert.eval()

        # Log Model Identities
        self._log_model_identity("RoBERTa V2.1", roberta_path, self.model_roberta)
        self._log_model_identity("DeBERTa-v3 V2.1", CFG.deberta_path, self.model_deberta)
        self._log_model_identity("DistilBERT V2.1", CFG.distilbert_path, self.model_distilbert)
        logger.info(f"XGBoost Stylometrics | Path: {CFG.xgboost_path} | Classes: {getattr(self.model_xgboost, 'classes_', 'Unknown')}")

        # Calibration parameters
        self.temperatures = {}
        self.platt_calibrator_xgb = None

        temp_path = CFG.checkpoint_dir / "temperatures.json"
        if temp_path.exists():
            try:
                import json
                with open(temp_path, "r", encoding="utf-8") as f:
                    self.temperatures = json.load(f)
                logger.info(f"Loaded calibrated temperatures: {self.temperatures}")
            except Exception as e:
                logger.warning(f"Failed to load calibrated temperatures: {e}")

        platt_path = CFG.checkpoint_dir / "platt_calibrator_xgb.joblib"
        if platt_path.exists():
            try:
                import calibration
                self.platt_calibrator_xgb = joblib.load(platt_path)
                logger.info("Loaded Platt Scaler for XGBoost calibration.")
            except Exception as e:
                logger.warning(f"Failed to load Platt Scaler for XGBoost: {e}")

        # Voting weights fallback
        if CFG.voting_config_path.exists():
            v_cfg = load_json(CFG.voting_config_path)
            self.voting_weights = v_cfg["normalized_weights"]
        else:
            w = [CFG.roberta_weight, CFG.deberta_weight, CFG.distilbert_weight, CFG.xgboost_weight]
            sum_w = sum(w)
            self.voting_weights = {
                "roberta": CFG.roberta_weight / sum_w,
                "deberta": CFG.deberta_weight / sum_w,
                "distilbert": CFG.distilbert_weight / sum_w,
                "xgboost": CFG.xgboost_weight / sum_w
            }

        # ── 2. Load & Validate Stacking Meta-Classifier ───────────────────────
        if not CFG.meta_model_path.exists():
            raise FileNotFoundError(f"CRITICAL ERROR: Stacking Meta-Classifier not found at {CFG.meta_model_path}")

        logger.info(f"Loading Stacking Meta-Classifier: {CFG.meta_model_path}")
        self.meta_model = joblib.load(str(CFG.meta_model_path))

        meta_classes = [0, 1]
        if hasattr(self.meta_model, "models") and len(self.meta_model.models) > 0:
            fold_m = self.meta_model.models[0]
            if hasattr(fold_m, "classes_"):
                meta_classes = list(fold_m.classes_)

        if meta_classes != [0, 1]:
            raise ValueError(f"Incompatible Meta-Classifier classes: {meta_classes}. Expected [0, 1].")

        self.human_index = 0
        self.ai_index = 1

        logger.info(f"Stacking Meta-Classifier ready. Classes: {meta_classes} (0=Human, 1=AI)")

        # Log Diagnostic Runtime Signature
        sig = self.get_runtime_signature()
        logger.info(f"""
================================================================================
HYBRID ENSEMBLE RUNTIME SIGNATURE
================================================================================
Pipeline Source: {sig['pipeline_source']}
Class Name     : {sig['class_name']}
Device         : {sig['device']}

Models:
  RoBERTa        : {sig['roberta_path']} ({sig['roberta_class']})
  DeBERTa        : {sig['deberta_path']} ({sig['deberta_class']})
  DistilBERT     : {sig['distilbert_path']} ({sig['distilbert_class']})
  XGBoost        : {sig['xgboost_path']}
  Meta-Classifier: {sig['meta_classifier_path']}
================================================================================
""")
        logger.info("Hybrid Ensemble Inference Pipeline fully initialized.")

    def get_runtime_signature(self) -> Dict[str, str]:
        """Return diagnostic dictionary of loaded runtime modules and model paths."""
        import inspect
        return {
            "pipeline_source": str(inspect.getfile(self.__class__)),
            "class_name": self.__class__.__name__,
            "device": str(self.device),
            "roberta_path": str(self.roberta_path_actual),
            "roberta_class": self.model_roberta.__class__.__name__,
            "deberta_path": str(CFG.deberta_path),
            "deberta_class": self.model_deberta.__class__.__name__,
            "distilbert_path": str(CFG.distilbert_path),
            "distilbert_class": self.model_distilbert.__class__.__name__,
            "xgboost_path": str(CFG.xgboost_path),
            "meta_classifier_path": str(CFG.meta_model_path)
        }

    def _log_model_identity(self, name: str, path: Path, model) -> None:
        num_params = sum(p.numel() for p in model.parameters())
        id2label = getattr(model.config, "id2label", None)
        num_labels = getattr(model.config, "num_labels", 2)
        logger.info(
            f"MODEL IDENTITY | Name: {name:<16} | Path: {path.resolve()} | "
            f"Class: {model.__class__.__name__:<35} | Params: {num_params:,} | "
            f"Labels: {num_labels} | id2label: {id2label}"
        )

    # ── Inference helpers ────────────────────────────────────────────────────

    def _predict_transformer(self, model, tokenizer, text: str, model_name: str) -> np.ndarray:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu().numpy()

        if model_name in self.temperatures and self.temperatures[model_name] != 1.0:
            T = self.temperatures[model_name]
            logits = logits / T

        probs = softmax(logits, axis=-1)[0]  # shape (2,) [P(Human), P(AI)]
        return probs

    def _predict_xgboost(self, text: str) -> np.ndarray:
        from feature_engineering.utils import extract_features
        import pandas as pd
        feat_dict = extract_features(text)
        feat_df = pd.DataFrame([feat_dict])
        raw_probs = self.model_xgboost.predict_proba(feat_df)[0]  # [P(Human), P(AI)]

        if self.platt_calibrator_xgb is not None:
            cal_probs = self.platt_calibrator_xgb.predict_proba(raw_probs.reshape(1, -1))[0]
            return cal_probs
        return raw_probs

    # ── Core Prediction Pipeline ─────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Execute full inference pipeline and return canonical result object.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        t0 = time.time()
        words = text.strip().split()
        word_count = len(words)
        char_count = len(text)
        sentences = [s for s in text.split('.') if s.strip()]
        sentence_count = len(sentences) if len(sentences) > 0 else 1

        # 1. Base Model Inferences
        probs = {
            "roberta":    self._predict_transformer(self.model_roberta, self.tokenizer_roberta, text, "roberta"),
            "deberta":    self._predict_transformer(self.model_deberta, self.tokenizer_deberta, text, "deberta"),
            "distilbert": self._predict_transformer(self.model_distilbert, self.tokenizer_distilbert, text, "distilbert"),
            "xgboost":    self._predict_xgboost(text)
        }

        # Format Model Breakdown
        model_breakdown = {}
        display_name_map = {
            "roberta": "RoBERTa-base",
            "deberta": "DeBERTa-v3",
            "distilbert": "DistilBERT",
            "xgboost": "XGBoost (Stylometrics)"
        }
        for m in probs:
            p_ai = float(probs[m][1])
            p_hu = float(probs[m][0])
            pred_str = "AI Generated" if p_ai >= 0.50 else "Human Written"
            entry = {
                "ai_probability": p_ai,
                "human_probability": p_hu,
                "prediction": pred_str
            }
            model_breakdown[m] = entry
            model_breakdown[display_name_map[m]] = entry

        # 2. Meta-Feature Vector Construction
        dict_probs_2d = {m: probs[m].reshape(1, -1) for m in probs}
        meta_features = prepare_stacking_features(dict_probs_2d, [text])

        if meta_features.shape[1] != 18:
            raise ValueError(f"CRITICAL ERROR: Expected 18 meta-features, got shape {meta_features.shape}.")

        # 3. Authoritative Stacking Meta-Classifier Inference
        stacking_probs = self.meta_model.predict_proba(dict_probs_2d, [text])[0]

        human_prob = float(stacking_probs[self.human_index])
        ai_prob = float(stacking_probs[self.ai_index])

        pred_id = 1 if ai_prob >= human_prob else 0
        prediction = self.ID2LABEL[pred_id]
        confidence = float(max(ai_prob, human_prob))

        # 4. Diagnostic Consensus Calculation
        base_models = ["roberta", "deberta", "distilbert", "xgboost"]
        ai_votes = sum(1 for m in base_models if probs[m][1] >= 0.50)
        human_votes = len(base_models) - ai_votes
        total_models = len(base_models)

        if ai_votes == 4:
            consensus_state = "UNANIMOUS AGREEMENT (4/4 MODELS AGREE)"
        elif ai_votes == 3:
            consensus_state = "SPLIT CONSENSUS (3/4 MODELS AGREE)"
        elif ai_votes == 2:
            consensus_state = "TIED CONSENSUS (2/4 MODELS AGREE)"
        elif ai_votes == 1:
            consensus_state = "SPLIT CONSENSUS (3/4 MODELS AGREE)"
        else:
            consensus_state = "UNANIMOUS AGREEMENT (4/4 MODELS AGREE)"

        # Secondary Voting calculations for analytics
        soft_prob = np.mean([probs[m] for m in probs], axis=0)
        weighted_prob = (
            probs["roberta"] * self.voting_weights["roberta"] +
            probs["deberta"] * self.voting_weights["deberta"] +
            probs["distilbert"] * self.voting_weights["distilbert"] +
            probs["xgboost"] * self.voting_weights["xgboost"]
        )

        elapsed_ms = (time.time() - t0) * 1000

        # Construct Canonical Result Object
        result = {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "confidence": confidence,
            "strategy_used": "Stacking Meta-Classifier",
            "latency_ms": round(elapsed_ms, 1),
            "text": text[:150] + " …" if len(text) > 150 else text,

            "model_breakdown": model_breakdown,

            "consensus": {
                "ai_votes": ai_votes,
                "human_votes": human_votes,
                "total_models": total_models,
                "agree_count": max(ai_votes, human_votes),
                "state": consensus_state,
                "votes": {m: ("AI" if probs[m][1] >= 0.50 else "Human") for m in base_models}
            },

            "meta_classifier": {
                "classes": [0, 1],
                "raw_probabilities": [human_prob, ai_prob],
                "ai_probability": ai_prob,
                "human_probability": human_prob
            },

            # Backwards compatibility fields
            "base_probs": {m: probs[m] for m in probs},
            "base_predictions": model_breakdown,
            "soft_voting_probs": soft_prob,
            "weighted_voting_probs": weighted_prob,
            "stacking_probs": stacking_probs,
            "final_probs": np.array([human_prob, ai_prob]),
            "meta_features": meta_features[0].tolist()
        }

        # Validate Schema Before Return
        validate_result_schema(result)

        # Structured Debug Logging
        logger.debug(f"""
=== HYBRID ENSEMBLE DEBUG ===
Input Text: Chars={char_count}, Words={word_count}, Sentences={sentence_count}

Base Model Predictions:
  - RoBERTa    ({self.roberta_path_actual}): AI={probs['roberta'][1]*100:.2f}%, Hu={probs['roberta'][0]*100:.2f}% -> {model_breakdown['roberta']['prediction']}
  - DeBERTa    ({CFG.deberta_path}): AI={probs['deberta'][1]*100:.2f}%, Hu={probs['deberta'][0]*100:.2f}% -> {model_breakdown['deberta']['prediction']}
  - DistilBERT ({CFG.distilbert_path}): AI={probs['distilbert'][1]*100:.2f}%, Hu={probs['distilbert'][0]*100:.2f}% -> {model_breakdown['distilbert']['prediction']}
  - XGBoost    ({CFG.xgboost_path}): AI={probs['xgboost'][1]*100:.2f}%, Hu={probs['xgboost'][0]*100:.2f}% -> {model_breakdown['xgboost']['prediction']}

Diagnostic Consensus:
  - AI Votes: {ai_votes} | Human Votes: {human_votes}
  - Consensus State: {consensus_state}

Meta Features (18 vector):
  {meta_features[0].tolist()}

Stacking Meta-Classifier:
  - Raw predict_proba: [Human={human_prob:.4f}, AI={ai_prob:.4f}]
  - Authoritative Verdict: {prediction} ({confidence*100:.2f}%)
=============================
""")

        return result

    # ── Formatting ───────────────────────────────────────────────────────────

    def format_result(self, result: Dict) -> str:
        """Format output for CLI display."""
        bp = result["base_probs"]
        pred_lbl = result["prediction"]
        confidence = result["confidence"] * 100
        cons = result["consensus"]

        lines = [
            "==========================================================",
            "BASE MODELS BREAKDOWN:",
            f"  RoBERTa-base : AI {bp['roberta'][1]*100:.1f}% | Human {bp['roberta'][0]*100:.1f}% -> {'AI' if bp['roberta'][1]>=0.5 else 'Human'}",
            f"  DeBERTa-v3   : AI {bp['deberta'][1]*100:.1f}% | Human {bp['deberta'][0]*100:.1f}% -> {'AI' if bp['deberta'][1]>=0.5 else 'Human'}",
            f"  DistilBERT   : AI {bp['distilbert'][1]*100:.1f}% | Human {bp['distilbert'][0]*100:.1f}% -> {'AI' if bp['distilbert'][1]>=0.5 else 'Human'}",
            f"  XGBoost      : AI {bp['xgboost'][1]*100:.1f}% | Human {bp['xgboost'][0]*100:.1f}% -> {'AI' if bp['xgboost'][1]>=0.5 else 'Human'}",
            "",
            "DIAGNOSTIC CONSENSUS:",
            f"  State: {cons['state']} ({cons['ai_votes']} AI / {cons['human_votes']} Human)",
            "  Note: Consensus is diagnostic only and does not override Stacking Meta-Classifier.",
            "",
            "FINAL AUTHORITATIVE HYBRID ENSEMBLE PREDICTION:",
            f"  Result      : {pred_lbl}",
            f"  Confidence  : {confidence:.2f}% (P(AI)={result['ai_probability']*100:.2f}%, P(Human)={result['human_probability']*100:.2f}%)",
            f"  Engine      : {result['strategy_used']} | Latency: {result['latency_ms']:.1f} ms",
            "=========================================================="
        ]
        return "\n".join(lines)


# ── CLI Entry ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using the Hybrid Ensemble Model.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="Academic text string to classify.")
    group.add_argument("--file", type=str, help="Path to a txt file with one sample per line.")
    group.add_argument("--interactive", action="store_true", help="Start interactive terminal REPL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = HybridInferencePipeline()

    if args.text:
        result = pipeline.predict(args.text)
        print("\n" + pipeline.format_result(result))

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Target text file not found: {file_path}")
            sys.exit(1)

        texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logger.info(f"Classifying {len(texts):,} lines from {file_path} …")

        for i, text in enumerate(texts, start=1):
            print(f"\n[{i}] Text: {text[:80]} …")
            result = pipeline.predict(text)
            print(pipeline.format_result(result))

    else:
        print("\nHybrid Ensemble Academic Text Detector CLI (type 'quit' to exit)\n")
        while True:
            try:
                text = input("Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            try:
                result = pipeline.predict(text)
                print("\n" + pipeline.format_result(result) + "\n")
            except Exception as exc:
                print(f"Error classifying text: {exc}")


if __name__ == "__main__":
    main()
