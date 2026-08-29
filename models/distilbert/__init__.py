"""
models/distilbert/__init__.py
============================
DistilBERT fine-tuning package for AI-Generated Text Detection.

Exposes the public API for programmatic use:

    from models.distilbert import DistilBERTInference, CFG

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

from config import CFG
from inference import DistilBERTInference

__all__ = ["CFG", "DistilBERTInference"]
