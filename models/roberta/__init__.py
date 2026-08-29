"""
models/roberta/__init__.py
==========================
RoBERTa fine-tuning package for AI-Generated Text Detection.

Exposes the public API for programmatic use:

    from models.roberta import RoBERTaInference, CFG

Author  : Bharanidharan K
Project : Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text
"""

from config import CFG
from inference import RoBERTaInference

__all__ = ["CFG", "RoBERTaInference"]
