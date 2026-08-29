"""
Baseline Models Package
========================
Exports the training, evaluation, and inference classes.

Note: imports are deferred (lazy) to avoid circular-import issues when
``train.py`` is executed directly as a script (i.e. __main__).
"""

# Lazy imports — only resolved when the package is used as a library.
# Running ``python models/baseline/train.py`` does NOT trigger these.
def _lazy_imports():
    from models.baseline.train     import BaselineModelTrainer  # noqa: F401
    from models.baseline.evaluate  import ModelEvaluator         # noqa: F401
    from models.baseline.inference import ModelInference         # noqa: F401


__all__ = ["BaselineModelTrainer", "ModelEvaluator", "ModelInference"]
