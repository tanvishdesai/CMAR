"""Public API for av-robustbench."""

from __future__ import annotations

from av_robustbench.certification import CertificationResult, SmoothedAVClassifier
from av_robustbench.core import AttackResult, RobustnessCard
from av_robustbench.evaluate import benchmark
from av_robustbench.models import AVDetector, list_models, load_model, register_model

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AVDetector",
    "AttackResult",
    "CertificationResult",
    "RobustnessCard",
    "SmoothedAVClassifier",
    "benchmark",
    "list_models",
    "load_model",
    "register_model",
]

