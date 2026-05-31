"""CertAV: Certified adversarial robustness via multimodal randomized smoothing."""

from cmar.certification.smoothing import SmoothedClassifier
from cmar.certification.core import CertificationResult

__all__ = ["SmoothedClassifier", "CertificationResult"]
