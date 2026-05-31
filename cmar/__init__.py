"""CMAR: Cross-Modal Adversarial Robustness for AV deepfake detection.

Includes CertAV: Certified Adversarial Robustness via Multimodal
Randomized Smoothing.
"""

from cmar.config import (
    AudioConfig,
    CacheConfig,
    DataSplitConfig,
    ModelConfig,
    SmoothingConfig,
    TrainConfig,
)

__all__ = [
    "AudioConfig",
    "CacheConfig",
    "DataSplitConfig",
    "ModelConfig",
    "SmoothingConfig",
    "TrainConfig",
]
