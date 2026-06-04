"""Phase 2 research utilities for CertAV.

This package contains experiment-facing helpers for the post-ICASSP CertAV
directions: PCA/anisotropic smoothing and conformal prediction.
"""

from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES, PCANoise

__all__ = ["ANISOTROPIC_NOISE_MODES", "PCANoise"]
