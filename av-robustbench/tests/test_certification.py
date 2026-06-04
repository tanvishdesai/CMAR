from __future__ import annotations

import torch

from av_robustbench.certification import (
    SmoothedAVClassifier,
    certified_radius,
    lower_confidence_bound_exact,
)
from av_robustbench.models.adapters import TorchFeatureDetector


class ConstantOne(torch.nn.Module):
    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": torch.full((visual.shape[0],), 10.0, device=visual.device)}


def test_lower_confidence_bound_edges() -> None:
    assert lower_confidence_bound_exact(0, 10, 0.001) == 0.0
    assert 0.0 < lower_confidence_bound_exact(10, 10, 0.001) < 1.0


def test_certified_radius_boundary() -> None:
    assert certified_radius(0.25, 0.5) == 0.0
    assert certified_radius(0.25, 0.99) > 0.0


def test_smoothed_constant_detector_certifies() -> None:
    detector = TorchFeatureDetector(ConstantOne(), name="constant")
    smoothed = SmoothedAVClassifier(detector, sigma=0.25, device="cpu")
    visual = torch.zeros(4, 3)
    audio = torch.zeros(5, 3)
    result = smoothed.certify(visual, audio, 1, n0=8, n=20, alpha=0.01, batch_size=5, seed=123)
    assert result.predicted_class == 1
    assert result.correct
    assert not result.abstained
    assert result.certified_radius > 0.0

