"""Smoothed classifier for multimodal (audio-visual) deepfake detection.

Wraps a base CMAR classifier to create a randomized-smoothing-based certified
classifier following Cohen et al. (2019), extended to joint audio-visual
feature spaces.

Usage:
    base_model = CMAR(config)  # trained with Gaussian noise augmentation
    smoothed = SmoothedClassifier(base_model, sigma=0.25, device='cuda')
    result = smoothed.certify(visual_feat, audio_feat, true_label=1)
    print(f"Predicted: {result.predicted_class}, Radius: {result.certified_radius:.3f}")
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

from cmar.certification.core import (
    CertificationResult,
    lower_confidence_bound_exact,
    certified_radius,
)


class SmoothedClassifier:
    """A smoothed classifier wrapping a base CMAR model.

    The smoothed classifier adds isotropic Gaussian noise to the input
    features and uses Monte Carlo sampling to estimate class probabilities,
    then computes certified L2 robustness radii.

    Supports three noise modes:
    - 'joint': noise added to both visual and audio features (default)
    - 'visual_only': noise added only to visual features
    - 'audio_only': noise added only to audio features

    The certification radius applies to the feature space that receives noise.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sigma: float,
        device: torch.device | str = "cuda",
        noise_mode: str = "joint",
    ) -> None:
        """
        Args:
            base_model: trained CMAR model (base classifier f)
            sigma: noise standard deviation for Gaussian smoothing
            device: torch device
            noise_mode: 'joint', 'visual_only', or 'audio_only'
        """
        self.base_model = base_model
        self.sigma = sigma
        self.device = torch.device(device)
        self.noise_mode = noise_mode
        assert noise_mode in ("joint", "visual_only", "audio_only"), \
            f"Invalid noise_mode: {noise_mode}"
        self.base_model.to(self.device)
        self.base_model.eval()

    def _add_noise(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise according to noise_mode."""
        if self.noise_mode in ("joint", "visual_only"):
            visual = visual + torch.randn_like(visual) * self.sigma
        if self.noise_mode in ("joint", "audio_only"):
            audio = audio + torch.randn_like(audio) * self.sigma
        return visual, audio

    @torch.no_grad()
    def _sample_predictions(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        n_samples: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Run n_samples noisy forward passes and collect binary predictions.

        Args:
            visual: (1, T_v, D_v) visual features for a single sample
            audio: (1, T_a, D_a) audio features for a single sample
            n_samples: number of Monte Carlo samples
            batch_size: how many noisy copies to process at once

        Returns:
            counts: np.array of shape (2,) with [count_class_0, count_class_1]
        """
        counts = np.zeros(2, dtype=int)

        for start in range(0, n_samples, batch_size):
            this_batch = min(batch_size, n_samples - start)

            # Expand to batch
            vis_batch = visual.expand(this_batch, -1, -1).clone()
            aud_batch = audio.expand(this_batch, -1, -1).clone()

            # Add noise
            vis_noisy, aud_noisy = self._add_noise(vis_batch, aud_batch)

            # Forward pass
            out = self.base_model(vis_noisy, aud_noisy)
            logits = out["logits"]  # (this_batch, 1) or (this_batch,)
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs >= 0.5).long().cpu().numpy()

            counts[0] += int(np.sum(preds == 0))
            counts[1] += int(np.sum(preds == 1))

        return counts

    def predict(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        n_samples: int = 100,
        batch_size: int = 64,
    ) -> int:
        """Predict the class of an input using the smoothed classifier.

        Args:
            visual: (1, T_v, D_v) or (T_v, D_v)
            audio: (1, T_a, D_a) or (T_a, D_a)
            n_samples: number of Monte Carlo samples for prediction
            batch_size: batch size for processing

        Returns:
            Predicted class (0 = real, 1 = fake)
        """
        visual = visual.unsqueeze(0) if visual.dim() == 2 else visual
        audio = audio.unsqueeze(0) if audio.dim() == 2 else audio
        visual = visual.to(self.device)
        audio = audio.to(self.device)

        counts = self._sample_predictions(visual, audio, n_samples, batch_size)
        return int(np.argmax(counts))

    def certify(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        true_label: int,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 64,
    ) -> CertificationResult:
        """Certify the robustness of the prediction for a single sample.

        Two-phase procedure:
        1. Prediction phase (n0 samples): identify top class cA
        2. Certification phase (n samples): estimate pA with confidence bound

        Args:
            visual: (1, T_v, D_v) or (T_v, D_v)
            audio: (1, T_a, D_a) or (T_a, D_a)
            true_label: ground truth label (0 or 1)
            n0: number of samples for prediction phase
            n: number of samples for certification phase
            alpha: significance level for confidence bound (default 0.001 = 99.9%)
            batch_size: batch size for forward passes

        Returns:
            CertificationResult with predicted class, certified radius, etc.
        """
        visual = visual.unsqueeze(0) if visual.dim() == 2 else visual
        audio = audio.unsqueeze(0) if audio.dim() == 2 else audio
        visual = visual.to(self.device)
        audio = audio.to(self.device)

        # Phase 1: Prediction
        counts0 = self._sample_predictions(visual, audio, n0, batch_size)
        cA = int(np.argmax(counts0))

        # Phase 2: Certification
        counts = self._sample_predictions(visual, audio, n, batch_size)
        nA = int(counts[cA])

        # Compute lower bound on pA
        pA_lower = lower_confidence_bound_exact(nA, n, alpha)

        # Compute certified radius
        if pA_lower > 0.5:
            radius = certified_radius(self.sigma, pA_lower)
            return CertificationResult(
                predicted_class=cA,
                certified_radius=radius,
                correct=(cA == true_label),
                pA_lower=pA_lower,
                counts_top=nA,
                counts_total=n,
                abstained=False,
            )
        else:
            # ABSTAIN: not confident enough
            return CertificationResult(
                predicted_class=-1,
                certified_radius=0.0,
                correct=False,
                pA_lower=pA_lower,
                counts_top=nA,
                counts_total=n,
                abstained=True,
            )

    def certify_dataset(
        self,
        visual_features: list[torch.Tensor],
        audio_features: list[torch.Tensor],
        labels: list[int],
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> list[CertificationResult]:
        """Certify all samples in a dataset.

        Args:
            visual_features: list of (T_v, D_v) tensors
            audio_features: list of (T_a, D_a) tensors
            labels: list of ground truth labels
            n0, n, alpha, batch_size: certification parameters
            verbose: whether to print progress

        Returns:
            List of CertificationResult for each sample
        """
        results = []
        n_total = len(visual_features)
        for i in range(n_total):
            result = self.certify(
                visual_features[i],
                audio_features[i],
                labels[i],
                n0=n0,
                n=n,
                alpha=alpha,
                batch_size=batch_size,
            )
            results.append(result)
            if verbose and (i + 1) % 50 == 0:
                n_correct = sum(1 for r in results if r.certified_correct)
                n_abstain = sum(1 for r in results if r.abstained)
                print(
                    f"  [{i+1}/{n_total}] "
                    f"certified_correct={n_correct}/{i+1} "
                    f"({100*n_correct/(i+1):.1f}%) "
                    f"abstained={n_abstain}/{i+1}"
                )
        return results
