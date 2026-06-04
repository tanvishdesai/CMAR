"""Randomized smoothing statistics for audio-visual classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm


@dataclass
class CertificationResult:
    """Result of certifying one sample."""

    predicted_class: int
    certified_radius: float
    correct: bool
    pA_lower: float
    counts_top: int
    counts_total: int
    abstained: bool
    true_label: int | None = None
    sample_id: str | None = None
    counts: list[int] = field(default_factory=list)
    sigma: float | None = None
    noise_mode: str = "joint"
    alpha: float = 0.001
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def certified_correct(self) -> bool:
        return bool(self.correct and not self.abstained and self.certified_radius > 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "true_label": self.true_label,
            "predicted_class": self.predicted_class,
            "certified_radius": self.certified_radius,
            "correct": self.correct,
            "certified_correct": self.certified_correct,
            "pA_lower": self.pA_lower,
            "counts_top": self.counts_top,
            "counts_total": self.counts_total,
            "counts": self.counts,
            "abstained": self.abstained,
            "sigma": self.sigma,
            "noise_mode": self.noise_mode,
            "alpha": self.alpha,
            "metadata": self.metadata,
        }


def lower_confidence_bound_exact(k: int, n: int, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound."""

    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def certified_radius(sigma: float, pA_lower: float) -> float:
    """Certified L2 radius from Cohen et al. randomized smoothing."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if pA_lower <= 0.5:
        return 0.0
    return float(sigma * norm.ppf(pA_lower))


def certified_accuracy_at_radius(results: list[CertificationResult], radius: float) -> float:
    if not results:
        return 0.0
    return float(
        np.mean(
            [
                result.correct and not result.abstained and result.certified_radius >= radius
                for result in results
            ]
        )
    )


def certified_accuracy_curve(
    results: list[CertificationResult],
    radii: list[float] | None = None,
) -> dict[str, list[float]]:
    if radii is None:
        max_radius = max([r.certified_radius for r in results], default=2.0)
        upper = max(2.0, float(np.ceil(max_radius * 20.0) / 20.0))
        radii = [round(i * 0.05, 4) for i in range(int(upper / 0.05) + 1)]
    return {
        "radii": radii,
        "certified_accuracy": [certified_accuracy_at_radius(results, r) for r in radii],
    }


def mean_certified_radius(results: list[CertificationResult], *, correct_only: bool = False) -> float:
    selected = [
        result.certified_radius
        for result in results
        if not result.abstained and (not correct_only or result.correct)
    ]
    return float(np.mean(selected)) if selected else 0.0


def summarize_certification(
    results: list[CertificationResult],
    radii: list[float] | None = None,
) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n_samples": 0,
            "accuracy": 0.0,
            "abstention_rate": 0.0,
            "mean_certified_radius": 0.0,
            "certified_accuracy_curve": {"radii": radii or [], "certified_accuracy": []},
        }
    return {
        "n_samples": n,
        "accuracy": float(np.mean([r.correct and not r.abstained for r in results])),
        "abstention_rate": float(np.mean([r.abstained for r in results])),
        "mean_certified_radius": mean_certified_radius(results),
        "mean_correct_certified_radius": mean_certified_radius(results, correct_only=True),
        "certified_accuracy_at_radii": {
            f"r_{r:.2f}": certified_accuracy_at_radius(results, r)
            for r in (radii or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
        },
        "certified_accuracy_curve": certified_accuracy_curve(results, radii),
    }

