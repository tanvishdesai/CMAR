from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def degradation_robustness_accuracy_ratio(clean_score: float, condition_scores: Iterable[float]) -> float:
    scores = [float(score) for score in condition_scores if not np.isnan(float(score))]
    if clean_score <= 0 or not scores:
        return float("nan")
    return float(np.mean(scores) / clean_score)


def certified_robustness_accuracy_ratio(clean_accuracy: float, mean_certified_radius: float) -> float:
    denominator = max(1e-12, 1.0 - float(clean_accuracy))
    return float(mean_certified_radius / denominator)


def robustness_accuracy_ratio(
    clean_score: float,
    robust_scores: Iterable[float] | None = None,
    *,
    mean_certified_radius: float | None = None,
) -> float:
    if mean_certified_radius is not None:
        return certified_robustness_accuracy_ratio(clean_score, mean_certified_radius)
    if robust_scores is None:
        return float("nan")
    return degradation_robustness_accuracy_ratio(clean_score, robust_scores)

