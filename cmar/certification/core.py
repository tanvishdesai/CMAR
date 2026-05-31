"""Core statistical utilities for randomized smoothing certification.

Implements the Clopper-Pearson confidence bounds and certified radius
computation following Cohen et al. (2019) "Certified Adversarial Robustness
via Randomized Smoothing".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm, binom


@dataclass
class CertificationResult:
    """Result of certifying a single sample."""

    predicted_class: int  # -1 means ABSTAIN
    certified_radius: float  # 0.0 if abstained
    correct: bool  # whether predicted_class matches true label
    pA_lower: float  # lower confidence bound on top-class probability
    counts_top: int  # number of times top class was predicted
    counts_total: int  # total samples
    abstained: bool  # True if classifier abstained

    @property
    def certified_correct(self) -> bool:
        """True if prediction is both correct and certified (not abstained)."""
        return self.correct and not self.abstained


def lower_confidence_bound(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson lower confidence bound.

    Returns the lower bound on the true probability p such that
    P(Binomial(n, p) >= k) >= 1 - alpha.

    Args:
        k: number of successes (times top class was predicted)
        n: total number of trials
        alpha: significance level (e.g. 0.001 for 99.9% confidence)

    Returns:
        Lower bound on p
    """
    if k == 0:
        return 0.0
    return float(binom.ppf(alpha, n, float(k) / n) / n)


def lower_confidence_bound_exact(k: int, n: int, alpha: float) -> float:
    """Exact Clopper-Pearson lower confidence bound using beta distribution.

    This is the standard method used in Cohen et al. (2019).
    """
    from scipy.stats import beta as beta_dist
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def certified_radius(sigma: float, pA_lower: float) -> float:
    """Compute the certified L2 radius.

    R = sigma * Phi^{-1}(pA_lower)

    where Phi^{-1} is the inverse of the standard Gaussian CDF.

    Args:
        sigma: noise standard deviation used in smoothing
        pA_lower: lower confidence bound on the top-class probability

    Returns:
        Certified L2 radius. Returns 0.0 if pA_lower <= 0.5.
    """
    if pA_lower <= 0.5:
        return 0.0
    return float(sigma * norm.ppf(pA_lower))


def certified_accuracy_at_radius(
    results: list[CertificationResult],
    radius: float,
) -> float:
    """Fraction of samples that are correctly classified AND certified at radius r.

    Args:
        results: list of CertificationResult from certifying a dataset
        radius: the L2 radius to evaluate at

    Returns:
        Certified accuracy (fraction of correctly certified samples)
    """
    if len(results) == 0:
        return 0.0
    n_certified_correct = sum(
        1 for r in results
        if r.correct and not r.abstained and r.certified_radius >= radius
    )
    return n_certified_correct / len(results)


def certified_accuracy_curve(
    results: list[CertificationResult],
    radii: Optional[list[float]] = None,
) -> dict[str, list[float]]:
    """Compute certified accuracy at multiple radii.

    Args:
        results: list of CertificationResult
        radii: list of radii to evaluate at. If None, uses 0.0 to 2.0 in steps of 0.05

    Returns:
        Dict with 'radii' and 'certified_accuracy' lists
    """
    if radii is None:
        radii = [round(r * 0.05, 3) for r in range(41)]  # 0.0 to 2.0
    accs = [certified_accuracy_at_radius(results, r) for r in radii]
    return {"radii": radii, "certified_accuracy": accs}
