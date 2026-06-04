"""Conformal prediction helpers for CertAV Phase 2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cmar.certification.core import lower_confidence_bound_exact


LABELS = [0, 1]


@dataclass(frozen=True)
class ConformalConfigKey:
    alpha: float
    radius: float
    score_type: str

    def as_key(self) -> str:
        return f"alpha_{self.alpha:.3f}__r_{self.radius:.2f}__score_{self.score_type}"


def class_probability(
    counts: Iterable[int],
    label: int,
    *,
    score_type: str,
    cp_alpha: float,
    robust_radius: float = 0.0,
    sigma: float = 1.0,
) -> float:
    """Return the probability estimate used by a conformal score."""

    counts_arr = np.asarray(list(counts), dtype=int)
    n = int(np.sum(counts_arr))
    if n <= 0:
        return 0.5
    k = int(counts_arr[int(label)])
    p_raw = k / n
    p_cp = lower_confidence_bound_exact(k, n, cp_alpha)

    if score_type == "raw":
        p = p_raw
    elif score_type == "cp":
        p = p_cp
    elif score_type == "log":
        p = p_raw
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")

    if robust_radius > 0:
        if p_cp > 0.5:
            from scipy.stats import norm

            class_radius = sigma * float(norm.ppf(p_cp))
        else:
            class_radius = 0.0
        p = p_cp if class_radius >= robust_radius else 0.5

    return float(np.clip(p, 1e-12, 1.0))


def nonconformity_score(
    counts: Iterable[int],
    label: int,
    *,
    score_type: str,
    cp_alpha: float,
    robust_radius: float = 0.0,
    sigma: float = 1.0,
    temperature: float = 1.0,
) -> float:
    p = class_probability(
        counts,
        label,
        score_type=score_type,
        cp_alpha=cp_alpha,
        robust_radius=robust_radius,
        sigma=sigma,
    )
    if score_type == "log":
        return float(-math.log(p) / max(temperature, 1e-12))
    return float(1.0 - p)


def conformal_quantile(scores: Iterable[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile."""

    arr = np.sort(np.asarray(list(scores), dtype=float))
    n = int(arr.size)
    if n == 0:
        raise ValueError("Cannot calibrate conformal threshold with zero scores.")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(arr[rank - 1])


def prediction_set(
    counts: Iterable[int],
    qhat: float,
    *,
    score_type: str,
    cp_alpha: float,
    robust_radius: float = 0.0,
    sigma: float = 1.0,
    temperature: float = 1.0,
) -> list[int]:
    included: list[int] = []
    for label in LABELS:
        score = nonconformity_score(
            counts,
            label,
            score_type=score_type,
            cp_alpha=cp_alpha,
            robust_radius=robust_radius,
            sigma=sigma,
            temperature=temperature,
        )
        if score <= qhat:
            included.append(label)
    return included or LABELS.copy()


def summarize_prediction_sets(rows: list[dict], radius: float = 0.0) -> dict[str, float | int | None]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "coverage": None,
            "singleton_rate": None,
            "mean_set_size": None,
        }

    coverage = np.asarray([bool(row["covered"]) for row in rows], dtype=float)
    set_sizes = np.asarray([int(row["set_size"]) for row in rows], dtype=float)
    labels = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
    abstained = np.asarray([bool(row.get("certav_abstained", False)) for row in rows], dtype=bool)
    cert_radius = np.asarray([float(row.get("certified_radius", 0.0)) for row in rows], dtype=float)

    def mean_or_none(mask: np.ndarray) -> float | None:
        return float(np.mean(coverage[mask])) if np.any(mask) else None

    certified_mask = cert_radius >= radius
    return {
        "n": n,
        "coverage": float(np.mean(coverage)),
        "singleton_rate": float(np.mean(set_sizes == 1)),
        "mean_set_size": float(np.mean(set_sizes)),
        "coverage_real": mean_or_none(labels == 0),
        "coverage_fake": mean_or_none(labels == 1),
        "coverage_certified_at_radius": mean_or_none(certified_mask),
        "coverage_uncertified_at_radius": mean_or_none(~certified_mask),
        "coverage_certav_abstained": mean_or_none(abstained),
    }
