from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def binary_metrics(
    labels: Iterable[Any],
    logits_or_scores: Iterable[Any],
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels_arr = np.asarray(labels).reshape(-1).astype(int)
    values = np.asarray(logits_or_scores).reshape(-1).astype(float)
    scores = sigmoid_np(values) if from_logits else values
    preds = (scores >= threshold).astype(int)
    output = {
        "accuracy": float(accuracy_score(labels_arr, preds)) if labels_arr.size else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(labels_arr, preds))
        if len(np.unique(labels_arr)) >= 2
        else float("nan"),
        "auc": float("nan"),
        "eer": float("nan"),
        "ap": float("nan"),
    }
    if labels_arr.size and len(np.unique(labels_arr)) >= 2:
        output.update(
            {
                "auc": float(roc_auc_score(labels_arr, scores)),
                "eer": compute_eer(labels_arr, scores),
                "ap": float(average_precision_score(labels_arr, scores)),
            }
        )
    return output


def bootstrap_ci(
    labels: Iterable[Any],
    logits_or_scores: Iterable[Any],
    *,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    from_logits: bool = True,
) -> dict[str, dict[str, float]]:
    labels_arr = np.asarray(labels).reshape(-1).astype(int)
    values = np.asarray(logits_or_scores).reshape(-1).astype(float)
    base = binary_metrics(labels_arr, values, from_logits=from_logits)
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {key: [] for key in base}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels_arr), size=len(labels_arr))
        if len(np.unique(labels_arr[indices])) < 2:
            continue
        sample = binary_metrics(labels_arr[indices], values[indices], from_logits=from_logits)
        for key, value in sample.items():
            boot[key].append(value)
    out: dict[str, dict[str, float]] = {}
    for key, value in base.items():
        arr = np.asarray(boot[key], dtype=float)
        out[key] = {
            "mean": value,
            "ci95": 1.96 * float(np.nanstd(arr)) if arr.size else float("nan"),
        }
    return out

