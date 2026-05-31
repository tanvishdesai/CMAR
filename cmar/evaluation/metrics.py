from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def binary_metrics(labels: Iterable[float], logits_or_scores: Iterable[float], from_logits: bool = True) -> Dict[str, float]:
    labels_arr = np.asarray(labels).astype(int)
    values = np.asarray(logits_or_scores).astype(float)
    scores = sigmoid_np(values) if from_logits else values
    if len(np.unique(labels_arr)) < 2:
        return {"auc": float("nan"), "eer": float("nan"), "ap": float("nan")}
    return {
        "auc": float(roc_auc_score(labels_arr, scores)),
        "eer": compute_eer(labels_arr, scores),
        "ap": float(average_precision_score(labels_arr, scores)),
    }


def bootstrap_ci(
    labels: Iterable[float],
    logits_or_scores: Iterable[float],
    n_bootstrap: int = 1000,
    seed: int = 2026,
    from_logits: bool = True,
) -> Dict[str, Dict[str, float]]:
    labels_arr = np.asarray(labels).astype(int)
    values = np.asarray(logits_or_scores).astype(float)
    rng = np.random.default_rng(seed)
    base = binary_metrics(labels_arr, values, from_logits=from_logits)
    boot = {"auc": [], "eer": [], "ap": []}
    n = len(labels_arr)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels_arr[idx])) < 2:
            continue
        metrics = binary_metrics(labels_arr[idx], values[idx], from_logits=from_logits)
        for key in boot:
            boot[key].append(metrics[key])
    out: Dict[str, Dict[str, float]] = {}
    for key, value in base.items():
        arr = np.asarray(boot[key], dtype=float)
        ci = 1.96 * float(np.nanstd(arr)) if arr.size else float("nan")
        out[key] = {"mean": value, "ci95": ci}
    return out


def robustness_accuracy_ratio(condition_auc: float, clean_auc: float) -> float:
    return float(condition_auc / clean_auc) if clean_auc else float("nan")


def delta_auc(clean_auc: float, condition_auc: float) -> float:
    return float(clean_auc - condition_auc)


def ttda_gain(with_ttda_auc: float, without_ttda_auc: float) -> float:
    return float(with_ttda_auc - without_ttda_auc)


def cmrr(
    clean_auc: float,
    visual_attack_auc: float,
    audio_attack_auc: float,
    both_attack_auc: float,
) -> float:
    if clean_auc == 0 or both_attack_auc == 0:
        return float("nan")
    visual_ratio = visual_attack_auc / clean_auc
    audio_ratio = audio_attack_auc / clean_auc
    both_ratio = both_attack_auc / clean_auc
    return float((visual_ratio + audio_ratio) / both_ratio)
