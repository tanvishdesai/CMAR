from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cmar.evaluation.metrics import binary_metrics, bootstrap_ci, sigmoid_np


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_features: bool = False,
) -> Dict[str, object]:
    model.eval()
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    clip_ids: List[str] = []
    categories: List[str] = []
    features: List[torch.Tensor] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = move_batch(batch, device)
        out = model(batch["visual"], batch["audio"], return_features=return_features)
        logits.append(out["logits"].detach().cpu())
        labels.append(batch["label"].detach().cpu())
        clip_ids.extend(batch["clip_id"])
        categories.extend(batch["av_category"])
        if return_features:
            features.append(out["features"].detach().cpu())
    result: Dict[str, object] = {
        "clip_id": clip_ids,
        "av_category": categories,
        "logits": torch.cat(logits).numpy(),
        "labels": torch.cat(labels).numpy(),
    }
    if return_features:
        result["features"] = torch.cat(features).numpy()
    return result


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_bootstrap: int = 1000,
) -> Dict[str, object]:
    preds = predict_logits(model, loader, device)
    labels = preds["labels"]
    logits = preds["logits"]
    metrics = binary_metrics(labels, logits)
    ci = bootstrap_ci(labels, logits, n_bootstrap=n_bootstrap) if n_bootstrap > 0 else {}
    return {"metrics": metrics, "ci": ci, "predictions": preds}


def evaluate_by_category(
    labels: np.ndarray,
    logits: np.ndarray,
    categories: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    categories_arr = np.asarray(list(categories))
    for category in sorted(set(categories_arr)):
        mask = categories_arr == category
        if mask.sum() < 2:
            continue
        out[category] = binary_metrics(labels[mask], logits[mask])
    return out


def evaluate_category_contrasts(
    labels: np.ndarray,
    logits: np.ndarray,
    categories: Iterable[str],
    real_category: str = "RR",
) -> Dict[str, Dict[str, float]]:
    """Compute meaningful FakeAVCeleb category metrics.

    FakeAVCeleb's AV categories are label-pure: RR is real, while FR/RF/FF are
    fake. AUC within one category is therefore undefined. The useful contrast is
    each fake category against RR.
    """

    out: Dict[str, Dict[str, float]] = {}
    categories_arr = np.asarray(list(categories))
    labels_arr = np.asarray(labels)
    logits_arr = np.asarray(logits)
    for category in sorted(set(categories_arr)):
        if category == real_category:
            continue
        mask = (categories_arr == real_category) | (categories_arr == category)
        if mask.sum() < 2 or len(np.unique(labels_arr[mask])) < 2:
            continue
        metrics = binary_metrics(labels_arr[mask], logits_arr[mask])
        metrics["n_real"] = int((categories_arr[mask] == real_category).sum())
        metrics["n_fake"] = int((categories_arr[mask] == category).sum())
        out[f"{category}_vs_{real_category}"] = metrics
    return out


def category_operating_points(
    labels: np.ndarray,
    logits: np.ndarray,
    categories: Iterable[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Report per-category rates at a fixed probability threshold."""

    out: Dict[str, Dict[str, float]] = {}
    categories_arr = np.asarray(list(categories))
    labels_arr = np.asarray(labels).astype(int)
    scores = sigmoid_np(np.asarray(logits, dtype=float))
    preds = (scores >= threshold).astype(int)
    for category in sorted(set(categories_arr)):
        mask = categories_arr == category
        if mask.sum() == 0:
            continue
        cat_labels = labels_arr[mask]
        cat_preds = preds[mask]
        row = {
            "n": int(mask.sum()),
            "positive_rate": float(cat_labels.mean()),
            "mean_score": float(scores[mask].mean()),
            "predicted_fake_rate": float(cat_preds.mean()),
        }
        if len(np.unique(cat_labels)) == 1:
            if cat_labels[0] == 1:
                row["fake_recall_at_threshold"] = float(cat_preds.mean())
            else:
                row["false_positive_rate_at_threshold"] = float(cat_preds.mean())
        out[category] = row
    return out


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, object]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint
