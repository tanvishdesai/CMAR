from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from av_robustbench.attacks.base import get_logits
from av_robustbench.metrics.binary import binary_metrics


def evaluate_clean(
    model: torch.nn.Module,
    dataset: Iterable[Any],
    *,
    max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    logits_all = []
    labels_all = []
    clip_ids = []
    for index, item in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        visual, audio, label, clip_id = _unpack(item)
        visual = _ensure_batch(visual).to(device)
        audio = _ensure_batch(audio).to(device)
        label_tensor = _ensure_label_batch(label).to(device)
        with torch.no_grad():
            logits = get_logits(model, visual, audio).detach().cpu()
        logits_all.append(logits.reshape(-1))
        labels_all.append(label_tensor.detach().cpu().reshape(-1))
        clip_ids.append(clip_id)
    if not logits_all:
        return {"n_samples": 0, "accuracy": 0.0, "auc": float("nan"), "eer": float("nan"), "ap": float("nan")}
    logits = torch.cat(logits_all).reshape(-1)
    labels = torch.cat(labels_all).reshape(-1)
    metrics = binary_metrics(labels.numpy(), logits.numpy(), from_logits=True)
    metrics.update(
        {
            "n_samples": int(labels.numel()),
            "logits": logits.tolist(),
            "labels": labels.tolist(),
            "clip_ids": clip_ids[: int(labels.numel())],
        }
    )
    return metrics


def _unpack(item: Any) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
    if isinstance(item, dict):
        return item["visual"], item["audio"], item["label"], str(item.get("clip_id", ""))
    if isinstance(item, tuple | list) and len(item) >= 3:
        return item[0], item[1], item[2], str(item[3]) if len(item) >= 4 else ""
    raise TypeError("Dataset items must be dicts or `(visual, audio, label)` tuples.")


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim in {1, 2, 4}:
        return tensor.unsqueeze(0)
    return tensor


def _ensure_label_batch(label: torch.Tensor | int | float) -> torch.Tensor:
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    if label.ndim == 0:
        label = label.reshape(1)
    return label
