from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from av_robustbench.attacks.base import BaseAttack, get_logits
from av_robustbench.core import AttackResult


def evaluate_under_attack(
    model: torch.nn.Module,
    dataset: Iterable[Any],
    attacks: list[BaseAttack],
    *,
    max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, AttackResult]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    results: dict[str, AttackResult] = {}
    for attack in attacks:
        clean_logits = []
        adv_logits = []
        labels_all = []
        for index, item in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break
            visual, audio, label = _unpack(item)
            visual = _ensure_batch(visual).to(device)
            audio = _ensure_batch(audio).to(device)
            label = _ensure_label_batch(label).to(device)
            with torch.no_grad():
                clean = get_logits(model, visual, audio).detach().cpu()
            adv_visual, adv_audio = attack.attack(model, visual, audio, label)
            with torch.no_grad():
                adv = get_logits(model, adv_visual, adv_audio).detach().cpu()
            clean_logits.append(clean.reshape(clean.shape[0], -1) if clean.ndim > 1 else clean.reshape(-1, 1))
            adv_logits.append(adv.reshape(adv.shape[0], -1) if adv.ndim > 1 else adv.reshape(-1, 1))
            labels_all.append(label.detach().cpu().reshape(-1))
        if clean_logits:
            clean_cat = torch.cat(clean_logits, dim=0)
            adv_cat = torch.cat(adv_logits, dim=0)
            labels_cat = torch.cat(labels_all, dim=0)
        else:
            clean_cat = torch.empty(0)
            adv_cat = torch.empty(0)
            labels_cat = torch.empty(0)
        results[attack.name] = AttackResult(
            clean_logits=clean_cat,
            adversarial_logits=adv_cat,
            labels=labels_cat,
            eps=attack.eps,
            threat_model=attack.threat_model,
            attack_name=attack.name,
        )
    return results


def _unpack(item: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(item, dict):
        return item["visual"], item["audio"], item["label"]
    if isinstance(item, tuple | list) and len(item) >= 3:
        return item[0], item[1], item[2]
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
