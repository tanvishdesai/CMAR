from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from av_robustbench.attacks import AutoAttackAV, BaseAttack, PGDAttack, PGDAttackL2, SquareAttack
from av_robustbench.attacks.evaluate import evaluate_under_attack
from av_robustbench.certification import certify_multi_sigma
from av_robustbench.core import RobustnessCard
from av_robustbench.datasets import FeatureCacheDataset
from av_robustbench.degradations.battery import evaluate_degraded_feature_caches
from av_robustbench.evaluate.clean import evaluate_clean
from av_robustbench.utils.io import ensure_dir, write_json


def benchmark(
    model: torch.nn.Module,
    dataset: Any,
    *,
    model_name: str | None = None,
    dataset_name: str | None = None,
    attacks: list[str | BaseAttack] | None = None,
    certify: bool = False,
    sigmas: list[float] | None = None,
    degrade: bool = False,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_samples: int | None = None,
    device: str | torch.device | None = None,
    certification_kwargs: dict[str, Any] | None = None,
) -> RobustnessCard:
    """Run clean, adversarial, certified, and degradation evaluation."""

    model_name = model_name or getattr(model, "name", model.__class__.__name__)
    dataset_name = dataset_name or getattr(dataset, "dataset_name", dataset.__class__.__name__)
    clean = evaluate_clean(model, dataset, max_samples=max_samples, device=device)
    attack_outputs: dict[str, Any] = {}
    if attacks:
        attack_objects = [_build_attack(attack) if isinstance(attack, str) else attack for attack in attacks]
        raw_attack_results = evaluate_under_attack(
            model,
            dataset,
            attack_objects,
            max_samples=max_samples,
            device=device,
        )
        attack_outputs = {name: result.to_dict() for name, result in raw_attack_results.items()}

    certification_outputs: dict[str, Any] = {}
    if certify:
        certification_kwargs = certification_kwargs or {}
        certification_outputs = certify_multi_sigma(
            model,
            dataset,
            sigmas=sigmas or [0.25],
            max_samples=max_samples,
            device=device,
            **certification_kwargs,
        )

    degradation_outputs: dict[str, Any] = {}
    if degrade:
        resolved_cache = cache_dir
        if resolved_cache is None and isinstance(dataset, FeatureCacheDataset):
            resolved_cache = dataset.cache_dir
        if resolved_cache is None:
            raise ValueError("degrade=True requires `cache_dir` or a FeatureCacheDataset.")
        degradation_outputs = evaluate_degraded_feature_caches(
            model,
            resolved_cache,
            max_samples=max_samples,
            device=device,
        )

    card = RobustnessCard(
        model_name=model_name,
        dataset=dataset_name,
        clean_metrics=_strip_vectors(clean),
        certification=certification_outputs,
        attacks=attack_outputs,
        degradations=degradation_outputs,
        metadata={"max_samples": max_samples},
    )
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        write_json(card.to_dict(), Path(output_dir) / "robustness_card.json")
        write_json(clean, Path(output_dir) / "clean_predictions.json")
    return card


def _build_attack(name: str) -> BaseAttack:
    normalized = name.lower().replace("-", "_")
    if normalized in {"pgd", "pgd_linf", "linf"}:
        return PGDAttack()
    if normalized in {"pgd_l2", "l2"}:
        return PGDAttackL2()
    if normalized in {"square", "square_attack"}:
        return SquareAttack()
    if normalized in {"autoattack", "autoattack_av", "aa"}:
        return AutoAttackAV()
    raise KeyError(f"Unknown attack `{name}`.")


def _strip_vectors(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k not in {"logits", "labels", "clip_ids"}}

