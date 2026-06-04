from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from av_robustbench.datasets.feature_cache import FeatureCacheDataset
from av_robustbench.degradations.specs import DEGRADATION_SPECS
from av_robustbench.evaluate.clean import evaluate_clean
from av_robustbench.metrics.rar import degradation_robustness_accuracy_ratio


@dataclass
class DegradationBattery:
    conditions: list[str] = field(default_factory=lambda: list(DEGRADATION_SPECS))

    def run_feature_caches(
        self,
        model: torch.nn.Module,
        cache_dir: str | Path,
        *,
        manifest_csv: str | Path | None = None,
        split: str = "test",
        max_samples: int | None = None,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        return evaluate_degraded_feature_caches(
            model,
            cache_dir,
            conditions=self.conditions,
            manifest_csv=manifest_csv,
            split=split,
            max_samples=max_samples,
            device=device,
        )


def evaluate_degraded_feature_caches(
    model: torch.nn.Module,
    cache_dir: str | Path,
    *,
    conditions: list[str] | None = None,
    manifest_csv: str | Path | None = None,
    split: str = "test",
    max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    conditions = conditions or list(DEGRADATION_SPECS)
    clean_ds = FeatureCacheDataset(cache_dir, manifest_csv, split=split, condition="clean", allow_partial_cache=True)
    clean = evaluate_clean(model, clean_ds, max_samples=max_samples, device=device)
    condition_metrics: dict[str, Any] = {}
    for condition in conditions:
        ds = FeatureCacheDataset(
            cache_dir,
            manifest_csv,
            split=split,
            condition=condition,
            allow_partial_cache=True,
        )
        condition_metrics[condition] = evaluate_clean(model, ds, max_samples=max_samples, device=device)
    robust_scores = [
        metrics.get("auc", float("nan"))
        for metrics in condition_metrics.values()
    ]
    return {
        "clean": clean,
        "conditions": condition_metrics,
        "degradation_rar_auc": degradation_robustness_accuracy_ratio(clean.get("auc", float("nan")), robust_scores),
    }

