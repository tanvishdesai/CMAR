from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_robustbench.utils.io import to_jsonable, write_json


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _binary_or_multiclass_predictions(logits: Any) -> np.ndarray:
    arr = _as_numpy(logits)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        return (arr.reshape(-1) >= 0.0).astype(int)
    return arr.argmax(axis=1).astype(int)


@dataclass
class AttackResult:
    """Outputs and summary helpers for one attack evaluation."""

    clean_logits: Any
    adversarial_logits: Any
    labels: Any
    eps: float
    threat_model: str
    attack_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def clean_accuracy(self) -> float:
        preds = _binary_or_multiclass_predictions(self.clean_logits)
        labels = _as_numpy(self.labels).reshape(-1).astype(int)
        return float(np.mean(preds == labels)) if labels.size else 0.0

    def adversarial_accuracy(self) -> float:
        preds = _binary_or_multiclass_predictions(self.adversarial_logits)
        labels = _as_numpy(self.labels).reshape(-1).astype(int)
        return float(np.mean(preds == labels)) if labels.size else 0.0

    def attack_success_rate(self) -> float:
        clean_preds = _binary_or_multiclass_predictions(self.clean_logits)
        adv_preds = _binary_or_multiclass_predictions(self.adversarial_logits)
        labels = _as_numpy(self.labels).reshape(-1).astype(int)
        if labels.size == 0:
            return 0.0
        clean_correct = clean_preds == labels
        if not np.any(clean_correct):
            return 0.0
        successful = clean_correct & (adv_preds != labels)
        return float(np.sum(successful) / np.sum(clean_correct))

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "threat_model": self.threat_model,
            "eps": self.eps,
            "clean_accuracy": self.clean_accuracy(),
            "adversarial_accuracy": self.adversarial_accuracy(),
            "attack_success_rate": self.attack_success_rate(),
            "metadata": to_jsonable(self.metadata),
            "clean_logits": to_jsonable(self.clean_logits),
            "adversarial_logits": to_jsonable(self.adversarial_logits),
            "labels": to_jsonable(self.labels),
        }


@dataclass
class RobustnessCard:
    """Structured robustness report for a model/dataset pair."""

    model_name: str
    dataset: str
    clean_metrics: dict[str, Any] = field(default_factory=dict)
    certification: dict[str, Any] = field(default_factory=dict)
    attacks: dict[str, Any] = field(default_factory=dict)
    degradations: dict[str, Any] = field(default_factory=dict)
    cross_dataset: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "av-robustbench-card-v1",
            "model_name": self.model_name,
            "dataset": self.dataset,
            "clean_metrics": to_jsonable(self.clean_metrics),
            "certification": to_jsonable(self.certification),
            "attacks": to_jsonable(self.attacks),
            "degradations": to_jsonable(self.degradations),
            "cross_dataset": to_jsonable(self.cross_dataset),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> Path:
        return write_json(self.to_dict(), path)

    def to_markdown(self) -> str:
        lines = [
            f"# Robustness Card: {self.model_name}",
            "",
            f"- Dataset: `{self.dataset}`",
        ]
        if self.clean_metrics:
            lines.extend(["", "## Clean Evaluation", "", _markdown_table(self.clean_metrics)])
        if self.attacks:
            lines.extend(["", "## Adversarial Evaluation", "", _nested_markdown_table(self.attacks)])
        if self.certification:
            lines.extend(["", "## Certified Evaluation", "", _nested_markdown_table(self.certification)])
        if self.degradations:
            lines.extend(["", "## Degradation Evaluation", "", _nested_markdown_table(self.degradations)])
        return "\n".join(lines).rstrip() + "\n"

    def to_latex(self) -> str:
        rows = []
        for section_name, values in [
            ("Clean", self.clean_metrics),
            ("Certification", self.certification),
            ("Attacks", self.attacks),
            ("Degradations", self.degradations),
        ]:
            if not values:
                continue
            flat = _flatten(values)
            for key, value in flat.items():
                if isinstance(value, int | float | np.number):
                    rows.append((section_name, key.replace("_", r"\_"), f"{float(value):.4f}"))
        body = "\n".join(f"{section} & {metric} & {value} \\\\" for section, metric, value in rows)
        return (
            "\\begin{tabular}{lll}\n"
            "\\toprule\n"
            "Section & Metric & Value \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            out.update(_flatten(item, new_key))
        else:
            out[new_key] = item
    return out


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict | list | tuple):
        return "`...`"
    return str(value)


def _markdown_table(values: Mapping[str, Any]) -> str:
    lines = ["| Metric | Value |", "|:--|--:|"]
    for key, value in values.items():
        lines.append(f"| `{key}` | {_format_cell(value)} |")
    return "\n".join(lines)


def _nested_markdown_table(values: Mapping[str, Any]) -> str:
    flat = _flatten(values)
    return _markdown_table(flat)
