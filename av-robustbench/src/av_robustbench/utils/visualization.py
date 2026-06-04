from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from av_robustbench.utils.io import ensure_dir


def plot_certified_accuracy_curves(
    curves: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: str | Path,
    *,
    title: str = "Certified Accuracy Curves",
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, curve in curves.items():
        ax.plot(curve["radii"], curve["certified_accuracy"], marker="o", ms=2.5, label=label)
    ax.set_xlabel("L2 radius")
    ax.set_ylabel("Certified accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_attack_bar_chart(
    attack_metrics: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    *,
    metric: str = "adversarial_accuracy",
    title: str = "Attack Robustness",
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    names = list(attack_metrics)
    values = [float(attack_metrics[name].get(metric, np.nan)) for name in names]
    fig, ax = plt.subplots(figsize=(max(6.0, len(names) * 0.8), 4.0))
    ax.bar(names, values, color="#4f7cac")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_degradation_heatmap(
    degradation_metrics: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    *,
    metrics: Sequence[str] = ("auc", "accuracy"),
    title: str = "Degradation Robustness",
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    conditions = list(degradation_metrics)
    data = np.asarray(
        [[float(degradation_metrics[c].get(m, np.nan)) for m in metrics] for c in conditions],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(max(5.5, len(metrics) * 1.2), max(4.0, len(conditions) * 0.35)))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ").title() for m in metrics])
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path

