from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from av_robustbench.core import RobustnessCard
from av_robustbench.utils.io import read_json, write_json


@dataclass
class LeaderboardEntry:
    name: str
    dataset: str
    clean_auc: float | None = None
    clean_accuracy: float | None = None
    robust_acc_eps010: float | None = None
    robust_acc_eps020: float | None = None
    cert_acc_r025: float | None = None
    cert_acc_r100: float | None = None
    certified_radius_mean: float | None = None
    degradation_avg_auc: float | None = None
    cross_dataset_acc: float | None = None
    paper_url: str | None = None
    code_url: str | None = None
    submission_date: str = field(default_factory=lambda: date.today().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "clean_auc": self.clean_auc,
            "clean_accuracy": self.clean_accuracy,
            "robust_acc_eps010": self.robust_acc_eps010,
            "robust_acc_eps020": self.robust_acc_eps020,
            "cert_acc_r025": self.cert_acc_r025,
            "cert_acc_r100": self.cert_acc_r100,
            "certified_radius_mean": self.certified_radius_mean,
            "degradation_avg_auc": self.degradation_avg_auc,
            "cross_dataset_acc": self.cross_dataset_acc,
            "paper_url": self.paper_url,
            "code_url": self.code_url,
            "submission_date": self.submission_date,
            "metadata": self.metadata,
        }


def create_leaderboard_entry(
    card: RobustnessCard,
    *,
    model_name: str | None = None,
    paper_url: str | None = None,
    code_url: str | None = None,
) -> LeaderboardEntry:
    attacks = card.attacks or {}
    certification = card.certification or {}
    degradations = card.degradations or {}
    return LeaderboardEntry(
        name=model_name or card.model_name,
        dataset=card.dataset,
        clean_auc=_float_or_none(card.clean_metrics.get("auc")),
        clean_accuracy=_float_or_none(card.clean_metrics.get("accuracy")),
        robust_acc_eps010=_attack_acc_for_eps(attacks, 0.10),
        robust_acc_eps020=_attack_acc_for_eps(attacks, 0.20),
        cert_acc_r025=_cert_acc_at_radius(certification, "r_0.25"),
        cert_acc_r100=_cert_acc_at_radius(certification, "r_1.00"),
        certified_radius_mean=_mean_cert_radius(certification),
        degradation_avg_auc=_degradation_avg_auc(degradations),
        cross_dataset_acc=_float_or_none(card.cross_dataset.get("accuracy")),
        paper_url=paper_url,
        code_url=code_url,
    )


def validate_entry(entry: LeaderboardEntry | dict[str, Any]) -> None:
    data = entry.to_dict() if isinstance(entry, LeaderboardEntry) else entry
    required = ["name", "dataset", "submission_date"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Missing required leaderboard fields: {missing}")
    for key, value in data.items():
        if key.endswith("auc") or key.endswith("acc") or key.endswith("accuracy"):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1], got {value}")


def update_leaderboard(
    leaderboard_path: str | Path,
    entry: LeaderboardEntry,
    *,
    replace: bool = True,
) -> Path:
    validate_entry(entry)
    leaderboard_path = Path(leaderboard_path)
    if leaderboard_path.exists():
        data = read_json(leaderboard_path)
    else:
        data = {"schema_version": "av-robustbench-leaderboard-v1", "models": []}
    models = data.setdefault("models", [])
    if replace:
        models[:] = [item for item in models if item.get("name") != entry.name or item.get("dataset") != entry.dataset]
    models.append(entry.to_dict())
    models.sort(key=lambda item: (item.get("dataset", ""), item.get("name", "")))
    return write_json(data, leaderboard_path)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attack_acc_for_eps(attacks: dict[str, Any], eps: float) -> float | None:
    best = None
    for attack in attacks.values():
        if abs(float(attack.get("eps", -999.0)) - eps) < 1e-6:
            value = _float_or_none(attack.get("adversarial_accuracy"))
            best = value if best is None else max(best, value or 0.0)
    return best


def _cert_acc_at_radius(certification: dict[str, Any], radius_key: str) -> float | None:
    values = []
    for sigma_result in certification.values():
        summary = sigma_result.get("summary", {}) if isinstance(sigma_result, dict) else {}
        at_radii = summary.get("certified_accuracy_at_radii", {})
        if radius_key in at_radii:
            values.append(float(at_radii[radius_key]))
    return max(values) if values else None


def _mean_cert_radius(certification: dict[str, Any]) -> float | None:
    values = []
    for sigma_result in certification.values():
        summary = sigma_result.get("summary", {}) if isinstance(sigma_result, dict) else {}
        if "mean_certified_radius" in summary:
            values.append(float(summary["mean_certified_radius"]))
    return max(values) if values else None


def _degradation_avg_auc(degradations: dict[str, Any]) -> float | None:
    conditions = degradations.get("conditions", {}) if isinstance(degradations, dict) else {}
    values = [
        float(metrics["auc"])
        for metrics in conditions.values()
        if isinstance(metrics, dict) and metrics.get("auc") is not None
    ]
    return sum(values) / len(values) if values else None

