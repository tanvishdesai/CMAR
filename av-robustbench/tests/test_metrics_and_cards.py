from __future__ import annotations

import json

from av_robustbench.core import AttackResult, RobustnessCard
from av_robustbench.metrics import binary_metrics, degradation_robustness_accuracy_ratio


def test_binary_metrics() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [-2, -1, 1, 2])
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_attack_result_summary() -> None:
    result = AttackResult(
        clean_logits=[-1, 1],
        adversarial_logits=[1, -1],
        labels=[0, 1],
        eps=0.1,
        threat_model="feature_Linf",
    )
    assert result.clean_accuracy() == 1.0
    assert result.adversarial_accuracy() == 0.0
    assert result.attack_success_rate() == 1.0


def test_robustness_card_serializes() -> None:
    card = RobustnessCard("m", "d", clean_metrics={"auc": 1.0})
    data = json.loads(card.to_json())
    assert data["model_name"] == "m"
    assert "Clean Evaluation" in card.to_markdown()


def test_degradation_rar() -> None:
    assert degradation_robustness_accuracy_ratio(1.0, [0.5, 1.0]) == 0.75

