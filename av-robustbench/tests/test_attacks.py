from __future__ import annotations

import torch

from av_robustbench.attacks import (
    AutoAttackAV,
    PGDAttack,
    PGDAttackL2,
    SquareAttack,
    evaluate_under_attack,
)
from av_robustbench.models.adapters import TorchFeatureDetector


class SumDetector(torch.nn.Module):
    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": visual.reshape(visual.shape[0], -1).sum(dim=1) + audio.reshape(audio.shape[0], -1).sum(dim=1)}


def test_pgd_linf_respects_budget() -> None:
    model = TorchFeatureDetector(SumDetector())
    visual = torch.zeros(2, 3, 4)
    audio = torch.zeros(2, 2, 4)
    labels = torch.zeros(2)
    attack = PGDAttack(eps_value=0.05, n_steps=3, attack_target="both")
    adv_v, adv_a = attack.attack(model, visual, audio, labels)
    assert torch.max(torch.abs(adv_v - visual)) <= 0.0501
    assert torch.max(torch.abs(adv_a - audio)) <= 0.0501


def test_pgd_l2_respects_joint_budget() -> None:
    model = TorchFeatureDetector(SumDetector())
    visual = torch.zeros(2, 3, 4)
    audio = torch.zeros(2, 2, 4)
    labels = torch.zeros(2)
    attack = PGDAttackL2(eps_value=0.5, n_steps=3, attack_target="both")
    adv_v, adv_a = attack.attack(model, visual, audio, labels)
    joint = torch.cat([(adv_v - visual).reshape(2, -1), (adv_a - audio).reshape(2, -1)], dim=1)
    assert torch.all(joint.norm(p=2, dim=1) <= 0.5001)


def test_square_attack_runs() -> None:
    model = TorchFeatureDetector(SumDetector())
    visual = torch.zeros(1, 3, 4)
    audio = torch.zeros(1, 2, 4)
    labels = torch.zeros(1)
    attack = SquareAttack(eps_value=0.05, n_queries=5)
    adv_v, adv_a = attack.attack(model, visual, audio, labels)
    assert adv_v.shape == visual.shape
    assert adv_a.shape == audio.shape


def test_square_attack_respects_budget_and_target() -> None:
    model = TorchFeatureDetector(SumDetector())
    visual = torch.zeros(1, 3, 4)
    audio = torch.zeros(1, 2, 4)
    labels = torch.zeros(1)
    attack = SquareAttack(eps_value=0.05, n_queries=5, attack_target="visual", seed=123)
    adv_v, adv_a = attack.attack(model, visual, audio, labels)
    assert torch.max(torch.abs(adv_v - visual)) <= 0.0501
    assert torch.equal(adv_a, audio)


def test_autoattack_ensemble_runs_and_respects_budget() -> None:
    model = TorchFeatureDetector(SumDetector())
    visual = torch.zeros(2, 3, 4)
    audio = torch.zeros(2, 2, 4)
    labels = torch.zeros(2)
    attack = AutoAttackAV(
        eps_value=0.05,
        attacks=[
            PGDAttack(eps_value=0.05, n_steps=2, random_start=False),
            PGDAttackL2(eps_value=0.05, n_steps=2, random_start=False),
            SquareAttack(eps_value=0.05, n_queries=3, seed=123),
        ],
    )
    adv_v, adv_a = attack.attack(model, visual, audio, labels)
    assert attack.threat_model == "feature_autoattack_style"
    assert torch.max(torch.abs(adv_v - visual)) <= 0.0501
    assert torch.max(torch.abs(adv_a - audio)) <= 0.0501


def test_evaluate_under_attack_returns_result() -> None:
    model = TorchFeatureDetector(SumDetector())
    dataset = [
        {"visual": torch.zeros(3, 4), "audio": torch.zeros(2, 4), "label": torch.tensor(0.0), "clip_id": "a"},
        {"visual": torch.ones(3, 4), "audio": torch.ones(2, 4), "label": torch.tensor(1.0), "clip_id": "b"},
    ]
    results = evaluate_under_attack(model, dataset, [PGDAttack(eps_value=0.01, n_steps=1)], device="cpu")
    assert "pgd_linf" in results
    assert results["pgd_linf"].clean_accuracy() >= 0.5
