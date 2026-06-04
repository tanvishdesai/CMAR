from __future__ import annotations

from dataclasses import dataclass, field

import torch

from av_robustbench.attacks.base import BaseAttack, get_logits, normalize_labels, per_sample_loss
from av_robustbench.attacks.pgd import PGDAttack, PGDAttackL2
from av_robustbench.attacks.square_attack import SquareAttack


@dataclass
class AutoAttackAV(BaseAttack):
    """AutoAttack-style deterministic ensemble for AV feature models."""

    eps_value: float = 0.1
    attacks: list[BaseAttack] = field(default_factory=list)
    name: str = "autoattack_av"

    def __post_init__(self) -> None:
        if not self.attacks:
            self.attacks = [
                PGDAttack(eps_value=self.eps_value, n_steps=40, random_start=False, name="apgd_ce_linf"),
                PGDAttack(eps_value=self.eps_value, n_steps=40, random_start=True, name="apgd_restart_linf"),
                PGDAttackL2(eps_value=max(self.eps_value, 1e-6), n_steps=40, random_start=True),
                SquareAttack(eps_value=self.eps_value, n_queries=500),
            ]

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return "feature_autoattack_style"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = normalize_labels(labels, visual.device)
        best_v = visual.detach().clone()
        best_a = audio.detach().clone()
        with torch.no_grad():
            best_loss = per_sample_loss(get_logits(model, best_v, best_a), labels)
        for attack in self.attacks:
            cand_v, cand_a = attack.attack(model, visual, audio, labels)
            with torch.no_grad():
                cand_loss = per_sample_loss(get_logits(model, cand_v, cand_a), labels)
            improve = cand_loss >= best_loss
            view_v = (improve.shape[0],) + (1,) * (best_v.ndim - 1)
            view_a = (improve.shape[0],) + (1,) * (best_a.ndim - 1)
            best_v = torch.where(improve.view(view_v), cand_v, best_v)
            best_a = torch.where(improve.view(view_a), cand_a, best_a)
            best_loss = torch.where(improve, cand_loss, best_loss)
        return best_v.detach(), best_a.detach()

