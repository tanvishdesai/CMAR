from __future__ import annotations

from dataclasses import dataclass

import torch

from av_robustbench.attacks.base import AttackTarget, BaseAttack
from av_robustbench.attacks.pgd import PGDAttack


@dataclass
class CrossModalTransferAttack(BaseAttack):
    """Generate perturbations on a surrogate and transfer them to a target model."""

    surrogate_model: torch.nn.Module
    eps_value: float = 0.1
    n_steps: int = 20
    step_size: float | None = None
    attack_target: AttackTarget = "visual"
    name: str = "cross_modal_transfer"

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return f"feature_transfer/{self.attack_target}"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        pgd = PGDAttack(
            eps_value=self.eps_value,
            n_steps=self.n_steps,
            step_size=self.step_size,
            random_start=True,
            attack_target=self.attack_target,
            name=self.name,
        )
        return pgd.attack(self.surrogate_model, visual, audio, labels)

