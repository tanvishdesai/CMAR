from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from av_robustbench.attacks.base import (
    AttackTarget,
    BaseAttack,
    classification_loss,
    get_logits,
    normalize_labels,
)


@dataclass
class PGDInputSpace(BaseAttack):
    """PGD through differentiable raw-input encoders into an AV detector."""

    visual_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None
    audio_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None
    eps_value: float = 4.0 / 255.0
    n_steps: int = 20
    step_size: float | None = None
    random_start: bool = True
    attack_target: AttackTarget = "visual"
    visual_bounds: tuple[float, float] = (0.0, 1.0)
    audio_bounds: tuple[float, float] = (-1.0, 1.0)
    name: str = "pgd_input"

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return f"input_Linf/{self.attack_target}"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attack_target in {"visual", "both"} and self.visual_encoder is None:
            raise ValueError("visual_encoder is required when attacking visual raw inputs.")
        if self.attack_target in {"audio", "both"} and self.audio_encoder is None:
            raise ValueError("audio_encoder is required when attacking audio raw inputs.")
        step_size = self.step_size or (2.0 * self.eps_value / max(1, self.n_steps))
        labels = normalize_labels(labels, visual.device)
        visual0 = visual.detach()
        audio0 = audio.detach()
        visual_adv = visual0.clone()
        audio_adv = audio0.clone()
        if self.random_start:
            if self.attack_target in {"visual", "both"}:
                visual_adv = torch.clamp(
                    visual_adv + torch.empty_like(visual_adv).uniform_(-self.eps_value, self.eps_value),
                    *self.visual_bounds,
                )
            if self.attack_target in {"audio", "both"}:
                audio_adv = torch.clamp(
                    audio_adv + torch.empty_like(audio_adv).uniform_(-self.eps_value, self.eps_value),
                    *self.audio_bounds,
                )
        for _ in range(self.n_steps):
            visual_adv = visual_adv.detach()
            audio_adv = audio_adv.detach()
            visual_adv.requires_grad_(self.attack_target in {"visual", "both"})
            audio_adv.requires_grad_(self.attack_target in {"audio", "both"})
            visual_features = self.visual_encoder(visual_adv) if self.visual_encoder else visual_adv
            audio_features = self.audio_encoder(audio_adv) if self.audio_encoder else audio_adv
            logits = get_logits(model, visual_features, audio_features)
            loss = classification_loss(logits, labels)
            grad_inputs = [x for x in (visual_adv, audio_adv) if x.requires_grad]
            grads = torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)
            grad_iter = iter(grads)
            if self.attack_target in {"visual", "both"}:
                grad = next(grad_iter)
                visual_adv = visual_adv + step_size * grad.sign()
                visual_adv = torch.max(torch.min(visual_adv, visual0 + self.eps_value), visual0 - self.eps_value)
                visual_adv = torch.clamp(visual_adv, *self.visual_bounds)
            if self.attack_target in {"audio", "both"}:
                grad = next(grad_iter)
                audio_adv = audio_adv + step_size * grad.sign()
                audio_adv = torch.max(torch.min(audio_adv, audio0 + self.eps_value), audio0 - self.eps_value)
                audio_adv = torch.clamp(audio_adv, *self.audio_bounds)
        return visual_adv.detach(), audio_adv.detach()
