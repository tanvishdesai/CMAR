from __future__ import annotations

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
class PGDAttack(BaseAttack):
    """L-infinity PGD in AV feature space."""

    eps_value: float = 0.1
    n_steps: int = 20
    step_size: float | None = None
    random_start: bool = True
    attack_target: AttackTarget = "both"
    name: str = "pgd_linf"

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return f"feature_Linf/{self.attack_target}"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attack_target not in {"visual", "audio", "both"}:
            raise ValueError("attack_target must be visual, audio, or both")
        step_size = self.step_size or (2.0 * self.eps_value / max(1, self.n_steps))
        visual0 = visual.detach()
        audio0 = audio.detach()
        labels = normalize_labels(labels, visual0.device)
        visual_adv = visual0.clone()
        audio_adv = audio0.clone()
        if self.random_start:
            if self.attack_target in {"visual", "both"}:
                visual_adv = visual_adv + torch.empty_like(visual_adv).uniform_(-self.eps_value, self.eps_value)
            if self.attack_target in {"audio", "both"}:
                audio_adv = audio_adv + torch.empty_like(audio_adv).uniform_(-self.eps_value, self.eps_value)
        for _ in range(self.n_steps):
            visual_adv = visual_adv.detach()
            audio_adv = audio_adv.detach()
            visual_adv.requires_grad_(self.attack_target in {"visual", "both"})
            audio_adv.requires_grad_(self.attack_target in {"audio", "both"})
            logits = get_logits(model, visual_adv, audio_adv)
            loss = classification_loss(logits, labels)
            grad_inputs = [x for x in (visual_adv, audio_adv) if x.requires_grad]
            grads = torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)
            grad_iter = iter(grads)
            if self.attack_target in {"visual", "both"}:
                grad = next(grad_iter)
                visual_adv = visual_adv + step_size * grad.sign()
                visual_adv = torch.max(torch.min(visual_adv, visual0 + self.eps_value), visual0 - self.eps_value)
            if self.attack_target in {"audio", "both"}:
                grad = next(grad_iter)
                audio_adv = audio_adv + step_size * grad.sign()
                audio_adv = torch.max(torch.min(audio_adv, audio0 + self.eps_value), audio0 - self.eps_value)
        return visual_adv.detach(), audio_adv.detach()


@dataclass
class PGDAttackL2(BaseAttack):
    """L2 PGD in AV feature space with joint projection for both modalities."""

    eps_value: float = 1.0
    n_steps: int = 20
    step_size: float | None = None
    random_start: bool = True
    attack_target: AttackTarget = "both"
    name: str = "pgd_l2"

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return f"feature_L2/{self.attack_target}"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attack_target not in {"visual", "audio", "both"}:
            raise ValueError("attack_target must be visual, audio, or both")
        step_size = self.step_size or (self.eps_value / max(1, self.n_steps // 2))
        visual0 = visual.detach()
        audio0 = audio.detach()
        labels = normalize_labels(labels, visual0.device)
        visual_adv = visual0.clone()
        audio_adv = audio0.clone()
        if self.random_start:
            visual_delta = torch.zeros_like(visual0)
            audio_delta = torch.zeros_like(audio0)
            if self.attack_target in {"visual", "both"}:
                visual_delta = torch.randn_like(visual0)
            if self.attack_target in {"audio", "both"}:
                audio_delta = torch.randn_like(audio0)
            visual_delta, audio_delta = _project_joint_l2(
                visual_delta,
                audio_delta,
                self.eps_value * torch.rand(visual0.shape[0], device=visual0.device),
                self.attack_target,
            )
            visual_adv = visual0 + visual_delta
            audio_adv = audio0 + audio_delta
        for _ in range(self.n_steps):
            visual_adv = visual_adv.detach()
            audio_adv = audio_adv.detach()
            visual_adv.requires_grad_(self.attack_target in {"visual", "both"})
            audio_adv.requires_grad_(self.attack_target in {"audio", "both"})
            logits = get_logits(model, visual_adv, audio_adv)
            loss = classification_loss(logits, labels)
            grad_inputs = [x for x in (visual_adv, audio_adv) if x.requires_grad]
            grads = torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)
            grad_v = torch.zeros_like(visual_adv)
            grad_a = torch.zeros_like(audio_adv)
            grad_iter = iter(grads)
            if self.attack_target in {"visual", "both"}:
                grad_v = next(grad_iter)
            if self.attack_target in {"audio", "both"}:
                grad_a = next(grad_iter)
            grad_v, grad_a = _normalize_joint_l2(grad_v, grad_a, self.attack_target)
            visual_adv = visual_adv + step_size * grad_v
            audio_adv = audio_adv + step_size * grad_a
            delta_v, delta_a = _project_joint_l2(
                visual_adv - visual0,
                audio_adv - audio0,
                torch.full((visual0.shape[0],), self.eps_value, device=visual0.device),
                self.attack_target,
            )
            visual_adv = visual0 + delta_v
            audio_adv = audio0 + delta_a
        return visual_adv.detach(), audio_adv.detach()


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _normalize_joint_l2(
    visual: torch.Tensor,
    audio: torch.Tensor,
    target: AttackTarget,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    if target in {"visual", "both"}:
        parts.append(_flatten_batch(visual))
    if target in {"audio", "both"}:
        parts.append(_flatten_batch(audio))
    if not parts:
        return torch.zeros_like(visual), torch.zeros_like(audio)
    norm = torch.cat(parts, dim=1).norm(p=2, dim=1).clamp_min(eps)
    shape_v = (visual.shape[0],) + (1,) * (visual.ndim - 1)
    shape_a = (audio.shape[0],) + (1,) * (audio.ndim - 1)
    return (
        visual / norm.view(shape_v) if target in {"visual", "both"} else torch.zeros_like(visual),
        audio / norm.view(shape_a) if target in {"audio", "both"} else torch.zeros_like(audio),
    )


def _project_joint_l2(
    visual_delta: torch.Tensor,
    audio_delta: torch.Tensor,
    eps: torch.Tensor,
    target: AttackTarget,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    if target in {"visual", "both"}:
        parts.append(_flatten_batch(visual_delta))
    if target in {"audio", "both"}:
        parts.append(_flatten_batch(audio_delta))
    if not parts:
        return torch.zeros_like(visual_delta), torch.zeros_like(audio_delta)
    norm = torch.cat(parts, dim=1).norm(p=2, dim=1).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(norm), eps.to(norm.device) / norm)
    shape_v = (visual_delta.shape[0],) + (1,) * (visual_delta.ndim - 1)
    shape_a = (audio_delta.shape[0],) + (1,) * (audio_delta.ndim - 1)
    return (
        visual_delta * factor.view(shape_v) if target in {"visual", "both"} else torch.zeros_like(visual_delta),
        audio_delta * factor.view(shape_a) if target in {"audio", "both"} else torch.zeros_like(audio_delta),
    )

