from __future__ import annotations

from dataclasses import dataclass

import torch

from av_robustbench.attacks.base import (
    AttackTarget,
    BaseAttack,
    get_logits,
    normalize_labels,
    per_sample_loss,
)


@dataclass
class SquareAttack(BaseAttack):
    """Score-based Square Attack variant for feature tensors.

    The attack perturbs contiguous temporal/feature blocks in cached AV
    features and accepts proposals that increase the untargeted loss.
    """

    eps_value: float = 0.1
    n_queries: int = 1000
    p_init: float = 0.25
    attack_target: AttackTarget = "both"
    seed: int = 2026
    name: str = "square"

    @property
    def eps(self) -> float:
        return self.eps_value

    @property
    def threat_model(self) -> str:
        return f"feature_Linf_blackbox/{self.attack_target}"

    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = normalize_labels(labels, visual.device)
        generator = torch.Generator(device=visual.device)
        generator.manual_seed(self.seed)
        visual0 = visual.detach()
        audio0 = audio.detach()
        visual_adv = visual0.clone()
        audio_adv = audio0.clone()
        if self.attack_target in {"visual", "both"}:
            visual_adv = visual_adv + torch.empty_like(visual_adv).uniform_(
                -self.eps_value, self.eps_value, generator=generator
            )
            visual_adv = torch.max(torch.min(visual_adv, visual0 + self.eps_value), visual0 - self.eps_value)
        if self.attack_target in {"audio", "both"}:
            audio_adv = audio_adv + torch.empty_like(audio_adv).uniform_(
                -self.eps_value, self.eps_value, generator=generator
            )
            audio_adv = torch.max(torch.min(audio_adv, audio0 + self.eps_value), audio0 - self.eps_value)
        with torch.no_grad():
            current_loss = per_sample_loss(get_logits(model, visual_adv, audio_adv), labels)
        for query in range(self.n_queries):
            p = self._schedule(query)
            cand_v = visual_adv.clone()
            cand_a = audio_adv.clone()
            if self.attack_target in {"visual", "both"}:
                cand_v = _random_block_proposal(cand_v, visual0, self.eps_value, p, generator)
            if self.attack_target in {"audio", "both"}:
                cand_a = _random_block_proposal(cand_a, audio0, self.eps_value, p, generator)
            with torch.no_grad():
                proposed_loss = per_sample_loss(get_logits(model, cand_v, cand_a), labels)
            accept = proposed_loss >= current_loss
            view_v = (accept.shape[0],) + (1,) * (visual_adv.ndim - 1)
            view_a = (accept.shape[0],) + (1,) * (audio_adv.ndim - 1)
            visual_adv = torch.where(accept.view(view_v), cand_v, visual_adv)
            audio_adv = torch.where(accept.view(view_a), cand_a, audio_adv)
            current_loss = torch.where(accept, proposed_loss, current_loss)
        return visual_adv.detach(), audio_adv.detach()

    def _schedule(self, query: int) -> float:
        progress = query / max(1, self.n_queries)
        return max(0.01, self.p_init * (1.0 - progress) ** 2)


def _random_block_proposal(
    current: torch.Tensor,
    clean: torch.Tensor,
    eps: float,
    p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    candidate = current.clone()
    bsz = candidate.shape[0]
    if candidate.ndim < 2:
        noise = torch.sign(torch.randn(candidate.shape, generator=generator, device=candidate.device))
        return torch.max(torch.min(clean + eps * noise, clean + eps), clean - eps)
    temporal = candidate.shape[1]
    features = candidate.shape[2] if candidate.ndim >= 3 else candidate.shape[1]
    t_width = max(1, int(round(temporal * p)))
    f_width = max(1, int(round(features * p)))
    for b in range(bsz):
        t0 = int(
            torch.randint(
                0,
                max(1, temporal - t_width + 1),
                (1,),
                generator=generator,
                device=candidate.device,
            ).item()
        )
        f0 = int(
            torch.randint(
                0,
                max(1, features - f_width + 1),
                (1,),
                generator=generator,
                device=candidate.device,
            ).item()
        )
        sign = 1.0 if torch.rand((), generator=generator, device=candidate.device).item() >= 0.5 else -1.0
        if candidate.ndim >= 3:
            candidate[b, t0 : t0 + t_width, f0 : f0 + f_width] = (
                clean[b, t0 : t0 + t_width, f0 : f0 + f_width] + sign * eps
            )
        else:
            candidate[b, f0 : f0 + f_width] = clean[b, f0 : f0 + f_width] + sign * eps
    return torch.max(torch.min(candidate, clean + eps), clean - eps)
