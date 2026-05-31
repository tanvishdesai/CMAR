from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch.nn import functional as F


AttackTarget = Literal["visual", "audio", "both"]


@dataclass
class FeatureAttackConfig:
    target: AttackTarget = "visual"
    eps: float = 0.05
    step_size: Optional[float] = None
    steps: int = 20
    random_start: bool = True


class FeaturePGDAttacker:
    """White-box PGD over cached feature tensors.

    This is a fast proxy for adversarial evaluation in the feature-cache path.
    Input-space PGD for the final paper should be run through raw encoders; the
    encoder wrappers are provided, but cached-feature PGD is the practical smoke
    test and ablation tool.
    """

    def __init__(self, model: torch.nn.Module, config: FeatureAttackConfig) -> None:
        self.model = model
        self.config = config
        self.step_size = config.step_size or config.eps / max(1, config.steps // 2)

    def attack(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        v0 = visual.detach()
        a0 = audio.detach()
        v_adv = v0.clone()
        a_adv = a0.clone()
        if self.config.random_start:
            if self.config.target in {"visual", "both"}:
                v_adv = v_adv + torch.empty_like(v_adv).uniform_(-self.config.eps, self.config.eps)
            if self.config.target in {"audio", "both"}:
                a_adv = a_adv + torch.empty_like(a_adv).uniform_(-self.config.eps, self.config.eps)

        for _ in range(self.config.steps):
            v_adv.requires_grad_(self.config.target in {"visual", "both"})
            a_adv.requires_grad_(self.config.target in {"audio", "both"})
            logits = self.model(v_adv, a_adv)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            grads = torch.autograd.grad(
                loss,
                [x for x in (v_adv, a_adv) if x.requires_grad],
                retain_graph=False,
                create_graph=False,
            )
            grad_iter = iter(grads)
            if self.config.target in {"visual", "both"}:
                grad_v = next(grad_iter)
                v_adv = v_adv.detach() + self.step_size * grad_v.sign()
                v_adv = torch.max(torch.min(v_adv, v0 + self.config.eps), v0 - self.config.eps)
            else:
                v_adv = v_adv.detach()
            if self.config.target in {"audio", "both"}:
                grad_a = next(grad_iter)
                a_adv = a_adv.detach() + self.step_size * grad_a.sign()
                a_adv = torch.max(torch.min(a_adv, a0 + self.config.eps), a0 - self.config.eps)
            else:
                a_adv = a_adv.detach()
        return {"visual": v_adv.detach(), "audio": a_adv.detach()}


class FeatureFGSMAttacker(FeaturePGDAttacker):
    def __init__(self, model: torch.nn.Module, target: AttackTarget, eps: float) -> None:
        super().__init__(
            model,
            FeatureAttackConfig(target=target, eps=eps, step_size=eps, steps=1, random_start=False),
        )


@dataclass
class RawVisualAttackConfig:
    eps: float = 4.0 / 255.0
    step_size: Optional[float] = None
    steps: int = 20
    random_start: bool = True


class RawVisualPGDAttacker:
    """Input-space PGD for the visual stream.

    Frames must be RGB float tensors in [0, 1] with shape (B, F, 3, H, W).
    The attack differentiates through DINOv2, then through the cached-feature
    CMAR head/fusion model. Audio is supplied as cached Whisper features.
    """

    def __init__(
        self,
        visual_encoder: torch.nn.Module,
        cmar_model: torch.nn.Module,
        config: RawVisualAttackConfig,
    ) -> None:
        self.visual_encoder = visual_encoder.eval()
        self.cmar_model = cmar_model.eval()
        self.config = config
        self.step_size = config.step_size or config.eps / max(1, config.steps // 2)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def _encode_visual(self, frames01: torch.Tensor) -> torch.Tensor:
        bsz, n_frames, channels, height, width = frames01.shape
        mean = self.mean.to(frames01.device, frames01.dtype)
        std = self.std.to(frames01.device, frames01.dtype)
        normalized = (frames01 - mean) / std
        flat = normalized.reshape(bsz * n_frames, channels, height, width)
        feats = self.visual_encoder(flat)
        return feats.reshape(bsz, n_frames, -1)

    def attack(
        self,
        frames01: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        clean = frames01.detach()
        adv = clean.clone()
        if self.config.random_start:
            adv = torch.clamp(
                adv + torch.empty_like(adv).uniform_(-self.config.eps, self.config.eps),
                0.0,
                1.0,
            )
        for _ in range(self.config.steps):
            adv.requires_grad_(True)
            visual_features = self._encode_visual(adv)
            logits = self.cmar_model(visual_features, audio_features)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            (grad,) = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)
            adv = adv.detach() + self.step_size * grad.sign()
            adv = torch.max(torch.min(adv, clean + self.config.eps), clean - self.config.eps)
            adv = torch.clamp(adv, 0.0, 1.0)
        return adv.detach()


@dataclass
class WhisperInputFeatureAttackConfig:
    eps: float = 0.05
    step_size: Optional[float] = None
    steps: int = 20
    random_start: bool = True


class WhisperInputFeaturePGDAttacker:
    """PGD over Whisper log-mel input features.

    This is not waveform-space PGD, but it is useful before implementing a
    fully differentiable waveform-to-Whisper frontend.
    """

    def __init__(
        self,
        audio_encoder: torch.nn.Module,
        cmar_model: torch.nn.Module,
        config: WhisperInputFeatureAttackConfig,
    ) -> None:
        self.audio_encoder = audio_encoder.eval()
        self.cmar_model = cmar_model.eval()
        self.config = config
        self.step_size = config.step_size or config.eps / max(1, config.steps // 2)

    def attack(
        self,
        whisper_input_features: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        clean = whisper_input_features.detach()
        adv = clean.clone()
        if self.config.random_start:
            adv = adv + torch.empty_like(adv).uniform_(-self.config.eps, self.config.eps)
        for _ in range(self.config.steps):
            adv.requires_grad_(True)
            audio_features = self.audio_encoder(adv)
            logits = self.cmar_model(visual_features, audio_features)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            (grad,) = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)
            adv = adv.detach() + self.step_size * grad.sign()
            adv = torch.max(torch.min(adv, clean + self.config.eps), clean - self.config.eps)
        return adv.detach()
