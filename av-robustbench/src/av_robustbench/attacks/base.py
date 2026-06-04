from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch

AttackTarget = Literal["visual", "audio", "both"]


class BaseAttack(ABC):
    """Base interface for attacks used by av-robustbench."""

    name: str = "base_attack"

    @abstractmethod
    def attack(
        self,
        model: torch.nn.Module,
        visual: torch.Tensor,
        audio: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial visual/audio tensors."""

    @property
    @abstractmethod
    def threat_model(self) -> str:
        """Human-readable threat model identifier."""

    @property
    @abstractmethod
    def eps(self) -> float:
        """Perturbation budget."""


def call_model(model: torch.nn.Module, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "predict"):
        out = model.predict(visual, audio)
    else:
        out = model(visual, audio)
    if isinstance(out, dict):
        if "logits" not in out:
            if "probs" in out:
                probs = out["probs"]
                if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[-1] == 1):
                    out = dict(out)
                    out["logits"] = torch.logit(probs.reshape(-1).clamp(1e-7, 1 - 1e-7))
                else:
                    out = dict(out)
                    out["logits"] = torch.log(probs.clamp_min(1e-12))
            else:
                raise ValueError("Model output must contain `logits` or `probs`.")
        return out
    return {"logits": out}


def get_logits(model: torch.nn.Module, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    logits = call_model(model, visual, audio)["logits"]
    return logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, device=visual.device)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))
    return torch.nn.functional.cross_entropy(logits, labels.long().reshape(-1))


def per_sample_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            labels.float().reshape(-1),
            reduction="none",
        )
    return torch.nn.functional.cross_entropy(logits, labels.long().reshape(-1), reduction="none")


def normalize_labels(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    labels = labels.to(device)
    if labels.ndim == 0:
        labels = labels.reshape(1)
    return labels

