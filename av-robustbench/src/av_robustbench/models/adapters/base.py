from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn


class AVDetector(nn.Module, ABC):
    """Interface every av-robustbench model adapter implements."""

    @abstractmethod
    def predict(self, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return at least `{"logits": Tensor}` for a visual/audio batch."""

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.predict(visual, audio)

    @property
    @abstractmethod
    def input_type(self) -> str:
        """Either `features` or `raw`."""

    @property
    @abstractmethod
    def feature_dims(self) -> dict[str, tuple[int, ...]]:
        """Feature tensor contract for feature-space models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> AVDetector:
        """Load the adapter from a checkpoint."""


def normalize_detector_output(output: Any) -> dict[str, torch.Tensor]:
    if isinstance(output, dict):
        if "logits" not in output and "probs" not in output:
            raise ValueError("Detector output dict must include `logits` or `probs`.")
        if "logits" in output and not isinstance(output["logits"], torch.Tensor):
            output = dict(output)
            output["logits"] = torch.as_tensor(output["logits"])
        if "probs" in output and not isinstance(output["probs"], torch.Tensor):
            output = dict(output)
            output["probs"] = torch.as_tensor(output["probs"])
        return output
    logits = output if isinstance(output, torch.Tensor) else torch.as_tensor(output)
    return {"logits": logits}


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        return torch.sigmoid(logits.reshape(-1))
    return torch.softmax(logits, dim=-1)

