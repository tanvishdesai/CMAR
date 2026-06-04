from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from av_robustbench.models.adapters.base import (
    AVDetector,
    logits_to_probs,
    normalize_detector_output,
)


class TorchFeatureDetector(AVDetector):
    """Adapter for a PyTorch feature-space AV detector."""

    def __init__(
        self,
        module: nn.Module | Callable[[torch.Tensor, torch.Tensor], Any],
        *,
        name: str = "torch_feature_detector",
        feature_dims: dict[str, tuple[int, ...]] | None = None,
        input_type: str = "features",
    ) -> None:
        super().__init__()
        if input_type not in {"features", "raw"}:
            raise ValueError("input_type must be `features` or `raw`")
        self.module = module
        self._name = name
        self._feature_dims = feature_dims or {}
        self._input_type = input_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_type(self) -> str:
        return self._input_type

    @property
    def feature_dims(self) -> dict[str, tuple[int, ...]]:
        return self._feature_dims

    def predict(self, visual: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.module(visual, audio)
        normalized = normalize_detector_output(output)
        if "probs" not in normalized and "logits" in normalized:
            normalized["probs"] = logits_to_probs(normalized["logits"])
        return normalized

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> TorchFeatureDetector:
        module = kwargs.pop("module", None)
        if module is None:
            raise ValueError("TorchFeatureDetector.from_checkpoint requires `module=`.")
        checkpoint = torch.load(path, map_location=kwargs.pop("map_location", "cpu"), weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint must contain a state dict or `model_state`.")
        module.load_state_dict(state_dict)
        return cls(module=module, **kwargs)
