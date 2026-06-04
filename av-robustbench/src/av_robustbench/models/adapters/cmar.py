from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from av_robustbench.models.adapters.base import (
    AVDetector,
    logits_to_probs,
    normalize_detector_output,
)


class CMARAdapter(AVDetector):
    """Adapter for the CMAR feature-cache architecture used by CertAV."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        name: str = "cmar",
        feature_dims: dict[str, tuple[int, ...]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self._name = name
        self._feature_dims = feature_dims or {"visual": (16, 384), "audio": (64, 384)}
        self.metadata = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_type(self) -> str:
        return "features"

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
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> CMARAdapter:
        try:
            from cmar.config import ModelConfig
            from cmar.models.cmar import CMAR, CMARVisualOnly
        except ImportError as exc:
            raise ImportError(
                "CMARAdapter requires the `cmar` package to be importable. "
                "Install CertAV/CMAR or pass a ready module through TorchFeatureDetector."
            ) from exc

        map_location = kwargs.pop("map_location", "cpu")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
        model_config_kwargs = kwargs.pop("model_config", None) or checkpoint_dict.get("model_config", {})
        if hasattr(model_config_kwargs, "__dict__") and not isinstance(model_config_kwargs, dict):
            model_config_kwargs = dict(model_config_kwargs.__dict__)
        visual_only = bool(kwargs.pop("visual_only", model_config_kwargs.get("visual_only", False)))
        model_fields = getattr(ModelConfig, "__dataclass_fields__", {})
        config = ModelConfig(**{k: v for k, v in model_config_kwargs.items() if k in model_fields})
        config.visual_only = visual_only
        module = CMARVisualOnly(config) if visual_only else CMAR(config)

        state_dict = _extract_state_dict(checkpoint)
        module.load_state_dict(state_dict)
        module.eval()
        return cls(
            module,
            name=kwargs.pop("name", "cmar"),
            feature_dims=kwargs.pop("feature_dims", None),
            metadata={"checkpoint": str(path), **kwargs.pop("metadata", {})},
        )


class CertAVAdapter(CMARAdapter):
    """CMAR adapter with CertAV smoothing metadata."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        training_sigma: float | None = None,
        noise_mode: str = "joint",
        name: str = "certav",
        feature_dims: dict[str, tuple[int, ...]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        if training_sigma is not None:
            metadata["training_sigma"] = training_sigma
        metadata["noise_mode"] = noise_mode
        super().__init__(module, name=name, feature_dims=feature_dims, metadata=metadata)
        self.training_sigma = training_sigma
        self.noise_mode = noise_mode

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> CertAVAdapter:
        base = CMARAdapter.from_checkpoint(path, **kwargs)
        return cls(
            base.module,
            name=kwargs.get("name", "certav"),
            training_sigma=kwargs.get("training_sigma"),
            noise_mode=kwargs.get("noise_mode", "joint"),
            feature_dims=base.feature_dims,
            metadata=base.metadata,
        )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_module_prefix(value)
        if all(isinstance(k, str) for k in checkpoint):
            tensor_values = [isinstance(v, torch.Tensor) for v in checkpoint.values()]
            if tensor_values and all(tensor_values):
                return _strip_module_prefix(checkpoint)
    raise ValueError("Checkpoint does not contain a loadable model state dict.")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
