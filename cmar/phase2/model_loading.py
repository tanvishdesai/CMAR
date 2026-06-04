"""Small loading helpers shared by Phase 2 scripts."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from cmar.config import ModelConfig


def model_config_from_checkpoint(ckpt: dict[str, Any], args: Namespace | None = None) -> ModelConfig:
    """Rebuild ``ModelConfig`` from a checkpoint with optional CLI overrides."""

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    args = args or Namespace()

    def arg_or_cfg(name: str, default: Any) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return model_cfg.get(name, default)

    return ModelConfig(
        visual_dim=int(arg_or_cfg("visual_dim", 384)),
        audio_dim=int(arg_or_cfg("audio_dim", 384)),
        hidden_dim=int(arg_or_cfg("hidden_dim", 256)),
        n_segments=int(arg_or_cfg("n_segments", 8)),
        max_frames=int(arg_or_cfg("max_frames", 16)),
        cmcm_layers=int(arg_or_cfg("cmcm_layers", 2)),
        attention_heads=int(arg_or_cfg("attention_heads", 8)),
        dropout=float(arg_or_cfg("dropout", 0.3)),
    )
