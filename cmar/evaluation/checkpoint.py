from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from cmar.config import ModelConfig
from cmar.models import CMAR, CMARVisualOnly


def _model_config_from_dict(data: dict) -> ModelConfig:
    allowed = {field.name for field in ModelConfig.__dataclass_fields__.values()}
    return ModelConfig(**{key: value for key, value in data.items() if key in allowed})


def model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, ModelConfig, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    cfg_dict = checkpoint.get("config", {}).get("model", {})
    config = _model_config_from_dict(cfg_dict) if cfg_dict else ModelConfig()
    model = CMARVisualOnly(config) if config.visual_only else CMAR(config)
    model.to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()
    return model, config, checkpoint
