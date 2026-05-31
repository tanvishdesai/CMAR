from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class WhisperTinyFeatureExtractor(nn.Module):
    """Whisper encoder wrapper for cache generation and optional raw experiments."""

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        tune_layernorm: bool = False,
    ) -> None:
        super().__init__()
        from transformers import WhisperModel

        self.model_name = model_name
        self.model = WhisperModel.from_pretrained(model_name)
        self.encoder = self.model.encoder
        self.freeze_except_layernorm(tune_layernorm=tune_layernorm)

    def freeze_except_layernorm(self, tune_layernorm: bool = False) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        if tune_layernorm:
            for module in self.encoder.modules():
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        param.requires_grad = True

    def layernorm_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.encoder.modules():
            if isinstance(module, nn.LayerNorm):
                yield from module.parameters()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state
