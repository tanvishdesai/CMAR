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


class HFAudioFeatureExtractor(nn.Module):
    """Generic waveform encoder for HuBERT/WavLM-style models."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def extract_waveform(
        self,
        processor,
        waveform,
        sample_rate: int,
        device: torch.device,
    ) -> torch.Tensor:
        inputs = processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0).detach().cpu()


def build_audio_feature_extractor(model_name: str = "openai/whisper-tiny") -> tuple[nn.Module, object]:
    """Build an audio encoder and its processor by public model name."""

    if "whisper" in model_name.lower():
        from transformers import WhisperFeatureExtractor

        return (
            WhisperTinyFeatureExtractor(model_name=model_name, tune_layernorm=False),
            WhisperFeatureExtractor.from_pretrained(model_name),
        )

    from transformers import AutoFeatureExtractor

    return HFAudioFeatureExtractor(model_name), AutoFeatureExtractor.from_pretrained(model_name)
