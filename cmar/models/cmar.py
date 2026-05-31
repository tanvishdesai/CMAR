from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from cmar.config import ModelConfig
from cmar.models.classifier import ClassificationHead
from cmar.models.cmcm import CrossModalConsistencyModule
from cmar.models.temporal_aggregation import AudioTemporalAggregator, VisualTemporalAggregator


class CMAR(nn.Module):
    """Feature-cached CMAR model.

    Inputs are cached DINOv2 frame features and cached Whisper temporal features.
    This is the fast Kaggle training path: backbones are frozen at preprocessing
    time and reruns only train aggregation, CMCM, and the classifier.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.visual = VisualTemporalAggregator(
            input_dim=self.config.visual_dim,
            hidden_dim=self.config.hidden_dim,
            n_segments=self.config.n_segments,
            max_frames=self.config.max_frames,
            n_heads=self.config.attention_heads,
            dropout=0.1,
        )
        self.audio = AudioTemporalAggregator(
            input_dim=self.config.audio_dim,
            hidden_dim=self.config.hidden_dim,
            n_segments=self.config.n_segments,
        )
        self.cmcm = CrossModalConsistencyModule(
            hidden_dim=self.config.hidden_dim,
            n_heads=self.config.attention_heads,
            num_layers=self.config.cmcm_layers,
            dropout=0.1,
        )
        self.classifier = ClassificationHead(
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        visual_segments = self.visual(visual_features)
        audio_segments = self.audio(audio_features)
        fused, attention = self.cmcm(
            visual_segments,
            audio_segments,
            return_attention=return_attention,
        )
        logits = self.classifier(fused)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            out["features"] = fused
            out["visual_segments"] = visual_segments
            out["audio_segments"] = audio_segments
        if attention is not None:
            out["attention_v_to_a"] = attention["v_to_a"]
            out["attention_a_to_v"] = attention["a_to_v"]
        return out


class CMARVisualOnly(nn.Module):
    """Visual-only ablation using cached DINOv2 features."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(visual_only=True)
        self.visual = VisualTemporalAggregator(
            input_dim=self.config.visual_dim,
            hidden_dim=self.config.hidden_dim,
            n_segments=self.config.n_segments,
            max_frames=self.config.max_frames,
            n_heads=self.config.attention_heads,
            dropout=0.1,
        )
        self.classifier = ClassificationHead(
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        visual_segments = self.visual(visual_features)
        logits = self.classifier(visual_segments)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            out["features"] = visual_segments
            out["visual_segments"] = visual_segments
        return out


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
