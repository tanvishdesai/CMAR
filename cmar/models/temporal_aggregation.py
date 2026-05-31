from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class VisualTemporalAggregator(nn.Module):
    """Maps per-frame DINOv2 features to aligned temporal segments."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        n_segments: int = 8,
        max_frames: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_segments = n_segments
        self.max_frames = max_frames
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 3:
            raise ValueError("visual_features must have shape (batch, frames, dim)")
        bsz, n_frames, _ = visual_features.shape
        if n_frames > self.max_frames:
            visual_features = visual_features[:, : self.max_frames]
            n_frames = self.max_frames
        x = self.proj(visual_features)
        x = x + self.pos_embed[:, :n_frames]
        x = self.encoder(x)

        if n_frames == self.n_segments:
            return self.norm(x)

        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.n_segments)
        x = x.transpose(1, 2).contiguous()
        return self.norm(x)


class AudioTemporalAggregator(nn.Module):
    """Projects Whisper temporal features and pools them to aligned segments."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        n_segments: int = 8,
    ) -> None:
        super().__init__()
        self.n_segments = n_segments
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        if audio_features.ndim == 2:
            audio_features = audio_features.unsqueeze(1)
        if audio_features.ndim != 3:
            raise ValueError("audio_features must have shape (batch, time, dim)")
        x = self.proj(audio_features)
        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.n_segments)
        x = x.transpose(1, 2).contiguous()
        return self.norm(x)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Small helper kept for experiments that need deterministic positions."""

    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
