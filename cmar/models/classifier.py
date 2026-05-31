from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        pooled = segment_features.mean(dim=1)
        return self.net(pooled).squeeze(-1)
