from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_v_to_a = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_a_to_v = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_v1 = nn.LayerNorm(hidden_dim)
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_v2 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.ffn_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        v_query = self.norm_v1(visual)
        a_key = self.norm_a1(audio)
        v_delta, v_attn = self.attn_v_to_a(
            query=v_query,
            key=a_key,
            value=a_key,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        visual = visual + v_delta
        visual = visual + self.ffn_v(self.norm_v2(visual))

        a_query = self.norm_a1(audio)
        v_key = self.norm_v1(visual)
        a_delta, a_attn = self.attn_a_to_v(
            query=a_query,
            key=v_key,
            value=v_key,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        audio = audio + a_delta
        audio = audio + self.ffn_a(self.norm_a2(audio))

        weights = None
        if need_weights:
            weights = {"v_to_a": v_attn.detach(), "a_to_v": a_attn.detach()}
        return visual, audio, weights


class CrossModalConsistencyModule(nn.Module):
    """Bidirectional cross-modal attention followed by segment-wise fusion."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        all_weights = []
        for layer in self.layers:
            visual, audio, weights = layer(visual, audio, need_weights=return_attention)
            if weights is not None:
                all_weights.append(weights)
        fused = self.fusion(torch.cat([visual, audio], dim=-1))
        if not return_attention:
            return fused, None
        packed = {
            "v_to_a": torch.stack([w["v_to_a"] for w in all_weights], dim=1),
            "a_to_v": torch.stack([w["a_to_v"] for w in all_weights], dim=1),
        }
        return fused, packed
