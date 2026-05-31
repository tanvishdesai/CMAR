from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def symmetric_binary_kl(clean_logits: torch.Tensor, degraded_logits: torch.Tensor) -> torch.Tensor:
    clean_logits = torch.nan_to_num(clean_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    degraded_logits = torch.nan_to_num(degraded_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    clean_p = torch.sigmoid(clean_logits).clamp(1e-6, 1 - 1e-6)
    degraded_p = torch.sigmoid(degraded_logits).clamp(1e-6, 1 - 1e-6)
    clean_dist = torch.stack([1 - clean_p, clean_p], dim=-1)
    degraded_dist = torch.stack([1 - degraded_p, degraded_p], dim=-1)
    clean_log = clean_dist.log()
    degraded_log = degraded_dist.log()
    kl_cd = F.kl_div(clean_log, degraded_dist, reduction="batchmean")
    kl_dc = F.kl_div(degraded_log, clean_dist, reduction="batchmean")
    return kl_cd + kl_dc


@dataclass
class CMARLoss:
    consistency_weight: float = 0.3
    use_consistency: bool = True

    def __post_init__(self) -> None:
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(
        self,
        clean_logits: torch.Tensor,
        labels: torch.Tensor,
        degraded_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        clean_logits = torch.nan_to_num(clean_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        labels = labels.float()
        bce = self.bce(clean_logits, labels)
        consistency = torch.zeros((), device=clean_logits.device)
        if self.use_consistency and degraded_logits is not None:
            consistency = symmetric_binary_kl(clean_logits, degraded_logits)
        total = bce + self.consistency_weight * consistency
        total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
        return {
            "loss": total,
            "bce": bce.detach(),
            "consistency": consistency.detach(),
        }
