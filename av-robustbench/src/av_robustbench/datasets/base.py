from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class AVSample:
    visual: torch.Tensor
    audio: torch.Tensor
    label: int | float | torch.Tensor
    clip_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual": self.visual,
            "audio": self.audio,
            "label": torch.as_tensor(self.label),
            "clip_id": self.clip_id,
            "metadata": self.metadata,
        }


class AVDataset(Dataset, ABC):
    """Dataset contract used by av-robustbench."""

    dataset_name: str = "av_dataset"

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a dict with visual, audio, label, clip_id."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""

    @property
    @abstractmethod
    def split(self) -> str:
        """Dataset split name."""

    @property
    def n_samples(self) -> int:
        return len(self)

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for index in range(len(self)):
            label = self[index]["label"]
            label_int = int(label.item() if isinstance(label, torch.Tensor) else label)
            counts[label_int] = counts.get(label_int, 0) + 1
        return counts


def collate_av_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    visuals = [_as_tensor(item["visual"]) for item in batch]
    audios = [_as_tensor(item["audio"]) for item in batch]
    labels = torch.stack([_as_tensor(item["label"]).float().reshape(()) for item in batch])
    output = {
        "visual": pad_sequence(visuals, batch_first=True),
        "audio": pad_sequence(audios, batch_first=True),
        "label": labels,
        "clip_id": [str(item.get("clip_id", index)) for index, item in enumerate(batch)],
        "metadata": [item.get("metadata", {}) for item in batch],
    }
    return output


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

