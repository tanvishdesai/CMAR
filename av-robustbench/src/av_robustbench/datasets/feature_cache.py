from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from av_robustbench.datasets.base import AVDataset


def cache_feature_path(
    cache_dir: str | Path,
    modality: str,
    split: str,
    clip_id: str,
    *,
    condition: str = "clean",
) -> Path:
    root = Path(cache_dir)
    if condition == "clean":
        return root / "features" / modality / split / f"{clip_id}.pt"
    return root / "features" / "degraded_test" / condition / modality / f"{clip_id}.pt"


class FeatureCacheDataset(AVDataset):
    """Generic cached-feature dataset compatible with the CMAR cache layout."""

    dataset_name = "feature_cache"

    def __init__(
        self,
        cache_dir: str | Path,
        manifest_csv: str | Path | None = None,
        *,
        split: str = "test",
        condition: str = "clean",
        visual_only: bool = False,
        allow_partial_cache: bool = False,
        label_column: str = "label",
        clip_id_column: str = "clip_id",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_csv = Path(manifest_csv) if manifest_csv else self.cache_dir / "manifests" / f"{split}.csv"
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        self.manifest = pd.read_csv(self.manifest_csv)
        if "split" in self.manifest.columns:
            self.manifest = self.manifest[self.manifest["split"].astype(str) == split].reset_index(drop=True)
        self._split = split
        self.condition = condition
        self.visual_only = visual_only
        self.label_column = label_column
        self.clip_id_column = clip_id_column
        if allow_partial_cache:
            self.manifest = self._filter_available_rows()
        if self.manifest.empty:
            raise RuntimeError(f"No rows available in {self.manifest_csv} for split={split}.")

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        clip_id = str(row[self.clip_id_column])
        visual = self._load_feature("visual", clip_id)
        if self.visual_only:
            audio = torch.empty(1, visual.shape[-1], dtype=torch.float32)
        else:
            audio = self._load_feature("audio", clip_id)
        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        metadata = {
            key: _json_scalar(row[key])
            for key in self.manifest.columns
            if key not in {self.label_column, self.clip_id_column}
        }
        return {
            "visual": visual,
            "audio": audio,
            "label": label,
            "clip_id": clip_id,
            "metadata": metadata,
        }

    def coverage_report(self) -> dict[str, Any]:
        total = len(self.manifest)
        visual_ok = 0
        audio_ok = 0
        for _, row in self.manifest.iterrows():
            clip_id = str(row[self.clip_id_column])
            if self._feature_exists("visual", clip_id):
                visual_ok += 1
            if self.visual_only or self._feature_exists("audio", clip_id):
                audio_ok += 1
        return {
            "cache_dir": str(self.cache_dir),
            "manifest_csv": str(self.manifest_csv),
            "split": self.split,
            "condition": self.condition,
            "rows": total,
            "visual_available": visual_ok,
            "audio_available": audio_ok,
            "complete": visual_ok == total and audio_ok == total,
        }

    def _feature_exists(self, modality: str, clip_id: str) -> bool:
        path = cache_feature_path(self.cache_dir, modality, self.split, clip_id, condition=self.condition)
        if path.exists():
            return True
        if self.condition != "clean":
            return cache_feature_path(self.cache_dir, modality, self.split, clip_id, condition="clean").exists()
        return False

    def _filter_available_rows(self) -> pd.DataFrame:
        keep = []
        for _, row in self.manifest.iterrows():
            clip_id = str(row[self.clip_id_column])
            keep.append(
                self._feature_exists("visual", clip_id)
                and (self.visual_only or self._feature_exists("audio", clip_id))
            )
        return self.manifest.loc[keep].reset_index(drop=True)

    def _load_feature(self, modality: str, clip_id: str) -> torch.Tensor:
        path = cache_feature_path(self.cache_dir, modality, self.split, clip_id, condition=self.condition)
        if not path.exists() and self.condition != "clean":
            path = cache_feature_path(self.cache_dir, modality, self.split, clip_id, condition="clean")
        if not path.exists():
            raise FileNotFoundError(f"Missing {modality} feature for {clip_id}: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        return tensor.float() if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor, dtype=torch.float32)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value

