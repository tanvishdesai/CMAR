from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from cmar.utils.cache import feature_path
from cmar.utils.io import load_tensor


def feature_space_augment(
    visual: torch.Tensor,
    audio: torch.Tensor,
    p_noise: float = 0.5,
    noise_std: float = 0.015,
    temporal_drop_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap consistency augmentation for cached features.

    Pixel/audio codec transforms must happen before feature extraction, so the
    fast training path uses this feature-level proxy for the consistency loss.
    The real degraded test sets are still generated from raw media in Session 1.
    """

    visual_aug = visual.clone()
    audio_aug = audio.clone()
    if random.random() < p_noise:
        visual_aug = visual_aug + torch.randn_like(visual_aug) * noise_std
    if random.random() < p_noise:
        audio_aug = audio_aug + torch.randn_like(audio_aug) * noise_std
    if temporal_drop_prob > 0:
        if visual_aug.ndim == 2 and visual_aug.shape[0] > 1:
            mask = torch.rand(visual_aug.shape[0]) < temporal_drop_prob
            visual_aug[mask] = 0
        if audio_aug.ndim == 2 and audio_aug.shape[0] > 1:
            mask = torch.rand(audio_aug.shape[0]) < temporal_drop_prob
            audio_aug[mask] = 0
    return visual_aug, audio_aug


def cache_coverage_report(
    cache_dir: str | Path,
    manifest_csv: str | Path,
    split: Optional[str] = None,
    condition: str = "clean",
    visual_only: bool = False,
    max_examples: int = 10,
) -> Dict[str, object]:
    cache_dir = Path(cache_dir)
    manifest_csv = Path(manifest_csv)
    manifest = pd.read_csv(manifest_csv)
    if split is not None and "split" in manifest.columns:
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    inferred_split = split or manifest_csv.stem

    rows = len(manifest)
    missing_visual: List[str] = []
    missing_audio: List[str] = []
    available_mask = []
    for _, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        v_path = feature_path(cache_dir, "visual", inferred_split, clip_id, condition=condition)
        if not v_path.exists():
            v_path = feature_path(cache_dir, "visual", inferred_split, clip_id, condition="clean")
        a_path = feature_path(cache_dir, "audio", inferred_split, clip_id, condition=condition)
        if not a_path.exists():
            a_path = feature_path(cache_dir, "audio", inferred_split, clip_id, condition="clean")

        has_visual = v_path.exists()
        has_audio = True if visual_only else a_path.exists()
        available_mask.append(has_visual and has_audio)
        if not has_visual and len(missing_visual) < max_examples:
            missing_visual.append(clip_id)
        if not has_audio and len(missing_audio) < max_examples:
            missing_audio.append(clip_id)

    available = int(sum(available_mask))
    return {
        "manifest_csv": str(manifest_csv),
        "split": inferred_split,
        "condition": condition,
        "total_rows": rows,
        "available_rows": available,
        "missing_rows": rows - available,
        "missing_visual_examples": missing_visual,
        "missing_audio_examples": missing_audio,
        "complete": available == rows,
    }


class CachedAVDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        manifest_csv: str | Path,
        split: Optional[str] = None,
        condition: str = "clean",
        return_degraded: bool = False,
        feature_augmentation: bool = False,
        visual_only: bool = False,
        allow_partial_cache: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_csv = Path(manifest_csv)
        self.manifest = pd.read_csv(self.manifest_csv)
        if split is not None and "split" in self.manifest.columns:
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.split = split or self._infer_split()
        self.condition = condition
        self.return_degraded = return_degraded
        self.feature_augmentation = feature_augmentation
        self.visual_only = visual_only
        self.allow_partial_cache = allow_partial_cache
        if allow_partial_cache:
            self.manifest = self._filter_available_rows()
            if self.manifest.empty:
                raise RuntimeError(
                    f"No cached feature pairs are available for {self.manifest_csv} "
                    f"split={self.split} condition={self.condition}."
                )

    def _infer_split(self) -> str:
        stem = self.manifest_csv.stem
        if stem in {"train", "val", "test"}:
            return stem
        if "test" in stem:
            return "test"
        return stem

    def __len__(self) -> int:
        return len(self.manifest)

    def _feature_exists(self, modality: str, row: pd.Series) -> bool:
        clip_id = str(row["clip_id"])
        path = feature_path(self.cache_dir, modality, self.split, clip_id, condition=self.condition)
        if not path.exists():
            path = feature_path(self.cache_dir, modality, self.split, clip_id, condition="clean")
        return path.exists()

    def _filter_available_rows(self) -> pd.DataFrame:
        keep = []
        for _, row in self.manifest.iterrows():
            has_visual = self._feature_exists("visual", row)
            has_audio = True if self.visual_only else self._feature_exists("audio", row)
            keep.append(has_visual and has_audio)
        return self.manifest.loc[keep].reset_index(drop=True)

    def _load_modality(self, modality: str, row: pd.Series) -> torch.Tensor:
        clip_id = str(row["clip_id"])
        path = feature_path(self.cache_dir, modality, self.split, clip_id, condition=self.condition)
        if not path.exists():
            path = feature_path(self.cache_dir, modality, self.split, clip_id, condition="clean")
        if not path.exists():
            raise FileNotFoundError(f"Missing {modality} feature for {clip_id}: {path}")
        return load_tensor(path, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        visual = self._load_modality("visual", row)
        audio = torch.empty(1, 384)
        if not self.visual_only:
            audio = self._load_modality("audio", row)
        item: Dict[str, object] = {
            "clip_id": str(row["clip_id"]),
            "visual": visual,
            "audio": audio,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "av_category": str(row.get("av_category", "")),
        }
        if self.return_degraded:
            if self.feature_augmentation:
                visual_deg, audio_deg = feature_space_augment(visual, audio)
            else:
                visual_deg, audio_deg = visual.clone(), audio.clone()
            item["visual_degraded"] = visual_deg
            item["audio_degraded"] = audio_deg
        return item


def collate_av_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    visuals = [item["visual"] for item in batch]
    audios = [item["audio"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    visual_batch = pad_sequence(visuals, batch_first=True)
    audio_batch = pad_sequence(audios, batch_first=True)
    out: Dict[str, object] = {
        "clip_id": [item["clip_id"] for item in batch],
        "visual": visual_batch,
        "audio": audio_batch,
        "label": labels,
        "av_category": [item["av_category"] for item in batch],
    }
    if "visual_degraded" in batch[0]:
        out["visual_degraded"] = pad_sequence([item["visual_degraded"] for item in batch], batch_first=True)
        out["audio_degraded"] = pad_sequence([item["audio_degraded"] for item in batch], batch_first=True)
    return out
