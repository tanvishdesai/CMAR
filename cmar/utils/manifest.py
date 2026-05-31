from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from cmar.config import DataSplitConfig
from cmar.utils.io import ensure_dir


FAKEAVCELEB_CATEGORIES = {
    "RealVideo-RealAudio": ("RR", 0),
    "FakeVideo-RealAudio": ("FR", 1),
    "RealVideo-FakeAudio": ("RF", 1),
    "FakeVideo-FakeAudio": ("FF", 1),
}


def stable_id(text: str, prefix: str = "") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem.replace(" ", "_")
    stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    return f"{prefix}{stem}_{digest}"


def resolve_fakeavceleb_root(root: str | Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"FakeAVCeleb root does not exist: {root}")
    if all((root / name).exists() for name in FAKEAVCELEB_CATEGORIES):
        return root
    for child in root.rglob("RealVideo-RealAudio"):
        candidate = child.parent
        if all((candidate / name).exists() for name in FAKEAVCELEB_CATEGORIES):
            return candidate
    raise FileNotFoundError(
        "Could not find FakeAVCeleb category folders under "
        f"{root}. Expected RealVideo-RealAudio, FakeVideo-RealAudio, "
        "RealVideo-FakeAudio, FakeVideo-FakeAudio."
    )


def discover_fakeavceleb_videos(root: str | Path) -> pd.DataFrame:
    root = resolve_fakeavceleb_root(root)
    rows: List[Dict[str, object]] = []
    for category, (av_category, label) in FAKEAVCELEB_CATEGORIES.items():
        category_dir = root / category
        for path in category_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
                continue
            if any(part.lower() in {"frames", "moved"} for part in path.relative_to(root).parts):
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            race = parts[1] if len(parts) > 2 else "unknown"
            gender = parts[2] if len(parts) > 3 else "unknown"
            person_id = parts[3] if len(parts) > 4 else "unknown"
            rows.append(
                {
                    "clip_id": stable_id(str(rel).replace("\\", "/"), prefix=f"{av_category}_"),
                    "label": label,
                    "av_category": av_category,
                    "video_path": str(path),
                    "audio_path": str(path),
                    "source_dataset": "FakeAVCeleb",
                    "source_category": category,
                    "race": race,
                    "gender": gender,
                    "person_id": person_id,
                }
            )
    if not rows:
        raise RuntimeError(f"No videos found under FakeAVCeleb root: {root}")
    return pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)


def _take_split(
    rows: pd.DataFrame,
    train_n: int,
    val_n: int,
    test_n: int,
    rng: random.Random,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = list(rows.index)
    rng.shuffle(idx)
    requested = train_n + val_n + test_n
    if len(idx) >= requested:
        train_idx = idx[:train_n]
        val_idx = idx[train_n : train_n + val_n]
        test_idx = idx[train_n + val_n : requested]
    else:
        n = len(idx)
        train_end = max(1, int(round(n * 0.70)))
        val_end = min(n, train_end + max(1, int(round(n * 0.15))))
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
    return rows.loc[train_idx], rows.loc[val_idx], rows.loc[test_idx]


def build_fakeavceleb_splits(
    root: str | Path,
    output_manifest_dir: str | Path,
    split_config: Optional[DataSplitConfig] = None,
) -> Dict[str, pd.DataFrame]:
    split_config = split_config or DataSplitConfig()
    rng = random.Random(split_config.seed)
    df = discover_fakeavceleb_videos(root)
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    real_train, real_val, real_test = _take_split(
        real,
        split_config.train_real,
        split_config.val_real,
        split_config.test_real,
        rng,
    )
    fake_train, fake_val, fake_test = _take_split(
        fake,
        split_config.train_fake,
        split_config.val_fake,
        split_config.test_fake,
        rng,
    )
    splits = {
        "train": pd.concat([real_train, fake_train], ignore_index=True),
        "val": pd.concat([real_val, fake_val], ignore_index=True),
        "test": pd.concat([real_test, fake_test], ignore_index=True),
    }
    manifest_dir = ensure_dir(output_manifest_dir)
    for split, split_df in splits.items():
        split_df = split_df.sample(frac=1.0, random_state=split_config.seed).reset_index(drop=True)
        split_df["split"] = split
        splits[split] = split_df
        split_df.to_csv(manifest_dir / f"{split}.csv", index=False)
    return splits


def _metadata_items(obj: object) -> Iterable[Dict[str, object]]:
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ("root", "data", "items", "videos"):
            if key in obj:
                yield from _metadata_items(obj[key])
                return
        if "file" in obj:
            yield obj
        else:
            for value in obj.values():
                yield from _metadata_items(value)


def build_lavdf_manifest(
    root: str | Path,
    output_manifest_dir: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    root = Path(root)
    metadata_path = root / "metadata.min.json"
    if not metadata_path.exists():
        metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No LAV-DF metadata JSON found under {root}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    rows: List[Dict[str, object]] = []
    for item in _metadata_items(metadata):
        item_split = str(item.get("split", "")).lower()
        rel_file = str(item.get("file", ""))
        if not rel_file or (item_split and item_split != split.lower()):
            continue
        video_path = root / rel_file
        modify_video = bool(item.get("modify_video", False))
        modify_audio = bool(item.get("modify_audio", False))
        n_fakes = int(item.get("n_fakes", 0) or 0)
        label = int(n_fakes > 0 or modify_video or modify_audio)
        if not label:
            av_category = "RR"
        elif modify_video and modify_audio:
            av_category = "FF"
        elif modify_video:
            av_category = "FR"
        elif modify_audio:
            av_category = "RF"
        else:
            av_category = "FF"
        rows.append(
            {
                "clip_id": stable_id(rel_file, prefix=f"LAVDF_{av_category}_"),
                "label": label,
                "av_category": av_category,
                "video_path": str(video_path),
                "audio_path": str(video_path),
                "source_dataset": "LAV-DF",
                "duration": item.get("duration"),
                "split": split,
            }
        )
    if not rows:
        raise RuntimeError(f"No LAV-DF rows discovered for split={split} under {root}")
    df = pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)
    manifest_dir = ensure_dir(output_manifest_dir)
    df.to_csv(manifest_dir / f"lavdf_{split}.csv", index=False)
    return df
