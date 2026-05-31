from __future__ import annotations

import gc
import time
import warnings
import shutil
import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm.auto import tqdm

from cmar.config import CacheConfig, DEGRADED_CONDITIONS
from cmar.evaluation.degradations import (
    DEGRADATION_SPECS,
    degrade_audio,
    degrade_frames,
    h264_roundtrip_frames,
)
from cmar.models.audio_encoder import WhisperTinyFeatureExtractor
from cmar.models.visual_encoder import DINOv2FeatureExtractor
from cmar.utils.io import ensure_dir, save_tensor, write_json


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class ExtractionBudget:
    max_new_rows: int = 0
    max_runtime_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    new_rows: int = 0
    stopped: bool = False

    def mark_row(self, wrote_feature: bool) -> None:
        if wrote_feature:
            self.new_rows += 1
        if self.max_new_rows > 0 and self.new_rows >= self.max_new_rows:
            self.stopped = True
        if self.max_runtime_seconds > 0 and (time.time() - self.started_at) >= self.max_runtime_seconds:
            self.stopped = True


def cleanup_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage_mb() -> str:
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        return f"{rss:.0f}MB"
    except Exception:
        pass
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "N/A"


def pool_feature_tokens(features: torch.Tensor, max_tokens: int) -> torch.Tensor:
    features = torch.as_tensor(features).float()
    if max_tokens <= 0 or features.ndim != 2 or features.shape[0] <= max_tokens:
        return features.contiguous()
    pooled = F.adaptive_avg_pool1d(features.t().unsqueeze(0), max_tokens)
    return pooled.squeeze(0).t().contiguous()


def sample_video_frames(
    video_path: str | Path,
    n_frames: int = 16,
) -> List[np.ndarray]:
    import cv2

    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        warnings.warn(f"Could not open video: {path}")
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(n_frames)]

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        indices = list(range(n_frames))
    else:
        indices = np.linspace(0, max(0, total - 1), n_frames).round().astype(int).tolist()

    frames: List[np.ndarray] = []
    for index in indices:
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def frames_to_tensor(frames: Iterable[np.ndarray], image_size: int = 224) -> torch.Tensor:
    tensors = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").resize(
            (image_size, image_size),
            Image.BICUBIC,
        )
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def load_audio(
    media_path: str | Path,
    sample_rate: int = 16000,
    max_seconds: float = 10.0,
) -> np.ndarray:
    target_len = int(sample_rate * max_seconds)
    media_path = Path(media_path)

    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-t",
            str(max_seconds),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode == 0 and proc.stdout:
                waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                if waveform.size > 0:
                    if waveform.size < target_len:
                        waveform = np.pad(waveform, (0, target_len - waveform.size))
                    elif waveform.size > target_len:
                        waveform = waveform[:target_len]
                    return waveform.astype(np.float32)
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            if stderr:
                warnings.warn(f"ffmpeg audio decode failed for {media_path}: {stderr[:240]}")
        except Exception as exc:  # noqa: BLE001 - fall back to librosa below.
            warnings.warn(f"ffmpeg audio decode failed for {media_path}: {exc}")

    import librosa

    try:
        waveform, _ = librosa.load(str(media_path), sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001 - keep preprocessing robust on bad clips.
        warnings.warn(f"Could not load audio from {media_path}: {exc}")
        waveform = np.zeros(target_len, dtype=np.float32)
    waveform = waveform.astype(np.float32)
    if waveform.size < target_len:
        waveform = np.pad(waveform, (0, target_len - waveform.size))
    elif waveform.size > target_len:
        waveform = waveform[:target_len]
    return waveform


@torch.inference_mode()
def extract_visual_features(
    visual_encoder: DINOv2FeatureExtractor,
    frames: List[np.ndarray],
    device: torch.device,
    image_size: int = 224,
) -> torch.Tensor:
    visual_encoder.eval()
    batch = frames_to_tensor(frames, image_size=image_size).to(device)
    features = visual_encoder(batch)
    return features.detach().cpu()


@torch.inference_mode()
def extract_audio_features(
    audio_encoder: WhisperTinyFeatureExtractor,
    whisper_processor,
    waveform: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    audio_encoder.eval()
    inputs = whisper_processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    features = audio_encoder(input_features)
    return features.squeeze(0).detach().cpu()


def build_extractors(
    device: torch.device,
    image_size: int = 224,
    load_visual: bool = True,
    load_audio: bool = True,
) -> Tuple[Optional[DINOv2FeatureExtractor], Optional[WhisperTinyFeatureExtractor], object | None]:
    from transformers import WhisperFeatureExtractor

    visual_encoder = None
    if load_visual:
        visual_encoder = DINOv2FeatureExtractor(
            pretrained=True,
            tune_layernorm=False,
            image_size=image_size,
        ).to(device)
    audio_encoder = None
    whisper_processor = None
    if load_audio:
        audio_encoder = WhisperTinyFeatureExtractor(tune_layernorm=False).to(device)
        whisper_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    return visual_encoder, audio_encoder, whisper_processor


def feature_path(
    cache_dir: str | Path,
    modality: str,
    split: str,
    clip_id: str,
    condition: str = "clean",
) -> Path:
    root = Path(cache_dir)
    if condition == "clean":
        return root / "features" / modality / split / f"{clip_id}.pt"
    return root / "features" / "degraded_test" / condition / modality / f"{clip_id}.pt"


def _save_visual(
    row: pd.Series,
    split: str,
    out_path: Path,
    visual_encoder: DINOv2FeatureExtractor,
    device: torch.device,
    config: CacheConfig,
    condition: str = "clean",
    source_video: Optional[str | Path] = None,
) -> None:
    if visual_encoder is None:
        raise RuntimeError("Visual extractor was not loaded but visual features were requested.")
    video_path = source_video or row["video_path"]
    frames = sample_video_frames(video_path, n_frames=config.n_frames)
    if condition == "d11_h264_crf28":
        frames = h264_roundtrip_frames(frames, crf=28)
    elif condition != "clean":
        frames = degrade_frames(frames, condition, seed=int(row.name) + 2026)
    features = extract_visual_features(
        visual_encoder,
        frames,
        device=device,
        image_size=config.image_size,
    )
    save_tensor(features, out_path, dtype=config.feature_dtype)
    del frames, features


def _save_audio(
    row: pd.Series,
    split: str,
    out_path: Path,
    audio_encoder: WhisperTinyFeatureExtractor,
    whisper_processor,
    device: torch.device,
    config: CacheConfig,
    condition: str = "clean",
    source_media: Optional[str | Path] = None,
) -> None:
    if audio_encoder is None or whisper_processor is None:
        raise RuntimeError("Audio extractor was not loaded but audio features were requested.")
    media_path = source_media or row["audio_path"]
    waveform = load_audio(
        media_path,
        sample_rate=config.audio_sr,
        max_seconds=config.audio_max_seconds,
    )
    if condition != "clean":
        waveform = degrade_audio(
            waveform,
            sample_rate=config.audio_sr,
            condition=condition,
            seed=int(row.name) + 2026,
        )
        target_len = int(config.audio_sr * config.audio_max_seconds)
        if waveform.size < target_len:
            waveform = np.pad(waveform, (0, target_len - waveform.size))
        elif waveform.size > target_len:
            waveform = waveform[:target_len]
    features = extract_audio_features(
        audio_encoder,
        whisper_processor,
        waveform,
        sample_rate=config.audio_sr,
        device=device,
    )
    features = pool_feature_tokens(features, config.max_audio_tokens)
    save_tensor(features, out_path, dtype=config.feature_dtype)
    del waveform, features


def extract_clean_manifest(
    manifest: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    visual_encoder: Optional[DINOv2FeatureExtractor],
    audio_encoder: Optional[WhisperTinyFeatureExtractor],
    whisper_processor,
    device: torch.device,
    config: CacheConfig,
    budget: Optional[ExtractionBudget] = None,
    chunk_size: int = 0,
) -> Dict[str, float]:
    start = time.time()
    processed = 0
    budget = budget or ExtractionBudget()
    chunk_size = chunk_size if chunk_size > 0 else len(manifest) + 1
    for row_index, (_, row) in enumerate(
        tqdm(manifest.iterrows(), total=len(manifest), desc=f"clean {split}", mininterval=10.0),
        start=1,
    ):
        if budget.stopped:
            break
        clip_id = str(row["clip_id"])
        v_path = feature_path(cache_dir, "visual", split, clip_id)
        a_path = feature_path(cache_dir, "audio", split, clip_id)
        wrote_feature = False
        if config.overwrite or not v_path.exists():
            _save_visual(row, split, v_path, visual_encoder, device, config)
            wrote_feature = True
        if config.overwrite or not a_path.exists():
            _save_audio(row, split, a_path, audio_encoder, whisper_processor, device, config)
            wrote_feature = True
        budget.mark_row(wrote_feature)
        processed += 1
        if row_index % chunk_size == 0:
            cleanup_memory(device)
            print(f"[cleanup] clean {split}: row={row_index}/{len(manifest)} mem={get_memory_usage_mb()}")
    elapsed = time.time() - start
    cleanup_memory(device)
    return {
        "clips": processed,
        "seconds": elapsed,
        "seconds_per_clip": elapsed / max(1, processed),
        "stopped": budget.stopped,
        "new_rows": budget.new_rows,
    }


def extract_degraded_test_manifest(
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    visual_encoder: Optional[DINOv2FeatureExtractor],
    audio_encoder: Optional[WhisperTinyFeatureExtractor],
    whisper_processor,
    device: torch.device,
    config: CacheConfig,
    conditions: Optional[List[str]] = None,
    budget: Optional[ExtractionBudget] = None,
    chunk_size: int = 0,
) -> Dict[str, Dict[str, float]]:
    conditions = conditions or DEGRADED_CONDITIONS
    budget = budget or ExtractionBudget()
    chunk_size = chunk_size if chunk_size > 0 else len(manifest) + 1
    timings: Dict[str, Dict[str, float]] = {}
    for condition in conditions:
        if budget.stopped:
            break
        spec = DEGRADATION_SPECS[condition]
        start = time.time()
        processed = 0
        for row_index, (_, row) in enumerate(
            tqdm(manifest.iterrows(), total=len(manifest), desc=condition, mininterval=10.0),
            start=1,
        ):
            if budget.stopped:
                break
            clip_id = str(row["clip_id"])
            wrote_feature = False

            if spec.visual:
                v_path = feature_path(cache_dir, "visual", "test", clip_id, condition=condition)
                if config.overwrite or not v_path.exists():
                    _save_visual(
                        row,
                        "test",
                        v_path,
                        visual_encoder,
                        device,
                        config,
                        condition=condition,
                    )
                    wrote_feature = True
            if spec.audio:
                a_path = feature_path(cache_dir, "audio", "test", clip_id, condition=condition)
                if config.overwrite or not a_path.exists():
                    _save_audio(
                        row,
                        "test",
                        a_path,
                        audio_encoder,
                        whisper_processor,
                        device,
                        config,
                        condition=condition,
                    )
                    wrote_feature = True
            budget.mark_row(wrote_feature)
            processed += 1
            if row_index % chunk_size == 0:
                cleanup_memory(device)
                print(f"[cleanup] {condition}: row={row_index}/{len(manifest)} mem={get_memory_usage_mb()}")
        elapsed = time.time() - start
        timings[condition] = {
            "clips": processed,
            "seconds": elapsed,
            "seconds_per_clip": elapsed / max(1, processed),
            "stopped": budget.stopped,
            "new_rows": budget.new_rows,
        }
        cleanup_memory(device)
    return timings


def write_cache_metadata(
    cache_dir: str | Path,
    config: CacheConfig,
    split_sizes: Dict[str, int],
    timings: Dict[str, object],
) -> None:
    metadata = {
        "project": "CMAR",
        "cache_version": "v1",
        "feature_contract": {
            "visual": "(16, 384) DINOv2-Small CLS features per sampled frame",
            "audio": (
                f"(<= {config.max_audio_tokens}, 384) pooled Whisper-Tiny encoder "
                "temporal features, float16 by default"
            ),
            "training_note": (
                "Default training consumes cached features; DINOv2/Whisper LN tuning "
                "requires optional raw-mode experiments because cached features detach "
                "the backbones from the graph."
            ),
        },
        "config": config.__dict__,
        "split_sizes": split_sizes,
        "timings": timings,
        "degraded_conditions": {
            name: spec.__dict__ for name, spec in DEGRADATION_SPECS.items()
        },
    }
    write_json(metadata, Path(cache_dir) / "metadata.json")


def verify_cache_shapes(cache_dir: str | Path, manifest_dir: str | Path, split: str = "train") -> Dict[str, object]:
    manifest = pd.read_csv(Path(manifest_dir) / f"{split}.csv")
    if manifest.empty:
        raise RuntimeError(f"Manifest {split}.csv is empty")
    row = manifest.iloc[0]
    v = torch.load(feature_path(cache_dir, "visual", split, row["clip_id"]), map_location="cpu")
    a = torch.load(feature_path(cache_dir, "audio", split, row["clip_id"]), map_location="cpu")
    return {
        "clip_id": row["clip_id"],
        "visual_shape": list(v.shape),
        "audio_shape": list(a.shape),
        "visual_dtype": str(v.dtype),
        "audio_dtype": str(a.dtype),
    }
