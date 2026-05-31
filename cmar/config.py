from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataSplitConfig:
    """Deterministic FakeAVCeleb split sizes used by the paper protocol."""

    seed: int = 2026
    train_real: int = 350
    val_real: int = 75
    test_real: int = 75
    train_fake: int = 3500
    val_fake: int = 750
    test_fake: int = 750


@dataclass
class CacheConfig:
    """Paths and feature extraction settings for reusable Kaggle caches."""

    dataset_root: Optional[str] = None
    lavdf_root: Optional[str] = None
    output_dir: str = "/kaggle/working/cmar_cache"
    n_frames: int = 16
    image_size: int = 224
    audio_sr: int = 16000
    audio_max_seconds: float = 10.0
    max_audio_tokens: int = 64
    feature_dtype: str = "float16"
    overwrite: bool = False
    extract_degraded_test: bool = True
    include_train_degraded: bool = False


@dataclass
class AudioConfig:
    whisper_name: str = "openai/whisper-tiny"
    sample_rate: int = 16000
    max_seconds: float = 10.0


@dataclass
class ModelConfig:
    visual_dim: int = 384
    audio_dim: int = 384
    hidden_dim: int = 256
    n_segments: int = 8
    max_frames: int = 16
    cmcm_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.3
    visual_only: bool = False
    return_attention: bool = False


@dataclass
class TrainConfig:
    cache_dir: str = "/kaggle/input/cmar-features-v1/cmar_cache"
    output_dir: str = "/kaggle/working/cmar_runs/full"
    seed: int = 2026
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 30
    grad_accum_steps: int = 4
    lr: float = 5e-4
    ln_lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_epochs: int = 3
    early_stop_patience: int = 5
    consistency_weight: float = 0.3
    use_consistency: bool = True
    feature_augmentation: bool = True
    amp: bool = False
    monitor_metric: str = "auc"
    save_every: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)


def to_dict(config: Any) -> Dict[str, Any]:
    """Dataclass-to-dict helper that also stringifies Paths."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(asdict(config))


@dataclass
class SmoothingConfig:
    """Configuration for randomized smoothing certification."""

    sigma: float = 0.25  # Gaussian noise std dev
    n0: int = 100  # samples for prediction phase
    n: int = 1000  # samples for certification phase
    alpha: float = 0.001  # significance level (99.9% confidence)
    batch_size: int = 64  # batch size for Monte Carlo sampling
    noise_mode: str = "joint"  # 'joint', 'visual_only', 'audio_only'


DEGRADED_CONDITIONS: List[str] = [
    "d1_jpeg75",
    "d2_jpeg50",
    "d3_resize075",
    "d4_resize050",
    "d5_vnoise001",
    "d6_vnoise002",
    "d7_mp3_128k",
    "d8_mp3_64k",
    "d9_anoise_30db",
    "d10_anoise_20db",
    "d11_h264_crf28",
    "d12_social",
]
