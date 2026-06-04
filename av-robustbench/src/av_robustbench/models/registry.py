from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from av_robustbench.models.adapters.base import AVDetector
from av_robustbench.models.adapters.cmar import CertAVAdapter, CMARAdapter


class ModelNotAvailableError(RuntimeError):
    """Raised when a registered model has no resolvable checkpoint."""


@dataclass
class ModelMetadata:
    name: str
    adapter_class: type[AVDetector]
    dataset: str | None = None
    threat_model: str | None = None
    checkpoint_url: str | None = None
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    paper: str | None = None
    paper_url: str | None = None
    architecture: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_class": f"{self.adapter_class.__module__}.{self.adapter_class.__name__}",
            "dataset": self.dataset,
            "threat_model": self.threat_model,
            "checkpoint_url": self.checkpoint_url,
            "hf_repo_id": self.hf_repo_id,
            "hf_filename": self.hf_filename,
            "paper": self.paper,
            "paper_url": self.paper_url,
            "architecture": self.architecture,
            "tags": self.tags,
            **self.extra,
        }


_REGISTRY: dict[str, ModelMetadata] = {}


def register_model(
    name: str,
    adapter_class: type[AVDetector],
    metadata: dict[str, Any] | ModelMetadata | None = None,
    *,
    overwrite: bool = False,
) -> ModelMetadata:
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Model `{name}` is already registered.")
    if isinstance(metadata, ModelMetadata):
        item = metadata
    else:
        metadata = dict(metadata or {})
        item = ModelMetadata(
            name=name,
            adapter_class=adapter_class,
            dataset=metadata.pop("dataset", None),
            threat_model=metadata.pop("threat_model", None),
            checkpoint_url=metadata.pop("checkpoint_url", None),
            hf_repo_id=metadata.pop("hf_repo_id", None),
            hf_filename=metadata.pop("hf_filename", None),
            paper=metadata.pop("paper", None),
            paper_url=metadata.pop("paper_url", None),
            architecture=metadata.pop("architecture", None),
            tags=list(metadata.pop("tags", [])),
            extra=dict(metadata),
        )
    _REGISTRY[name] = item
    return item


def clear_registry() -> None:
    _REGISTRY.clear()
    _register_builtin_models()


def get_model_metadata(name: str) -> ModelMetadata:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model `{name}`. Available: {', '.join(sorted(_REGISTRY))}") from exc


def list_models() -> list[dict[str, Any]]:
    return [metadata.to_dict() for metadata in sorted(_REGISTRY.values(), key=lambda item: item.name)]


def load_model(
    name: str,
    *,
    dataset: str | None = None,
    threat_model: str | None = None,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    **adapter_kwargs: Any,
) -> AVDetector:
    metadata = get_model_metadata(name)
    if dataset is not None and metadata.dataset is not None and metadata.dataset != dataset:
        raise ValueError(f"Model `{name}` is registered for dataset `{metadata.dataset}`, not `{dataset}`.")
    if threat_model is not None and metadata.threat_model is not None and metadata.threat_model != threat_model:
        raise ValueError(
            f"Model `{name}` is registered for threat model `{metadata.threat_model}`, not `{threat_model}`."
        )

    resolved_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resolved_path is None:
        resolved_path = _download_registered_checkpoint(metadata, cache_dir=cache_dir)
    if resolved_path is None:
        raise ModelNotAvailableError(
            f"Model `{name}` is registered but no checkpoint path or downloadable checkpoint is configured. "
            "Pass `checkpoint_path=...` or register HuggingFace metadata with `hf_repo_id` and `hf_filename`."
        )
    return metadata.adapter_class.from_checkpoint(
        resolved_path,
        name=name,
        **metadata.extra,
        **adapter_kwargs,
    )


def _download_registered_checkpoint(
    metadata: ModelMetadata,
    *,
    cache_dir: str | Path | None = None,
) -> Path | None:
    if metadata.hf_repo_id and metadata.hf_filename:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Loading HuggingFace-hosted models requires `pip install av-robustbench[models]`."
            ) from exc
        return Path(
            hf_hub_download(
                repo_id=metadata.hf_repo_id,
                filename=metadata.hf_filename,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
    if metadata.checkpoint_url:
        raise ModelNotAvailableError(
            "Direct URL checkpoint download is not enabled by default. "
            "Register the checkpoint on HuggingFace Hub or pass `checkpoint_path=...`."
        )
    return None


def _register_builtin_models() -> None:
    # These are adapter entries, not fabricated weights. They become loadable
    # when a local checkpoint or HuggingFace checkpoint metadata is supplied.
    register_model(
        "cmar_baseline",
        CMARAdapter,
        {
            "dataset": "fakeavceleb",
            "architecture": "CMAR feature-cache detector",
            "tags": ["cmar", "feature-space", "baseline"],
        },
        overwrite=True,
    )
    register_model(
        "certav_sigma025",
        CertAVAdapter,
        {
            "dataset": "fakeavceleb",
            "architecture": "CMAR + randomized smoothing",
            "training_sigma": 0.25,
            "noise_mode": "joint",
            "tags": ["certav", "certified", "feature-space"],
        },
        overwrite=True,
    )
    register_model(
        "certav_sigma100",
        CertAVAdapter,
        {
            "dataset": "fakeavceleb",
            "architecture": "CMAR + randomized smoothing",
            "training_sigma": 1.0,
            "noise_mode": "joint",
            "tags": ["certav", "certified", "feature-space"],
        },
        overwrite=True,
    )
    register_model(
        "cmar_pgd_at",
        CMARAdapter,
        {
            "dataset": "fakeavceleb",
            "architecture": "CMAR adversarially trained with feature-space PGD",
            "tags": ["cmar", "pgd-at", "feature-space"],
        },
        overwrite=True,
    )


_register_builtin_models()
