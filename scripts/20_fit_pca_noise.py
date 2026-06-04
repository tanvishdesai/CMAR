#!/usr/bin/env python3
"""Fit PCA bases for Phase 2 anisotropic smoothing.

The PCA is fitted only on the training split. For the default joint feature
space, each clip contributes one vector:

    [mean_t visual_features ; mean_t audio_features]

The resulting artifact is consumed by anisotropic training and certification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.training.dataset import CachedAVDataset
from cmar.utils.seed import seed_everything


def pooled_feature_matrix(
    cache_dir: Path,
    split: str,
    feature_space: str,
    max_samples: int | None,
) -> tuple[np.ndarray, int, int]:
    ds = CachedAVDataset(
        cache_dir,
        cache_dir / "manifests" / f"{split}.csv",
        condition="clean",
        return_degraded=False,
    )
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    rows: list[np.ndarray] = []
    visual_dim = 0
    audio_dim = 0

    for idx in range(n):
        item = ds[idx]
        visual = item["visual"].float().mean(dim=0).numpy()
        audio = item["audio"].float().mean(dim=0).numpy()
        visual_dim = int(visual.shape[0])
        audio_dim = int(audio.shape[0])
        if feature_space == "joint":
            row = np.concatenate([visual, audio], axis=0)
        elif feature_space == "visual":
            row = visual
        elif feature_space == "audio":
            row = audio
        else:
            raise ValueError(f"Unsupported feature_space: {feature_space}")
        rows.append(row.astype(np.float64, copy=False))

    if not rows:
        raise RuntimeError("No training features were loaded for PCA.")
    return np.stack(rows, axis=0), visual_dim, audio_dim


def fit_covariance_pca(features: np.ndarray) -> dict[str, np.ndarray | int | float]:
    n_samples, dim = features.shape
    if n_samples < 2:
        raise ValueError("Need at least two samples to fit PCA.")

    mean = features.mean(axis=0)
    centered = features - mean
    cov = (centered.T @ centered) / max(1, n_samples - 1)
    cov = (cov + cov.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = eigenvectors[:, order].T

    total = float(eigenvalues.sum())
    if total <= 0:
        explained = np.zeros_like(eigenvalues)
    else:
        explained = eigenvalues / total
    cumulative = np.cumsum(explained)

    def dim_at(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "eigenvalues": eigenvalues.astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
        "cumulative_variance": cumulative.astype(np.float32),
        "dim_at_80pct": dim_at(0.80),
        "dim_at_90pct": dim_at(0.90),
        "dim_at_95pct": dim_at(0.95),
        "ambient_dim": dim,
        "n_samples": n_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA artifact for anisotropic CertAV")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--feature-space", choices=["joint", "visual", "audio"], default="joint")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    print(f"Loading pooled {args.feature_space} features from {cache_dir} split={args.split}")
    features, visual_dim, audio_dim = pooled_feature_matrix(
        cache_dir,
        split=args.split,
        feature_space=args.feature_space,
        max_samples=args.max_samples,
    )
    print(f"Feature matrix: {features.shape}")

    pca = fit_covariance_pca(features)
    artifact = {
        **pca,
        "feature_space": args.feature_space,
        "visual_dim": visual_dim,
        "audio_dim": audio_dim,
        "cache_dir": str(cache_dir),
        "split": args.split,
        "seed": args.seed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)

    summary = {
        "artifact": str(output_path),
        "feature_space": args.feature_space,
        "ambient_dim": int(pca["ambient_dim"]),
        "n_samples": int(pca["n_samples"]),
        "visual_dim": visual_dim,
        "audio_dim": audio_dim,
        "dim_at_80pct": int(pca["dim_at_80pct"]),
        "dim_at_90pct": int(pca["dim_at_90pct"]),
        "dim_at_95pct": int(pca["dim_at_95pct"]),
        "top20_explained_variance_ratio": pca["explained_variance_ratio"][:20].tolist(),
        "top50_cumulative_variance": pca["cumulative_variance"][:50].tolist(),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPCA artifact written:")
    print(f"  artifact: {output_path}")
    print(f"  summary:  {summary_path}")
    print(f"  dim@90%:  {summary['dim_at_90pct']} / {summary['ambient_dim']}")


if __name__ == "__main__":
    main()
