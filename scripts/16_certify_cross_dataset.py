#!/usr/bin/env python3
"""Certify CertAV models on LAV-DF (cross-dataset zero-shot evaluation).

Loads a CertAV model trained on FakeAVCeleb and certifies it on
LAV-DF test features WITHOUT any fine-tuning. Demonstrates the
generalization of certified robustness to a completely unseen dataset.

Prerequisites:
    - LAV-DF features must be pre-extracted into a cache directory
      using 01_preprocess_features.py with --lavdf-root
    - The cache must contain manifests/lavdf_test.csv

Usage:
    python scripts/16_certify_cross_dataset.py \
        --checkpoint /path/to/certav/sigma_1.00/best.pt \
        --sigma 1.00 \
        --lavdf-cache-dir /kaggle/input/.../lavdf_cache \
        --output /kaggle/working/cert_lavdf_1.00.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import ModelConfig
from cmar.models.cmar import CMAR
from cmar.certification.smoothing import SmoothedClassifier
from cmar.certification.core import (
    certified_accuracy_curve,
    certified_accuracy_at_radius,
)
from cmar.training.dataset import CachedAVDataset
from cmar.utils.seed import seed_everything


def load_lavdf_test_data(
    cache_dir: Path,
    max_samples: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
    """Load LAV-DF test features from cache.

    The manifest is expected at cache_dir/manifests/lavdf_test.csv.
    Falls back to test.csv if lavdf_test.csv doesn't exist (in case
    a dedicated LAV-DF cache was created with test.csv).
    """
    manifest_path = cache_dir / "manifests" / "lavdf_test.csv"
    if not manifest_path.exists():
        manifest_path = cache_dir / "manifests" / "test.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No LAV-DF manifest found. Looked for:\n"
            f"  {cache_dir / 'manifests' / 'lavdf_test.csv'}\n"
            f"  {cache_dir / 'manifests' / 'test.csv'}"
        )

    ds = CachedAVDataset(
        cache_dir, manifest_path,
        condition="clean",
        return_degraded=False,
        allow_partial_cache=True,  # LAV-DF cache may be incomplete
    )

    visual_features = []
    audio_features = []
    labels = []
    clip_ids = []

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in range(n):
        try:
            item = ds[i]
            visual_features.append(item["visual"])
            audio_features.append(item["audio"])
            labels.append(int(item["label"].item()))
            clip_ids.append(item["clip_id"])
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [warn] Skipping sample {i}: {e}")
            continue

    return visual_features, audio_features, labels, clip_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset certification on LAV-DF")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to CertAV model checkpoint (trained on FakeAVCeleb)")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--noise-mode", type=str, default="joint",
                        choices=["joint", "visual_only", "audio_only"])
    parser.add_argument("--lavdf-cache-dir", type=str, required=True,
                        help="Path to LAV-DF feature cache")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of LAV-DF samples to certify")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cmcm-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (trained on FakeAVCeleb)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(cmcm_layers=args.cmcm_layers)
    model = CMAR(model_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Training σ: {ckpt.get('sigma', 'unknown')}")

    # Create smoothed classifier
    smoothed = SmoothedClassifier(
        model, sigma=args.sigma, device=device,
        noise_mode=args.noise_mode,
    )

    # Load LAV-DF test data
    lavdf_cache = Path(args.lavdf_cache_dir)
    print(f"Loading LAV-DF test data from {lavdf_cache}...")
    visual_features, audio_features, labels, clip_ids = load_lavdf_test_data(
        lavdf_cache, max_samples=args.max_samples,
    )
    print(f"Loaded {len(labels)} LAV-DF test samples")
    print(f"  Real: {sum(1 for l in labels if l == 0)}, Fake: {sum(1 for l in labels if l == 1)}")

    if len(labels) == 0:
        print("[ERROR] No LAV-DF samples loaded. Check cache directory.")
        sys.exit(1)

    # Run certification
    print(f"\nCross-dataset certification: FakeAVCeleb-trained → LAV-DF test")
    print(f"σ={args.sigma}, n0={args.n0}, n={args.n}, α={args.alpha}")
    start_time = time.time()

    results = smoothed.certify_dataset(
        visual_features, audio_features, labels,
        n0=args.n0, n=args.n, alpha=args.alpha,
        batch_size=args.batch_size, verbose=True,
    )

    elapsed = time.time() - start_time

    # Compute metrics
    n_total = len(results)
    n_correct = sum(1 for r in results if r.correct and not r.abstained)
    n_abstained = sum(1 for r in results if r.abstained)
    radii = [r.certified_radius for r in results if not r.abstained]

    curve = certified_accuracy_curve(results)
    key_radii = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    cert_acc_at_radii = {
        f"r_{r:.2f}": certified_accuracy_at_radius(results, r)
        for r in key_radii
    }

    per_sample = []
    for i, r in enumerate(results):
        per_sample.append({
            "clip_id": clip_ids[i],
            "true_label": labels[i],
            "predicted_class": r.predicted_class,
            "correct": r.correct,
            "abstained": r.abstained,
            "certified_radius": round(r.certified_radius, 6),
            "pA_lower": round(r.pA_lower, 6),
        })

    output = {
        "config": {
            "sigma": args.sigma,
            "noise_mode": args.noise_mode,
            "n0": args.n0,
            "n": args.n,
            "alpha": args.alpha,
            "checkpoint": args.checkpoint,
            "dataset": "LAV-DF",
            "cross_dataset": True,
        },
        "summary": {
            "total_samples": n_total,
            "correct_predictions": n_correct,
            "abstained": n_abstained,
            "accuracy": n_correct / n_total if n_total else 0,
            "abstain_rate": n_abstained / n_total if n_total else 0,
            "mean_certified_radius": float(np.mean(radii)) if radii else 0.0,
            "median_certified_radius": float(np.median(radii)) if radii else 0.0,
            "elapsed_seconds": round(elapsed, 1),
        },
        "certified_accuracy_at_radii": cert_acc_at_radii,
        "certified_accuracy_curve": curve,
        "per_sample_results": per_sample,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET CERTIFICATION (LAV-DF)")
    print(f"σ={args.sigma}, mode={args.noise_mode}")
    print(f"{'='*60}")
    print(f"Total samples: {n_total}")
    print(f"Correct: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"Abstained: {n_abstained}/{n_total} ({100*n_abstained/n_total:.1f}%)")
    if radii:
        print(f"Mean certified radius: {np.mean(radii):.4f}")
    print(f"\nCertified accuracy at key radii:")
    for r_name, acc in cert_acc_at_radii.items():
        print(f"  {r_name}: {acc:.4f} ({100*acc:.1f}%)")
    print(f"\nResults saved to: {output_path}")
    print(f"Time elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
