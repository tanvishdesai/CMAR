#!/usr/bin/env python3
"""Certify a trained CertAV model using randomized smoothing.

Runs the two-phase certification procedure (predict + certify) on the
FakeAVCeleb test set and outputs certified accuracy at multiple L2 radii.

Usage:
    python scripts/11_certify.py \
        --checkpoint /kaggle/working/certav/sigma_0.25/best.pt \
        --sigma 0.25 \
        --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
        --output certav_cert_0.25.json

    # Quick test with fewer samples
    python scripts/11_certify.py \
        --checkpoint ./test_run/best.pt --sigma 0.25 \
        --n0 50 --n 100 --max-samples 20 --output test_cert.json
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
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify CertAV model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise std dev (must match training sigma)")
    parser.add_argument("--noise-mode", type=str, default="joint",
                        choices=["joint", "visual_only", "audio_only"])
    parser.add_argument("--cache-dir", type=str,
                        default="/kaggle/input/cmar-features-clean-v1/cmar_cache")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for certification results")
    parser.add_argument("--n0", type=int, default=100,
                        help="Number of samples for prediction phase")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples for certification phase")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Significance level (0.001 = 99.9%% confidence)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for Monte Carlo sampling")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples to certify")
    parser.add_argument("--condition", type=str, default="clean",
                        help="Test condition (clean, d12_social, etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cmcm-layers", type=int, default=2)
    return parser.parse_args()


def load_test_data(
    cache_dir: Path,
    condition: str = "clean",
    max_samples: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
    """Load test features and labels from cache.

    Returns:
        visual_features, audio_features, labels, clip_ids
    """
    manifest_csv = cache_dir / "manifests" / "test.csv"
    ds = CachedAVDataset(
        cache_dir, manifest_csv,
        condition=condition,
        return_degraded=False,
    )

    visual_features = []
    audio_features = []
    labels = []
    clip_ids = []

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in range(n):
        item = ds[i]
        visual_features.append(item["visual"])
        audio_features.append(item["audio"])
        labels.append(int(item["label"].item()))
        clip_ids.append(item["clip_id"])

    return visual_features, audio_features, labels, clip_ids


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(cmcm_layers=args.cmcm_layers)
    model = CMAR(model_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Create smoothed classifier
    smoothed = SmoothedClassifier(
        model, sigma=args.sigma, device=device,
        noise_mode=args.noise_mode,
    )

    # Load test data
    cache_dir = Path(args.cache_dir)
    print(f"Loading test data (condition={args.condition})...")
    visual_features, audio_features, labels, clip_ids = load_test_data(
        cache_dir, condition=args.condition, max_samples=args.max_samples,
    )
    print(f"Loaded {len(labels)} test samples")

    # Run certification
    print(f"\nCertifying with σ={args.sigma}, n0={args.n0}, n={args.n}, "
          f"α={args.alpha}, noise_mode={args.noise_mode}")
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
    n_certified_correct = sum(1 for r in results if r.certified_correct)
    radii = [r.certified_radius for r in results if not r.abstained]

    # Certified accuracy curve
    curve = certified_accuracy_curve(results)

    # Key radii
    key_radii = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    cert_acc_at_radii = {
        f"r_{r:.2f}": certified_accuracy_at_radius(results, r)
        for r in key_radii
    }

    # Per-sample results
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

    # Summary
    output = {
        "config": {
            "sigma": args.sigma,
            "noise_mode": args.noise_mode,
            "n0": args.n0,
            "n": args.n,
            "alpha": args.alpha,
            "condition": args.condition,
            "checkpoint": args.checkpoint,
        },
        "summary": {
            "total_samples": n_total,
            "correct_predictions": n_correct,
            "abstained": n_abstained,
            "accuracy": n_correct / n_total if n_total else 0,
            "abstain_rate": n_abstained / n_total if n_total else 0,
            "certified_correct": n_certified_correct,
            "certified_accuracy_at_0": cert_acc_at_radii.get("r_0.00", 0),
            "mean_certified_radius": float(np.mean(radii)) if radii else 0.0,
            "median_certified_radius": float(np.median(radii)) if radii else 0.0,
            "max_certified_radius": float(np.max(radii)) if radii else 0.0,
            "elapsed_seconds": round(elapsed, 1),
        },
        "certified_accuracy_at_radii": cert_acc_at_radii,
        "certified_accuracy_curve": curve,
        "per_sample_results": per_sample,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CERTIFICATION RESULTS (σ={args.sigma}, mode={args.noise_mode})")
    print(f"{'='*60}")
    print(f"Total samples: {n_total}")
    print(f"Correct predictions: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"Abstained: {n_abstained}/{n_total} ({100*n_abstained/n_total:.1f}%)")
    print(f"Mean certified radius: {np.mean(radii):.4f}" if radii else "No certified samples")
    print(f"\nCertified accuracy at key radii:")
    for r_name, acc in cert_acc_at_radii.items():
        print(f"  {r_name}: {acc:.4f} ({100*acc:.1f}%)")
    print(f"\nResults saved to: {output_path}")
    print(f"Time elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
