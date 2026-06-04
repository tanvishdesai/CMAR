#!/usr/bin/env python3
"""Calibrate CertAV conformal prediction thresholds on the validation split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.certification.core import certified_radius, lower_confidence_bound_exact
from cmar.certification.smoothing import SmoothedClassifier
from cmar.models.cmar import CMAR
from cmar.phase2.conformal import ConformalConfigKey, conformal_quantile, nonconformity_score
from cmar.phase2.model_loading import model_config_from_checkpoint
from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES
from cmar.training.dataset import CachedAVDataset
from cmar.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate conformal CertAV thresholds")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--noise-mode", choices=["joint", "visual_only", "audio_only", *ANISOTROPIC_NOISE_MODES.keys()], default="joint")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.25, 0.50, 1.00])
    parser.add_argument("--score-types", nargs="+", choices=["raw", "cp", "log"], default=["raw", "cp", "log"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--cp-alpha", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cmcm-layers", type=int, default=None)
    parser.add_argument("--visual-dim", type=int, default=None)
    parser.add_argument("--audio-dim", type=int, default=None)
    parser.add_argument("--pca-noise-path", type=str, default=None)
    parser.add_argument("--pca-top-k", type=int, default=None)
    parser.add_argument("--pca-off-sigma", type=float, default=1e-3)
    parser.add_argument("--no-pca-equalize-budget", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CMAR(model_config_from_checkpoint(ckpt, args)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pca_noise_path = args.pca_noise_path
    if args.noise_mode in ANISOTROPIC_NOISE_MODES and pca_noise_path is None:
        candidate = Path(args.checkpoint).resolve().parent / "pca_noise.pt"
        if candidate.exists():
            pca_noise_path = str(candidate)
        else:
            raise SystemExit(f"--noise-mode {args.noise_mode} requires --pca-noise-path")

    smoothed = SmoothedClassifier(
        model,
        sigma=args.sigma,
        device=device,
        noise_mode=args.noise_mode,
        pca_noise_path=pca_noise_path,
        pca_top_k=args.pca_top_k,
        pca_off_sigma=args.pca_off_sigma,
        pca_equalize_budget=not args.no_pca_equalize_budget,
    )

    cache_dir = Path(args.cache_dir)
    ds = CachedAVDataset(
        cache_dir,
        cache_dir / "manifests" / f"{args.split}.csv",
        condition="clean",
        return_degraded=False,
    )
    n_items = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    print(f"Calibrating on {n_items} samples from split={args.split}")

    samples: list[dict] = []
    score_buckets: dict[str, list[float]] = {}
    for score_type in args.score_types:
        for radius in args.radii:
            for alpha in args.alphas:
                score_buckets[ConformalConfigKey(alpha, radius, score_type).as_key()] = []

    for idx in range(n_items):
        item = ds[idx]
        label = int(item["label"].item())
        estimated = smoothed.estimate_probabilities(
            item["visual"],
            item["audio"],
            n_samples=args.n,
            batch_size=args.batch_size,
        )
        counts = estimated["counts"]
        p_true_lower = lower_confidence_bound_exact(int(counts[label]), args.n, args.cp_alpha)
        r_true = certified_radius(args.sigma, p_true_lower)
        sample_row = {
            "clip_id": item["clip_id"],
            "true_label": label,
            "counts": counts,
            "probabilities": estimated["probabilities"],
            "p_true_lower": p_true_lower,
            "true_class_radius": r_true,
        }
        samples.append(sample_row)

        for score_type in args.score_types:
            for radius in args.radii:
                score = nonconformity_score(
                    counts,
                    label,
                    score_type=score_type,
                    cp_alpha=args.cp_alpha,
                    robust_radius=radius,
                    sigma=args.sigma,
                )
                for alpha in args.alphas:
                    score_buckets[ConformalConfigKey(alpha, radius, score_type).as_key()].append(score)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{n_items}] calibrated samples")

    thresholds = []
    for score_type in args.score_types:
        for radius in args.radii:
            base_scores = score_buckets[ConformalConfigKey(args.alphas[0], radius, score_type).as_key()]
            for alpha in args.alphas:
                qhat = conformal_quantile(base_scores, alpha)
                thresholds.append(
                    {
                        "key": ConformalConfigKey(alpha, radius, score_type).as_key(),
                        "alpha": alpha,
                        "radius": radius,
                        "score_type": score_type,
                        "qhat": qhat,
                        "n_calibration": len(base_scores),
                    }
                )

    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "cache_dir": args.cache_dir,
            "split": args.split,
            "sigma": args.sigma,
            "noise_mode": args.noise_mode,
            "pca_noise_path": pca_noise_path,
            "n": args.n,
            "cp_alpha": args.cp_alpha,
            "alphas": args.alphas,
            "radii": args.radii,
            "score_types": args.score_types,
            "seed": args.seed,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "thresholds": thresholds,
        "calibration_samples": samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Calibration written to {output_path}")
    for row in thresholds:
        print(f"  {row['key']}: qhat={row['qhat']:.6f}")


if __name__ == "__main__":
    main()
