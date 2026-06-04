#!/usr/bin/env python3
"""Manifold analysis: measure intrinsic dimensionality and noise alignment.

This script provides empirical evidence for WHY feature-space smoothing
on frozen foundation models avoids the accuracy-robustness tradeoff.

Measurements:
1. Intrinsic dimensionality of DINOv2/Whisper feature space (PCA)
2. Feature-space noise-manifold alignment: how much Gaussian noise stays
   within the data manifold vs being "wasted" in orthogonal directions
3. Clean prediction stability under noise: how σ-noise affects confidence

These numbers support the theoretical argument in the paper.

Usage:
    python scripts/18_manifold_analysis.py \
        --cache-dir /path/to/cmar_cache \
        --output /kaggle/working/manifold_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import ModelConfig
from cmar.models.cmar import CMAR
from cmar.training.dataset import CachedAVDataset
from cmar.utils.seed import seed_everything


def load_features(cache_dir: Path, split: str = "train", max_samples: int = 1000):
    """Load visual and audio features from cache."""
    manifest_csv = cache_dir / "manifests" / f"{split}.csv"
    ds = CachedAVDataset(cache_dir, manifest_csv, condition="clean", return_degraded=False)

    visual_feats = []
    audio_feats = []
    labels = []
    n = min(len(ds), max_samples)

    for i in range(n):
        item = ds[i]
        # Average over temporal dimension to get a single feature vector
        v = item["visual"].mean(dim=0)  # (D_v,)
        a = item["audio"].mean(dim=0)   # (D_a,)
        visual_feats.append(v.numpy())
        audio_feats.append(a.numpy())
        labels.append(int(item["label"].item()))

    return np.array(visual_feats), np.array(audio_feats), np.array(labels)


def measure_intrinsic_dimensionality(features: np.ndarray, thresholds=[0.90, 0.95, 0.99]):
    """Measure intrinsic dimensionality via PCA.

    Returns the number of principal components needed to explain
    X% of variance. Lower numbers = more structured = better for smoothing.
    """
    pca = PCA(n_components=min(features.shape[0], features.shape[1]))
    pca.fit(features)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    results = {
        "ambient_dim": features.shape[1],
        "n_samples": features.shape[0],
        "explained_variance_ratio": pca.explained_variance_ratio_[:20].tolist(),
        "cumulative_variance": cumvar[:50].tolist(),
    }

    for t in thresholds:
        n_components = int(np.searchsorted(cumvar, t) + 1)
        results[f"dim_at_{int(t*100)}pct"] = n_components

    return results


def measure_noise_manifold_alignment(features: np.ndarray, sigma: float, n_trials: int = 100):
    """Measure how Gaussian noise aligns with the data manifold.

    Key idea: If features lie on a low-dim manifold in 384-d space,
    Gaussian noise N(0,σ²I) will mostly have components ORTHOGONAL
    to the manifold (wasted noise). The fraction of noise that stays
    within the manifold is the "alignment ratio".

    High alignment → noise barely moves samples off-manifold → accuracy preserved.
    Low alignment → noise destroys manifold structure → accuracy drops.

    We measure alignment by projecting noisy features onto the top-K
    PCA components (the manifold approximation) and computing the
    fraction of noise energy that survives the projection.
    """
    pca = PCA(n_components=min(50, features.shape[1]))
    pca.fit(features)

    results = {}
    for k in [5, 10, 20, 50]:
        if k > pca.n_components_:
            continue

        # Project onto top-k components
        components = pca.components_[:k]  # (k, D)

        alignment_ratios = []
        for _ in range(n_trials):
            # Sample a random feature
            idx = np.random.randint(len(features))
            x = features[idx]

            # Add noise
            noise = np.random.randn(len(x)) * sigma
            x_noisy = x + noise

            # Project noise onto manifold
            noise_proj = components @ noise  # (k,)
            noise_energy = np.sum(noise ** 2)
            proj_energy = np.sum(noise_proj ** 2)

            alignment_ratios.append(proj_energy / noise_energy if noise_energy > 0 else 0)

        results[f"k{k}"] = {
            "mean_alignment": float(np.mean(alignment_ratios)),
            "std_alignment": float(np.std(alignment_ratios)),
            "expected_alignment": k / features.shape[1],  # Random baseline
        }

    return results


def measure_prediction_stability(
    features_v: np.ndarray,
    features_a: np.ndarray,
    labels: np.ndarray,
    model: CMAR,
    device: torch.device,
    sigmas: list[float] = [0.12, 0.25, 0.50, 1.00],
    n_noise_samples: int = 50,
    max_samples: int = 200,
):
    """Measure how stable predictions are under Gaussian noise.

    For each sigma, add noise N(0,σ²I) to features and check:
    - What fraction of predictions change?
    - How does confidence (sigmoid output) change?
    """
    model.eval()
    n = min(len(labels), max_samples)

    results = {}
    for sigma in sigmas:
        pred_changes = 0
        confidence_drops = []

        for i in range(n):
            v = torch.tensor(features_v[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D)
            a = torch.tensor(features_a[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D)

            # Clean prediction
            with torch.no_grad():
                out = model(v, a)
                clean_conf = torch.sigmoid(out["logits"].view(-1)).item()
                clean_pred = int(clean_conf >= 0.5)

            # Noisy predictions
            noisy_preds = []
            noisy_confs = []
            for _ in range(n_noise_samples):
                v_noisy = v + torch.randn_like(v) * sigma
                a_noisy = a + torch.randn_like(a) * sigma
                with torch.no_grad():
                    out = model(v_noisy, a_noisy)
                    conf = torch.sigmoid(out["logits"].view(-1)).item()
                    noisy_preds.append(int(conf >= 0.5))
                    noisy_confs.append(conf)

            # Majority vote
            majority_pred = 1 if sum(noisy_preds) > n_noise_samples // 2 else 0
            if majority_pred != clean_pred:
                pred_changes += 1

            # Confidence stability
            confidence_drops.append(abs(clean_conf - np.mean(noisy_confs)))

        results[f"sigma_{sigma:.2f}"] = {
            "prediction_flip_rate": pred_changes / n,
            "mean_confidence_change": float(np.mean(confidence_drops)),
            "std_confidence_change": float(np.std(confidence_drops)),
        }
        print(f"  σ={sigma:.2f}: flip_rate={pred_changes/n:.4f}, "
              f"mean_conf_change={np.mean(confidence_drops):.4f}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifold analysis for CertAV")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: CMAR checkpoint for prediction stability analysis")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cmcm-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    print("Loading features...")
    visual_feats, audio_feats, labels = load_features(
        cache_dir, split="train", max_samples=args.max_samples,
    )
    print(f"Loaded {len(labels)} samples")
    print(f"Visual features shape: {visual_feats.shape}")
    print(f"Audio features shape: {audio_feats.shape}")

    results = {"n_samples": len(labels)}

    # 1. Intrinsic dimensionality
    print("\n=== Intrinsic Dimensionality ===")
    print("  Visual features:")
    results["intrinsic_dim_visual"] = measure_intrinsic_dimensionality(visual_feats)
    for k, v in results["intrinsic_dim_visual"].items():
        if k.startswith("dim_at"):
            print(f"    {k}: {v} (out of {visual_feats.shape[1]})")

    print("  Audio features:")
    results["intrinsic_dim_audio"] = measure_intrinsic_dimensionality(audio_feats)
    for k, v in results["intrinsic_dim_audio"].items():
        if k.startswith("dim_at"):
            print(f"    {k}: {v} (out of {audio_feats.shape[1]})")

    # Joint features
    joint_feats = np.concatenate([visual_feats, audio_feats], axis=1)
    print("  Joint features:")
    results["intrinsic_dim_joint"] = measure_intrinsic_dimensionality(joint_feats)
    for k, v in results["intrinsic_dim_joint"].items():
        if k.startswith("dim_at"):
            print(f"    {k}: {v} (out of {joint_feats.shape[1]})")

    # 2. Noise-manifold alignment
    print("\n=== Noise-Manifold Alignment ===")
    for sigma in [0.25, 0.50, 1.00]:
        print(f"\n  σ = {sigma}:")
        print("  Visual:")
        align_v = measure_noise_manifold_alignment(visual_feats, sigma)
        results[f"alignment_visual_sigma_{sigma:.2f}"] = align_v
        for k, v in align_v.items():
            print(f"    {k}: actual={v['mean_alignment']:.4f}, "
                  f"random_baseline={v['expected_alignment']:.4f}, "
                  f"ratio={v['mean_alignment']/v['expected_alignment']:.2f}x")

        print("  Audio:")
        align_a = measure_noise_manifold_alignment(audio_feats, sigma)
        results[f"alignment_audio_sigma_{sigma:.2f}"] = align_a
        for k, v in align_a.items():
            print(f"    {k}: actual={v['mean_alignment']:.4f}, "
                  f"random_baseline={v['expected_alignment']:.4f}, "
                  f"ratio={v['mean_alignment']/v['expected_alignment']:.2f}x")

    # 3. Prediction stability (if checkpoint provided)
    if args.checkpoint:
        print("\n=== Prediction Stability ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = CMAR(ModelConfig(cmcm_layers=args.cmcm_layers))
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        results["prediction_stability"] = measure_prediction_stability(
            visual_feats, audio_feats, labels, model, device,
            max_samples=min(200, len(labels)),
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  MANIFOLD ANALYSIS COMPLETE")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print key takeaways
    v_dim95 = results["intrinsic_dim_visual"].get("dim_at_95pct", "?")
    a_dim95 = results["intrinsic_dim_audio"].get("dim_at_95pct", "?")
    j_dim95 = results["intrinsic_dim_joint"].get("dim_at_95pct", "?")
    print(f"\nKey findings:")
    print(f"  Visual intrinsic dim (95%): {v_dim95} / {visual_feats.shape[1]} ambient")
    print(f"  Audio intrinsic dim (95%):  {a_dim95} / {audio_feats.shape[1]} ambient")
    print(f"  Joint intrinsic dim (95%):  {j_dim95} / {joint_feats.shape[1]} ambient")
    print(f"\n  → If intrinsic dim << ambient dim, Gaussian noise mostly")
    print(f"    projects ORTHOGONAL to the manifold, explaining why")
    print(f"    σ=1.00 doesn't hurt accuracy.")


if __name__ == "__main__":
    main()
