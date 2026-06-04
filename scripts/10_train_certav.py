#!/usr/bin/env python3
"""Train CertAV: CMAR base classifier with Gaussian noise augmentation.

This script trains the CMAR architecture with Gaussian noise injected into
the cached feature space during training. The resulting model is designed
to be used inside a SmoothedClassifier for certified robustness.

Usage:
    # Train with joint noise sigma=0.25
    python scripts/10_train_certav.py \
        --sigma 0.25 \
        --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
        --output-dir /kaggle/working/certav/sigma_0.25

    # Train with visual-only noise
    python scripts/10_train_certav.py \
        --sigma 0.25 --noise-mode visual_only \
        --output-dir /kaggle/working/certav/visonly_0.25

    # Quick smoke test
    python scripts/10_train_certav.py --sigma 0.25 --epochs 2 --output-dir ./test_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import ModelConfig, TrainConfig
from cmar.models.cmar import CMAR, count_parameters
from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES, PCANoise
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.training.noise_augmented_trainer import fit_certav
from cmar.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CertAV noise-augmented model")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Gaussian noise std dev for training")
    parser.add_argument("--noise-mode", type=str, default="joint",
                        choices=["joint", "visual_only", "audio_only", *ANISOTROPIC_NOISE_MODES.keys()],
                        help="Which modalities receive noise")
    parser.add_argument("--pca-noise-path", type=str, default=None,
                        help="PCA artifact from scripts/20_fit_pca_noise.py for anisotropic modes")
    parser.add_argument("--pca-top-k", type=int, default=None,
                        help="Top-k PCA components for anisotropic strategies 2/3")
    parser.add_argument("--pca-off-sigma", type=float, default=1e-3,
                        help="Small off-subspace sigma for anisotropic strategies 2/3")
    parser.add_argument("--no-pca-equalize-budget", action="store_true",
                        help="Do not normalize anisotropic trace to D*sigma^2")
    parser.add_argument("--cache-dir", type=str,
                        default="/kaggle/input/cmar-features-clean-v1/cmar_cache",
                        help="Path to cached features")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cmcm-layers", type=int, default=2)
    parser.add_argument("--visual-dim", type=int, default=None,
                        help="Override cached visual feature dimension")
    parser.add_argument("--audio-dim", type=int, default=None,
                        help="Override cached audio feature dimension")
    parser.add_argument("--allow-partial-cache", action="store_true")
    parser.add_argument("--cache-report-only", action="store_true",
                        help="Only print cache coverage, don't train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    manifest_dir = cache_dir / "manifests"

    # Check cache coverage
    from cmar.training.dataset import cache_coverage_report
    for split in ["train", "val"]:
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[ERROR] Manifest not found: {csv_path}")
            sys.exit(1)
        report = cache_coverage_report(cache_dir, csv_path, condition="clean")
        print(f"Cache coverage ({split}): {report['available_rows']}/{report['total_rows']}")
        if not report["complete"] and not args.allow_partial_cache:
            print(f"[ERROR] Incomplete cache for {split}. Use --allow-partial-cache for smoke test.")
            sys.exit(1)

    if args.cache_report_only:
        print("Cache report complete. Exiting (--cache-report-only was set).")
        return

    # Create datasets
    train_ds = CachedAVDataset(
        cache_dir, manifest_dir / "train.csv",
        condition="clean",
        return_degraded=False,
        allow_partial_cache=args.allow_partial_cache,
    )

    first_item = train_ds[0]
    inferred_visual_dim = int(first_item["visual"].shape[-1])
    inferred_audio_dim = int(first_item["audio"].shape[-1])

    # Setup config after inspecting cache dimensions so encoder-family caches
    # can be trained without editing Python code.
    model_config = ModelConfig(
        visual_dim=args.visual_dim or inferred_visual_dim,
        audio_dim=args.audio_dim or inferred_audio_dim,
        cmcm_layers=args.cmcm_layers,
    )
    train_config = TrainConfig(
        cache_dir=str(cache_dir),
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=0.01,
        warmup_epochs=3,
        early_stop_patience=args.patience,
        consistency_weight=0.0,  # No consistency loss for CertAV
        use_consistency=False,
        feature_augmentation=False,  # Noise is handled by trainer
        amp=not args.no_amp,
        model=model_config,
    )
    val_ds = CachedAVDataset(
        cache_dir, manifest_dir / "val.csv",
        condition="clean",
        return_degraded=False,
        allow_partial_cache=args.allow_partial_cache,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_av_batch, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_av_batch, pin_memory=True,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMAR(model_config).to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"Training with sigma={args.sigma}, noise_mode={args.noise_mode}")
    print(f"Feature dims: visual={model_config.visual_dim}, audio={model_config.audio_dim}")
    print(f"Device: {device}")

    pca_noise = None
    if args.noise_mode in ANISOTROPIC_NOISE_MODES:
        if args.pca_noise_path is None:
            raise SystemExit(f"--noise-mode {args.noise_mode} requires --pca-noise-path")
        pca_noise = PCANoise.from_file(
            args.pca_noise_path,
            sigma=args.sigma,
            strategy=args.noise_mode,
            device=device,
            top_k=args.pca_top_k,
            off_sigma=args.pca_off_sigma,
            equalize_budget=not args.no_pca_equalize_budget,
        )
        print("PCA noise metadata:", pca_noise.metadata())

    # Train
    best_metrics = fit_certav(
        model, train_loader, val_loader, train_config, device,
        sigma=args.sigma,
        noise_mode=args.noise_mode,
        pca_noise=pca_noise,
        pca_noise_path=args.pca_noise_path,
    )

    print("\n=== Training Complete ===")
    print(f"Best val AUC: {best_metrics.get('val_auc', 'N/A')}")
    print(f"Best val EER: {best_metrics.get('val_eer', 'N/A')}")
    print(f"Checkpoint: {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
