from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.models import CMAR, CMARVisualOnly
from cmar.training.dataset import CachedAVDataset, cache_coverage_report, collate_av_batch
from cmar.training.trainer import fit
from cmar.utils.seed import seed_everything
from scripts.common import add_path_overrides, apply_train_overrides, load_train_config


def format_cache_report(name: str, report: dict[str, object]) -> str:
    lines = [
        f"{name}: {report['available_rows']}/{report['total_rows']} rows available "
        f"({report['missing_rows']} missing)",
    ]
    if report["missing_visual_examples"]:
        lines.append(f"  missing visual examples: {', '.join(report['missing_visual_examples'])}")
    if report["missing_audio_examples"]:
        lines.append(f"  missing audio examples: {', '.join(report['missing_audio_examples'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMAR from cached features")
    parser.add_argument("--config", default="configs/train_cmar.json")
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help=(
            "Train only on rows whose cached feature files exist. Use this only "
            "for smoke tests; real experiments should use a complete cache."
        ),
    )
    parser.add_argument(
        "--cache-report-only",
        action="store_true",
        help="Print train/val cache coverage and exit without training.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override config learning rate.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    add_path_overrides(parser)
    args = parser.parse_args()

    config = apply_train_overrides(load_train_config(args.config), args)
    if args.lr is not None:
        config.lr = args.lr
    if args.no_amp:
        config.amp = False
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_dir = Path(config.cache_dir) / "manifests"
    train_report = cache_coverage_report(
        config.cache_dir,
        manifest_dir / "train.csv",
        split="train",
        visual_only=config.model.visual_only,
    )
    val_report = cache_coverage_report(
        config.cache_dir,
        manifest_dir / "val.csv",
        split="val",
        visual_only=config.model.visual_only,
    )
    print(format_cache_report("train cache", train_report))
    print(format_cache_report("val cache", val_report))
    if args.cache_report_only:
        return
    if (not train_report["complete"] or not val_report["complete"]) and not args.allow_partial_cache:
        raise SystemExit(
            "\nCache is incomplete, so training was stopped before DataLoader workers started.\n"
            "This is expected if preprocessing was run with --max-new-rows slices.\n\n"
            "Resume clean preprocessing until train and val are complete, for example:\n"
            "python /kaggle/working/CMAR/scripts/01_preprocess_features.py \\\n"
            "  --config /kaggle/working/CMAR/configs/preprocess_fakeavceleb.json \\\n"
            "  --dataset-root /kaggle/input/datasets/shreyaty08/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 \\\n"
            "  --lavdf-root /kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF \\\n"
            "  --output-dir /kaggle/working/cmar_cache \\\n"
            "  --no-degraded --max-new-rows 200 --max-runtime-seconds 900 --chunk-size 50\n\n"
            "For a quick smoke test only, rerun training with --allow-partial-cache."
        )
    train_ds = CachedAVDataset(
        config.cache_dir,
        manifest_dir / "train.csv",
        split="train",
        return_degraded=config.use_consistency,
        feature_augmentation=config.feature_augmentation,
        visual_only=config.model.visual_only,
        allow_partial_cache=args.allow_partial_cache,
    )
    val_ds = CachedAVDataset(
        config.cache_dir,
        manifest_dir / "val.csv",
        split="val",
        visual_only=config.model.visual_only,
        allow_partial_cache=args.allow_partial_cache,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_av_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_av_batch,
    )
    model = CMARVisualOnly(config.model) if config.model.visual_only else CMAR(config.model)
    model.to(device)
    best = fit(model, train_loader, val_loader, config, device)
    print("Best metrics:", best)


if __name__ == "__main__":
    main()
