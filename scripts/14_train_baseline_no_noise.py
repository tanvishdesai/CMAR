#!/usr/bin/env python3
"""Train baseline CMAR without noise augmentation (sigma=0).

This trains the identical CMAR architecture used by CertAV but WITHOUT
any Gaussian noise injection during training. The resulting model is
the control condition: when wrapped in a SmoothedClassifier, it should
produce very small certified radii and/or high abstention rates, proving
that noise-augmented training is essential.

Usage:
    python scripts/14_train_baseline_no_noise.py \
        --cache-dir /kaggle/input/.../cmar_cache \
        --output-dir /kaggle/working/baseline_no_noise \
        --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import ModelConfig, TrainConfig, to_dict
from cmar.models.cmar import CMAR, count_parameters
from cmar.training.dataset import CachedAVDataset, cache_coverage_report, collate_av_batch
from cmar.evaluation.metrics import binary_metrics
from cmar.utils.seed import seed_everything
from cmar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMAR baseline (no noise)")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cmcm-layers", type=int, default=2)
    parser.add_argument("--visual-dim", type=int, default=None)
    parser.add_argument("--audio-dim", type=int, default=None)
    return parser.parse_args()


def train_one_epoch(
    model, loader, optimizer, criterion, device,
    grad_accum_steps=1, max_grad_norm=1.0,
):
    """Standard training epoch - NO noise injection."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader, start=1):
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        out = model(visual, audio)
        loss = criterion(out["logits"], labels) / grad_accum_steps

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach()) * grad_accum_steps
        n_batches += 1

    return {"loss": total_loss / max(1, n_batches)}


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    """Standard evaluation - NO noise."""
    model.eval()
    all_logits, all_labels = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in loader:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        out = model(visual, audio)
        logits = out["logits"].view(-1)
        total_loss += float(criterion(logits, labels.float()).detach())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    metrics = binary_metrics(labels_np, logits_np)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    manifest_dir = cache_dir / "manifests"

    # Check cache
    for split in ["train", "val"]:
        report = cache_coverage_report(cache_dir, manifest_dir / f"{split}.csv", condition="clean")
        print(f"Cache coverage ({split}): {report['available_rows']}/{report['total_rows']}")

    output_dir = ensure_dir(args.output_dir)

    # Datasets
    train_ds = CachedAVDataset(cache_dir, manifest_dir / "train.csv", condition="clean", return_degraded=False)
    val_ds = CachedAVDataset(cache_dir, manifest_dir / "val.csv", condition="clean", return_degraded=False)
    first_item = train_ds[0]
    model_config = ModelConfig(
        visual_dim=args.visual_dim or int(first_item["visual"].shape[-1]),
        audio_dim=args.audio_dim or int(first_item["audio"].shape[-1]),
        cmcm_layers=args.cmcm_layers,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_av_batch, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_av_batch, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMAR(model_config).to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"Training BASELINE (sigma=0.0, no noise), seed={args.seed}")
    print(f"Device: {device}")

    # Optimizer & scheduler
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_metrics = {}
    patience = 0

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "val_loss", "val_auc", "val_eer"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_accum_steps=args.grad_accum,
            )
            val_metrics = evaluate_epoch(model, val_loader, device)
            scheduler.step()

            row = {
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_eer": val_metrics["eer"],
            }
            writer.writerow(row)
            f.flush()

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_metrics = row.copy()
                patience = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": row,
                    "config": {"model": to_dict(model_config)},
                    "sigma": 0.0,
                    "noise_mode": "none",
                }, output_dir / "best.pt")
            else:
                patience += 1

            print(f"epoch={epoch:03d} sigma=0.00 mode=none "
                  f"train_loss={row['train_loss']:.4f} "
                  f"val_auc={row['val_auc']:.4f} val_eer={row['val_eer']:.4f}")

            if patience >= args.patience:
                print(f"Early stopping after {patience} epochs without AUC improvement.")
                break

    write_json(best_metrics, output_dir / "best_metrics.json")
    print(f"\n=== Baseline Training Complete ===")
    print(f"Best val AUC: {best_metrics.get('val_auc', 'N/A')}")
    print(f"Checkpoint: {output_dir}/best.pt")


if __name__ == "__main__":
    main()
