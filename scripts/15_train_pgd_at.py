#!/usr/bin/env python3
"""Train CMAR with PGD Adversarial Training (AT) baseline.

Adversarial training is the dominant empirical defense. This script
trains CMAR by generating PGD adversarial examples at each step and
training on a mix of clean and adversarial features. The resulting
model should be empirically robust but provide NO provable certificates.

Usage:
    python scripts/15_train_pgd_at.py \
        --cache-dir /kaggle/input/.../cmar_cache \
        --output-dir /kaggle/working/baseline_pgd_at \
        --at-eps 0.1 --at-steps 7 --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import ModelConfig, to_dict
from cmar.models.cmar import CMAR, count_parameters
from cmar.training.dataset import CachedAVDataset, cache_coverage_report, collate_av_batch
from cmar.evaluation.metrics import binary_metrics
from cmar.utils.seed import seed_everything
from cmar.utils.io import ensure_dir, write_json


def pgd_attack_batch(
    model: torch.nn.Module,
    visual: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_steps: int = 7,
    step_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate PGD adversarial features for a batch.

    Attacks both modalities simultaneously (L_inf threat model in feature space).
    """
    if step_size is None:
        step_size = 2.5 * eps / n_steps

    vis_adv = visual.clone().detach().requires_grad_(True)
    aud_adv = audio.clone().detach().requires_grad_(True)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(n_steps):
        out = model(vis_adv, aud_adv)
        loss = criterion(out["logits"].view(-1), labels.float())
        loss.backward()

        with torch.no_grad():
            if vis_adv.grad is not None:
                vis_adv = vis_adv + step_size * vis_adv.grad.sign()
                vis_adv = (visual + (vis_adv - visual).clamp(-eps, eps)).detach().requires_grad_(True)
            if aud_adv.grad is not None:
                aud_adv = aud_adv + step_size * aud_adv.grad.sign()
                aud_adv = (audio + (aud_adv - audio).clamp(-eps, eps)).detach().requires_grad_(True)

    return vis_adv.detach(), aud_adv.detach()


def train_one_epoch_at(
    model, loader, optimizer, criterion, device,
    at_eps, at_steps, at_alpha,
    grad_accum_steps=1, max_grad_norm=1.0,
):
    """One epoch of PGD adversarial training.

    For each batch:
    1. Generate PGD adversarial features
    2. Compute loss on adversarial features (50%) + clean features (50%)
    3. Update model
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader, start=1):
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        # Generate adversarial features
        model.eval()  # deterministic forward for attack
        vis_adv, aud_adv = pgd_attack_batch(
            model, visual, audio, labels,
            eps=at_eps, n_steps=at_steps, step_size=at_alpha,
        )
        model.train()

        # Forward on adversarial features
        out_adv = model(vis_adv, aud_adv)
        loss_adv = criterion(out_adv["logits"].view(-1), labels.float())

        # Forward on clean features
        out_clean = model(visual, audio)
        loss_clean = criterion(out_clean["logits"].view(-1), labels.float())

        # Combined loss (50/50 clean + adversarial)
        loss = (0.5 * loss_clean + 0.5 * loss_adv) / grad_accum_steps

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
    """Standard (clean) evaluation."""
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

    metrics = binary_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_logits).numpy(),
    )
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGD Adversarial Training for CMAR")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--at-eps", type=float, default=0.1,
                        help="L_inf perturbation budget for PGD during training")
    parser.add_argument("--at-steps", type=int, default=7,
                        help="Number of PGD steps")
    parser.add_argument("--at-alpha", type=float, default=None,
                        help="PGD step size (default: 2.5*eps/steps)")
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.at_alpha is None:
        args.at_alpha = 2.5 * args.at_eps / args.at_steps

    cache_dir = Path(args.cache_dir)
    manifest_dir = cache_dir / "manifests"

    for split in ["train", "val"]:
        report = cache_coverage_report(cache_dir, manifest_dir / f"{split}.csv", condition="clean")
        print(f"Cache coverage ({split}): {report['available_rows']}/{report['total_rows']}")

    output_dir = ensure_dir(args.output_dir)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMAR(model_config).to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"PGD-AT Training: eps={args.at_eps}, steps={args.at_steps}, alpha={args.at_alpha:.4f}")
    print(f"Device: {device}")

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
            train_metrics = train_one_epoch_at(
                model, train_loader, optimizer, criterion, device,
                at_eps=args.at_eps, at_steps=args.at_steps, at_alpha=args.at_alpha,
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
                    "at_eps": args.at_eps,
                    "at_steps": args.at_steps,
                }, output_dir / "best.pt")
            else:
                patience += 1

            print(f"epoch={epoch:03d} PGD-AT eps={args.at_eps} "
                  f"train_loss={row['train_loss']:.4f} "
                  f"val_auc={row['val_auc']:.4f} val_eer={row['val_eer']:.4f}")

            if patience >= args.patience:
                print(f"Early stopping after {patience} epochs without AUC improvement.")
                break

    write_json(best_metrics, output_dir / "best_metrics.json")
    write_json({"at_eps": args.at_eps, "at_steps": args.at_steps, "at_alpha": args.at_alpha, "seed": args.seed}, output_dir / "at_config.json")
    print(f"\n=== PGD-AT Training Complete ===")
    print(f"Best val AUC: {best_metrics.get('val_auc', 'N/A')}")
    print(f"Checkpoint: {output_dir}/best.pt")


if __name__ == "__main__":
    main()
