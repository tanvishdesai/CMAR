from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cmar.config import TrainConfig, to_dict
from cmar.evaluation.metrics import binary_metrics
from cmar.models.cmar import count_parameters
from cmar.training.losses import CMARLoss
from cmar.utils.io import ensure_dir, write_json


def make_optimizer(model: torch.nn.Module, config: TrainConfig) -> AdamW:
    ln_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name.lower() or "layernorm" in name.lower():
            ln_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": config.lr})
    if ln_params:
        groups.append({"params": ln_params, "lr": config.ln_lr})
    return AdamW(groups, weight_decay=config.weight_decay)


def lr_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": to_dict(config),
            "parameter_count": count_parameters(model),
        },
        path,
    )


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CMARLoss,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "bce": 0.0, "consistency": 0.0}
    n_batches = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch = _move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch["visual"], batch["audio"])
            degraded_logits = None
            if "visual_degraded" in batch:
                degraded_out = model(batch["visual_degraded"], batch["audio_degraded"])
                degraded_logits = degraded_out["logits"]
            losses = criterion(out["logits"], batch["label"], degraded_logits)
            loss = losses["loss"] / grad_accum_steps
        if not torch.isfinite(loss):
            print(f"[warn] skipping non-finite train loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        out = model(batch["visual"], batch["audio"])
        logits = torch.nan_to_num(out["logits"].float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        total_loss += float(criterion(logits, batch["label"].float()).detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    metrics = binary_metrics(labels, logits)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    output_dir = ensure_dir(config.output_dir)
    write_json(to_dict(config), output_dir / "train_config.json")
    write_json(count_parameters(model), output_dir / "parameter_count.json")

    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_lambda(epoch, config.epochs, config.warmup_epochs),
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    criterion = CMARLoss(
        consistency_weight=config.consistency_weight,
        use_consistency=config.use_consistency,
    )

    log_path = output_dir / "training_log.csv"
    best_auc = -1.0
    best_metrics: Dict[str, float] = {}
    patience = 0
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_bce",
                "train_consistency",
                "val_loss",
                "val_auc",
                "val_eer",
                "val_ap",
            ],
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler if scaler is not None and scaler.is_enabled() else None,
                grad_accum_steps=config.grad_accum_steps,
                max_grad_norm=config.max_grad_norm,
                use_amp=use_amp,
            )
            val_metrics = evaluate_epoch(model, val_loader, device)
            if not math.isfinite(train_metrics["loss"]):
                print("[warn] non-finite train loss; stopping to preserve last best checkpoint.")
                break
            if not math.isfinite(val_metrics["auc"]):
                print("[warn] non-finite validation AUC; stopping to preserve last best checkpoint.")
                break
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_bce": train_metrics["bce"],
                "train_consistency": train_metrics["consistency"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_eer": val_metrics["eer"],
                "val_ap": val_metrics["ap"],
            }
            writer.writerow(row)
            f.flush()

            is_best = val_metrics["auc"] > best_auc
            if is_best:
                best_auc = val_metrics["auc"]
                best_metrics = row
                patience = 0
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, row, config)
            else:
                patience += 1
            if epoch % config.save_every == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, row, config)
            print(
                f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
                f"val_auc={row['val_auc']:.4f} val_eer={row['val_eer']:.4f}"
            )
            if patience >= config.early_stop_patience:
                print(f"Early stopping after {patience} epochs without AUC improvement.")
                break
    write_json(best_metrics, output_dir / "best_metrics.json")
    return best_metrics
