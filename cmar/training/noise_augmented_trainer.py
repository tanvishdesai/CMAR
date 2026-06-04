"""Gaussian Noise Augmented Trainer for CertAV.

Trains the base CMAR classifier with Gaussian noise injection into the
feature space, which is necessary for the smoothed classifier to produce
useful certified radii.

Key difference from the standard CMAR trainer:
- No consistency loss (just BCE)
- Gaussian noise added to features at each forward pass during training
- Supports per-modality noise control
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cmar.config import TrainConfig, to_dict
from cmar.evaluation.metrics import binary_metrics
from cmar.models.cmar import count_parameters
from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES, PCANoise
from cmar.utils.io import ensure_dir, write_json


def make_optimizer_certav(model: torch.nn.Module, config: TrainConfig) -> AdamW:
    """Create optimizer for CertAV training.

    Uses a single learning rate for all parameters since we don't have
    separate LN-tuning in the cached feature path.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def lr_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Cosine annealing with linear warmup."""
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
    sigma: float,
    noise_mode: str,
    pca_noise_metadata: dict | None = None,
) -> None:
    """Save checkpoint with smoothing metadata."""
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
            "sigma": sigma,
            "noise_mode": noise_mode,
            "pca_noise": pca_noise_metadata,
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


def add_gaussian_noise(
    visual: torch.Tensor,
    audio: torch.Tensor,
    sigma: float,
    noise_mode: str = "joint",
    pca_noise: PCANoise | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to features according to noise_mode.

    Args:
        visual: (B, T_v, D_v) visual features
        audio: (B, T_a, D_a) audio features
        sigma: noise standard deviation
        noise_mode: 'joint', 'visual_only', or 'audio_only'

    Returns:
        Tuple of (noisy_visual, noisy_audio)
    """
    if pca_noise is not None:
        return pca_noise.add_noise(visual, audio)
    if noise_mode in ("joint", "visual_only"):
        visual = visual + torch.randn_like(visual) * sigma
    if noise_mode in ("joint", "audio_only"):
        audio = audio + torch.randn_like(audio) * sigma
    return visual, audio


def train_one_epoch_certav(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    sigma: float,
    noise_mode: str = "joint",
    pca_noise: PCANoise | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train one epoch with Gaussian noise augmentation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch = _move_batch(batch, device)
        visual = batch["visual"]
        audio = batch["audio"]
        labels = batch["label"]

        # Add Gaussian noise
        visual_noisy, audio_noisy = add_gaussian_noise(
            visual, audio, sigma, noise_mode, pca_noise=pca_noise
        )

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(visual_noisy, audio_noisy)
            logits = out["logits"]
            loss = criterion(logits, labels) / grad_accum_steps

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

        total_loss += float(loss.detach().cpu()) * grad_accum_steps
        n_batches += 1

    return {"loss": total_loss / max(1, n_batches)}


@torch.no_grad()
def evaluate_epoch_certav(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sigma: float,
    noise_mode: str = "joint",
    pca_noise: PCANoise | None = None,
    n_noise_samples: int = 10,
) -> Dict[str, float]:
    """Evaluate with noise-averaged predictions.

    During validation, we average predictions over n_noise_samples noisy copies
    to approximate the smoothed classifier's behavior.
    """
    model.eval()
    all_avg_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        visual = batch["visual"]
        audio = batch["audio"]
        labels = batch["label"]
        bs = visual.shape[0]

        # Average predictions over multiple noisy samples
        prob_sum = torch.zeros(bs, device=device)
        for _ in range(n_noise_samples):
            vis_noisy, aud_noisy = add_gaussian_noise(
                visual, audio, sigma, noise_mode, pca_noise=pca_noise
            )
            out = model(vis_noisy, aud_noisy)
            logits = out["logits"].view(-1)
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            prob_sum += torch.sigmoid(logits)

        avg_probs = prob_sum / n_noise_samples
        # Compute loss using logits from last sample (for monitoring only)
        total_loss += float(criterion(logits, labels.float()).detach().cpu())
        all_avg_probs.append(avg_probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_avg_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # binary_metrics expects logits or scores; pass from_logits=False since these are probabilities
    metrics = binary_metrics(labels_np, probs, from_logits=False)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def fit_certav(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    sigma: float,
    noise_mode: str = "joint",
    pca_noise: PCANoise | None = None,
    pca_noise_path: str | Path | None = None,
) -> Dict[str, float]:
    """Full training loop for CertAV noise-augmented model.

    Args:
        model: CMAR base model
        train_loader: training data loader
        val_loader: validation data loader
        config: training configuration
        device: torch device
        sigma: Gaussian noise standard deviation for training
        noise_mode: 'joint', 'visual_only', or 'audio_only'

    Returns:
        Best validation metrics dict
    """
    output_dir = ensure_dir(config.output_dir)
    if noise_mode in ANISOTROPIC_NOISE_MODES and pca_noise is None:
        raise ValueError(f"{noise_mode} requires a PCANoise instance.")

    pca_noise_metadata = pca_noise.metadata() if pca_noise is not None else None
    if pca_noise_path is not None and pca_noise is not None:
        copied = output_dir / "pca_noise.pt"
        if Path(pca_noise_path).resolve() != copied.resolve():
            shutil.copy2(pca_noise_path, copied)
        pca_noise_metadata = {**(pca_noise_metadata or {}), "checkpoint_artifact": str(copied)}

    write_json(
        {
            **to_dict(config),
            "sigma": sigma,
            "noise_mode": noise_mode,
            "pca_noise": pca_noise_metadata,
        },
        output_dir / "train_config.json",
    )
    write_json(count_parameters(model), output_dir / "parameter_count.json")

    optimizer = make_optimizer_certav(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_lambda(epoch, config.epochs, config.warmup_epochs),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    log_path = output_dir / "training_log.csv"
    best_auc = -1.0
    best_metrics: Dict[str, float] = {}
    patience = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "lr", "train_loss",
                "val_loss", "val_auc", "val_eer", "val_ap",
            ],
        )
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch_certav(
                model, train_loader, optimizer, criterion, device,
                sigma=sigma,
                noise_mode=noise_mode,
                pca_noise=pca_noise,
                grad_accum_steps=config.grad_accum_steps,
                max_grad_norm=config.max_grad_norm,
                scaler=scaler if scaler is not None and scaler.is_enabled() else None,
                use_amp=use_amp,
            )
            val_metrics = evaluate_epoch_certav(
                model, val_loader, device,
                sigma=sigma,
                noise_mode=noise_mode,
                pca_noise=pca_noise,
                n_noise_samples=10,
            )

            if not math.isfinite(train_metrics["loss"]):
                print("[warn] non-finite train loss; stopping.")
                break
            if not math.isfinite(val_metrics.get("auc", float("nan"))):
                print("[warn] non-finite val AUC; stopping.")
                break

            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
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
                best_metrics = row.copy()
                patience = 0
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer,
                    epoch, row, config, sigma, noise_mode,
                    pca_noise_metadata=pca_noise_metadata,
                )
            else:
                patience += 1

            if epoch % config.save_every == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                    epoch, row, config, sigma, noise_mode,
                    pca_noise_metadata=pca_noise_metadata,
                )

            print(
                f"epoch={epoch:03d} sigma={sigma:.2f} mode={noise_mode} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_auc={row['val_auc']:.4f} val_eer={row['val_eer']:.4f}"
            )

            if patience >= config.early_stop_patience:
                print(f"Early stopping after {patience} epochs without AUC improvement.")
                break

    write_json(best_metrics, output_dir / "best_metrics.json")
    return best_metrics
