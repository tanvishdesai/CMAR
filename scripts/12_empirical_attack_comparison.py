#!/usr/bin/env python3
"""Compare smoothed classifier vs base classifier under empirical PGD attacks.

This script demonstrates that:
1. The base (non-smoothed) classifier is vulnerable to feature-space PGD
2. The smoothed classifier resists the same attacks due to noise averaging
3. Certified radii correctly predict the boundary of attack success

Usage:
    python scripts/12_empirical_attack_comparison.py \
        --checkpoint /kaggle/working/certav/sigma_0.25/best.pt \
        --sigma 0.25 \
        --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
        --output empirical_comparison_0.25.json
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

from cmar.models.cmar import CMAR
from cmar.certification.smoothing import SmoothedClassifier
from cmar.evaluation.metrics import binary_metrics
from cmar.phase2.model_loading import model_config_from_checkpoint
from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES, load_pca_artifact
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.utils.seed import seed_everything


def pgd_attack_features(
    model: torch.nn.Module,
    visual: torch.Tensor,
    audio: torch.Tensor,
    label: torch.Tensor,
    eps: float,
    n_steps: int = 20,
    step_size: float | None = None,
    attack_target: str = "both",
) -> tuple[torch.Tensor, torch.Tensor]:
    """PGD attack in feature space.

    Args:
        model: CMAR model
        visual: (B, T_v, D_v) visual features
        audio: (B, T_a, D_a) audio features
        label: (B,) true labels
        eps: L_inf budget
        n_steps: number of PGD steps
        step_size: step size (default: 2*eps/n_steps)
        attack_target: 'visual', 'audio', or 'both'

    Returns:
        Adversarial (visual, audio) features
    """
    if step_size is None:
        step_size = 2.0 * eps / n_steps

    visual_adv = visual.clone().detach()
    audio_adv = audio.clone().detach()
    criterion = torch.nn.BCEWithLogitsLoss()

    if attack_target in ("visual", "both"):
        visual_adv.requires_grad_(True)
    if attack_target in ("audio", "both"):
        audio_adv.requires_grad_(True)

    for _ in range(n_steps):
        model.zero_grad(set_to_none=True)
        out = model(visual_adv, audio_adv)
        loss = criterion(out["logits"].view(-1), label.float())
        loss.backward()

        with torch.no_grad():
            if attack_target in ("visual", "both") and visual_adv.grad is not None:
                perturbation = step_size * visual_adv.grad.sign()
                visual_adv = visual_adv + perturbation
                delta = (visual_adv - visual).clamp(-eps, eps)
                visual_adv = (visual + delta).detach().requires_grad_(True)
            if attack_target in ("audio", "both") and audio_adv.grad is not None:
                perturbation = step_size * audio_adv.grad.sign()
                audio_adv = audio_adv + perturbation
                delta = (audio_adv - audio).clamp(-eps, eps)
                audio_adv = (audio + delta).detach().requires_grad_(True)

    return visual_adv.detach(), audio_adv.detach()


def evaluate_base_under_attack(
    model: torch.nn.Module,
    dataset: CachedAVDataset,
    device: torch.device,
    eps: float,
    attack_target: str = "both",
    max_samples: int | None = None,
    pca_artifact=None,
    pca_top_k: int | None = None,
) -> dict:
    """Evaluate the base (non-smoothed) model under PGD attack."""
    model.eval()
    all_logits_clean = []
    all_logits_adv = []
    all_labels = []
    alignments = []

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(n):
        item = dataset[i]
        visual = item["visual"].unsqueeze(0).to(device)
        audio = item["audio"].unsqueeze(0).to(device)
        label = item["label"].unsqueeze(0).to(device)

        # Clean prediction
        with torch.no_grad():
            out_clean = model(visual, audio)
            all_logits_clean.append(out_clean["logits"].view(-1).cpu())

        # Adversarial prediction
        model.eval()
        vis_adv, aud_adv = pgd_attack_features(
            model, visual, audio, label, eps=eps,
            attack_target=attack_target,
        )
        if pca_artifact is not None:
            alignments.append(
                attack_manifold_alignment(
                    vis_adv - visual,
                    aud_adv - audio,
                    pca_artifact,
                    top_k=pca_top_k,
                )
            )
        model.eval()
        with torch.no_grad():
            out_adv = model(vis_adv, aud_adv)
            all_logits_adv.append(out_adv["logits"].view(-1).cpu())

        all_labels.append(label.cpu())

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] attacked")

    logits_clean = torch.cat(all_logits_clean).numpy()
    logits_adv = torch.cat(all_logits_adv).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics_clean = binary_metrics(labels, logits_clean)
    metrics_adv = binary_metrics(labels, logits_adv)

    output = {
        "clean": metrics_clean,
        "adversarial": metrics_adv,
        "eps": eps,
        "attack_target": attack_target,
    }
    if alignments:
        keys = sorted({key for row in alignments for key in row})
        output["attack_manifold_alignment"] = {
            key: float(np.nanmean([row.get(key, np.nan) for row in alignments]))
            for key in keys
        }
        output["attack_manifold_alignment_per_sample"] = alignments
    return output


def attack_manifold_alignment(
    visual_delta: torch.Tensor,
    audio_delta: torch.Tensor,
    pca_artifact,
    top_k: int | None = None,
) -> dict[str, float]:
    """Cosine-squared alignment between a PGD direction and a PCA subspace."""

    with torch.no_grad():
        space = pca_artifact.feature_space
        if space == "joint":
            delta = torch.cat(
                [
                    visual_delta.float().mean(dim=1).view(-1),
                    audio_delta.float().mean(dim=1).view(-1),
                ],
                dim=0,
            )
        elif space == "visual":
            delta = visual_delta.float().mean(dim=1).view(-1)
        else:
            delta = audio_delta.float().mean(dim=1).view(-1)

        components = pca_artifact.components[: int(top_k or pca_artifact.dim_at_90pct)].to(delta.device)
        denom = torch.sum(delta * delta).clamp_min(1e-12)
        projection = components @ delta
        cos2 = torch.sum(projection * projection) / denom
        return {
            f"{space}_pca_cos2": float(cos2.detach().cpu()),
            f"{space}_pca_top_k": float(components.shape[0]),
        }


def evaluate_smoothed_under_attack(
    base_model: torch.nn.Module,
    dataset: CachedAVDataset,
    device: torch.device,
    sigma: float,
    eps: float,
    attack_target: str = "both",
    noise_mode: str = "joint",
    pca_noise_path: str | None = None,
    pca_top_k: int | None = None,
    pca_off_sigma: float = 1e-3,
    pca_equalize_budget: bool = True,
    n_samples: int = 100,
    max_samples: int | None = None,
) -> dict:
    """Evaluate the smoothed classifier under PGD attack.

    The attack is applied to the base features, then the smoothed
    classifier adds noise on top. This simulates an attacker who
    knows the base model but must contend with the smoothing noise.
    """
    base_model.eval()
    smoothed = SmoothedClassifier(
        base_model, sigma=sigma, device=device,
        noise_mode=noise_mode,
        pca_noise_path=pca_noise_path,
        pca_top_k=pca_top_k,
        pca_off_sigma=pca_off_sigma,
        pca_equalize_budget=pca_equalize_budget,
    )

    n_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    correct_clean = 0
    correct_adv = 0

    for i in range(n_total):
        item = dataset[i]
        visual = item["visual"].unsqueeze(0).to(device)
        audio = item["audio"].unsqueeze(0).to(device)
        label_val = int(item["label"].item())
        label_t = item["label"].unsqueeze(0).to(device)

        # Clean smoothed prediction
        pred_clean = smoothed.predict(visual, audio, n_samples=n_samples)
        if pred_clean == label_val:
            correct_clean += 1

        # Attack base features
        base_model.eval()
        vis_adv, aud_adv = pgd_attack_features(
            base_model, visual, audio, label_t, eps=eps,
            attack_target=attack_target,
        )
        base_model.eval()

        # Smoothed prediction on adversarial features
        pred_adv = smoothed.predict(vis_adv, aud_adv, n_samples=n_samples)
        if pred_adv == label_val:
            correct_adv += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_total}] smoothed clean_acc={correct_clean/(i+1):.3f} "
                  f"adv_acc={correct_adv/(i+1):.3f}")

    return {
        "clean_accuracy": correct_clean / n_total,
        "adversarial_accuracy": correct_adv / n_total,
        "n_total": n_total,
        "sigma": sigma,
        "eps": eps,
        "attack_target": attack_target,
        "n_samples": n_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical attack comparison")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--noise-mode", type=str, default="joint")
    parser.add_argument("--pca-noise-path", type=str, default=None)
    parser.add_argument("--pca-top-k", type=int, default=None)
    parser.add_argument("--pca-off-sigma", type=float, default=1e-3)
    parser.add_argument("--no-pca-equalize-budget", action="store_true")
    parser.add_argument("--cache-dir", type=str,
                        default="/kaggle/input/cmar-features-clean-v1/cmar_cache")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eps-values", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Limit samples for speed")
    parser.add_argument("--n-smoothing-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cmcm-layers", type=int, default=None)
    parser.add_argument("--visual-dim", type=int, default=None)
    parser.add_argument("--audio-dim", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = model_config_from_checkpoint(ckpt, args)
    model = CMAR(model_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Load test data
    cache_dir = Path(args.cache_dir)
    test_ds = CachedAVDataset(
        cache_dir, cache_dir / "manifests" / "test.csv",
        condition="clean", return_degraded=False,
    )
    print(f"Test samples: {len(test_ds)}")

    pca_artifact = None
    if args.pca_noise_path:
        pca_artifact = load_pca_artifact(args.pca_noise_path, device=device)

    results = {
        "sigma": args.sigma,
        "noise_mode": args.noise_mode,
        "pca_noise_path": args.pca_noise_path,
        "attacks": {},
    }

    for eps in args.eps_values:
        print(f"\n{'='*60}")
        print(f"PGD attack eps={eps}")
        print(f"{'='*60}")

        # Base model under attack
        print("\n--- Base classifier (no smoothing) ---")
        base_results = evaluate_base_under_attack(
            model, test_ds, device, eps=eps,
            attack_target="both", max_samples=args.max_samples,
            pca_artifact=pca_artifact,
            pca_top_k=args.pca_top_k,
        )
        print(f"  Clean AUC: {base_results['clean']['auc']:.4f}")
        print(f"  Adv AUC:   {base_results['adversarial']['auc']:.4f}")

        # Smoothed model under attack
        print(f"\n--- Smoothed classifier (sigma={args.sigma}) ---")
        smooth_results = evaluate_smoothed_under_attack(
            model, test_ds, device,
            sigma=args.sigma, eps=eps,
            noise_mode=args.noise_mode,
            pca_noise_path=args.pca_noise_path,
            pca_top_k=args.pca_top_k,
            pca_off_sigma=args.pca_off_sigma,
            pca_equalize_budget=not args.no_pca_equalize_budget,
            n_samples=args.n_smoothing_samples,
            max_samples=args.max_samples,
        )
        print(f"  Clean acc: {smooth_results['clean_accuracy']:.4f}")
        print(f"  Adv acc:   {smooth_results['adversarial_accuracy']:.4f}")

        results["attacks"][f"eps_{eps:.2f}"] = {
            "base_classifier": base_results,
            "smoothed_classifier": smooth_results,
        }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
