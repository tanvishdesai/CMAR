#!/usr/bin/env python3
"""Evaluate conformal CertAV prediction sets on test or cross-dataset splits."""

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
from cmar.phase2.conformal import prediction_set, summarize_prediction_sets
from cmar.phase2.model_loading import model_config_from_checkpoint
from cmar.phase2.pca_noise import ANISOTROPIC_NOISE_MODES
from cmar.training.dataset import CachedAVDataset
from cmar.utils.seed import seed_everything


def pgd_attack_features(
    model: torch.nn.Module,
    visual: torch.Tensor,
    audio: torch.Tensor,
    label: torch.Tensor,
    eps: float,
    n_steps: int = 20,
    step_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-projected PGD attack in feature space.

    Uses L2 norm to be consistent with the L2 certification radius.
    The old L∞ version created L2 perturbations of up to eps*sqrt(D),
    far exceeding the certified L2 ball.
    """
    if step_size is None:
        step_size = 2.5 * eps / max(1, n_steps)

    criterion = torch.nn.BCEWithLogitsLoss()
    visual_adv = visual.clone().detach().requires_grad_(True)
    audio_adv = audio.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        model.zero_grad(set_to_none=True)
        out = model(visual_adv, audio_adv)
        loss = criterion(out["logits"].view(-1), label.float())
        loss.backward()
        with torch.no_grad():
            # Collect gradients into a joint perturbation vector
            grad_v = visual_adv.grad if visual_adv.grad is not None else torch.zeros_like(visual_adv)
            grad_a = audio_adv.grad if audio_adv.grad is not None else torch.zeros_like(audio_adv)

            # L2-normalized gradient step (joint over both modalities)
            grad_norm = torch.sqrt(
                (grad_v ** 2).sum() + (grad_a ** 2).sum()
            ).clamp_min(1e-12)
            visual_adv = visual_adv + step_size * grad_v / grad_norm
            audio_adv = audio_adv + step_size * grad_a / grad_norm

            # Project back onto L2 ball of radius eps
            delta_v = visual_adv - visual
            delta_a = audio_adv - audio
            delta_norm = torch.sqrt(
                (delta_v ** 2).sum() + (delta_a ** 2).sum()
            ).clamp_min(1e-12)
            if delta_norm > eps:
                scale = eps / delta_norm
                delta_v = delta_v * scale
                delta_a = delta_a * scale
            visual_adv = (visual + delta_v).detach().requires_grad_(True)
            audio_adv = (audio + delta_a).detach().requires_grad_(True)

    return visual_adv.detach(), audio_adv.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conformal CertAV prediction sets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--noise-mode", choices=["joint", "visual_only", "audio_only", *ANISOTROPIC_NOISE_MODES.keys()], default="joint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--condition", type=str, default="clean")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--cp-alpha", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attack-eps-values", type=float, nargs="*", default=[])
    parser.add_argument("--attack-steps", type=int, default=20)
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

    with Path(args.calibration).open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    thresholds = calibration["thresholds"]
    cp_alpha = float(args.cp_alpha or calibration.get("config", {}).get("cp_alpha", 0.001))

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
        condition=args.condition,
        return_degraded=False,
    )
    n_items = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    variants: list[tuple[str, float | None]] = [("clean", None)]
    variants.extend([(f"pgd_eps_{eps:.2f}", eps) for eps in args.attack_eps_values])

    summaries: list[dict] = []
    per_sample_by_variant: dict[str, dict[str, list[dict]]] = {}

    for variant_name, eps in variants:
        print(f"\nEvaluating variant={variant_name} on {n_items} samples")
        rows_by_key = {row["key"]: [] for row in thresholds}

        for idx in range(n_items):
            item = ds[idx]
            label = int(item["label"].item())
            visual = item["visual"]
            audio = item["audio"]

            if eps is not None:
                model.eval()
                visual_adv, audio_adv = pgd_attack_features(
                    model,
                    visual.unsqueeze(0).to(device),
                    audio.unsqueeze(0).to(device),
                    item["label"].unsqueeze(0).to(device),
                    eps=eps,
                    n_steps=args.attack_steps,
                )
                model.eval()
                visual_eval = visual_adv.squeeze(0).cpu()
                audio_eval = audio_adv.squeeze(0).cpu()
            else:
                visual_eval = visual
                audio_eval = audio

            estimated = smoothed.estimate_probabilities(
                visual_eval,
                audio_eval,
                n_samples=args.n,
                batch_size=args.batch_size,
            )
            counts = estimated["counts"]
            top_class = int(torch.tensor(counts).argmax().item())
            p_top_lower = lower_confidence_bound_exact(int(counts[top_class]), args.n, cp_alpha)
            cert_radius = certified_radius(args.sigma, p_top_lower)
            certav_abstained = p_top_lower <= 0.5

            for threshold in thresholds:
                pred_set = prediction_set(
                    counts,
                    threshold["qhat"],
                    score_type=threshold["score_type"],
                    cp_alpha=cp_alpha,
                    robust_radius=threshold["radius"],
                    sigma=args.sigma,
                )
                rows_by_key[threshold["key"]].append(
                    {
                        "clip_id": item["clip_id"],
                        "true_label": label,
                        "prediction_set": pred_set,
                        "set_size": len(pred_set),
                        "covered": label in pred_set,
                        "counts": counts,
                        "probabilities": estimated["probabilities"],
                        "certified_radius": cert_radius,
                        "certav_abstained": certav_abstained,
                    }
                )

            if (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{n_items}] evaluated")

        per_sample_by_variant[variant_name] = rows_by_key
        for threshold in thresholds:
            summary = summarize_prediction_sets(rows_by_key[threshold["key"]], radius=threshold["radius"])
            summaries.append(
                {
                    **threshold,
                    "variant": variant_name,
                    "attack_eps": eps,
                    **summary,
                }
            )

    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "cache_dir": args.cache_dir,
            "calibration": args.calibration,
            "split": args.split,
            "condition": args.condition,
            "sigma": args.sigma,
            "noise_mode": args.noise_mode,
            "pca_noise_path": pca_noise_path,
            "n": args.n,
            "cp_alpha": cp_alpha,
            "attack_eps_values": args.attack_eps_values,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "summaries": summaries,
        "per_sample_results": per_sample_by_variant,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nConformal evaluation written to {output_path}")
    for row in summaries:
        print(
            f"  {row['variant']} {row['key']}: "
            f"coverage={row['coverage']:.3f}, singleton={row['singleton_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
