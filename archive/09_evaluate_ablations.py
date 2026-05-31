from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.evaluation.checkpoint import model_from_checkpoint
from cmar.evaluation.evaluate import evaluate_model
from cmar.evaluation.metrics import binary_metrics, delta_auc, robustness_accuracy_ratio
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.utils.io import write_json


@torch.no_grad()
def evaluate_masked(model, loader, device, mask_visual: bool = False, mask_audio: bool = False) -> Dict[str, float]:
    logits_all = []
    labels_all = []
    model.eval()
    for batch in loader:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        if mask_visual:
            visual = torch.zeros_like(visual)
        if mask_audio:
            audio = torch.zeros_like(audio)
        logits = model(visual, audio)["logits"]
        logits_all.append(logits.detach().cpu())
        labels_all.append(batch["label"])
    return binary_metrics(torch.cat(labels_all).numpy(), torch.cat(logits_all).numpy())


def evaluate_checkpoint(
    name: str,
    checkpoint: Path,
    cache_dir: Path,
    conditions: List[str],
    device: torch.device,
    batch_size: int,
    bootstrap: int,
    include_masking: bool,
) -> Dict[str, object]:
    model, config, _ = model_from_checkpoint(checkpoint, device)
    manifest_csv = cache_dir / "manifests" / "test.csv"
    results: Dict[str, object] = {
        "checkpoint": str(checkpoint),
        "visual_only": config.visual_only,
        "conditions": {},
    }
    clean_auc = None
    for condition in conditions:
        ds = CachedAVDataset(
            cache_dir,
            manifest_csv,
            split="test",
            condition=condition,
            visual_only=config.visual_only,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_av_batch)
        evaluated = evaluate_model(model, loader, device, n_bootstrap=bootstrap)
        metrics = evaluated["metrics"]
        if condition == "clean":
            clean_auc = metrics["auc"]
        elif clean_auc is not None:
            metrics["rar"] = robustness_accuracy_ratio(metrics["auc"], clean_auc)
            metrics["delta_auc"] = delta_auc(clean_auc, metrics["auc"])
        results["conditions"][condition] = {
            "metrics": metrics,
            "ci": evaluated.get("ci", {}),
        }
        if include_masking and not config.visual_only and condition in {"clean", "d12_social"}:
            results["conditions"][condition]["modality_masking"] = {
                "audio_only_probe_mask_visual": evaluate_masked(
                    model, loader, device, mask_visual=True
                ),
                "visual_only_probe_mask_audio": evaluate_masked(
                    model, loader, device, mask_audio=True
                ),
            }
        print(name, condition, metrics)
    return results


def write_csv_summary(results: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "condition", "auc", "eer", "ap", "rar", "delta_auc"],
        )
        writer.writeheader()
        for model_name, model_results in results["models"].items():
            for condition, payload in model_results["conditions"].items():
                metrics = payload["metrics"]
                writer.writerow(
                    {
                        "model": model_name,
                        "condition": condition,
                        "auc": metrics.get("auc"),
                        "eer": metrics.get("eer"),
                        "ap": metrics.get("ap"),
                        "rar": metrics.get("rar"),
                        "delta_auc": metrics.get("delta_auc"),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained CMAR ablations on test conditions")
    parser.add_argument("--cache-dir", default="/kaggle/input/cmar-features-clean-v1/cmar_cache")
    parser.add_argument("--full-checkpoint", default="/kaggle/working/cmar_runs/full_final/best.pt")
    parser.add_argument("--ablation-root", default="/kaggle/working/cmar_runs/ablations")
    parser.add_argument("--conditions", nargs="*", default=["clean", "d12_social", "d11_h264_crf28"])
    parser.add_argument("--output", default="/kaggle/working/ablation-results.json")
    parser.add_argument("--summary-csv", default="/kaggle/working/ablation-results.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--include-modality-masking", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    checkpoints: Dict[str, Path] = {}
    full_checkpoint = Path(args.full_checkpoint)
    if full_checkpoint.exists():
        checkpoints["full_cmar"] = full_checkpoint
    ablation_root = Path(args.ablation_root)
    for name in ["visual_only", "no_consistency", "cmcm_1", "cmcm_4"]:
        path = ablation_root / name / "best.pt"
        if path.exists():
            checkpoints[name] = path
        else:
            print(f"[skip] missing checkpoint for {name}: {path}")
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found for ablation evaluation.")

    results: Dict[str, object] = {
        "protocol": {
            "conditions": args.conditions,
            "note": "Ablation test-set evaluation; validation AUC alone should not drive paper claims.",
        },
        "models": {},
    }
    for name, checkpoint in checkpoints.items():
        results["models"][name] = evaluate_checkpoint(
            name,
            checkpoint,
            cache_dir,
            args.conditions,
            device,
            args.batch_size,
            args.bootstrap,
            args.include_modality_masking,
        )
        write_json(results, args.output)
        write_csv_summary(results, Path(args.summary_csv))
    print("Wrote", args.output)
    print("Wrote", args.summary_csv)


if __name__ == "__main__":
    main()
