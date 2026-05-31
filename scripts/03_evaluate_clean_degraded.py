from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.config import DEGRADED_CONDITIONS
from cmar.evaluation.degradations import DEGRADATION_SPECS
from cmar.evaluation.checkpoint import model_from_checkpoint
from cmar.evaluation.evaluate import (
    category_operating_points,
    evaluate_category_contrasts,
    evaluate_model,
    predict_logits,
)
from cmar.evaluation.metrics import binary_metrics, delta_auc, robustness_accuracy_ratio, sigmoid_np
from cmar.training.dataset import CachedAVDataset, cache_coverage_report, collate_av_batch
from cmar.utils.cache import feature_path
from cmar.utils.io import write_json


@torch.no_grad()
def evaluate_cached_ensemble(
    model,
    cache_dir: Path,
    manifest_csv: Path,
    device: torch.device,
    batch_size: int,
    ensemble_conditions: List[str],
) -> Dict[str, object]:
    """Average cached clean/degraded predictions for an audit-only ensemble.

    This is not a true runtime TTDA implementation because the feature-cache path
    cannot generate fresh degradations of arbitrary test inputs. It is useful as
    a cached robustness probe and is deliberately named as such in the output.
    """

    score_sum = None
    labels = None
    for condition in ensemble_conditions:
        ds = CachedAVDataset(cache_dir, manifest_csv, split="test", condition=condition)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_av_batch)
        preds = predict_logits(model, loader, device)
        scores = sigmoid_np(np.asarray(preds["logits"], dtype=float))
        score_sum = scores if score_sum is None else score_sum + scores
        labels = preds["labels"]

    metrics = binary_metrics(labels, score_sum / len(ensemble_conditions), from_logits=False)
    return {
        "metrics": metrics,
        "ensemble_conditions": ensemble_conditions,
        "note": "Audit-only cached ensemble; not a final runtime TTDA result.",
    }


def condition_coverage_report(
    cache_dir: Path,
    manifest_csv: Path,
    condition: str,
    visual_only: bool = False,
) -> Dict[str, object]:
    manifest = pd.read_csv(manifest_csv)
    rows = len(manifest)
    spec = DEGRADATION_SPECS.get(condition)
    if condition == "clean" or spec is None:
        visual_condition = "clean"
        audio_condition = "clean"
    else:
        visual_condition = condition if spec.visual else "clean"
        audio_condition = "clean" if visual_only or not spec.audio else condition

    missing_visual: List[str] = []
    missing_audio: List[str] = []
    available = 0
    for _, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        v_path = feature_path(cache_dir, "visual", "test", clip_id, condition=visual_condition)
        a_path = feature_path(cache_dir, "audio", "test", clip_id, condition=audio_condition)
        has_visual = v_path.exists()
        has_audio = True if visual_only else a_path.exists()
        if has_visual and has_audio:
            available += 1
        else:
            if not has_visual and len(missing_visual) < 10:
                missing_visual.append(clip_id)
            if not has_audio and len(missing_audio) < 10:
                missing_audio.append(clip_id)
    return {
        "condition": condition,
        "visual_condition_required": visual_condition,
        "audio_condition_required": audio_condition,
        "total_rows": rows,
        "available_rows": available,
        "missing_rows": rows - available,
        "missing_visual_examples": missing_visual,
        "missing_audio_examples": missing_audio,
        "complete": available == rows,
    }


@torch.no_grad()
def evaluate_with_modality_mask(
    model,
    cache_dir: Path,
    manifest_csv: Path,
    device: torch.device,
    batch_size: int,
    condition: str,
    mask_visual: bool = False,
    mask_audio: bool = False,
) -> Dict[str, float]:
    ds = CachedAVDataset(cache_dir, manifest_csv, split="test", condition=condition)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_av_batch)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMAR on clean and degraded cached features")
    parser.add_argument("--cache-dir", default="/kaggle/input/cmar-features-v1/cmar_cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/kaggle/working/cmar-results-clean-degraded.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--include-ttda", action="store_true")
    parser.add_argument(
        "--include-cached-ensemble",
        action="store_true",
        help="Run the audit-only cached clean/degraded ensemble. --include-ttda is kept as an alias.",
    )
    parser.add_argument(
        "--include-modality-masking",
        action="store_true",
        help="Evaluate full CMAR with one modality zeroed out on clean and D12.",
    )
    parser.add_argument(
        "--allow-clean-fallback",
        action="store_true",
        help="Allow old behavior where a missing degraded feature silently falls back to clean.",
    )
    parser.add_argument(
        "--skip-lavdf",
        action="store_true",
        help="Skip optional LAV-DF evaluation even if manifests/lavdf_test.csv exists.",
    )
    parser.add_argument(
        "--strict-lavdf",
        action="store_true",
        help="Fail if manifests/lavdf_test.csv exists but LAV-DF cached features are incomplete.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    manifest_csv = cache_dir / "manifests" / "test.csv"
    model, _, _ = model_from_checkpoint(args.checkpoint, device)
    output_path = Path(args.output)

    results: Dict[str, object] = {
        "protocol": {
            "feature_cache_evaluation": True,
            "strict_degraded_cache": not args.allow_clean_fallback,
            "notes": [
                "Clean/degraded metrics use cached DINOv2 and Whisper features.",
                "Category AUC is reported as each fake category contrasted against RR.",
            ],
        },
        "coverage": {},
    }
    clean_auc = None
    for condition in ["clean"] + DEGRADED_CONDITIONS:
        coverage = condition_coverage_report(cache_dir, manifest_csv, condition)
        results["coverage"][condition] = coverage
        if not coverage["complete"] and not args.allow_clean_fallback:
            write_json(results, output_path)
            raise FileNotFoundError(
                f"Cache coverage incomplete for {condition}: "
                f"{coverage['available_rows']}/{coverage['total_rows']} rows. "
                "Rerun preprocessing for the missing condition or pass "
                "--allow-clean-fallback for a smoke test only."
            )
        ds = CachedAVDataset(cache_dir, manifest_csv, split="test", condition=condition)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_av_batch)
        result = evaluate_model(model, loader, device, n_bootstrap=args.bootstrap)
        metrics = result["metrics"]
        if condition == "clean":
            clean_auc = metrics["auc"]
            preds = result["predictions"]
            results["category_contrasts"] = evaluate_category_contrasts(
                preds["labels"],
                preds["logits"],
                preds["av_category"],
            )
            results["category_operating_points"] = category_operating_points(
                preds["labels"],
                preds["logits"],
                preds["av_category"],
            )
        else:
            metrics["rar"] = robustness_accuracy_ratio(metrics["auc"], clean_auc)
            metrics["delta_auc"] = delta_auc(clean_auc, metrics["auc"])
        results[condition] = {"metrics": metrics, "ci": result["ci"]}
        print(condition, metrics)
        write_json(results, output_path)

    if args.include_ttda or args.include_cached_ensemble:
        ensemble_conditions = ["clean", "d1_jpeg75", "d3_resize075", "d5_vnoise001", "d7_mp3_128k"]
        results["clean_cached_ensemble"] = evaluate_cached_ensemble(
            model,
            cache_dir,
            manifest_csv,
            device,
            args.batch_size,
            ensemble_conditions,
        )
        print("clean_cached_ensemble", results["clean_cached_ensemble"])
        clean_ensemble_auc = results["clean_cached_ensemble"]["metrics"]["auc"]
        results["clean_cached_ensemble"]["metrics"]["gain_vs_clean_auc"] = float(clean_ensemble_auc - clean_auc)
        if args.include_ttda:
            results["clean_ttda"] = {
                "deprecated_alias_for": "clean_cached_ensemble",
                "reason": "The cached-feature path does not implement true runtime TTDA.",
            }
        write_json(results, output_path)

    if args.include_modality_masking:
        masking = {}
        for condition in ["clean", "d12_social"]:
            masking[condition] = {
                "full": results[condition]["metrics"],
                "audio_only_probe_mask_visual": evaluate_with_modality_mask(
                    model, cache_dir, manifest_csv, device, args.batch_size, condition, mask_visual=True
                ),
                "visual_only_probe_mask_audio": evaluate_with_modality_mask(
                    model, cache_dir, manifest_csv, device, args.batch_size, condition, mask_audio=True
                ),
                "zero_both_sanity": evaluate_with_modality_mask(
                    model,
                    cache_dir,
                    manifest_csv,
                    device,
                    args.batch_size,
                    condition,
                    mask_visual=True,
                    mask_audio=True,
                ),
            }
        results["modality_masking"] = masking
        write_json(results, output_path)

    lavdf_csv = cache_dir / "manifests" / "lavdf_test.csv"
    if args.skip_lavdf:
        results["lavdf_clean"] = {"skipped": True, "reason": "--skip-lavdf was set"}
    elif lavdf_csv.exists():
        report = cache_coverage_report(cache_dir, lavdf_csv, split="test", condition="clean")
        if report["complete"]:
            ds = CachedAVDataset(cache_dir, lavdf_csv, split="test", condition="clean")
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_av_batch)
            results["lavdf_clean"] = evaluate_model(model, loader, device, n_bootstrap=args.bootstrap)["metrics"]
            print("lavdf_clean", results["lavdf_clean"])
        else:
            message = (
                "Skipping LAV-DF: lavdf_test.csv exists, but cached LAV-DF feature tensors "
                f"are incomplete ({report['available_rows']}/{report['total_rows']} rows)."
            )
            if args.strict_lavdf:
                raise FileNotFoundError(message)
            print(message)
            results["lavdf_clean"] = {"skipped": True, "reason": message, "coverage": report}

    write_json(results, output_path)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
