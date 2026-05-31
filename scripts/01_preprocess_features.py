from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.config import DataSplitConfig
from cmar.evaluation.degradations import DEGRADATION_SPECS
from cmar.utils.cache import (
    ExtractionBudget,
    build_extractors,
    extract_clean_manifest,
    extract_degraded_test_manifest,
    verify_cache_shapes,
    write_cache_metadata,
)
from cmar.utils.io import ensure_dir
from cmar.utils.manifest import build_fakeavceleb_splits, build_lavdf_manifest
from scripts.common import load_cache_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CMAR reusable feature cache")
    parser.add_argument("--config", default="configs/preprocess_fakeavceleb.json")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--lavdf-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-degraded", action="store_true")
    parser.add_argument("--degraded-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--max-new-rows", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-audio-tokens", type=int, default=None)
    args = parser.parse_args()

    config = load_cache_config(args.config)
    if args.dataset_root:
        config.dataset_root = args.dataset_root
    if args.lavdf_root:
        config.lavdf_root = args.lavdf_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_degraded:
        config.extract_degraded_test = False
    if args.no_degraded and args.degraded_only:
        raise ValueError("Use either --no-degraded or --degraded-only, not both.")
    if args.overwrite:
        config.overwrite = True
    if args.max_audio_tokens is not None:
        config.max_audio_tokens = args.max_audio_tokens
    if not config.dataset_root and not args.degraded_only:
        raise ValueError("Set --dataset-root or dataset_root in the preprocess config.")

    cache_dir = ensure_dir(config.output_dir)
    manifest_dir = ensure_dir(cache_dir / "manifests")
    if args.degraded_only:
        required = {split: manifest_dir / f"{split}.csv" for split in ("train", "val", "test")}
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "--degraded-only requires existing clean manifests in the cache. "
                "Missing:\n" + "\n".join(missing)
            )
        splits = {split: pd.read_csv(path) for split, path in required.items()}
        print("[manifest] --degraded-only: using existing manifests from", manifest_dir)
    else:
        splits = build_fakeavceleb_splits(
            config.dataset_root,
            manifest_dir,
            split_config=DataSplitConfig(),
        )
        if config.lavdf_root and Path(config.lavdf_root).exists():
            build_lavdf_manifest(config.lavdf_root, manifest_dir, split="test")

    requested_conditions = args.conditions
    if args.degraded_only and requested_conditions is None:
        requested_conditions = list(DEGRADATION_SPECS.keys())
    need_visual = True
    need_audio = True
    if args.degraded_only:
        specs = [DEGRADATION_SPECS[name] for name in requested_conditions or []]
        need_visual = any(spec.visual for spec in specs)
        need_audio = any(spec.audio for spec in specs)
        print(f"[extractors] degraded-only need_visual={need_visual} need_audio={need_audio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder, audio_encoder, whisper_processor = build_extractors(
        device,
        image_size=config.image_size,
        load_visual=need_visual,
        load_audio=need_audio,
    )

    budget = ExtractionBudget(
        max_new_rows=args.max_new_rows,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    requested_splits = set() if args.degraded_only else set(args.splits or [])
    timings = {"clean": {}, "degraded": {}}
    for split, df in splits.items():
        if split not in requested_splits:
            continue
        if budget.stopped:
            break
        timings["clean"][split] = extract_clean_manifest(
            df,
            split,
            cache_dir,
            visual_encoder,
            audio_encoder,
            whisper_processor,
            device,
            config,
            budget=budget,
            chunk_size=args.chunk_size,
        )
        if budget.stopped:
            print(
                f"[stop] Preprocessing stopped cleanly after {budget.new_rows} new rows. "
                "Rerun the same command to resume."
            )
            break
    if config.extract_degraded_test and not budget.stopped:
        timings["degraded"] = extract_degraded_test_manifest(
            splits["test"],
            cache_dir,
            visual_encoder,
            audio_encoder,
            whisper_processor,
            device,
            config,
            conditions=requested_conditions,
            budget=budget,
            chunk_size=args.chunk_size,
        )
        if budget.stopped:
            print(
                f"[stop] Preprocessing stopped cleanly after {budget.new_rows} new rows. "
                "Rerun the same command to resume."
            )
    split_sizes = {split: len(df) for split, df in splits.items()}
    write_cache_metadata(cache_dir, config, split_sizes, timings)
    shape_report = None
    if not args.degraded_only:
        shape_report = verify_cache_shapes(cache_dir, manifest_dir, split="train")
    print("Cache written to:", cache_dir)
    print("Split sizes:", split_sizes)
    if shape_report is not None:
        print("Shape check:", shape_report)


if __name__ == "__main__":
    main()
