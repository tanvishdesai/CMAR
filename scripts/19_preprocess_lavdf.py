#!/usr/bin/env python3
"""Preprocess LAV-DF test features (dedicated standalone script).

Unlike 01_preprocess_features.py (which is FakeAVCeleb-centric),
this script ONLY handles LAV-DF. It:
  1. Builds the lavdf_test.csv manifest from LAV-DF metadata.json
  2. Extracts DINOv2 + Whisper features for each test clip
  3. Saves them in the standard cache format

Usage on Kaggle:
    python scripts/19_preprocess_lavdf.py \
        --lavdf-root /kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF \
        --output-dir /kaggle/working/elevation_experiments/lavdf_cache \
        --max-samples 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmar.config import CacheConfig
from cmar.utils.cache import (
    ExtractionBudget,
    build_extractors_for_models,
    extract_clean_manifest,
    feature_path,
)
from cmar.utils.io import ensure_dir, write_json
from cmar.utils.manifest import build_lavdf_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LAV-DF test features")
    parser.add_argument("--lavdf-root", type=str, required=True,
                        help="Root of the LAV-DF dataset (contains metadata.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output cache directory for LAV-DF features")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max number of LAV-DF test samples to process")
    parser.add_argument("--max-runtime-seconds", type=int, default=5400,
                        help="Max total runtime in seconds (default: 90 min)")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--audio-max-seconds", type=float, default=10.0)
    parser.add_argument("--max-audio-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--visual-model-name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--audio-model-name", type=str, default="openai/whisper-tiny")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lavdf_root = Path(args.lavdf_root)
    cache_dir = Path(args.output_dir)
    manifest_dir = ensure_dir(cache_dir / "manifests")

    # Step 1: Build LAV-DF manifest
    print("=" * 60)
    print("  Step 1: Building LAV-DF test manifest")
    print("=" * 60)

    manifest = build_lavdf_manifest(lavdf_root, manifest_dir, split="test")
    print(f"  LAV-DF test samples discovered: {len(manifest)}")
    print(f"  Real: {(manifest['label']==0).sum()}, Fake: {(manifest['label']==1).sum()}")
    print(f"  Manifest saved to: {manifest_dir / 'lavdf_test.csv'}")

    # Also save as test.csv so 16_certify_cross_dataset.py can find it
    manifest.to_csv(manifest_dir / "test.csv", index=False)

    # Limit to max_samples
    if args.max_samples and len(manifest) > args.max_samples:
        manifest = manifest.head(args.max_samples).reset_index(drop=True)
        print(f"  Limited to {args.max_samples} samples")
        # Overwrite with limited manifest
        manifest.to_csv(manifest_dir / "lavdf_test.csv", index=False)
        manifest.to_csv(manifest_dir / "test.csv", index=False)

    # Check how many already exist
    existing = 0
    for _, row in manifest.iterrows():
        v = feature_path(cache_dir, "visual", "test", str(row["clip_id"]))
        a = feature_path(cache_dir, "audio", "test", str(row["clip_id"]))
        if v.exists() and a.exists():
            existing += 1
    print(f"  Already cached: {existing}/{len(manifest)}")
    if existing == len(manifest):
        print("  All features already extracted! Skipping extraction.")
        return

    # Step 2: Load encoders
    print("\n" + "=" * 60)
    print("  Step 2: Loading visual + audio encoders")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    visual_encoder, audio_encoder, whisper_processor = build_extractors_for_models(
        device,
        image_size=args.image_size,
        visual_model_name=args.visual_model_name,
        audio_model_name=args.audio_model_name,
    )
    print("  Encoders loaded")

    # Step 3: Extract features
    print("\n" + "=" * 60)
    print(f"  Step 3: Extracting features ({len(manifest) - existing} remaining)")
    print("=" * 60)

    config = CacheConfig(
        n_frames=args.n_frames,
        image_size=args.image_size,
        audio_sr=args.audio_sr,
        audio_max_seconds=args.audio_max_seconds,
        max_audio_tokens=args.max_audio_tokens,
        visual_model_name=args.visual_model_name,
        audio_model_name=args.audio_model_name,
        feature_dtype="float16",
        overwrite=False,
    )

    budget = ExtractionBudget(
        max_new_rows=args.max_samples,
        max_runtime_seconds=args.max_runtime_seconds,
    )

    # Use the split name "test" so features go to features/visual/test/ and
    # features/audio/test/ - matching what 16_certify_cross_dataset.py expects.
    timings = extract_clean_manifest(
        manifest=manifest,
        split="test",
        cache_dir=cache_dir,
        visual_encoder=visual_encoder,
        audio_encoder=audio_encoder,
        whisper_processor=whisper_processor,
        device=device,
        config=config,
        budget=budget,
        chunk_size=args.chunk_size,
    )

    # Step 4: Verify
    print("\n" + "=" * 60)
    print("  Step 4: Verification")
    print("=" * 60)

    final_count = 0
    missing_visual = 0
    missing_audio = 0
    for _, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        v = feature_path(cache_dir, "visual", "test", clip_id)
        a = feature_path(cache_dir, "audio", "test", clip_id)
        if v.exists() and a.exists():
            final_count += 1
        else:
            if not v.exists():
                missing_visual += 1
            if not a.exists():
                missing_audio += 1

    # Spot-check one feature shape
    sample_row = manifest.iloc[0]
    v_path = feature_path(cache_dir, "visual", "test", str(sample_row["clip_id"]))
    a_path = feature_path(cache_dir, "audio", "test", str(sample_row["clip_id"]))
    if v_path.exists() and a_path.exists():
        v_feat = torch.load(v_path, map_location="cpu")
        a_feat = torch.load(a_path, map_location="cpu")
        print(f"  Sample feature shapes:")
        print(f"    Visual: {list(v_feat.shape)} ({v_feat.dtype})")
        print(f"    Audio:  {list(a_feat.shape)} ({a_feat.dtype})")

    summary = {
        "total_manifest": len(manifest),
        "features_extracted": final_count,
        "missing_visual": missing_visual,
        "missing_audio": missing_audio,
        "extraction_timings": timings,
        "stopped_early": budget.stopped,
    }

    write_json(summary, cache_dir / "lavdf_preprocessing_summary.json")

    print(f"\n  Total in manifest: {len(manifest)}")
    print(f"  Successfully extracted: {final_count}/{len(manifest)}")
    if missing_visual:
        print(f"  Missing visual: {missing_visual}")
    if missing_audio:
        print(f"  Missing audio: {missing_audio}")
    if budget.stopped:
        print("  Stopped early (budget/time limit). Rerun to continue.")

    print(f"\n  Cache directory: {cache_dir}")
    print(f"  Manifest: {manifest_dir / 'lavdf_test.csv'}")
    print(f"\n{'='*60}")
    print(f"  LAV-DF PREPROCESSING {'COMPLETE' if final_count == len(manifest) else 'PARTIAL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
