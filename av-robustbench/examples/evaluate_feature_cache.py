from __future__ import annotations

import argparse
from pathlib import Path

from av_robustbench.datasets import FeatureCacheDataset
from av_robustbench.evaluate import benchmark, save_card_outputs
from av_robustbench.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CMAR-compatible feature-cache model.")
    parser.add_argument("--model", default="certav_sigma100")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("avrb_results"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model, checkpoint_path=args.checkpoint)
    dataset = FeatureCacheDataset(args.cache_dir, split="test", allow_partial_cache=True)
    card = benchmark(
        model,
        dataset,
        model_name=args.model,
        dataset_name="feature_cache",
        attacks=["pgd_linf", "pgd_l2"],
        certify=True,
        sigmas=[0.25, 1.0],
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        output_dir=args.output,
        certification_kwargs={"n0": 100, "n": 1000, "alpha": 0.001},
    )
    save_card_outputs(card, args.output)
    print(args.output / "robustness_card.json")


if __name__ == "__main__":
    main()

