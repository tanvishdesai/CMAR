from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.config import TrainConfig, to_dict
from cmar.models import CMAR, CMARVisualOnly
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.training.trainer import fit
from cmar.utils.io import write_json
from cmar.utils.seed import seed_everything
from scripts.common import load_train_config


def train_variant(name: str, config: TrainConfig, device: torch.device):
    manifest_dir = Path(config.cache_dir) / "manifests"
    train_ds = CachedAVDataset(
        config.cache_dir,
        manifest_dir / "train.csv",
        split="train",
        return_degraded=config.use_consistency,
        feature_augmentation=config.feature_augmentation,
        visual_only=config.model.visual_only,
    )
    val_ds = CachedAVDataset(
        config.cache_dir,
        manifest_dir / "val.csv",
        split="val",
        visual_only=config.model.visual_only,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_av_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_av_batch,
    )
    model = CMARVisualOnly(config.model) if config.model.visual_only else CMAR(config.model)
    model.to(device)
    print(f"Training ablation: {name}")
    return fit(model, train_loader, val_loader, config, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMAR ablation variants")
    parser.add_argument("--base-config", default="configs/train_cmar.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default="/kaggle/working/cmar_runs/ablations")
    parser.add_argument("--variants", nargs="*", default=["visual_only", "no_consistency", "cmcm_1", "cmcm_4"])
    args = parser.parse_args()

    base = load_train_config(args.base_config)
    if args.cache_dir:
        base.cache_dir = args.cache_dir
    seed_everything(base.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for variant in args.variants:
        cfg = copy.deepcopy(base)
        cfg.output_dir = str(Path(args.output_root) / variant)
        if variant == "visual_only":
            cfg.model.visual_only = True
            cfg.use_consistency = False
            cfg.consistency_weight = 0.0
            cfg.feature_augmentation = False
        elif variant == "no_consistency":
            cfg.use_consistency = False
            cfg.consistency_weight = 0.0
            cfg.feature_augmentation = False
        elif variant == "cmcm_1":
            cfg.model.cmcm_layers = 1
        elif variant == "cmcm_4":
            cfg.model.cmcm_layers = 4
        else:
            raise ValueError(f"Unknown ablation variant: {variant}")
        results[variant] = train_variant(variant, cfg, device)
    write_json(results, Path(args.output_root) / "ablation_training_summary.json")
    print("Ablation summary:", results)


if __name__ == "__main__":
    main()
