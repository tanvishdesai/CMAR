from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.evaluation.attacks import FeatureAttackConfig, FeatureFGSMAttacker, FeaturePGDAttacker
from cmar.evaluation.checkpoint import model_from_checkpoint
from cmar.evaluation.metrics import binary_metrics, cmrr, robustness_accuracy_ratio
from cmar.training.dataset import CachedAVDataset, collate_av_batch
from cmar.utils.io import write_json


ATTACKS = {
    "a1_pgd_visual_eps005": FeatureAttackConfig(target="visual", eps=0.05, steps=20),
    "a2_pgd_visual_eps010": FeatureAttackConfig(target="visual", eps=0.10, steps=20),
    "a3_pgd_visual_eps020": FeatureAttackConfig(target="visual", eps=0.20, steps=20),
    "a4_pgd_audio_eps005": FeatureAttackConfig(target="audio", eps=0.05, steps=20),
    "a5_pgd_audio_eps010": FeatureAttackConfig(target="audio", eps=0.10, steps=20),
    "a6_pgd_both_eps010": FeatureAttackConfig(target="both", eps=0.10, steps=20),
}


def evaluate_attack(model, loader, device, attacker) -> Dict[str, float]:
    logits_all = []
    labels_all = []
    model.eval()
    for batch in tqdm(loader, desc="attack", leave=False):
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        adv = attacker.attack(visual, audio, labels)
        with torch.no_grad():
            logits = model(adv["visual"], adv["audio"])["logits"]
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return binary_metrics(torch.cat(labels_all).numpy(), torch.cat(logits_all).numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-space adversarial evaluation for cached CMAR")
    parser.add_argument("--cache-dir", default="/kaggle/input/cmar-features-v1/cmar_cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/kaggle/working/adversarial-results.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = model_from_checkpoint(args.checkpoint, device)
    ds = CachedAVDataset(Path(args.cache_dir), Path(args.cache_dir) / "manifests" / "test.csv", split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_av_batch)

    clean_logits = []
    clean_labels = []
    for batch in tqdm(loader, desc="clean", leave=False):
        with torch.no_grad():
            logits = model(batch["visual"].to(device), batch["audio"].to(device))["logits"]
        clean_logits.append(logits.cpu())
        clean_labels.append(batch["label"])
    clean_metrics = binary_metrics(torch.cat(clean_labels).numpy(), torch.cat(clean_logits).numpy())
    results: Dict[str, object] = {
        "protocol": {
            "attack_space": "cached_feature",
            "valid_for_final_adversarial_claim": False,
            "warning": (
                "These attacks perturb cached DINOv2/Whisper feature tensors. "
                "They are useful stress tests, but they are not input-space "
                "PGD/FGSM attacks on video frames or audio waveforms."
            ),
            "next_required_protocol": (
                "Run raw visual PGD through DINOv2 with SSIM checks and raw "
                "audio/log-mel PGD with SNR/PESQ checks before making final "
                "adversarial robustness claims."
            ),
        },
        "clean": clean_metrics,
        "attacks": {},
    }
    print("clean", clean_metrics)

    for name, config in ATTACKS.items():
        attacker = FeaturePGDAttacker(model, config)
        metrics = evaluate_attack(model, loader, device, attacker)
        metrics["rar_vs_clean"] = robustness_accuracy_ratio(metrics["auc"], clean_metrics["auc"])
        metrics["attack_target"] = config.target
        metrics["feature_eps"] = config.eps
        metrics["steps"] = config.steps
        results["attacks"][name] = metrics
        print(name, metrics)

    for name, target in {"a7_fgsm_visual_eps010": "visual", "a8_fgsm_audio_eps010": "audio"}.items():
        attacker = FeatureFGSMAttacker(model, target=target, eps=0.10)
        metrics = evaluate_attack(model, loader, device, attacker)
        metrics["rar_vs_clean"] = robustness_accuracy_ratio(metrics["auc"], clean_metrics["auc"])
        metrics["attack_target"] = target
        metrics["feature_eps"] = 0.10
        metrics["steps"] = 1
        results["attacks"][name] = metrics
        print(name, metrics)

    both_auc = results["attacks"]["a6_pgd_both_eps010"]["auc"]
    if both_auc <= 1e-12:
        results["cmrr_feature_proxy"] = {
            "value": None,
            "valid": False,
            "reason": "Both-modality feature-space attack collapsed AUC to zero, so CMRR is undefined.",
        }
    else:
        results["cmrr_feature_proxy"] = {
            "value": cmrr(
                clean_metrics["auc"],
                results["attacks"]["a2_pgd_visual_eps010"]["auc"],
                results["attacks"]["a5_pgd_audio_eps010"]["auc"],
                results["attacks"]["a6_pgd_both_eps010"]["auc"],
            ),
            "valid": False,
            "reason": "Feature-space CMRR is a proxy diagnostic, not final input-space CMRR.",
        }
    write_json(results, args.output)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
