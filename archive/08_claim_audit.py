from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.utils.io import write_json


def load_json_optional(path: str | Path | None) -> Optional[dict[str, Any]]:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[skip] JSON not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric(data: Optional[dict[str, Any]], *path: str) -> Optional[float]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return num(cur)


def fmt(value: Optional[float], digits: int = 4) -> str:
    return "missing" if value is None else f"{value:.{digits}f}"


def assess_results(
    cmar: Optional[dict[str, Any]],
    adversarial: Optional[dict[str, Any]],
    ablations: Optional[dict[str, Any]],
    full_best: Optional[dict[str, Any]],
) -> Dict[str, Any]:
    clean_auc = metric(cmar, "clean", "metrics", "auc")
    d12_auc = metric(cmar, "d12_social", "metrics", "auc")
    d12_rar = metric(cmar, "d12_social", "metrics", "rar")
    d11_rar = metric(cmar, "d11_h264_crf28", "metrics", "rar")
    cached_ensemble_gain = metric(cmar, "clean_cached_ensemble", "metrics", "gain_vs_clean_auc")
    old_ttda_auc = metric(cmar, "clean_ttda", "auc")

    adv_protocol = adversarial.get("protocol", {}) if isinstance(adversarial, dict) else {}
    adv_is_final = bool(adv_protocol.get("valid_for_final_adversarial_claim", False))
    a2_auc = metric(adversarial, "attacks", "a2_pgd_visual_eps010", "auc")
    a5_auc = metric(adversarial, "attacks", "a5_pgd_audio_eps010", "auc")
    a6_auc = metric(adversarial, "attacks", "a6_pgd_both_eps010", "auc")

    full_val_auc = metric(full_best, "val_auc")
    visual_val_auc = metric(ablations, "visual_only", "val_auc")
    no_consistency_val_auc = metric(ablations, "no_consistency", "val_auc")
    cmcm_1_val_auc = metric(ablations, "cmcm_1", "val_auc")
    cmcm_4_val_auc = metric(ablations, "cmcm_4", "val_auc")

    degradation_supported = bool(clean_auc is not None and clean_auc >= 0.88 and d12_rar is not None and d12_rar >= 0.82)
    strong_adv_supported = bool(adv_is_final and a2_auc is not None and a5_auc is not None and a6_auc is not None)
    proxy_contradicts_naive_robustness = bool(
        not adv_is_final
        and a2_auc is not None
        and a5_auc is not None
        and a6_auc is not None
        and (a2_auc < 0.60 or a5_auc < 0.60 or a6_auc < 0.50)
    )
    multimodal_helped_val = bool(
        full_val_auc is not None
        and visual_val_auc is not None
        and full_val_auc > visual_val_auc
    )
    consistency_not_supported_on_val = bool(
        full_val_auc is not None
        and no_consistency_val_auc is not None
        and no_consistency_val_auc >= full_val_auc
    )

    if degradation_supported and not strong_adv_supported:
        recommended_direction = "pivot_to_robustness_characterization"
        direction_summary = (
            "Keep CMAR, but soften the thesis: claim strong robustness to realistic "
            "degradations and use adversarial experiments as a rigorous characterization "
            "until input-space attacks prove cross-modal protection."
        )
    elif degradation_supported and strong_adv_supported:
        recommended_direction = "restore_cross_modal_adversarial_claim_if_baselines_confirm"
        direction_summary = (
            "The original claim may be viable, but it still needs baseline comparison "
            "and seed aggregation before it is paper-safe."
        )
    else:
        recommended_direction = "method_or_dataset_rethink"
        direction_summary = (
            "Current evidence is not enough for either clean/degraded robustness or "
            "adversarial robustness. Prioritize diagnostics before more training."
        )

    findings: List[dict[str, str]] = []
    if degradation_supported:
        findings.append(
            {
                "status": "supported",
                "finding": (
                    f"Clean AUC {fmt(clean_auc)} and D12 RAR {fmt(d12_rar)} meet "
                    "the current minimum thresholds."
                ),
            }
        )
    else:
        findings.append(
            {
                "status": "weak",
                "finding": (
                    f"Clean AUC {fmt(clean_auc)} and D12 RAR {fmt(d12_rar)} do not "
                    "jointly clear the current minimum thresholds."
                ),
            }
        )
    if proxy_contradicts_naive_robustness:
        findings.append(
            {
                "status": "warning",
                "finding": (
                    "Feature-space attacks collapse performance enough that the phrase "
                    "'inherent adversarial robustness' is currently too strong."
                ),
            }
        )
    if multimodal_helped_val:
        findings.append(
            {
                "status": "supported",
                "finding": (
                    f"Full CMAR validation AUC {fmt(full_val_auc)} beats visual-only "
                    f"{fmt(visual_val_auc)}, so multimodal fusion is useful."
                ),
            }
        )
    if consistency_not_supported_on_val:
        findings.append(
            {
                "status": "warning",
                "finding": (
                    f"No-consistency validation AUC {fmt(no_consistency_val_auc)} is "
                    f"not worse than full CMAR {fmt(full_val_auc)}; the consistency "
                    "loss needs test-condition evidence or should be demoted."
                ),
            }
        )
    if cached_ensemble_gain is not None and cached_ensemble_gain < 0:
        findings.append(
            {
                "status": "warning",
                "finding": (
                    f"Cached ensemble gain is negative ({fmt(cached_ensemble_gain)}), "
                    "so TTDA should not be a contribution yet."
                ),
            }
        )
    elif old_ttda_auc is not None and clean_auc is not None and old_ttda_auc < clean_auc:
        findings.append(
            {
                "status": "warning",
                "finding": (
                    f"Older clean TTDA AUC {fmt(old_ttda_auc)} is below clean AUC "
                    f"{fmt(clean_auc)}; treat TTDA as failed until rerun."
                ),
            }
        )

    next_steps = [
        "Rerun clean/degraded evaluation with strict cache coverage, category contrasts, and modality masking.",
        "Evaluate ablation checkpoints on clean, D12, and the same modality-masking probes, not validation AUC only.",
        "Run baseline score evaluation for any available baseline or a transparent unimodal cached-feature baseline.",
        "Implement a small input-space visual PGD audit first, with SSIM and batch=1, before scaling.",
        "Add an audio attack audit only after the visual input-space path is validated.",
        "Run at least three seeds for the final selected claim.",
    ]

    return {
        "verdict": {
            "recommended_direction": recommended_direction,
            "summary": direction_summary,
            "original_strong_claim_supported_now": bool(strong_adv_supported),
            "degradation_claim_supported_now": degradation_supported,
            "feature_proxy_warns_against_naive_adversarial_claim": proxy_contradicts_naive_robustness,
        },
        "evidence": {
            "clean_auc": clean_auc,
            "d12_auc": d12_auc,
            "d12_rar": d12_rar,
            "d11_rar": d11_rar,
            "feature_proxy_a2_visual_auc": a2_auc,
            "feature_proxy_a5_audio_auc": a5_auc,
            "feature_proxy_a6_both_auc": a6_auc,
            "full_val_auc": full_val_auc,
            "visual_only_val_auc": visual_val_auc,
            "no_consistency_val_auc": no_consistency_val_auc,
            "cmcm_1_val_auc": cmcm_1_val_auc,
            "cmcm_4_val_auc": cmcm_4_val_auc,
            "cached_ensemble_gain_vs_clean_auc": cached_ensemble_gain,
        },
        "findings": findings,
        "claim_options": {
            "do_not_claim_yet": (
                "Cross-modal fusion provides inherent adversarial robustness against "
                "white-box input-space attacks."
            ),
            "current_safe_claim": (
                "Cached DINOv2/Whisper feature fusion with cross-modal attention is "
                "competitive on clean FakeAVCeleb and robust to common compression, "
                "resize, noise, and social-media degradation pipelines."
            ),
            "conditional_future_claim": (
                "If raw input-space attacks show single-modality attacks degrade CMAR "
                "less than unimodal or late-fusion baselines, the paper can restore the "
                "cross-modal redundancy adversarial claim."
            ),
            "fallback_paper_direction": (
                "A rigorous robustness characterization of audio-visual deepfake "
                "detectors showing that multimodal fusion helps real-world degradation "
                "but does not automatically confer adversarial robustness."
            ),
        },
        "next_steps": next_steps,
    }


def write_markdown(report: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict = report["verdict"]
    evidence = report["evidence"]
    lines = [
        "# CMAR Claim Audit",
        "",
        f"**Recommended direction:** `{verdict['recommended_direction']}`",
        "",
        verdict["summary"],
        "",
        "## Key Evidence",
        "",
        f"- Clean AUC: {fmt(evidence['clean_auc'])}",
        f"- D12 social RAR: {fmt(evidence['d12_rar'])}",
        f"- D11 H.264 RAR: {fmt(evidence['d11_rar'])}",
        f"- Feature-proxy A2 visual AUC: {fmt(evidence['feature_proxy_a2_visual_auc'])}",
        f"- Feature-proxy A5 audio AUC: {fmt(evidence['feature_proxy_a5_audio_auc'])}",
        f"- Feature-proxy A6 both AUC: {fmt(evidence['feature_proxy_a6_both_auc'])}",
        f"- Full validation AUC: {fmt(evidence['full_val_auc'])}",
        f"- Visual-only validation AUC: {fmt(evidence['visual_only_val_auc'])}",
        f"- No-consistency validation AUC: {fmt(evidence['no_consistency_val_auc'])}",
        "",
        "## Findings",
        "",
    ]
    for item in report["findings"]:
        lines.append(f"- **{item['status']}**: {item['finding']}")
    lines.extend(["", "## Claim Options", ""])
    for key, value in report["claim_options"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Next Steps", ""])
    for step in report["next_steps"]:
        lines.append(f"- {step}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether current CMAR results support the paper claim")
    parser.add_argument("--cmar-results", default="/kaggle/working/cmar-results-clean-degraded.json")
    parser.add_argument("--adversarial-results", default="/kaggle/working/adversarial-results.json")
    parser.add_argument("--ablation-summary", default="/kaggle/working/cmar_runs/ablations/ablation_training_summary.json")
    parser.add_argument("--full-best-metrics", default="/kaggle/working/cmar_runs/full_final/best_metrics.json")
    parser.add_argument("--output-json", default="/kaggle/working/claim-audit.json")
    parser.add_argument("--output-md", default="/kaggle/working/claim-audit.md")
    args = parser.parse_args()

    report = assess_results(
        cmar=load_json_optional(args.cmar_results),
        adversarial=load_json_optional(args.adversarial_results),
        ablations=load_json_optional(args.ablation_summary),
        full_best=load_json_optional(args.full_best_metrics),
    )
    write_json(report, args.output_json)
    write_markdown(report, args.output_md)
    print("Wrote", args.output_json)
    print("Wrote", args.output_md)
    print("Recommended direction:", report["verdict"]["recommended_direction"])


if __name__ == "__main__":
    main()
