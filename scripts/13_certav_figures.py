#!/usr/bin/env python3
"""Generate all paper figures for the CertAV paper.

Figures:
  Fig 1: Certified accuracy curves at different σ values (main result)
  Fig 2: Multimodal vs unimodal certification comparison
  Fig 3: Accuracy-radius tradeoff (clean acc vs mean certified radius)
  Fig 4: Empirical attack: base vs smoothed classifier
  Fig 5: Degradation robustness with/without smoothing

Usage:
    python scripts/13_certav_figures.py \
        --results-dir /kaggle/working/certav/ \
        --output-dir /kaggle/working/certav/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fig1_certified_accuracy_curves(results_dir: Path, output_dir: Path) -> None:
    """Fig 1: Certified accuracy vs L2 radius for different σ."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    sigmas = [0.12, 0.25, 0.50, 1.00]

    for i, sigma in enumerate(sigmas):
        cert_file = results_dir / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            cert_file = results_dir / f"sigma_{sigma:.2f}" / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            print(f"  [skip] {cert_file} not found")
            continue

        data = load_json(cert_file)
        curve = data.get("certified_accuracy_curve", {})
        radii = curve.get("radii", [])
        accs = curve.get("certified_accuracy", [])

        if radii and accs:
            ax.plot(radii, accs, color=colors[i % len(colors)],
                    linewidth=2, label=f"σ = {sigma}")

    ax.set_xlabel("Certified L₂ Radius", fontsize=13)
    ax.set_ylabel("Certified Accuracy", fontsize=13)
    ax.set_title("CertAV: Certified Accuracy vs Robustness Radius", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_certified_accuracy_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig1_certified_accuracy_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_certified_accuracy_curves")


def fig2_multimodal_vs_unimodal(results_dir: Path, output_dir: Path) -> None:
    """Fig 2: Joint vs visual-only vs audio-only certification."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sigma = 0.25
    configs = [
        ("joint", "CertAV-Joint", "#2196F3", "-"),
        ("visual_only", "CertAV-VisOnly", "#4CAF50", "--"),
        ("audio_only", "CertAV-AudOnly", "#FF9800", "-."),
    ]

    for mode, label, color, ls in configs:
        # Try multiple path patterns
        candidates = [
            results_dir / f"certav_cert_{sigma:.2f}_{mode}.json",
            results_dir / f"{mode}_{sigma:.2f}" / f"certav_cert_{sigma:.2f}.json",
            results_dir / f"sigma_{sigma:.2f}" / f"certav_cert_{sigma:.2f}_{mode}.json",
        ]
        if mode == "joint":
            candidates.insert(0, results_dir / f"certav_cert_{sigma:.2f}.json")
            candidates.insert(0, results_dir / f"sigma_{sigma:.2f}" / f"certav_cert_{sigma:.2f}.json")

        data = None
        for cand in candidates:
            if cand.exists():
                data = load_json(cand)
                break

        if data is None:
            print(f"  [skip] No cert results found for {mode}")
            continue

        curve = data.get("certified_accuracy_curve", {})
        radii = curve.get("radii", [])
        accs = curve.get("certified_accuracy", [])
        if radii and accs:
            ax.plot(radii, accs, color=color, linestyle=ls,
                    linewidth=2, label=label)

    ax.set_xlabel("Certified L₂ Radius", fontsize=13)
    ax.set_ylabel("Certified Accuracy", fontsize=13)
    ax.set_title(f"Multimodal vs Unimodal Certification (σ={sigma})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_multimodal_vs_unimodal.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig2_multimodal_vs_unimodal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_multimodal_vs_unimodal")


def fig3_accuracy_radius_tradeoff(results_dir: Path, output_dir: Path) -> None:
    """Fig 3: Clean accuracy vs mean certified radius across σ."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sigmas = [0.12, 0.25, 0.50, 1.00]
    clean_accs = []
    mean_radii = []
    labels_plot = []

    for sigma in sigmas:
        cert_file = results_dir / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            cert_file = results_dir / f"sigma_{sigma:.2f}" / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            continue

        data = load_json(cert_file)
        summary = data.get("summary", {})
        clean_accs.append(summary.get("accuracy", 0))
        mean_radii.append(summary.get("mean_certified_radius", 0))
        labels_plot.append(f"σ={sigma}")

    if clean_accs:
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
        for i, (acc, rad, lab) in enumerate(zip(clean_accs, mean_radii, labels_plot)):
            ax.scatter(rad, acc, s=150, color=colors[i % len(colors)],
                       zorder=5, edgecolor="black", linewidth=1.5)
            ax.annotate(lab, (rad, acc), textcoords="offset points",
                        xytext=(10, 5), fontsize=11)

        ax.plot(mean_radii, clean_accs, "--", color="gray", alpha=0.5, zorder=1)

    ax.set_xlabel("Mean Certified L₂ Radius", fontsize=13)
    ax.set_ylabel("Clean Accuracy (Smoothed)", fontsize=13)
    ax.set_title("Accuracy–Robustness Tradeoff", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_accuracy_radius_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig3_accuracy_radius_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_accuracy_radius_tradeoff")


def fig4_empirical_attack_comparison(results_dir: Path, output_dir: Path) -> None:
    """Fig 4: Base vs smoothed classifier under PGD attack."""
    import matplotlib.pyplot as plt

    attack_file = results_dir / "empirical_comparison.json"
    if not attack_file.exists():
        # Try sigma-specific path
        for sigma in [0.25, 0.50]:
            alt = results_dir / f"sigma_{sigma:.2f}" / f"empirical_comparison_{sigma:.2f}.json"
            if alt.exists():
                attack_file = alt
                break
    if not attack_file.exists():
        print("  [skip] No empirical comparison results found")
        return

    data = load_json(attack_file)
    attacks = data.get("attacks", {})

    eps_values = []
    base_aucs = []
    smooth_accs = []

    for eps_key, results in sorted(attacks.items()):
        eps_val = float(eps_key.replace("eps_", ""))
        eps_values.append(eps_val)
        base_res = results.get("base_classifier", {})
        smooth_res = results.get("smoothed_classifier", {})
        base_aucs.append(base_res.get("adversarial", {}).get("auc", 0))
        smooth_accs.append(smooth_res.get("adversarial_accuracy", 0))

    if not eps_values:
        print("  [skip] No attack data to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(eps_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_aucs, width, label="Base Classifier (AUC)",
                   color="#F44336", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width/2, smooth_accs, width, label="Smoothed Classifier (Acc)",
                   color="#2196F3", alpha=0.8, edgecolor="black")

    ax.set_xlabel("PGD Attack Budget (ε)", fontsize=13)
    ax.set_ylabel("Performance", fontsize=13)
    ax.set_title("Base vs Smoothed Classifier Under PGD Attack", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e:.2f}" for e in eps_values])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_empirical_attack_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig4_empirical_attack_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_empirical_attack_comparison")


def fig5_certified_accuracy_table(results_dir: Path, output_dir: Path) -> None:
    """Generate a summary table (saved as JSON) for paper tables."""
    sigmas = [0.12, 0.25, 0.50, 1.00]
    table_rows = []

    for sigma in sigmas:
        cert_file = results_dir / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            cert_file = results_dir / f"sigma_{sigma:.2f}" / f"certav_cert_{sigma:.2f}.json"
        if not cert_file.exists():
            continue

        data = load_json(cert_file)
        summary = data.get("summary", {})
        cert_at = data.get("certified_accuracy_at_radii", {})

        table_rows.append({
            "sigma": sigma,
            "clean_accuracy": summary.get("accuracy", 0),
            "abstain_rate": summary.get("abstain_rate", 0),
            "mean_radius": summary.get("mean_certified_radius", 0),
            "cert_acc_r0.00": cert_at.get("r_0.00", 0),
            "cert_acc_r0.25": cert_at.get("r_0.25", 0),
            "cert_acc_r0.50": cert_at.get("r_0.50", 0),
            "cert_acc_r1.00": cert_at.get("r_1.00", 0),
        })

    output_path = output_dir / "table_certified_accuracy.json"
    with open(output_path, "w") as f:
        json.dump(table_rows, f, indent=2)
    print(f"  ✓ table_certified_accuracy ({len(table_rows)} rows)")

    # Print table
    if table_rows:
        print("\n  σ      | Clean Acc | Abstain | Mean R | Cert@0.00 | Cert@0.25 | Cert@0.50 | Cert@1.00")
        print("  " + "-" * 90)
        for row in table_rows:
            print(f"  {row['sigma']:.2f}   | "
                  f"{row['clean_accuracy']:.3f}     | "
                  f"{row['abstain_rate']:.3f}   | "
                  f"{row['mean_radius']:.3f}  | "
                  f"{row['cert_acc_r0.00']:.3f}     | "
                  f"{row['cert_acc_r0.25']:.3f}     | "
                  f"{row['cert_acc_r0.50']:.3f}     | "
                  f"{row['cert_acc_r1.00']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CertAV figures")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating CertAV paper figures...")
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}\n")

    fig1_certified_accuracy_curves(results_dir, output_dir)
    fig2_multimodal_vs_unimodal(results_dir, output_dir)
    fig3_accuracy_radius_tradeoff(results_dir, output_dir)
    fig4_empirical_attack_comparison(results_dir, output_dir)
    fig5_certified_accuracy_table(results_dir, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
