#!/usr/bin/env python3
"""Summarize Phase 2 A+D and C results into paper-facing tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_cert_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary", {})
    config = data.get("config", {})
    cert_radii = data.get("certified_accuracy_at_radii", {})
    cert_radii_om = data.get("certified_accuracy_at_radii_onmanifold", {})
    return {
        "cert_path": str(path),
        "sigma": config.get("sigma"),
        "noise_mode": config.get("noise_mode"),
        "clean_accuracy": summary.get("accuracy"),
        "abstain_rate": summary.get("abstain_rate"),
        "mean_certified_radius": summary.get("mean_certified_radius"),
        "mean_certified_radius_l2": summary.get("mean_certified_radius_l2"),
        "mean_certified_radius_onmanifold": summary.get("mean_certified_radius_onmanifold"),
        "certified_accuracy_r_0_25": cert_radii.get("r_0.25"),
        "certified_accuracy_r_0_50": cert_radii.get("r_0.50"),
        "certified_accuracy_r_1_00": cert_radii.get("r_1.00"),
        "certified_accuracy_r_0_25_onmanifold": cert_radii_om.get("r_0.25"),
        "certified_accuracy_r_0_50_onmanifold": cert_radii_om.get("r_0.50"),
        "certified_accuracy_r_1_00_onmanifold": cert_radii_om.get("r_1.00"),
    }


def build_encoder_rows(phase2_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = phase2_dir / "encoder_study"
    if not root.exists():
        return rows
    for pair_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pca_summary = pair_dir / "pca_joint.summary.json"
        cert_json = pair_dir / "baseline_no_noise_cert.json"
        if not pca_summary.exists():
            alternatives = sorted(pair_dir.glob("*pca*.summary.json"))
            pca_summary = alternatives[0] if alternatives else pca_summary
        if not cert_json.exists():
            alternatives = sorted(pair_dir.glob("*cert*.json"))
            cert_json = alternatives[0] if alternatives else cert_json

        row: dict[str, Any] = {"encoder_pair": pair_dir.name}
        if pca_summary.exists():
            pca = load_json(pca_summary)
            row.update(
                {
                    "ambient_dim": pca.get("ambient_dim"),
                    "d_int_80": pca.get("dim_at_80pct"),
                    "d_int_90": pca.get("dim_at_90pct"),
                    "d_int_95": pca.get("dim_at_95pct"),
                    "d_int_ratio_90": (
                        pca.get("dim_at_90pct") / pca.get("ambient_dim")
                        if pca.get("ambient_dim") else None
                    ),
                }
            )
        if cert_json.exists():
            row.update(read_cert_summary(cert_json))
        rows.append(row)
    return rows


def build_anisotropic_rows(phase2_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = phase2_dir / "anisotropic"
    if not root.exists():
        return rows
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cert_json = run_dir / "certification.json"
        if not cert_json.exists():
            alternatives = sorted(run_dir.glob("*cert*.json"))
            cert_json = alternatives[0] if alternatives else cert_json
        if not cert_json.exists():
            continue
        row = {"strategy": run_dir.name, **read_cert_summary(cert_json)}
        data = load_json(cert_json)
        per = data.get("per_sample_results", [])
        log_volumes = [
            sample.get("certified_ellipsoid_log_volume")
            for sample in per
            if sample.get("certified_ellipsoid_log_volume") is not None
        ]
        if log_volumes:
            row["mean_ellipsoid_log_volume"] = float(np.mean(log_volumes))
        rows.append(row)
    return rows


def build_conformal_rows(phase2_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = phase2_dir / "conformal"
    if not root.exists():
        return rows
    for path in sorted(root.glob("*eval*.json")):
        data = load_json(path)
        for row in data.get("summaries", []):
            rows.append({"source": str(path), **row})
    return rows


def maybe_plot_scaling(rows: list[dict[str, Any]], output_dir: Path) -> None:
    valid = [
        row for row in rows
        if row.get("d_int_ratio_90") is not None and row.get("mean_certified_radius") is not None
    ]
    if len(valid) < 2:
        return
    import matplotlib.pyplot as plt

    x = np.asarray([row["d_int_ratio_90"] for row in valid], dtype=float)
    y = np.asarray([row["mean_certified_radius"] for row in valid], dtype=float)
    labels = [row["encoder_pair"] for row in valid]
    corr = float(np.corrcoef(x, y)[0, 1]) if len(valid) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(x, y, s=70, color="#2f6f73")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Intrinsic dimension ratio d_int / D at 90% variance")
    ax.set_ylabel("Mean certified radius")
    ax.set_title(f"Encoder Certifiability Scaling Law (corr={corr:.2f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_scaling_law.png", dpi=300)
    plt.close(fig)


def maybe_plot_anisotropic(rows: list[dict[str, Any]], output_dir: Path) -> None:
    valid = [
        row for row in rows
        if row.get("mean_certified_radius_onmanifold") is not None
    ]
    if not valid:
        return
    import matplotlib.pyplot as plt

    labels = [row["strategy"] for row in valid]
    worst = [row.get("mean_certified_radius") or 0.0 for row in valid]
    on = [row.get("mean_certified_radius_onmanifold") or 0.0 for row in valid]
    idx = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(idx - width / 2, worst, width, label="Worst-case L2")
    ax.bar(idx + width / 2, on, width, label="On-manifold")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean certified radius")
    ax.set_title("Anisotropic Smoothing Strategy Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_anisotropic_strategies.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 result directory")
    parser.add_argument("--phase2-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase2_dir = Path(args.phase2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_rows = build_encoder_rows(phase2_dir)
    anisotropic_rows = build_anisotropic_rows(phase2_dir)
    conformal_rows = build_conformal_rows(phase2_dir)

    write_csv(encoder_rows, output_dir / "phase2_encoder_scaling.csv")
    write_csv(anisotropic_rows, output_dir / "phase2_anisotropic.csv")
    write_csv(conformal_rows, output_dir / "phase2_conformal.csv")

    maybe_plot_scaling(encoder_rows, output_dir)
    maybe_plot_anisotropic(anisotropic_rows, output_dir)

    summary = {
        "phase2_dir": str(phase2_dir),
        "n_encoder_rows": len(encoder_rows),
        "n_anisotropic_rows": len(anisotropic_rows),
        "n_conformal_rows": len(conformal_rows),
        "outputs": {
            "encoder_scaling_csv": str(output_dir / "phase2_encoder_scaling.csv"),
            "anisotropic_csv": str(output_dir / "phase2_anisotropic.csv"),
            "conformal_csv": str(output_dir / "phase2_conformal.csv"),
        },
    }
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
