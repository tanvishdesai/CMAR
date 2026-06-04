#!/usr/bin/env python3
"""Compose feature-space certificates with empirical input-to-feature bounds.

This implements the practical Direction-B bridge discussed in the reviews:
estimate an empirical Lipschitz constant from input-space attack measurements,
then map feature-space certified radii back to approximate input-space radii.

The output is intentionally labeled empirical/probabilistic, not a worst-case
formal Lipschitz proof. It is a paper diagnostic and reviewer-facing bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def empirical_lipschitz_by_eps(input_attack: dict[str, Any], quantile: float) -> dict[str, dict[str, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in input_attack.get("per_sample_results", []):
        eps = float(row.get("eps", 0.0))
        if eps <= 0:
            continue
        displacement = float(row.get("feature_l2_displacement", 0.0))
        grouped[eps].append(displacement / eps)

    out: dict[str, dict[str, float]] = {}
    for eps, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out[f"eps_{eps:g}"] = {
            "eps": eps,
            "n": int(arr.size),
            "mean_L": float(np.mean(arr)),
            "median_L": float(np.median(arr)),
            "quantile_L": float(np.quantile(arr, quantile)),
            "max_L": float(np.max(arr)),
        }
    return out


def certification_rows(certification: dict[str, Any]) -> list[dict[str, Any]]:
    rows = certification.get("per_sample_results", [])
    if rows:
        return rows
    return []


def compose_rows(
    cert_rows: list[dict[str, Any]],
    lipschitz: dict[str, dict[str, float]],
    *,
    use_max: bool,
) -> list[dict[str, Any]]:
    composed: list[dict[str, Any]] = []
    for key, stats in lipschitz.items():
        L = stats["max_L"] if use_max else stats["quantile_L"]
        if L <= 0:
            continue
        for row in cert_rows:
            feature_radius = float(row.get("certified_radius", 0.0))
            input_radius = feature_radius / L
            composed.append(
                {
                    "clip_id": row.get("clip_id"),
                    "lipschitz_key": key,
                    "input_eps_source": stats["eps"],
                    "empirical_L": L,
                    "true_label": row.get("true_label"),
                    "predicted_class": row.get("predicted_class"),
                    "correct": bool(row.get("correct", False)),
                    "abstained": bool(row.get("abstained", False)),
                    "feature_certified_radius": feature_radius,
                    "composed_input_radius": input_radius,
                }
            )
    return composed


def summarize_composed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["lipschitz_key"]].append(row)

    summaries: dict[str, Any] = {}
    for key, group in grouped.items():
        radii = np.asarray([row["composed_input_radius"] for row in group], dtype=float)
        correct = np.asarray([row["correct"] and not row["abstained"] for row in group], dtype=bool)
        source_eps = float(group[0]["input_eps_source"])
        summaries[key] = {
            "n": len(group),
            "input_eps_source": source_eps,
            "empirical_L": float(group[0]["empirical_L"]),
            "mean_input_radius": float(np.mean(radii)),
            "median_input_radius": float(np.median(radii)),
            "max_input_radius": float(np.max(radii)),
            "composed_certified_accuracy_at_source_eps": float(np.mean(correct & (radii >= source_eps))),
            "composed_coverage_at_source_eps": float(np.mean(radii >= source_eps)),
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose input-space certificates from feature certificates")
    parser.add_argument("--certification-json", type=str, required=True)
    parser.add_argument("--input-attack-json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--use-max-lipschitz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    certification = load_json(args.certification_json)
    input_attack = load_json(args.input_attack_json)

    lipschitz = empirical_lipschitz_by_eps(input_attack, quantile=args.quantile)
    rows = compose_rows(
        certification_rows(certification),
        lipschitz,
        use_max=args.use_max_lipschitz,
    )
    output = {
        "config": {
            "certification_json": args.certification_json,
            "input_attack_json": args.input_attack_json,
            "quantile": args.quantile,
            "use_max_lipschitz": args.use_max_lipschitz,
            "interpretation": "empirical input-to-feature composition, not worst-case formal Lipschitz proof",
        },
        "empirical_lipschitz": lipschitz,
        "summary": summarize_composed(rows),
        "per_sample_results": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Composed certificate summary written to {output_path}")
    for key, row in output["summary"].items():
        print(
            f"  {key}: L={row['empirical_L']:.3f}, "
            f"mean_input_radius={row['mean_input_radius']:.6f}, "
            f"cert_acc@eps={row['composed_certified_accuracy_at_source_eps']:.3f}"
        )


if __name__ == "__main__":
    main()
