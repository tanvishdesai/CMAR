from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.utils.visualization import (
    plot_ablation_bars,
    plot_asymmetric_attack_bars,
    plot_category_auc,
    plot_rar_curves,
    plot_training_log,
)


def load_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        print(f"[skip] JSON not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CMAR paper figures from result files")
    parser.add_argument("--cmar-results", default=None)
    parser.add_argument("--adversarial-results", default=None)
    parser.add_argument("--ablation-csv", default=None)
    parser.add_argument("--baseline-results", default=None)
    parser.add_argument("--training-log", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/cmar_figures")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.training_log:
        if Path(args.training_log).exists():
            plot_training_log(args.training_log, output)
        else:
            print(f"[skip] training log not found: {args.training_log}")

    if args.cmar_results:
        cmar = load_json(args.cmar_results)
        if cmar:
            rar_rows = []
            for condition, value in cmar.items():
                if not isinstance(value, dict) or "metrics" not in value:
                    continue
                metrics = value["metrics"]
                if "rar" in metrics:
                    rar_rows.append({"model": "CMAR", "condition": condition, "rar": metrics["rar"]})
            if rar_rows:
                plot_rar_curves(pd.DataFrame(rar_rows), output)
            cat_rows = []
            category_source = cmar.get("category_contrasts") or cmar.get("per_category", {})
            for category, metrics in category_source.items():
                auc = metrics.get("auc")
                if auc is None or not math.isfinite(float(auc)):
                    continue
                cat_rows.append({"model": "CMAR", "category": category, "auc": auc})
            if cat_rows:
                plot_category_auc(pd.DataFrame(cat_rows), output)

    if args.adversarial_results:
        adv = load_json(args.adversarial_results)
        if adv:
            model_name = "CMAR"
            protocol = adv.get("protocol", {})
            if protocol.get("valid_for_final_adversarial_claim") is False:
                model_name = "CMAR feature proxy"
            rows = []
            for condition, metrics in adv.get("attacks", {}).items():
                rows.append({"model": model_name, "condition": condition, "auc": metrics["auc"]})
            if rows:
                plot_asymmetric_attack_bars(pd.DataFrame(rows), output)

    if args.ablation_csv:
        if Path(args.ablation_csv).exists():
            plot_ablation_bars(pd.read_csv(args.ablation_csv), output)
        else:
            print(f"[skip] ablation CSV not found: {args.ablation_csv}")

    print("Figures written to", output)


if __name__ == "__main__":
    main()
