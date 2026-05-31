from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.evaluation.metrics import binary_metrics, delta_auc, robustness_accuracy_ratio
from cmar.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external baseline score CSVs. CSV columns: "
            "model,condition,clip_id,label,score where score is P(fake)."
        )
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", default="/kaggle/working/baseline-results.json")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Baseline score CSV not found: {scores_path}\n"
            "This script does not run LipForensics/AASIST inference by itself; it only "
            "summarizes an already-generated CSV with columns "
            "model,condition,clip_id,label,score. Skip the baseline step until that CSV exists."
        )

    df = pd.read_csv(scores_path)
    required = {"model", "condition", "clip_id", "label", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required score columns: {sorted(missing)}")

    results = {}
    for model_name, model_df in df.groupby("model"):
        results[model_name] = {}
        clean_auc = None
        for condition, cond_df in model_df.groupby("condition"):
            metrics = binary_metrics(cond_df["label"], cond_df["score"], from_logits=False)
            if condition == "clean":
                clean_auc = metrics["auc"]
            elif clean_auc is not None:
                metrics["rar"] = robustness_accuracy_ratio(metrics["auc"], clean_auc)
                metrics["delta_auc"] = delta_auc(clean_auc, metrics["auc"])
            results[model_name][condition] = metrics
            print(model_name, condition, metrics)
    write_json(results, args.output)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
