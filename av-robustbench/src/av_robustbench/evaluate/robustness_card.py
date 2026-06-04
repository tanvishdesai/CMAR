from __future__ import annotations

from pathlib import Path

from av_robustbench.core import RobustnessCard
from av_robustbench.utils.io import ensure_dir, read_json, write_json


def load_card(path: str | Path) -> RobustnessCard:
    data = read_json(path)
    return RobustnessCard(
        model_name=data["model_name"],
        dataset=data["dataset"],
        clean_metrics=data.get("clean_metrics", {}),
        certification=data.get("certification", {}),
        attacks=data.get("attacks", {}),
        degradations=data.get("degradations", {}),
        cross_dataset=data.get("cross_dataset", {}),
        metadata=data.get("metadata", {}),
    )


def save_card_outputs(card: RobustnessCard, output_dir: str | Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths = {
        "json": write_json(card.to_dict(), output_dir / "robustness_card.json"),
        "markdown": output_dir / "robustness_card.md",
        "latex": output_dir / "robustness_card.tex",
    }
    paths["markdown"].write_text(card.to_markdown(), encoding="utf-8")
    paths["latex"].write_text(card.to_latex(), encoding="utf-8")
    return paths

