from __future__ import annotations

from pathlib import Path

from av_robustbench.core import RobustnessCard
from av_robustbench.evaluate.robustness_card import save_card_outputs


def write_report(card: RobustnessCard, output_dir: str | Path) -> dict[str, Path]:
    return save_card_outputs(card, output_dir)


def write_pdf_report(card: RobustnessCard, output_path: str | Path) -> Path:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError("PDF report generation requires `reportlab`.") from exc
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, f"Robustness Card: {card.model_name}")
    y -= 32
    c.setFont("Helvetica", 10)
    for line in card.to_markdown().splitlines():
        if y < 72:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 10)
        c.drawString(72, y, line[:110])
        y -= 14
    c.save()
    return output_path

