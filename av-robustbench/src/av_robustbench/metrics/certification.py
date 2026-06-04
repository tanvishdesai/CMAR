from __future__ import annotations

from av_robustbench.certification.core import CertificationResult, summarize_certification


def certification_metrics(
    results: list[CertificationResult],
    *,
    radii: list[float] | None = None,
) -> dict[str, object]:
    return summarize_certification(results, radii)

