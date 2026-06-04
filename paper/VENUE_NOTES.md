# Venue Notes: ICASSP 2027 Target

## Template and format

- Target venue: IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2027.
- Paper-kit status checked on 2026-05-31: the ICASSP 2027 CMS paper-kit endpoints were not yet public. The latest official reachable ICASSP template files were the ICASSP 2026 CMS files (`Template.tex`, `spconf.sty`, `IEEEbib.bst`, and `Template.pdf`).
- Draft implementation: use the official ICASSP `spconf` style and IEEE bibliography style, with the manuscript written as a conference paper rather than as a long technical report.

## Target positioning

The strongest ICASSP framing is not "another deepfake detector." The paper should be framed as a signal-processing robustness contribution for audio-visual media forensics:

1. Certified robustness for audio-visual deepfake detection under a joint feature-space threat model.
2. A practical certification route for frozen foundation representations, where raw-input certification is difficult and expensive.
3. A statistically repeated empirical study showing that the certifiability comes largely from representation geometry rather than only from noise-augmented training.
4. A benchmark-facing evaluation package (`av-robustbench`) that makes attacks, certificates, degradations, and robustness-card reporting reproducible.

## ICASSP-style expectations inferred from recent signal-processing papers

Recent ICASSP papers in audio, speech, and media forensics typically follow a compact structure:

- One precise technical claim in the title and abstract.
- A short introduction with clear motivation and 3-4 concrete contributions.
- Related work that establishes the gap without turning into a broad survey.
- A method section with equations and architecture details, but only details needed to reproduce the core idea.
- Experiments that lead with the primary table, then ablations, transfer/generalization, and stress tests.
- Claims stated in measured language, especially when results are based on a constrained threat model.
- A limitations paragraph or scoped conclusion when the method is not end-to-end over raw input.

## Claim boundaries for this draft

Use:

- "feature-space randomized smoothing"
- "joint audio-visual feature-space L2 certificate"
- "per-sample certified radii over frozen DINOv2 and Whisper representations"
- "feature-displacement validation" or "input-proxy stress test"
- "suggests that frozen foundation features are already unusually certifiable"
- "avoids the validation-AUC degradation observed under feature-space PGD adversarial training"

Avoid:

- "certified robustness to arbitrary pixel/audio-waveform perturbations"
- "first certified deepfake detector" without scope
- "noise augmentation is necessary"
- "no accuracy-robustness tradeoff" as a universal claim
- "input-space PGD proof" for the current feature-cache experiments

## Recommended contribution statement

This paper should claim that CertAV provides a reproducible feature-space certification pipeline for audio-visual deepfake detection, and that its strongest empirical finding is the combination of high certified accuracy, low abstention, cross-dataset transfer, and an intrinsic-dimension analysis explaining why large feature-space radii are possible in frozen multimodal representations.

