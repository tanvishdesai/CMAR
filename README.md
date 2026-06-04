# CertAV / CertAV-Bench

CertAV is a research codebase for certifiably robust audio-visual deepfake
detection. The current paper story is no longer the original CMAR-only project:
the active work is CertAV, the elevation experiments, and the companion
community benchmark package `av-robustbench`.

## Current State

The strongest current claim is that frozen DINOv2 and Whisper feature spaces are
surprisingly certifiable under randomized smoothing. Noise-augmented CertAV
models still provide the main protocol, but the elevation results show that the
feature geometry itself is a major source of certifiability.

Key evidence lives in:

- `agg-results-cmvrta/certav_aggregated/certav_master_results.json`
- `agg-results-cmvrta/certav_aggregated/figures/`
- `agg-results-cmvrta/certav-bench-v2-results/elevation_experiments/`
- `docs/certavbench-v2-mentor review.md`
- `docs/certav-elevt-v2-plan.md`
- `docs/writing_notes_related_work.md`

## Repository Map

```text
CMAR/
  cmar/                     Core CertAV model, training, evaluation, certification
  scripts/                  Reproducible preprocessing, training, certification, analysis
  configs/                  Dataset and training configuration
  agg-results-cmvrta/       Aggregated paper results and elevation experiment outputs
  av-robustbench/           Standalone benchmark package for AV robustness evaluation
  docs/                     Paper-facing notes, runbooks, dataset/cache documentation
  notebooks/                Kaggle/Colab cell wrappers and historical notebook drivers
```

The previous `archive/` folder and copied Colab cell files in `docs/` are legacy
artifacts. Notebook-style execution helpers now live under `notebooks/`.

## Setup

```bash
python -m pip install -r requirements.txt
```

For the benchmark package:

```bash
python -m pip install -e av-robustbench
```

Optional dependencies for model-zoo and degradation tooling:

```bash
python -m pip install -e "av-robustbench[all]"
```

## Main CertAV Workflow

Preprocess FakeAVCeleb features:

```bash
python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root /path/to/FakeAVCeleb \
  --output-dir /path/to/cmar_cache
```

Train a CertAV model:

```bash
python scripts/10_train_certav.py \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir /path/to/cmar_cache \
  --output-dir runs/certav_sigma100 \
  --seed 2026
```

Certify it:

```bash
python scripts/11_certify.py \
  --checkpoint runs/certav_sigma100/best.pt \
  --sigma 1.00 \
  --cache-dir /path/to/cmar_cache \
  --output runs/certav_sigma100/certification.json
```

Run empirical feature-space attacks:

```bash
python scripts/12_empirical_attack_comparison.py \
  --checkpoint runs/certav_sigma100/best.pt \
  --sigma 1.00 \
  --cache-dir /path/to/cmar_cache \
  --output runs/certav_sigma100/empirical_attacks.json
```

Generate paper figures from a results folder:

```bash
python scripts/13_certav_figures.py \
  --results-dir agg-results-cmvrta/certav_aggregated
```

## Elevation Experiments

The v2 elevation scripts add the baselines and validation experiments used in
the current paper narrative:

```bash
python scripts/14_train_baseline_no_noise.py --cache-dir /path/to/cmar_cache --output-dir runs/baseline_no_noise
python scripts/15_train_pgd_at.py --cache-dir /path/to/cmar_cache --output-dir runs/baseline_pgd_at
python scripts/16_certify_cross_dataset.py --checkpoint runs/certav_sigma100/best.pt --sigma 1.00 --lavdf-cache-dir /path/to/lavdf_cache --output runs/lavdf_cert.json
python scripts/17_input_space_attack.py --checkpoint runs/certav_sigma100/best.pt --sigma 1.00 --cache-dir /path/to/cmar_cache --output runs/input_space_attack.json
python scripts/18_manifold_analysis.py --cache-dir /path/to/cmar_cache --checkpoint runs/certav_sigma100/best.pt --output runs/manifold_analysis.json
python scripts/19_preprocess_lavdf.py --lavdf-root /path/to/LAV-DF --output-dir /path/to/lavdf_cache
```

For Kaggle-ready cells, use `notebooks/elevation_experiment_notebook.md`.

## av-robustbench

`av-robustbench` is the companion benchmark contribution. It provides:

- model adapters for feature-space audio-visual detectors
- PGD-Linf, joint PGD-L2, Square Attack, and AutoAttack-style ensembles
- randomized smoothing certification
- degradation battery evaluation
- robustness cards and leaderboard JSON helpers

Example:

```bash
av-robustbench evaluate \
  --model certav_sigma100 \
  --checkpoint /path/to/best.pt \
  --dataset feature_cache \
  --cache-dir /path/to/cmar_cache \
  --attacks pgd_linf pgd_l2 autoattack_av \
  --certify \
  --sigma 0.25 1.00 \
  --output results/robustness_card
```

Run benchmark tests:

```bash
cd av-robustbench
pytest
```

## Paper Drafting Notes

Start with `docs/README.md`, then read:

1. `docs/certavbench-v2-mentor review.md`
2. `docs/writing_notes_related_work.md`
3. `agg-results-cmvrta/certav_aggregated/certav_master_results.json`
4. `agg-results-cmvrta/certav-bench-v2-results/elevation_experiments/elevation_summary.json`

The paper should emphasize the validated results: strong certified radii,
input-space certificate hold rates, cross-dataset LAV-DF transfer, PGD-AT
comparison, and the manifold/intrinsic-dimensionality explanation.
