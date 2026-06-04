# Kaggle Runbook

This runbook describes the current CertAV / CertAV-Bench execution path. It is
script-first; notebook cell wrappers live in `notebooks/`.

## Inputs

Attach these Kaggle datasets as needed:

- CMAR/CertAV code snapshot.
- FakeAVCeleb feature cache, usually mounted as `cmar_cache`.
- Existing CertAV seed-run checkpoints for certification and elevation analysis.
- LAV-DF raw data only when rebuilding the LAV-DF cache.

The primary aggregate results already live in this repository under
`agg-results-cmvrta/`; do not rerun experiments unless you need fresh numbers.

## Setup

```bash
cp -r /kaggle/input/<your-cmar-code-dataset>/CMAR /kaggle/working/CMAR
cd /kaggle/working/CMAR
pip install -q -r requirements.txt
```

Optional environment check:

```bash
python scripts/00_environment_check.py \
  --load-models \
  --output /kaggle/working/cmar_environment.json
```

## Verify Feature Cache

```bash
python scripts/10_train_certav.py \
  --sigma 1.00 \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output-dir /kaggle/working/cache_probe \
  --cache-report-only
```

## Train CertAV

Run one seed per Kaggle session if you are reproducing the five-seed aggregate.

```bash
python scripts/10_train_certav.py \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output-dir /kaggle/working/certav_seed_2026/sigma_1.00 \
  --epochs 30 \
  --batch-size 8 \
  --grad-accum 4 \
  --patience 7 \
  --seed 2026
```

For the full original grid, run sigma values `0.12`, `0.25`, `0.50`, and
`1.00`, plus the visual-only and audio-only ablations at `0.25` and `1.00`.
The ready-to-copy version is `notebooks/certav_experiment_notebook.md`.

## Certify CertAV

```bash
python scripts/11_certify.py \
  --checkpoint /kaggle/working/certav_seed_2026/sigma_1.00/best.pt \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output /kaggle/working/certav_seed_2026/cert_sigma_1.00.json \
  --n0 100 \
  --n 1000 \
  --alpha 0.001 \
  --seed 2026
```

## Empirical Attacks

```bash
python scripts/12_empirical_attack_comparison.py \
  --checkpoint /kaggle/working/certav_seed_2026/sigma_1.00/best.pt \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output /kaggle/working/certav_seed_2026/empirical_attacks_1.00.json \
  --eps-values 0.05 0.10 0.20 \
  --max-samples 200
```

## Elevation Experiments

These are the current v2 experiments that support the draft narrative.

No-noise baseline:

```bash
python scripts/14_train_baseline_no_noise.py \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output-dir /kaggle/working/elevation_experiments/baseline_no_noise \
  --seed 2026
```

PGD adversarial-training baseline:

```bash
python scripts/15_train_pgd_at.py \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output-dir /kaggle/working/elevation_experiments/baseline_pgd_at \
  --at-eps 0.1 \
  --at-steps 7 \
  --seed 2026
```

LAV-DF preprocessing:

```bash
python scripts/19_preprocess_lavdf.py \
  --lavdf-root /kaggle/input/<lavdf-dataset>/LAV-DF \
  --output-dir /kaggle/working/lavdf_cache \
  --max-samples 500
```

Cross-dataset certification:

```bash
python scripts/16_certify_cross_dataset.py \
  --checkpoint /path/to/certav_sigma_1.00/best.pt \
  --sigma 1.00 \
  --lavdf-cache-dir /kaggle/working/lavdf_cache \
  --output /kaggle/working/elevation_experiments/cert_lavdf_1.00.json
```

Input-space attack pilot:

```bash
python scripts/17_input_space_attack.py \
  --checkpoint /path/to/certav_sigma_1.00/best.pt \
  --sigma 1.00 \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --output /kaggle/working/elevation_experiments/input_space_attack.json \
  --max-samples 100
```

Manifold analysis:

```bash
python scripts/18_manifold_analysis.py \
  --cache-dir /kaggle/input/<feature-cache-dataset>/cmar_cache \
  --checkpoint /path/to/certav_sigma_1.00/best.pt \
  --output /kaggle/working/elevation_experiments/manifold_analysis.json
```

The ready-to-copy version is `notebooks/elevation_experiment_notebook.md`.

## Phase 2 Experiments

The Phase 2 implementation adds:

- A+D: encoder-family PCA scaling-law runs and PCA-guided anisotropic smoothing.
- C: conformal calibration/evaluation on top of smoothed CertAV probabilities.

Start from `docs/PHASE2_KAGGLE_RUNBOOK.md`. The notebook cell wrapper is
`notebooks/phase2_final_kaggle_cells.md` for the complete start-to-finish
sequence. `notebooks/phase2_experiment_notebook.md` is the shorter legacy
wrapper.

## Aggregate And Plot

Use `notebooks/certav_aggregation_notebook.md` to aggregate the five seed runs.
For local figure regeneration from an already aggregated folder:

```bash
python scripts/13_certav_figures.py \
  --results-dir /kaggle/working/certav_aggregated \
  --output-dir /kaggle/working/certav_aggregated/figures
```

## Recommended Upload Artifacts

After each major run, publish or version these Kaggle datasets:

- `certav-seed-<seed>`: checkpoint folders, certification JSON, attack JSON, seed summary.
- `certav-aggregated`: master JSON, tables, and figures.
- `certav-bench-v2-results`: elevation experiment JSONs and trained baseline metrics.
- `lavdf-features-v1`: LAV-DF cache if you rebuild cross-dataset features.

## Notes

Historical Colab helpers were moved to `notebooks/`. They are fallback tools for
cache rebuilding, not part of the current paper-facing workflow.
