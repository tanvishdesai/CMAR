# Kaggle Runbook

This is the Kaggle execution sequence for the current CMAR state.

## Current State

As of May 29, 2026, preprocessing is complete. The Kaggle feature-cache dataset
shown as **CMAR Clean Features V1** now contains:

- clean visual features: `cmar_cache/features/visual/{train,val,test}/*.pt`
- clean audio features: `cmar_cache/features/audio/{train,val,test}/*.pt`
- all 12 degraded test conditions under `cmar_cache/features/degraded_test/`
- manifests and cache reports under `cmar_cache/manifests/` and `*.json`

The dataset slug used earlier was:

```text
vasuaashadesai/cmar-features-clean-v1
```

In a Kaggle notebook, the mounted path is usually:

```text
/kaggle/input/cmar-features-clean-v1/cmar_cache
```

If Kaggle mounts it with a different folder name, find it with:

```bash
find /kaggle/input -maxdepth 5 -type d -name cmar_cache
```

Do not rerun preprocessing for the main experiments unless you intentionally
want to rebuild the cache.

The v1 results changed the research posture slightly: clean/degraded robustness
looks promising, but cached feature-space attacks are not valid final evidence
for the original "inherent adversarial robustness" claim. The current workflow
therefore runs a claim audit after evaluation. The audit decides whether the
paper should keep the strong adversarial claim, soften it, or pivot toward a
robustness-characterization paper.

## Session Setup

Attach these Kaggle inputs:

- the CMAR code dataset or uploaded project zip
- the completed feature-cache dataset, **CMAR Clean Features V1**

Raw FakeAVCeleb and LAV-DF are no longer required for training or clean/degraded
evaluation because the feature cache already contains the extracted tensors.

Copy the code into the writable area. If the input contains a `CMAR/` folder:

```bash
cp -r /kaggle/input/<your-cmar-code-dataset>/CMAR /kaggle/working/CMAR
cd /kaggle/working/CMAR
pip install -q -r requirements.txt
```

The default requirements intentionally do not install `torchattacks`, because
the current PyPI release downgrades `requests` on Kaggle. CMAR's adversarial
script uses the manual feature-space PGD/FGSM implementation in
`cmar/evaluation/attacks.py`.

If the input contains `CMAR.zip` instead:

```bash
unzip -q /kaggle/input/<your-cmar-code-dataset>/CMAR.zip -d /kaggle/working
cd /kaggle/working/CMAR
pip install -q -r requirements.txt
```

Optional environment check:

```bash
python scripts/00_environment_check.py \
  --load-models \
  --output /kaggle/working/cmar_environment.json
```

## Step 1: Verify The Feature Cache

Run this before training:

```bash
cd /kaggle/working/CMAR
python scripts/02_train_cmar.py \
  --config configs/train_cmar.json \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --cache-report-only
```

Expected result:

- train clean cache complete
- val clean cache complete
- no missing visual/audio features

The degraded folders are not needed for training, but they are needed for
`scripts/03_evaluate_clean_degraded.py`.

## Step 2: Train CMAR

Use a fresh output directory if an older run produced `inf` or `nan` losses.
The current config disables AMP, uses a lower learning rate, clips gradients,
and sanitizes cached tensors.

```bash
cd /kaggle/working/CMAR
python scripts/02_train_cmar.py \
  --config configs/train_cmar.json \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --output-dir /kaggle/working/cmar_runs/full_final \
  --no-amp \
  --lr 0.0002
```

Training outputs:

```text
/kaggle/working/cmar_runs/full_final/best.pt
/kaggle/working/cmar_runs/full_final/training_log.csv
/kaggle/working/cmar_runs/full_final/train_config.json
/kaggle/working/cmar_runs/full_final/best_metrics.json
```

Upload this folder as a Kaggle dataset after training, for example:

```text
cmar-checkpoint-v1
```

## Step 3: Evaluate Clean And Degraded Conditions

If evaluation is in the same notebook immediately after training:

```bash
cd /kaggle/working/CMAR
python scripts/03_evaluate_clean_degraded.py \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --checkpoint /kaggle/working/cmar_runs/full_final/best.pt \
  --output /kaggle/working/cmar-results-clean-degraded.json \
  --include-cached-ensemble \
  --include-modality-masking \
  --skip-lavdf
```

If evaluation is in a new notebook, attach the uploaded checkpoint dataset and
use its `best.pt` path instead:

```bash
cd /kaggle/working/CMAR
python scripts/03_evaluate_clean_degraded.py \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --checkpoint /kaggle/input/cmar-checkpoint-v1/best.pt \
  --output /kaggle/working/cmar-results-clean-degraded.json \
  --include-cached-ensemble \
  --include-modality-masking \
  --skip-lavdf
```

Use `--skip-lavdf` with the current cache because `lavdf_test.csv` may exist
without precomputed LAV-DF feature tensors. Remove that flag only after creating
and attaching a cache that contains LAV-DF features.

The old `--include-ttda` flag is still accepted as a deprecated alias, but the
cached-feature path cannot implement true runtime TTDA. The output is now named
`clean_cached_ensemble` and should be treated as an audit-only ensemble probe.

By default, evaluation now checks that the required degraded cache files exist
for each condition. Use `--allow-clean-fallback` only for a smoke test.

Expected output:

```text
/kaggle/working/cmar-results-clean-degraded.json
```

Upload this JSON and the training folder as a results/checkpoint dataset.

## Step 4: Baselines

External baseline repositories and checkpoints are not bundled here, and
`scripts/04_evaluate_baselines.py` does not run baseline inference. Generate a
score CSV first with:

```text
model,condition,clip_id,label,score
```

Then run:

```bash
python scripts/04_evaluate_baselines.py \
  --scores /kaggle/working/baseline_scores.csv \
  --output /kaggle/working/baseline-results.json
```

## Step 5: Adversarial Evaluation

The implemented adversarial script attacks cached features as a fast proxy. This
is useful for debugging and ablations, and its JSON is explicitly marked
`valid_for_final_adversarial_claim=false`. Lower-level raw visual/audio PGD
helpers exist in `cmar/evaluation/attacks.py`, but the final paper should still
validate true waveform/input-space attacks on at least the main conditions.

```bash
python scripts/05_adversarial_evaluation.py \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --checkpoint /kaggle/input/cmar-checkpoint-v1/best.pt \
  --output /kaggle/working/adversarial-results.json
```

## Step 6: Ablations

```bash
python scripts/06_ablations.py \
  --base-config configs/train_cmar.json \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --output-root /kaggle/working/cmar_runs/ablations
```

Then evaluate those ablation checkpoints on the same test conditions:

```bash
python scripts/09_evaluate_ablations.py \
  --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
  --full-checkpoint /kaggle/working/cmar_runs/full_final/best.pt \
  --ablation-root /kaggle/working/cmar_runs/ablations \
  --conditions clean d12_social d11_h264_crf28 \
  --include-modality-masking \
  --output /kaggle/working/ablation-results.json \
  --summary-csv /kaggle/working/ablation-results.csv
```

## Step 7: Claim Audit

Run this before deciding what the paper should claim:

```bash
python scripts/08_claim_audit.py \
  --cmar-results /kaggle/working/cmar-results-clean-degraded.json \
  --adversarial-results /kaggle/working/adversarial-results.json \
  --ablation-summary /kaggle/working/cmar_runs/ablations/ablation_training_summary.json \
  --full-best-metrics /kaggle/working/cmar_runs/full_final/best_metrics.json \
  --output-json /kaggle/working/claim-audit.json \
  --output-md /kaggle/working/claim-audit.md
```

Read `claim-audit.md`. If it recommends `pivot_to_robustness_characterization`,
do not write the paper as if cross-modal fusion has already solved adversarial
robustness. Write the paper around real-world degradation robustness, modality
diagnostics, and a rigorous adversarial protocol instead.

## Step 8: Figures

```bash
python scripts/07_analysis_figures.py \
  --cmar-results /kaggle/working/cmar-results-clean-degraded.json \
  --adversarial-results /kaggle/working/adversarial-results.json \
  --ablation-csv /kaggle/working/ablation-results.csv \
  --training-log /kaggle/working/cmar_runs/full_final/training_log.csv \
  --output-dir /kaggle/working/cmar_figures
```

## Recommended Upload Artifacts

After each major run, create or version Kaggle datasets from the important
outputs:

- `cmar-features-clean-v1`: completed `cmar_cache/` with clean and degraded features
- `cmar-checkpoint-v1`: `best.pt`, `training_log.csv`, `train_config.json`, `best_metrics.json`
- `cmar-results-v1`: JSON result files and generated figures

## Historical Preprocessing Fallback

The Colab and sliced-preprocessing scripts remain available only if the feature
cache must be rebuilt:

- `docs/COLAB_PREPROCESS_CELL.py`: clean/all-in-one preprocessing with Drive mirror
- `docs/COLAB_UPLOAD_CLEAN_CACHE.py`: upload clean cache
- `docs/COLAB_PREPROCESS_DEGRADED_CELL.py`: sequential degraded preprocessing
- `docs/COLAB_PARALLEL_DEGRADED_WORKER.py`: parallel one-condition degraded worker

For the current experiment path, skip these and train directly from the
completed Kaggle cache.
