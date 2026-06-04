# CertAV Phase 2 Kaggle Runbook

This is the runnable sequence for the two Phase 2 paths from
`0106-reivews/CertAV_Phase2_Implementation_Guide.md`. For a single
copy-paste Kaggle notebook with all cells in order, use
`notebooks/phase2_final_kaggle_cells.md`.

- Path A+B/A+D: manifold-aware anisotropic smoothing, empirical
  input-to-feature composition, and encoder scaling-law support.
- Path C: conformal prediction on top of CertAV smoothing/certification.

The scripts are designed so every major run can be resumed or repeated from
Kaggle outputs. Use one seed first for smoke results, then repeat with 3-5 seeds
for paper tables.

## 0. Setup

```bash
cp -r /kaggle/input/<your-cmar-code-dataset>/CMAR /kaggle/working/CMAR
cd /kaggle/working/CMAR
pip install -q -r requirements.txt

export BASE_CACHE=/kaggle/input/<feature-cache-dataset>/cmar_cache
export PHASE2=/kaggle/working/phase2
mkdir -p $PHASE2
```

## 1. Path A+D: Encoder Scaling Law

Use the existing DINOv2-Small + Whisper-tiny cache as the first encoder pair.
For each new encoder pair, rebuild a separate cache with the same split
manifests and a unique output directory.

Example new cache: DINOv2-Base + Whisper-base.

```bash
export PAIR=dinov2base_whisperbase
export PAIR_DIR=$PHASE2/encoder_study/$PAIR

python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root /kaggle/input/<fakeavceleb-dataset>/FakeAVCeleb \
  --output-dir $PAIR_DIR/cmar_cache \
  --visual-model-name facebook/dinov2-base \
  --audio-model-name openai/whisper-base \
  --no-degraded \
  --splits train val test \
  --max-runtime-seconds 10800 \
  --chunk-size 50
```

Recommended four-pair grid:

```text
dinov2small_whispertiny    facebook/dinov2-small          openai/whisper-tiny
dinov2base_whisperbase     facebook/dinov2-base           openai/whisper-base
clip_whispertiny           openai/clip-vit-base-patch16   openai/whisper-tiny
dinov2small_hubert         facebook/dinov2-small          facebook/hubert-base-ls960
```

For each pair, fit the PCA basis, train the no-noise baseline, and certify it
at sigma 1.00.

```bash
python scripts/20_fit_pca_noise.py \
  --cache-dir $PAIR_DIR/cmar_cache \
  --feature-space joint \
  --output $PAIR_DIR/pca_joint.pt \
  --summary-output $PAIR_DIR/pca_joint.summary.json

python scripts/14_train_baseline_no_noise.py \
  --cache-dir $PAIR_DIR/cmar_cache \
  --output-dir $PAIR_DIR/baseline_no_noise \
  --seed 2026

python scripts/11_certify.py \
  --checkpoint $PAIR_DIR/baseline_no_noise/best.pt \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir $PAIR_DIR/cmar_cache \
  --output $PAIR_DIR/baseline_no_noise_cert.json \
  --n0 100 \
  --n 1000 \
  --alpha 0.001 \
  --seed 2026
```

This produces the Table 1 inputs: `d_int/D`, clean accuracy, and mean certified
radius for each encoder pair.

## 2. Path A: Anisotropic Smoothing

Start with the base cache or the best encoder pair from the scaling-law study.

```bash
export ANISO=$PHASE2/anisotropic
export CACHE=$BASE_CACHE

python scripts/20_fit_pca_noise.py \
  --cache-dir $CACHE \
  --feature-space joint \
  --output $ANISO/pca_joint.pt \
  --summary-output $ANISO/pca_joint.summary.json
```

Train and certify each strategy. Strategy names map to the guide as:

- `anisotropic_strat1`: eigenvalue-proportional covariance.
- `anisotropic_strat2`: top-k subspace projection.
- `anisotropic_strat3`: inverse-eigenvalue covariance.

```bash
for STRAT in anisotropic_strat1 anisotropic_strat2 anisotropic_strat3; do
  python scripts/10_train_certav.py \
    --sigma 1.00 \
    --noise-mode $STRAT \
    --pca-noise-path $ANISO/pca_joint.pt \
    --cache-dir $CACHE \
    --output-dir $ANISO/$STRAT \
    --epochs 30 \
    --batch-size 8 \
    --grad-accum 4 \
    --patience 7 \
    --seed 2026

  python scripts/11_certify.py \
    --checkpoint $ANISO/$STRAT/best.pt \
    --sigma 1.00 \
    --noise-mode $STRAT \
    --pca-noise-path $ANISO/pca_joint.pt \
    --cache-dir $CACHE \
    --output $ANISO/$STRAT/certification.json \
    --n0 100 \
    --n 1000 \
    --alpha 0.001 \
    --seed 2026
done
```

Run the PGD alignment diagnostic for the same PCA basis.

```bash
python scripts/12_empirical_attack_comparison.py \
  --checkpoint $ANISO/anisotropic_strat1/best.pt \
  --sigma 1.00 \
  --noise-mode anisotropic_strat1 \
  --pca-noise-path $ANISO/pca_joint.pt \
  --cache-dir $CACHE \
  --output $ANISO/anisotropic_strat1/attack_alignment.json \
  --eps-values 0.05 0.10 0.20 \
  --max-samples 200
```

## 2b. Optional Path B: Empirical Input-Space Composition

If you want the A+B framing from the reviews, run the existing input-space
attack pilot and compose the resulting empirical input-to-feature bound with a
feature-space certification file.

```bash
python scripts/17_input_space_attack.py \
  --checkpoint $ANISO/anisotropic_strat1/best.pt \
  --sigma 1.00 \
  --cache-dir $CACHE \
  --output $ANISO/anisotropic_strat1/input_space_attack.json \
  --max-samples 100

python scripts/24_compose_input_certificate.py \
  --certification-json $ANISO/anisotropic_strat1/certification.json \
  --input-attack-json $ANISO/anisotropic_strat1/input_space_attack.json \
  --output $ANISO/anisotropic_strat1/composed_input_certificate.json \
  --quantile 0.99
```

This reports empirical input-space radii using a 99th percentile Lipschitz
estimate. Use `--use-max-lipschitz` for a more conservative pilot estimate.

## 3. Path C: Conformal CertAV

Use a completed isotropic or anisotropic checkpoint. The clean baseline is the
isotropic sigma 1.00 CertAV model.

```bash
export CONF=$PHASE2/conformal
export ISO_CKPT=/kaggle/input/<certav-sigma100-run>/best.pt

python scripts/21_conformal_calibrate.py \
  --checkpoint $ISO_CKPT \
  --cache-dir $BASE_CACHE \
  --sigma 1.00 \
  --noise-mode joint \
  --output $CONF/calibration_sigma100.json \
  --alphas 0.05 0.10 0.20 \
  --radii 0.00 0.25 0.50 1.00 \
  --score-types raw cp log \
  --n 1000 \
  --cp-alpha 0.001 \
  --split val
```

Evaluate clean coverage and robust coverage under feature-space PGD.

```bash
python scripts/22_conformal_evaluate.py \
  --checkpoint $ISO_CKPT \
  --cache-dir $BASE_CACHE \
  --calibration $CONF/calibration_sigma100.json \
  --sigma 1.00 \
  --noise-mode joint \
  --output $CONF/test_eval_sigma100.json \
  --split test \
  --n 1000 \
  --attack-eps-values 0.25 0.50 1.00 \
  --attack-steps 20
```

For cross-dataset conformal coverage, first build the matching LAV-DF cache,
then run the same evaluator with `--cache-dir /path/to/lavdf_cache --split test`.

## 4. Aggregate Tables And Figures

Keep outputs in this layout:

```text
$PHASE2/
  encoder_study/<pair>/pca_joint.summary.json
  encoder_study/<pair>/baseline_no_noise_cert.json
  anisotropic/<strategy>/certification.json
  conformal/*eval*.json
```

Then run:

```bash
python scripts/23_phase2_summarize.py \
  --phase2-dir $PHASE2 \
  --output-dir $PHASE2/summary
```

The summary directory contains:

- `phase2_encoder_scaling.csv`
- `phase2_anisotropic.csv`
- `phase2_conformal.csv`
- `phase2_scaling_law.png`
- `phase2_anisotropic_strategies.png`

## 5. Recommended Execution Order

1. Run Path C first on an existing sigma 1.00 checkpoint. It is the fastest new
   result because it only needs calibration/evaluation.
2. Fit PCA and run anisotropic strategy 1 on the existing cache.
3. Add the Path B composition diagnostic if you want the A+B story.
4. Add strategies 2 and 3 once strategy 1 smoke tests pass.
5. Run the encoder-family preprocessing/training grid over multiple Kaggle
   sessions. Upload each pair directory as a Kaggle dataset.
6. Aggregate with `23_phase2_summarize.py`, inspect the tables, then repeat
   the strongest rows with 3-5 seeds.
