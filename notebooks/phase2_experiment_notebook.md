# Phase 2 Kaggle Notebook Cells

Copy these cells into Kaggle in order. Replace the dataset placeholders before
running.

## Cell 1: Setup

```python
import os

CODE_DATASET = "/kaggle/input/<your-cmar-code-dataset>/CMAR"
BASE_CACHE = "/kaggle/input/<feature-cache-dataset>/cmar_cache"
PHASE2 = "/kaggle/working/phase2"

!cp -r {CODE_DATASET} /kaggle/working/CMAR
%cd /kaggle/working/CMAR
!pip install -q -r requirements.txt
!mkdir -p {PHASE2}
```

## Cell 2: Fit PCA On Existing Cache

```python
ANISO = f"{PHASE2}/anisotropic"
!mkdir -p {ANISO}

!python scripts/20_fit_pca_noise.py \
  --cache-dir {BASE_CACHE} \
  --feature-space joint \
  --output {ANISO}/pca_joint.pt \
  --summary-output {ANISO}/pca_joint.summary.json
```

## Cell 3: Train Anisotropic Strategy 1

```python
!python scripts/10_train_certav.py \
  --sigma 1.00 \
  --noise-mode anisotropic_strat1 \
  --pca-noise-path {ANISO}/pca_joint.pt \
  --cache-dir {BASE_CACHE} \
  --output-dir {ANISO}/anisotropic_strat1 \
  --epochs 30 \
  --batch-size 8 \
  --grad-accum 4 \
  --patience 7 \
  --seed 2026
```

## Cell 4: Certify Anisotropic Strategy 1

```python
!python scripts/11_certify.py \
  --checkpoint {ANISO}/anisotropic_strat1/best.pt \
  --sigma 1.00 \
  --noise-mode anisotropic_strat1 \
  --pca-noise-path {ANISO}/pca_joint.pt \
  --cache-dir {BASE_CACHE} \
  --output {ANISO}/anisotropic_strat1/certification.json \
  --n0 100 \
  --n 1000 \
  --alpha 0.001 \
  --seed 2026
```

## Cell 5: Calibrate Conformal CertAV

Optional A+B composition diagnostic before moving to conformal:

```python
!python scripts/17_input_space_attack.py \
  --checkpoint {ANISO}/anisotropic_strat1/best.pt \
  --sigma 1.00 \
  --cache-dir {BASE_CACHE} \
  --output {ANISO}/anisotropic_strat1/input_space_attack.json \
  --max-samples 100

!python scripts/24_compose_input_certificate.py \
  --certification-json {ANISO}/anisotropic_strat1/certification.json \
  --input-attack-json {ANISO}/anisotropic_strat1/input_space_attack.json \
  --output {ANISO}/anisotropic_strat1/composed_input_certificate.json \
  --quantile 0.99
```

## Cell 6: Calibrate Conformal CertAV

```python
ISO_CKPT = "/kaggle/input/<certav-sigma100-run>/best.pt"
CONF = f"{PHASE2}/conformal"
!mkdir -p {CONF}

!python scripts/21_conformal_calibrate.py \
  --checkpoint {ISO_CKPT} \
  --cache-dir {BASE_CACHE} \
  --sigma 1.00 \
  --noise-mode joint \
  --output {CONF}/calibration_sigma100.json \
  --alphas 0.05 0.10 0.20 \
  --radii 0.00 0.25 0.50 1.00 \
  --score-types raw cp log \
  --n 1000 \
  --cp-alpha 0.001 \
  --split val
```

## Cell 7: Evaluate Conformal CertAV

```python
!python scripts/22_conformal_evaluate.py \
  --checkpoint {ISO_CKPT} \
  --cache-dir {BASE_CACHE} \
  --calibration {CONF}/calibration_sigma100.json \
  --sigma 1.00 \
  --noise-mode joint \
  --output {CONF}/test_eval_sigma100.json \
  --split test \
  --n 1000 \
  --attack-eps-values 0.25 0.50 1.00 \
  --attack-steps 20
```

## Cell 8: Encoder Pair Template

```python
FAKEAVCELEB_ROOT = "/kaggle/input/<fakeavceleb-dataset>/FakeAVCeleb"
PAIR = "dinov2base_whisperbase"
PAIR_DIR = f"{PHASE2}/encoder_study/{PAIR}"

!python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root {FAKEAVCELEB_ROOT} \
  --output-dir {PAIR_DIR}/cmar_cache \
  --visual-model-name facebook/dinov2-base \
  --audio-model-name openai/whisper-base \
  --no-degraded \
  --splits train val test \
  --max-runtime-seconds 10800 \
  --chunk-size 50

!python scripts/20_fit_pca_noise.py \
  --cache-dir {PAIR_DIR}/cmar_cache \
  --feature-space joint \
  --output {PAIR_DIR}/pca_joint.pt \
  --summary-output {PAIR_DIR}/pca_joint.summary.json

!python scripts/14_train_baseline_no_noise.py \
  --cache-dir {PAIR_DIR}/cmar_cache \
  --output-dir {PAIR_DIR}/baseline_no_noise \
  --seed 2026

!python scripts/11_certify.py \
  --checkpoint {PAIR_DIR}/baseline_no_noise/best.pt \
  --sigma 1.00 \
  --noise-mode joint \
  --cache-dir {PAIR_DIR}/cmar_cache \
  --output {PAIR_DIR}/baseline_no_noise_cert.json \
  --n0 100 \
  --n 1000 \
  --alpha 0.001
```

## Cell 9: Summarize

```python
!python scripts/23_phase2_summarize.py \
  --phase2-dir {PHASE2} \
  --output-dir {PHASE2}/summary
```
