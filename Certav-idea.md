# CertAV Implementation Walkthrough

## What Was Done

### New Code Written

| File | Purpose | Lines |
|:---|:---|:---|
| [core.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/certification/core.py) | Statistical utilities: Clopper-Pearson bounds, certified radius, certified accuracy metrics | ~110 |
| [smoothing.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/certification/smoothing.py) | `SmoothedClassifier` — wraps CMAR with Monte Carlo noise sampling, two-phase certification | ~215 |
| [__init__.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/certification/__init__.py) | Certification package init | 6 |
| [noise_augmented_trainer.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/training/noise_augmented_trainer.py) | Training loop with Gaussian noise injection, noise-averaged validation | ~290 |
| [10_train_certav.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/10_train_certav.py) | Training script for noise-augmented models | ~140 |
| [11_certify.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/11_certify.py) | Certification script with full result output | ~180 |
| [12_empirical_attack_comparison.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/12_empirical_attack_comparison.py) | PGD attack comparison: base vs smoothed classifier | ~250 |
| [13_certav_figures.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/13_certav_figures.py) | All paper figures: certified accuracy curves, multimodal vs unimodal, tradeoffs | ~270 |

### Modified Files

| File | Change |
|:---|:---|
| [config.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/config.py) | Added `SmoothingConfig` dataclass |
| [__init__.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/cmar/__init__.py) | Added `SmoothingConfig` export, updated docstring |
| [requirements.txt](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/requirements.txt) | Cleaned up, organized into sections |
| [README.md](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/README.md) | Complete rewrite for CertAV project |

### Archived (moved to `archive/`)

| File | Reason |
|:---|:---|
| `CONTEXT.md`, `PLANNING.md`, `TASK.md` | Old CMAR project docs |
| `cmar_runs/` | Old V1/V2 results |
| `04_evaluate_baselines.py` | CMAR-specific, not needed for CertAV |
| `06_ablations.py` | Replaced by CertAV ablation training |
| `07_analysis_figures.py` | Replaced by `13_certav_figures.py` |
| `08_claim_audit.py` | CMAR-specific audit |
| `09_evaluate_ablations.py` | CMAR-specific ablation eval |

### Kept As-Is (Reused by CertAV)

| File | Why Kept |
|:---|:---|
| `00_environment_check.py` | Still useful for hardware checks |
| `01_preprocess_features.py`, `01_preprocess_auto.py` | Feature extraction pipeline is reused |
| `02_train_cmar.py` | Can serve as baseline (no-noise) model training |
| `03_evaluate_clean_degraded.py` | Degradation evaluation still relevant |
| `05_adversarial_evaluation.py` | Feature-space attacks used for comparison |
| All `cmar/models/` files | Base classifier architecture is unchanged |
| All `cmar/evaluation/` files | Metrics and attack code reused |
| All `cmar/utils/` files | Caching, I/O, visualization reused |
| `cmar/training/dataset.py` | Dataset loading is unchanged |

---

## Final Repository Structure

```
CMAR/
├── cmar/
│   ├── __init__.py                    # Updated exports
│   ├── config.py                      # + SmoothingConfig
│   ├── certification/                 # 🆕 NEW PACKAGE
│   │   ├── __init__.py
│   │   ├── core.py                    # Clopper-Pearson, certified radius
│   │   └── smoothing.py              # SmoothedClassifier
│   ├── models/
│   │   ├── cmar.py                    # Base CMAR classifier (unchanged)
│   │   ├── cmcm.py                    # Cross-modal consistency (unchanged)
│   │   ├── temporal_aggregation.py    # Temporal aggregation (unchanged)
│   │   ├── classifier.py             # Classification head (unchanged)
│   │   └── ...
│   ├── training/
│   │   ├── dataset.py                 # CachedAVDataset (unchanged)
│   │   ├── noise_augmented_trainer.py # 🆕 Gaussian noise training
│   │   ├── trainer.py                 # Original trainer (kept for baseline)
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics.py                 # AUC, EER, AP (unchanged)
│   │   ├── attacks.py                 # PGD/FGSM (unchanged)
│   │   └── ...
│   └── utils/                         # I/O, cache, seed (unchanged)
├── scripts/
│   ├── 00_environment_check.py        # Hardware check
│   ├── 01_preprocess_features.py      # Feature extraction
│   ├── 01_preprocess_auto.py          # Auto preprocessing
│   ├── 02_train_cmar.py               # Original training (baseline)
│   ├── 03_evaluate_clean_degraded.py  # Degradation eval
│   ├── 05_adversarial_evaluation.py   # Feature attack eval
│   ├── 10_train_certav.py             # 🆕 CertAV training
│   ├── 11_certify.py                  # 🆕 Certification
│   ├── 12_empirical_attack_comparison.py # 🆕 Attack comparison
│   ├── 13_certav_figures.py           # 🆕 Paper figures
│   └── common.py
├── archive/                           # Old CMAR-specific files
├── docs/
├── configs/
├── paper/
├── README.md                          # Updated for CertAV
├── requirements.txt                   # Cleaned up
└── pyproject.toml
```

---

## Complete Execution Guide

> [!IMPORTANT]
> You need the **same cached DINOv2 + Whisper features** that were used for the original CMAR training. If you already have `cmar_cache/` with the `manifests/`, `visual/`, and `audio/` directories, you're good to go. If not, run Step 0 first.

### Step 0: Feature Preprocessing (Skip if cache already exists)

**Where:** Kaggle (needs the FakeAVCeleb dataset)

```python
# In a Kaggle notebook cell:
import subprocess, os
os.chdir('/kaggle/working')
# Clone or upload your repo
# Then run:
!python scripts/01_preprocess_features.py \
    --dataset-root /kaggle/input/fake-av-celeb \
    --output-dir /kaggle/working/cmar_cache
```

**Output:** `cmar_cache/` with `manifests/{train,val,test}.csv`, `visual/`, `audio/`

---

### Step 1: Train Noise-Augmented Models

**Where:** Kaggle or Colab (GPU needed, ~1-2h per sigma)

**What this does:** Trains the same CMAR architecture (temporal aggregation + CMCM cross-attention + classifier) but with Gaussian noise injected into the features during every training forward pass. This makes the model work well under the noise that the smoothed classifier adds during certification.

#### 1a. Train at multiple σ values (main experiment)

```bash
# σ = 0.12 (low noise, high clean accuracy, small certified radius)
python scripts/10_train_certav.py \
    --sigma 0.12 \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/sigma_0.12 \
    --epochs 30 --batch-size 8 --grad-accum 4 --patience 7

# σ = 0.25 (moderate noise, balanced tradeoff)
python scripts/10_train_certav.py \
    --sigma 0.25 \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/sigma_0.25 \
    --epochs 30 --batch-size 8 --grad-accum 4 --patience 7

# σ = 0.50 (higher noise, larger certified radius)
python scripts/10_train_certav.py \
    --sigma 0.50 \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/sigma_0.50 \
    --epochs 30 --batch-size 8 --grad-accum 4 --patience 7

# σ = 1.00 (high noise, largest radius, lower accuracy)
python scripts/10_train_certav.py \
    --sigma 1.00 \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/sigma_1.00 \
    --epochs 30 --batch-size 8 --grad-accum 4 --patience 7
```

#### 1b. Train ablation models (multimodal vs unimodal noise)

```bash
# Visual-only noise (σ_v=0.25, σ_a=0)
python scripts/10_train_certav.py \
    --sigma 0.25 --noise-mode visual_only \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/visonly_0.25 \
    --epochs 30

# Audio-only noise (σ_v=0, σ_a=0.25)
python scripts/10_train_certav.py \
    --sigma 0.25 --noise-mode audio_only \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output-dir /kaggle/working/certav/audonly_0.25 \
    --epochs 30
```

**Expected output per run:** `best.pt`, `training_log.csv`, `best_metrics.json`, `train_config.json`

> [!TIP]
> **Parallelization:** Each sigma trains independently. Run 4 Kaggle notebooks in parallel (one per σ) to finish all training in ~2 hours total instead of ~8 hours sequential.

---

### Step 2: Certify Each Model

**Where:** Kaggle or Colab (GPU recommended, ~30-60min per model)

**What this does:** For each test sample, the SmoothedClassifier runs 1,100 noisy forward passes (100 for prediction + 1000 for certification), estimates the class probability with a Clopper-Pearson confidence bound, and computes the certified ℓ₂ radius.

```bash
# Certify all σ variants
for SIGMA in 0.12 0.25 0.50 1.00; do
    python scripts/11_certify.py \
        --checkpoint /kaggle/working/certav/sigma_${SIGMA}/best.pt \
        --sigma $SIGMA \
        --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
        --output /kaggle/working/certav/certav_cert_${SIGMA}.json \
        --n0 100 --n 1000 --alpha 0.001 --batch-size 64
done

# Certify ablation models
python scripts/11_certify.py \
    --checkpoint /kaggle/working/certav/visonly_0.25/best.pt \
    --sigma 0.25 --noise-mode visual_only \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output /kaggle/working/certav/certav_cert_0.25_visual_only.json

python scripts/11_certify.py \
    --checkpoint /kaggle/working/certav/audonly_0.25/best.pt \
    --sigma 0.25 --noise-mode audio_only \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output /kaggle/working/certav/certav_cert_0.25_audio_only.json
```

**Expected output:** JSON files with:
- `summary`: accuracy, abstain rate, mean/median/max certified radius
- `certified_accuracy_at_radii`: certified acc at r = 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5
- `certified_accuracy_curve`: full curve for plotting
- `per_sample_results`: per-sample predictions, radii, and confidence

> [!NOTE]
> **Quick test mode:** Add `--max-samples 50 --n0 50 --n 200` to speed up certification for debugging. Full certification on 825 test samples at n=1000 takes ~30-60 min on T4.

---

### Step 3: Empirical Attack Comparison

**Where:** Kaggle or Colab (GPU required, ~20-30min)

**What this does:** Runs PGD attacks at ε ∈ {0.05, 0.10, 0.20} against both the bare CMAR model and the smoothed classifier. Shows that smoothing provides practical defense — not just theoretical certificates.

```bash
python scripts/12_empirical_attack_comparison.py \
    --checkpoint /kaggle/working/certav/sigma_0.25/best.pt \
    --sigma 0.25 \
    --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
    --output /kaggle/working/certav/empirical_comparison.json \
    --eps-values 0.05 0.10 0.20 \
    --max-samples 200 \
    --n-smoothing-samples 100
```

**Expected output:** JSON with base classifier AUC and smoothed classifier accuracy at each ε.

---

### Step 4: Generate Paper Figures

**Where:** Kaggle, Colab, or locally (no GPU needed)

```bash
python scripts/13_certav_figures.py \
    --results-dir /kaggle/working/certav/ \
    --output-dir /kaggle/working/certav/figures/
```

**Expected output:**
- `fig1_certified_accuracy_curves.{png,pdf}` — Main result: certified acc vs radius at different σ
- `fig2_multimodal_vs_unimodal.{png,pdf}` — Joint vs visual-only vs audio-only
- `fig3_accuracy_radius_tradeoff.{png,pdf}` — Clean accuracy vs mean certified radius
- `fig4_empirical_attack_comparison.{png,pdf}` — Base vs smoothed under PGD
- `table_certified_accuracy.json` — Summary table for the paper

---

### Step 5 (Optional): Degradation Robustness of Smoothed Classifier

**Where:** Kaggle (needs degraded features in cache)

```bash
# Run certification on degraded test conditions
for COND in d12_social d11_h264_crf28 d1_jpeg75; do
    python scripts/11_certify.py \
        --checkpoint /kaggle/working/certav/sigma_0.25/best.pt \
        --sigma 0.25 \
        --cache-dir /kaggle/input/cmar-features-clean-v1/cmar_cache \
        --condition $COND \
        --output /kaggle/working/certav/certav_cert_0.25_${COND}.json
done
```

---

## Execution Summary Table

| Step | Script | Time (T4 GPU) | Parallelizable? |
|:---|:---|:---|:---|
| 1a | `10_train_certav.py` × 4 sigmas | ~2h each | ✅ Yes (4 notebooks) |
| 1b | `10_train_certav.py` × 2 ablations | ~2h each | ✅ Yes |
| 2 | `11_certify.py` × 4+2 models | ~45min each | ✅ Yes |
| 3 | `12_empirical_attack_comparison.py` | ~30min | No |
| 4 | `13_certav_figures.py` | ~30s | N/A |

**Total wall-clock time:**
- Sequential: ~20 hours
- With 4 parallel notebooks: ~5-6 hours

---

## What to Expect in the Results

### Good Outcome (Paper-Worthy)
- σ=0.25: Clean accuracy ~75-82%, certified acc@0.25 > 60%, mean radius > 0.2
- Joint noise beats visual-only and audio-only at same σ (proves multimodal advantage)
- Smoothed classifier resists PGD attacks that destroy the base classifier

### Acceptable Outcome (Needs Reframing)
- Clean accuracy drops below 70% but certified radii are meaningful
- Would frame as: "certification is achievable but with meaningful accuracy cost"

### Concerning Outcome (May Need σ Tuning)
- All models abstain on >50% of samples → σ too high, reduce to 0.10
- Certified radii all near 0 → model not confident enough under noise

> [!TIP]
> If the initial σ=0.25 results look poor, try σ=0.15 or σ=0.10 as an intermediate value. The accuracy-radius tradeoff is the core finding — you want at least 3 points on the curve to tell a story.
