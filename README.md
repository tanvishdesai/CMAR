# CertAV: Certified Adversarial Robustness for Audio-Visual Deepfake Detection

**Multimodal Randomized Smoothing for Provable Robustness Guarantees**

## Overview

CertAV extends randomized smoothing (Cohen et al., 2019) to the joint audio-visual feature space for deepfake detection. By wrapping a cross-modal attention-based detector (CMAR architecture) inside a smoothed classifier, we provide **provable ℓ₂ robustness certificates** — mathematical guarantees that the detection remains correct under bounded adversarial perturbations.

### Key Contributions

1. **First certified AV deepfake detector**: Extends randomized smoothing to joint visual + audio feature spaces
2. **Multimodal noise augmented training**: Gaussian noise in the joint feature space exploits cross-modal redundancy
3. **Certified accuracy benchmarks**: Systematic measurement on FakeAVCeleb at multiple ℓ₂ radii
4. **Empirical + certified comparison**: Certified defense vs empirical attacks on the same architecture

## Architecture

```
Input: Cached DINOv2 visual features + Whisper audio features
         │
    ┌────┴─────┐
    │  Add N(0,σ²I) noise  │  × N Monte Carlo samples
    └────┬─────┘
         │
    ┌────┴─────┐
    │  Base CMAR Classifier  │
    │  (VisualTempAgg + AudioTempAgg + CMCM + Head)  │
    └────┬─────┘
         │
    ┌────┴─────┐
    │  Majority Vote → Certified Radius R = σ · Φ⁻¹(pA)  │
    └──────────┘
```

## Project Structure

```
CMAR/
├── cmar/                           # Core package
│   ├── certification/              # 🆕 Randomized smoothing
│   │   ├── core.py                 # Statistical utilities, Clopper-Pearson bounds
│   │   └── smoothing.py            # SmoothedClassifier wrapper
│   ├── models/                     # CMAR architecture (base classifier)
│   │   ├── cmar.py                 # Full CMAR model
│   │   ├── cmcm.py                 # Cross-Modal Consistency Module
│   │   ├── temporal_aggregation.py # Visual/Audio temporal aggregators
│   │   └── classifier.py          # Classification head
│   ├── training/
│   │   ├── dataset.py              # CachedAVDataset
│   │   ├── noise_augmented_trainer.py  # 🆕 Gaussian noise training loop
│   │   ├── trainer.py              # Original CMAR trainer
│   │   └── losses.py               # Loss functions
│   ├── evaluation/                 # Metrics and attacks
│   │   ├── metrics.py              # AUC, EER, AP, certification metrics
│   │   └── attacks.py              # Feature-space PGD/FGSM
│   └── utils/                      # I/O, caching, visualization
├── scripts/
│   ├── 00_environment_check.py     # Hardware and dependency check
│   ├── 01_preprocess_features.py   # DINOv2/Whisper feature extraction
│   ├── 02_train_cmar.py            # Original CMAR training
│   ├── 03_evaluate_clean_degraded.py
│   ├── 05_adversarial_evaluation.py
│   ├── 10_train_certav.py          # 🆕 Train with Gaussian noise augmentation
│   ├── 11_certify.py               # 🆕 Run certification procedure
│   ├── 12_empirical_attack_comparison.py  # 🆕 Smoothed vs base under PGD
│   └── 13_certav_figures.py        # 🆕 Generate all paper figures
├── configs/                        # YAML configurations
├── docs/                           # Documentation
├── paper/                          # LaTeX paper files
└── archive/                        # Old CMAR-specific files
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (T4 16GB or better)
- Cached DINOv2 + Whisper features (from preprocessing step)

### Installation

```bash
pip install -r requirements.txt
```

### Execution Sequence

#### Step 1: Feature Preprocessing (if not already done)
```bash
python scripts/01_preprocess_features.py \
    --dataset-root /path/to/FakeAVCeleb \
    --output-dir /path/to/cmar_cache
```

#### Step 2: Train Noise-Augmented Models
```bash
# Train at multiple noise levels
for SIGMA in 0.12 0.25 0.50 1.00; do
    python scripts/10_train_certav.py \
        --sigma $SIGMA \
        --cache-dir /path/to/cmar_cache \
        --output-dir ./certav_runs/sigma_${SIGMA}
done

# Train ablation models (visual-only and audio-only noise)
python scripts/10_train_certav.py --sigma 0.25 --noise-mode visual_only \
    --output-dir ./certav_runs/visonly_0.25
python scripts/10_train_certav.py --sigma 0.25 --noise-mode audio_only \
    --output-dir ./certav_runs/audonly_0.25
```

#### Step 3: Certify Each Model
```bash
for SIGMA in 0.12 0.25 0.50 1.00; do
    python scripts/11_certify.py \
        --checkpoint ./certav_runs/sigma_${SIGMA}/best.pt \
        --sigma $SIGMA \
        --cache-dir /path/to/cmar_cache \
        --output ./certav_runs/certav_cert_${SIGMA}.json
done
```

#### Step 4: Empirical Attack Comparison
```bash
python scripts/12_empirical_attack_comparison.py \
    --checkpoint ./certav_runs/sigma_0.25/best.pt \
    --sigma 0.25 \
    --cache-dir /path/to/cmar_cache \
    --output ./certav_runs/empirical_comparison.json
```

#### Step 5: Generate Figures
```bash
python scripts/13_certav_figures.py \
    --results-dir ./certav_runs/
```

## Key Concepts

### Randomized Smoothing
Given a base classifier f and noise level σ:
- **Smoothed classifier**: g(x) = argmax_c P[f(x + ε) = c], ε ~ N(0, σ²I)
- **Certified radius**: R = σ · Φ⁻¹(pA) where pA is the lower confidence bound on the top-class probability
- **Guarantee**: g(x + δ) = g(x) for all ‖δ‖₂ ≤ R

### Multimodal Extension
We add independent Gaussian noise to both visual and audio features:
- Visual: x_v' = x_v + ε_v, ε_v ~ N(0, σ²I)
- Audio: x_a' = x_a + ε_a, ε_a ~ N(0, σ²I)

The certified radius applies to the concatenated feature space.

## Citation

```bibtex
@article{certav2026,
    title={CertAV: Certified Adversarial Robustness for Audio-Visual Deepfake Detection via Multimodal Randomized Smoothing},
    year={2026}
}
```

## License

Research use only.
