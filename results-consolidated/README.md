# Consolidated Results

All experimental results from Phase 1 and Phase 2 of the CertAV project, organized for paper writing.

## Directory Structure

```
results-consolidated/
├── phase1/                              # Phase 1: Isotropic CertAV
│   ├── certav_master_results.json       # Main 5-seed results (σ=0.12-1.00)
│   ├── certification_joint_aggregated.json
│   ├── certification_ablation_aggregated.json
│   ├── training_aggregated.json
│   ├── degradation_aggregated.json
│   ├── empirical_attacks_aggregated.json
│   └── manifold-analysis/
│       └── manifold_analysis.json       # PCA/intrinsic dimension analysis
│
└── phase2/                              # Phase 2: Extensions
    ├── encoder-scaling/                 # Direction D: Encoder family study
    │   ├── encoder_scaling_all_5families.csv  # ⬅ MERGED: all 5 encoder pairs
    │   ├── encoder_scaling_baseline.csv       # DINOv2-S + Whisper-tiny only
    │   ├── encoder_scaling_4families.csv      # 4 new encoder pairs
    │   └── encoder_scaling_details.txt        # Full JSON details per encoder
    │
    ├── anisotropic/                     # Direction A: Anisotropic smoothing
    │   ├── phase2_anisotropic.csv       # 3 strategies comparison
    │   ├── phase2_anisotropic_strategies.png
    │   └── pca_joint.summary.json       # PCA basis used for anisotropic noise
    │
    └── conformal/                       # Direction C: Conformal prediction
        └── phase2_conformal.csv         # 288 rows: all α × r × score × variant combos
```

## Quick Reference: Key Numbers for Paper Tables

### Phase 1 Main Result (Table 1)
- Source: `phase1/certav_master_results.json`
- σ=1.00: 92.5±0.6% clean acc, 88.0% cert@r=1.0, mean R=2.215±0.032

### Encoder Scaling (Table 2)
- Source: `phase2/encoder-scaling/encoder_scaling_all_5families.csv`
- 5 encoder pairs, d_int/D range: 0.069–0.109, R_cert range: 1.948–2.207

### Anisotropic Smoothing (Table 3)
- Source: `phase2/anisotropic/phase2_anisotropic.csv`
- Strat 2 headline: R_manifold=7.632, cert@1.0=90.9%, abstain=0.0%

### Conformal Prediction (Table 4)
- Source: `phase2/conformal/phase2_conformal.csv`
- Primary: α=0.10, r=0, raw score, isotropic clean: coverage=96.8%, singleton=92.4%
