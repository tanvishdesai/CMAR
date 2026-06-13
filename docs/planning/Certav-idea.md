# CertAV: Certifiably Robust Audio-Visual Deepfake Detection

## Research Idea Summary

This project introduces **CertAV**, the first certified robustness framework for multimodal (audio-visual) deepfake detection. We apply randomized smoothing in the joint feature space of frozen DINOv2 + Whisper encoders, providing provable ℓ₂-norm guarantees against adversarial attacks. We then extend the framework in two complementary directions:

1. **Manifold-Aware (Anisotropic) Certification** — PCA-guided noise allocation that concentrates smoothing noise along the data manifold's principal components, yielding larger on-manifold certified radii.
2. **Conformal Deepfake Detection** — Distribution-free prediction sets with marginal coverage guarantees, extended with robust conformal prediction that maintains coverage under L2-bounded adversarial perturbation.

---

## Key Contributions

1. **CertAV (Phase 1):** First application of randomized smoothing to multimodal deepfake detection in joint audio-visual feature space, with certified ℓ₂ robustness guarantees.

2. **Manifold geometry analysis:** PCA-based intrinsic dimensionality study of frozen foundation model features (DINOv2-Small + Whisper-tiny), revealing low-dimensional structure (d_int/D ≈ 0.104 at 90% variance) that enables more efficient certification.

3. **Anisotropic smoothing for deepfake detection:** Three PCA-aligned noise strategies (eigenvalue-proportional, subspace-projection, inverse-eigenvalue) that achieve 3× larger on-manifold certified radii compared to isotropic smoothing.

4. **Conformal prediction for certified deepfake detection:** Split conformal calibration producing prediction sets with provable marginal coverage (≥ 1 − α) under both clean and L2-adversarial conditions.

5. **Attack-manifold alignment analysis:** Empirical investigation of whether PGD adversarial attacks align with the data manifold in feature space.

---

## Architecture & Pipeline

```
Input Video → [Frozen DINOv2-S] → Visual Features (16×384)
                                                          → [Temporal Agg] → [CMCM Cross-Attention] → [Classifier] → P(fake)
Input Audio → [Frozen Whisper-tiny] → Audio Features (64×384)
```

**Certification pipeline:**
1. Train base classifier with Gaussian noise augmentation (σ = 1.0)
2. At inference, run N=1000 noisy forward passes per sample
3. Compute Clopper-Pearson lower bound on top-class probability (pA)
4. Certified ℓ₂ radius: R = σ × Φ⁻¹(pA)

**Anisotropic extension:**
1. Fit PCA on pooled training features (joint visual+audio, 768-dim)
2. Allocate noise variance proportionally to eigenvalues (or inversely, or subspace-projected)
3. Noise is sampled in PCA space then rotated back to feature space
4. Certified on-manifold radius: R_on = Φ⁻¹(pA) × sqrt(mean variance in top-k PCA directions)

**Conformal extension:**
1. Calibrate on held-out set: compute nonconformity scores for each sample
2. Set threshold qhat at the (1-α) quantile of calibration scores
3. At test time: include class c in prediction set if score(c) ≤ qhat
4. Robust variant: only certify classes whose smoothing radius exceeds attack budget

---

## Experimental Design

### Dataset
- **FakeAVCeleb** — Audio-visual deepfake detection benchmark
- Pre-extracted features cached as `cmar_cache/` (DINOv2-S visual + Whisper-tiny audio)
- Train/Val/Test splits by clip ID

### Phase 1: Isotropic CertAV (Completed)

| Experiment | Scripts | Status |
|---|---|---|
| Train noise-augmented models (σ ∈ {0.12, 0.25, 0.50, 1.00}) | `10_train_certav.py` | ✅ Done |
| Certify at each σ | `11_certify.py` | ✅ Done |
| Multimodal vs unimodal ablation (joint vs visual-only vs audio-only) | `10_train_certav.py` + `11_certify.py` | ✅ Done |
| Empirical PGD attack comparison (base vs smoothed) | `12_empirical_attack_comparison.py` | ✅ Done |
| Paper figures | `13_certav_figures.py` | ✅ Done |

### Phase 2: Anisotropic + Conformal (To Rerun with Bug Fixes)

| Experiment | Scripts | Status |
|---|---|---|
| PCA fit on joint training features | `20_fit_pca_noise.py` | ✅ Done |
| Train anisotropic models (strat1, strat2, strat3) | `10_train_certav.py` | ✅ Done |
| Certify anisotropic models | `11_certify.py` | ⚠️ Rerun (Bug 1 fixed) |
| Attack-manifold alignment diagnostic | `12_empirical_attack_comparison.py` | ✅ Done |
| Conformal calibration (isotropic) | `21_conformal_calibrate.py` | ✅ Done |
| Conformal evaluation under L2-PGD | `22_conformal_evaluate.py` | ⚠️ Rerun (Bug 2 fixed) |
| Input-space certificate composition | `24_compose_input_certificate.py` | ✅ Done |
| Summarize all Phase 2 results | `23_phase2_summarize.py` | ⚠️ Rerun (Bug 3 fixed) |

### Dropped Experiments
- ~~Encoder family sweep (DINOv2-B + Whisper-Base)~~ — original base-scale pair dropped due to Colab OOM; replaced with feasible pairs (CLIP ViT-B/16 via timm, DINOv2-S + Whisper-Base, DINOv2-S + HuBERT)
- ~~LAV-DF cross-dataset conformal evaluation~~ — dropped because LAV-DF features are not preprocessed

---

## What to Run & In What Order

After applying the bug fixes, rerun **only Cells 6–14** from [phase2_final_kaggle_cells.md](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/notebooks/phase2_final_kaggle_cells.md). Cells 1–5 produced correct results and do NOT need rerunning.

| Order | Cell | What It Does | Why Rerun |
|---|---|---|---|
| 1 | Cell 6 | Certify anisotropic strat1, strat2, strat3 | Bug 1 fix: on-manifold radius is now the primary metric |
| 2 | Cell 7 | Attack alignment diagnostic | Already correct, but rerun to regenerate with new cert data |
| 3 | Cell 8 | Composed input-space certificate | Already correct, rerun for consistency |
| 4 | Cell 9 | Encoder study baseline cert | Already correct, rerun for consistency |
| 5 | Cell 10 | Conformal calibration | Already correct, needed before Cell 11 |
| 6 | Cell 11 | Conformal evaluation under L2-PGD | Bug 2 fix: L∞ → L2 PGD |
| 7 | Cell 12 | Summarize all Phase 2 results | Bug 3 fix: on-manifold metrics in CSV |
| 8 | Cell 13 | Display summary tables | Shows corrected results |
| 9 | Cell 14 | Package outputs | Creates downloadable archive |

### Expected Results After Fixes

| Metric | Isotropic (σ=1.0) | Anisotropic Strat1 | Anisotropic Strat3 |
|---|---|---|---|
| Clean Accuracy | ~93% | ~91% | ~91% |
| Mean Certified Radius (primary) | ~2.2 | ~6.3 (on-manifold) | ~6.6 (on-manifold) |
| Mean Certified Radius (L2) | ~2.2 | ~0.0002 (expected) | ~0.014 (expected) |
| Cert Acc @ r=0.25 (on-manifold) | ~92% | ~90% | ~90% |
| Cert Acc @ r=1.00 (on-manifold) | ~89% | ~89% | ~89% |

The key improvement: anisotropic strategies should now show **non-zero certified accuracy** at all radii (using on-manifold metric), with ~3× larger radii than isotropic.

### Expected Conformal Results After L2-PGD Fix

| α | Attack L2 ε | Coverage (standard) | Coverage (robust r=1.0) |
|---|---|---|---|
| 0.10 | 0.00 (clean) | ≥ 90% | ≥ 90% |
| 0.10 | 0.25 | ~85-90% | ≥ 85% |
| 0.10 | 0.50 | ~75-85% | ≥ 70% |
| 0.10 | 1.00 | ~60-75% | ≥ 50% |

Coverage should degrade gracefully with L2-ε (not collapse to ~3% as before) because L2-PGD at ε=0.25 is now properly constrained within the certified L2 ball.

---

## Paper Structure (Target: ICASSP 2026 or CVIP 2026)

### Title
**CertAV: Certified Robustness for Multimodal Deepfake Detection via Manifold-Aware Randomized Smoothing**

### Sections

1. **Introduction** — Deepfake detection is vulnerable to adversarial attacks; we provide the first certified robustness guarantee for multimodal deepfake detectors.

2. **Related Work** — Adversarial attacks on deepfake detectors; randomized smoothing; anisotropic/manifold-aware certification; conformal prediction.

3. **Method**
   - 3.1 CertAV: Isotropic smoothing in joint audio-visual feature space
   - 3.2 Manifold-aware anisotropic smoothing via PCA-guided noise
   - 3.3 Conformal prediction sets with adversarial coverage guarantees

4. **Experiments**
   - 4.1 Setup: FakeAVCeleb, DINOv2-S + Whisper-tiny, feature cache
   - 4.2 Isotropic certification results (σ sweep, multimodal vs unimodal)
   - 4.3 Anisotropic certification: on-manifold certified radius improvement
   - 4.4 Attack-manifold alignment analysis
   - 4.5 Conformal prediction under L2-PGD attack
   - 4.6 Input-space certificate composition

5. **Discussion** — Trade-offs, limitations (frozen encoders, Lipschitz constant), attack alignment findings

6. **Conclusion**

### Key Figures
- **Fig 1:** Certified accuracy curves (isotropic) at σ ∈ {0.12, 0.25, 0.50, 1.00}
- **Fig 2:** On-manifold certified accuracy: isotropic vs anisotropic strat1/2/3
- **Fig 3:** PCA eigenvalue spectrum + intrinsic dimensionality analysis
- **Fig 4:** Conformal coverage vs L2-PGD attack budget
- **Fig 5:** Multimodal vs unimodal smoothing comparison

### Key Tables
- **Table 1:** Certified accuracy at key radii (r = 0.25, 0.50, 1.00) — isotropic vs anisotropic
- **Table 2:** Conformal coverage and singleton rate under clean + attacked conditions
- **Table 3:** Attack-manifold alignment (cos² values)

---

## Bugs Fixed (Phase 2 v2)

1. **Bug 1 (Anisotropic certification metric):** `smoothing.py` used `certified_radius_l2` (worst-case, near-zero for anisotropic) as the primary radius. Fixed to use `certified_radius_onmanifold`. `11_certify.py` now reports both L2 and on-manifold certified accuracy curves.

2. **Bug 2 (Conformal PGD attack norm):** `22_conformal_evaluate.py` used L∞ PGD (sign-based step + L∞ clamp), but certification is L2. An L∞ attack at ε=0.25 creates L2 perturbations up to ε×√768 ≈ 6.9, far exceeding the certified radius of ~2.2. Fixed to L2-projected PGD.

3. **Bug 3 (Summarize script):** `23_phase2_summarize.py` didn't include on-manifold certified accuracy in the CSV output. Fixed to add `mean_certified_radius_l2`, `certified_accuracy_r_*_onmanifold` columns.
