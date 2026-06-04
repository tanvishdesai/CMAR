# CertAV Elevation Plan — Priority 1, 2 & 3

## Goal

Elevate CertAV from ICASSP-ready (40-55%) to CVPR-viable (25-35%) by adding baselines, cross-dataset validation, input-space attacks, and designing a genuine community benchmark.

---

## Proposed Changes

### Priority 1: Strengthen Core Claims

---

#### 1.1 Baseline: Certify Original CMAR (No Noise Augmentation)

**Why**: Prove that noise-augmented training is essential. Without it, the smoothed classifier should abstain on most samples or have tiny certified radii.

**What**: Train the CMAR model from scratch using `10_train_certav.py` but with `--sigma 0.0` (no noise), then certify it at σ=0.25 and σ=1.00.

##### [NEW] [14_train_baseline_no_noise.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/14_train_baseline_no_noise.py)

A thin wrapper that calls `10_train_certav.py`'s logic with σ=0. The key insight: we use the same architecture/trainer but inject zero noise. Then we certify the resulting model with the standard smoothing procedure to show it fails.

##### Changes to existing code: None required
The existing `10_train_certav.py` already accepts `--sigma 0.12`. We just need σ=0.0 — but the noise-augmented trainer will just skip adding noise. We'll add a dedicated script for clarity and to produce properly labeled outputs.

---

#### 1.2 Baseline: PGD Adversarial Training

**Why**: Compare certified defense (CertAV) vs the dominant empirical defense (adversarial training). Show that AT gives empirical resilience but NO provable certificates.

**What**: Train CMAR with PGD-AT (inner loop generates adversarial examples, outer loop updates on them). Then evaluate: (a) clean AUC, (b) PGD attack AUC, (c) try to certify it — certificates should be very small.

##### [NEW] [15_train_pgd_at.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/15_train_pgd_at.py)

PGD adversarial training script. Each training batch: generate PGD adversarial features → train on adversarial + clean mix.

---

#### 1.3 Cross-Dataset: LAV-DF Certification

**Why**: Show generalization. Even if numbers drop, proving the method works at all on an unseen dataset is valuable.

**What**: 
1. Preprocess LAV-DF test set → extract DINOv2+Whisper features into a cache (same format as FakeAVCeleb)
2. Run certification using our best FakeAVCeleb-trained models on the LAV-DF features (zero-shot)

**Important**: Yes, you DO need to preprocess LAV-DF. You need to extract DINOv2+Whisper features from the raw LAV-DF videos into the same cache format. The existing `01_preprocess_features.py` already supports LAV-DF via the `--lavdf-root` flag — it calls `build_lavdf_manifest()` to discover the videos and extracts features in the same format.

##### [NEW] [16_certify_cross_dataset.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/16_certify_cross_dataset.py)

Thin wrapper around `11_certify.py` that loads from a LAV-DF cache directory instead.

---

#### 1.4 Related Work Positioning Notes

##### [NEW] [writing_notes_related_work.md](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/docs/writing_notes_related_work.md)

Markdown file with citation suggestions, differentiation points, and framing guidance for the drafting phase.

---

### Priority 2: CVPR Elevation

---

#### 2.1 Input-Space PGD Attack Pilot

**Why**: The #1 reviewer concern will be: "Your attacks operate in feature space, not on actual images/audio. Are your certificates meaningful for real attacks?" This experiment directly answers that question.

**What**: Run PGD through the frozen DINOv2+Whisper encoders on 100 samples. This requires loading the actual raw video frames and audio, running a forward pass through the encoders with gradients enabled, computing PGD in input space, then re-extracting features and checking if the certified radius holds.

**Key challenge**: DINOv2 and Whisper are large models. We need gradients through them, which means they must be loaded (not from cache) and we need to track the computation graph. This is GPU-memory intensive → we process one sample at a time.

##### [NEW] [17_input_space_attack.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/17_input_space_attack.py)

End-to-end input-space PGD: loads raw frames → DINOv2 → CMAR; computes PGD gradients through the full pipeline.

---

#### 2.2 Theoretical Analysis (Answer to your question)

**Your question**: "How do I formalize it? Is this just drafting/math or do I need to run code?"

**Answer**: It's **both** — but the code part is much smaller than you think.

**What you need to do**:

1. **Empirical verification (CODE)**: Run a simple experiment that measures the intrinsic dimensionality of the feature space and how Gaussian noise distributes within it. Specifically:
   - Sample N features from your cache
   - Add Gaussian noise at σ=1.00
   - Compute the fraction of noisy features that stay within the data manifold (measured by cosine similarity to the nearest training feature)
   - Compare this to adding noise to raw pixels (which you can also measure)
   - This takes ~30 minutes of GPU time and produces 2-3 numbers that support your hypothesis

2. **Mathematical formalization (DRAFTING)**: Based on the empirical numbers above, write a short theoretical section (half a page) that argues:
   - Frozen DINOv2/Whisper features lie on a low-dimensional manifold within the 384-d feature space
   - Gaussian noise in this space projects onto the tangent space of the manifold with high probability
   - This is unlike raw pixel space where noise is truly isotropic in a very high-dimensional space
   - The accuracy-robustness tradeoff from Cohen et al. (2019) assumes worst-case noise directions; on a structured manifold, most noise directions are "benign"

You do NOT need to prove a formal theorem. A well-argued empirical observation with supporting measurements is sufficient for CVPR. A formal proof would elevate this to ICML/NeurIPS territory but is much harder and not required.

##### [NEW] [18_manifold_analysis.py](file:///c:/Users/DELL/Desktop/code_playground/Multi-Modal/deepshield/CMAR/scripts/18_manifold_analysis.py)

Script that computes intrinsic dimensionality and noise-manifold alignment metrics.

---

### Priority 3: Community Resource / Benchmark — Refined

---

#### The Problem with the Original Proposal

You're right — "releasing pre-extracted features + certification scripts" feels forced. Anyone CAN extract features on their own laptop. The value is too thin.

#### Refined Proposal: `av-robustbench` — An Adversarial Robustness Toolkit for AV Deepfake Detectors

Inspired by **RobustBench** (image classification robustness leaderboard) and **ForensicHub** (NeurIPS 2025 D&B accepted, unified forensic benchmark), here's a genuinely useful contribution:

**Core insight**: The deepfake detection community currently has NO standardized way to measure adversarial robustness. Everyone uses ad-hoc attack code, different ε budgets, different metrics. There is no "RobustBench for deepfake detectors."

**What `av-robustbench` would be**:

```
pip install av-robustbench
```

A Python package that provides:

1. **Standardized Attack Suite** (`av_robustbench.attacks`)
   - Feature-space PGD (L∞ and L₂)
   - Input-space PGD through frozen encoders
   - AutoAttack-style ensemble (APGD-CE + Square Attack adapted for AV)
   - Black-box transfer attacks from unimodal detectors
   
2. **Certification Engine** (`av_robustbench.certification`)
   - `SmoothedAVClassifier`: wraps ANY audio-visual model in a smoothed classifier
   - Multi-sigma certification with automatic curve generation
   - Supports arbitrary feature extractors (not just DINOv2/Whisper)
   - Output: standardized JSON with certified accuracy curves

3. **Degradation Battery** (`av_robustbench.degradations`)
   - 12 standardized degradation conditions (JPEG, H.264, social media simulation, etc.)
   - Consistent application pipeline
   - RAR (Robustness-Accuracy Ratio) computation

4. **Model Zoo** (`av_robustbench.models`)
   - Your CertAV as the first entry
   - Adapter interfaces for: CMAR, AVoiD-DF, LipForensics, AASIST, XceptionNet
   - Pre-trained checkpoints on HuggingFace

5. **Evaluation Protocol** (`av_robustbench.evaluate`)
   - Single command: `av-robustbench evaluate --model cmar --dataset fakeavceleb --attacks all`
   - Generates a standardized "robustness card" (like a model card but for adversarial robustness)
   - Leaderboard-compatible JSON output

6. **Leaderboard** (GitHub Pages / HuggingFace Spaces)
   - Public leaderboard comparing models on certified accuracy, empirical attack accuracy, and degradation robustness
   - Community can submit their own models

**Why this is genuinely valuable to the community**:
- **ForensicHub** (NeurIPS 2025 accepted) proved there's appetite for unified benchmarking in image forensics — but it covers NONE of the adversarial robustness or certification aspects
- **RobustBench** is the gold standard for image classification robustness but doesn't support multimodal or video
- **Nobody has combined AV deepfake detection + certified robustness + empirical attacks + degradations into one toolkit**
- Researchers spend weeks reimplementing attack code — this removes that friction
- The certification engine alone (wrap any model in a smoothed classifier) saves 100+ hours per research group

**Publication venue**: This would be a separate paper from CertAV, targeting:
- **NeurIPS 2027 Evaluations & Datasets Track** (the new name)
- **ACM MM 2027 Grand Challenge**
- Or just release it open-source alongside the CertAV paper and cite it

> [!IMPORTANT]
> **This is a significant engineering effort** (4-8 weeks). I recommend building it AFTER the CertAV paper is submitted to ICASSP (Sep 16). The toolkit can then reference the CertAV paper as its first benchmark entry.

---

## New Files Summary

| File | Type | Purpose |
|:---|:---|:---|
| `scripts/14_train_baseline_no_noise.py` | Python | Train CMAR without noise augmentation (baseline) |
| `scripts/15_train_pgd_at.py` | Python | PGD adversarial training baseline |
| `scripts/16_certify_cross_dataset.py` | Python | Certify on LAV-DF (cross-dataset) |
| `scripts/17_input_space_attack.py` | Python | Input-space PGD through DINOv2+Whisper |
| `scripts/18_manifold_analysis.py` | Python | Intrinsic dimensionality & manifold analysis |
| `docs/writing_notes_related_work.md` | Markdown | Citation & framing notes for drafting phase |
| `notebooks/elevation_experiment_notebook.md` | Markdown | Kaggle notebook with all cells for Priority 1+2 experiments |

---

## Execution Order

### Phase A: Pre-requisites (1 Kaggle session, ~1 hour)

1. **Preprocess LAV-DF features** — Extract DINOv2+Whisper features from LAV-DF test set
   - Use existing `01_preprocess_features.py` with `--lavdf-root` flag
   - Save as a Kaggle dataset (`lavdf-features-v1`)

### Phase B: Priority 1 Experiments (2-3 Kaggle sessions, ~6 hours total)

Run in a **single notebook** with the following cell order:

2. **Cell 1**: Configuration (paths, seeds)
3. **Cell 2**: Train baseline (no noise) — script 14 — ~30 min
4. **Cell 3**: Certify baseline at σ=0.25 and σ=1.00 — script 11 — ~10 min
5. **Cell 4**: Train PGD-AT baseline — script 15 — ~45 min
6. **Cell 5**: Evaluate PGD-AT (clean AUC + attacked AUC) — script 12 — ~20 min
7. **Cell 6**: Certify PGD-AT at σ=1.00 — script 11 — ~5 min
8. **Cell 7**: Certify on LAV-DF (σ=0.25, 0.50, 1.00) — script 16 — ~15 min

### Phase C: Priority 2 Experiments (1-2 Kaggle sessions, ~3 hours)

9. **Cell 8**: Input-space PGD attack pilot (100 samples) — script 17 — ~2 hours
10. **Cell 9**: Manifold analysis — script 18 — ~30 min

### Phase D: Aggregation (1 session, ~30 min)

11. **Cell 10**: Aggregate all new results into the master JSON
12. **Cell 11**: Generate updated figures

---

## Open Questions

> [!IMPORTANT]
> 1. **LAV-DF availability on Kaggle**: Do you have access to the LAV-DF dataset on Kaggle? If not, we need to find an alternative second dataset (Celeb-DF, DFDC).
>
> 2. **Seeds**: For baselines (no-noise, PGD-AT), should we run all 5 seeds or just 1 seed as a proof-of-concept? Running 5 seeds is more rigorous but 5× slower (~15 hours). Recommendation: Run **1 seed (2026)** first as pilot, then expand to 5 if results look promising.
>
> 3. **Existing checkpoints**: You mentioned you have all models saved from the 5-seed runs. Where are they saved on Kaggle? I need the exact dataset paths to reference them in the notebook.

## Verification Plan

### Automated Tests
- Each script outputs a JSON with structured results
- The aggregation notebook validates that all expected files exist
- Certification scripts print progress and final summary

### Manual Verification
- Compare baseline (no noise) certified accuracy to CertAV (should be dramatically worse)
- Compare PGD-AT empirical robustness to CertAV certified robustness
- Verify LAV-DF certification produces non-trivial numbers
- Verify input-space PGD matches or is bounded by feature-space certified radii
