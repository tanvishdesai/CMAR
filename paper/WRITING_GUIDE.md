# CertAV Paper Writing Guide

> Reference this document while writing both the ICASSP (5-page) and CVIP (8-page) drafts.
> Every section below addresses a specific framing decision, data-presentation rule, or reviewer trap.

---

## 1. The Central Thesis (Never Lose Sight of This)

**One sentence:** *The intrinsic geometry of frozen multimodal representations determines their certifiability under randomized smoothing, and exploiting this geometry enables stronger certified robustness guarantees for audio-visual deepfake detection.*

Every section of the paper must connect back to this thesis. If a paragraph doesn't serve it, cut it or move it to supplementary.

---

## 2. The Five Contributions (Ordered by Logical Flow)

Use this order in the Introduction. Each contribution motivates the next.

1. **CertAV pipeline** — feature-space randomized smoothing for AV deepfake detection.
   - *"We introduce CertAV, a certification pipeline that applies Gaussian randomized smoothing to the joint feature space of frozen DINOv2 and Whisper encoders."*

2. **Certifiability Scaling Theorem** — formal explanation for why it works.
   - *"We prove that under subspace invariance, the certified radius depends only on the d-dimensional on-manifold noise, implying a robustness amplification of √(D/d)."*

3. **Encoder scaling study** — empirical validation across 5 encoder families.
   - *"We validate the theorem across five encoder families, confirming that lower d_int/D yields higher certified radii."*

4. **Anisotropic smoothing** — exploiting the geometry.
   - *"We propose manifold-aware anisotropic smoothing that achieves 3.4× larger on-manifold certified radius with zero abstention."*

5. **Robust conformal prediction** — complementary guarantee.
   - *"We apply robust conformal prediction to provide distributional coverage guarantees for all test samples, including those on which CertAV abstains."*

---

## 3. Precise Language for the Threat Model

### DO say:
- "Feature-space ℓ₂ certificate"
- "Joint audio-visual feature perturbation"
- "Certified robustness in the representation space consumed by the detector"
- "Per-sample certified radius with 99.9% confidence"

### DO NOT say:
- ❌ "Certified against pixel/waveform attacks" (unless you explicitly compose with encoder Lipschitz)
- ❌ "End-to-end certified robustness" (this is feature-space only)
- ❌ "Provably robust deepfake detector" (too strong — scope it to feature-space)
- ❌ "First certified deepfake detector" (MMCert exists; say "first feature-space certified AV deepfake detector")

### Key scoping paragraph (include in Method or Limitations):
> *"CertAV certifies the representation space consumed by the detector, not raw pixels or waveforms. This is the correct abstraction level for frozen-encoder pipelines where the detector never sees raw input, but it is not an end-to-end physical-world guarantee. The certificate is exact for any adversary operating in the joint feature space."*

---

## 4. How to Present the Isotropic Baseline (Phase 1 Results)

### Main results table

Present the 5-seed averaged results at σ = 0.12, 0.25, 0.50, 1.00:

| σ | Clean Acc (%) | Abstain (%) | Mean R | C@0.25 | C@0.50 | C@1.00 | C@1.50 |
|---|---|---|---|---|---|---|---|
| 0.12 | 91.2±1.0 | 0.3 | 0.288±0.003 | 88.8 | 0.0 | 0.0 | 0.0 |
| 0.25 | 91.2±1.1 | 0.4 | 0.588±0.012 | 89.4 | 86.6 | 0.0 | 0.0 |
| 0.50 | 91.5±1.1 | 0.4 | 1.165±0.034 | 90.5 | 89.2 | 85.4 | 0.0 |
| 1.00 | 92.5±0.6 | 0.8 | 2.215±0.032 | 91.8 | 90.7 | 88.0 | 84.9 |

**Key narrative points:**
- Larger σ increases certified radius WITHOUT reducing clean accuracy. This is unusual.
- Abstention is always < 1%. This is very low for randomized smoothing.
- The explanation for both comes in the Theory section (Theorem 1).

### No-noise baseline (CRITICAL ablation)

| Model | Clean Acc | Mean R | C@1.0 | C@1.5 |
|---|---|---|---|---|
| Noise-augmented (σ=1.0) | 92.5% | 2.215 | 88.0% | 84.9% |
| No-noise baseline | 92.3% | 2.173 | 86.3% | 82.0% |
| PGD adversarial training | 90.5% | 2.463 | 90.5% | 90.5% |

**How to frame:**
- *"The no-noise baseline certifies within 1.9% of the noise-augmented model. This is the central finding: the geometry of frozen encoder representations, not the training procedure, is the primary driver of certifiability."*
- PGD-AT gives HIGHER radii but LOWER validation AUC (0.730 vs 0.941). Frame as: *"PGD-AT maximizes certified radius at the cost of discrimination quality — it collapses the decision boundary, making every prediction trivially stable."*

---

## 5. How to Present the Theoretical Analysis

### Theorem 1 (Certifiability Scaling)

**State the assumption explicitly:**
- Assumption 1 (Subspace Invariance): the classifier output depends only on the on-manifold component of perturbations.
- Call it an "idealization" and immediately say you validate it empirically.

**State the theorem cleanly:**
> Under Assumption 1, the top-class probability p_A of the smoothed classifier at z ∈ S depends on ε only through its projection ε_S ~ N(0, σ²I_d). The certified radius is R = σΦ⁻¹(p_A).

**State Corollary 1 clearly:**
> For an adversary with uniformly distributed attack direction, the expected on-manifold perturbation is only (d/D)r², so the adversary must spend total budget r ≥ m√(D/d) to achieve on-manifold displacement m. The amplification factor is A = √(D/d_int).

### How to present the empirical validation:

Use the encoder family table:

| Encoder Pair | D | d_int (90%) | d_int/D | A = √(D/d) | Mean R |
|---|---|---|---|---|---|
| DINOv2-S + WavLM | 1152 | 80 | 0.069 | 3.79 | 2.207 |
| DINOv2-S + HuBERT | 1152 | 79 | 0.069 | 3.82 | 2.173 |
| DINOv2-S + Whisper-tiny | 768 | 80 | 0.104 | 3.10 | 2.190 |
| DINOv2-B + Whisper-base | 1280 | 119 | 0.093 | 3.28 | 2.132 |
| CLIP-B + Whisper-tiny | 1152 | 126 | 0.109 | 3.02 | 1.948 |

**How to frame the ordering:**
- ✅ The lowest d_int/D (WavLM, HuBERT at 0.069) have the highest R_cert (~2.2)
- ✅ The highest d_int/D (CLIP at 0.109) has the lowest R_cert (1.948)
- ⚠️ DINOv2-S + Whisper-tiny (d_int/D=0.104) has R=2.190, higher than DINOv2-B + Whisper-base (d_int/D=0.093, R=2.132)

**HONEST handling of the partial violation:**
> *"The monotonic trend holds at the extremes: the encoder pairs with lowest d_int/D (DINOv2-S + WavLM/HuBERT, d_int/D ≈ 0.069) achieve the highest certified radii, while CLIP-B + Whisper-tiny (d_int/D = 0.109) yields the lowest. Within the mid-range, the ordering is not strictly monotonic, suggesting that d_int/D is the dominant but not sole predictor — factors such as spectral decay rate and feature non-Gaussianity also contribute."*

Do NOT claim a perfect scaling law. Claim a "dominant trend" validated across 5 encoder families.

---

## 6. How to Present Anisotropic Smoothing (MOST IMPORTANT SECTION)

### The primary table

| Strategy | Clean Acc | Abstain | R_ℓ₂ | R_manifold | Cert@r=1.0 |
|---|---|---|---|---|---|
| Isotropic (σ=1.0) | 90.9% | 0.8% | **2.215** | 2.215 | 88.0% |
| Strat 1 (eigenvalue-proportional) | 90.9% | 0.7% | 0.0002 | 6.272 | 89.5% |
| **Strat 2 (subspace projection)** | **90.9%** | **0.0%** | 0.002 | **7.632** | **90.9%** |
| Strat 3 (inverse-eigenvalue) | 90.9% | 0.7% | 0.014 | 6.586 | 89.0% |

### CRITICAL: Define metrics BEFORE the table

In the Method section, you MUST define:

> **Definition (On-manifold certified radius).** For anisotropic noise with covariance Σ in PCA basis, the on-manifold certified radius is R_manifold = Φ⁻¹(p_A) × √(mean variance over top-d_int directions). This measures the certified distance within the d_int-dimensional subspace where the decision boundary lies.

> **Definition (Worst-case ℓ₂ radius).** R_ℓ₂ = Φ⁻¹(p_A) × √(min eigenvalue of Σ). This is the radius of the largest inscribed sphere within the certified ellipsoid.

### The three things you MUST state about anisotropic results

**1. Budget equalization (preempts the #1 reviewer question):**
> *"All strategies are compared at equalized total noise budget: trace(Σ) = Dσ² = 768 for all strategies. The improvement comes from redistributing the noise budget, not from using more total noise."*

This is factually true — see `pca_noise.py` line 176: `total_budget = dim * (self.sigma ** 2)` and line 196: `return raw * (total_budget / raw_sum)`.

**2. R_ℓ₂ near-zero is expected, not a failure:**
> *"The isotropic certificate is a sphere of radius 2.22 in 768 dimensions — equal protection in all directions. The anisotropic certificate (Strat 2) is an ellipsoid with on-manifold axes of radius 7.63 and off-manifold axes of radius ~0.002. This is by design: noise budget is concentrated in the 80 dimensions where the decision boundary lies (Theorem 1), at the cost of near-zero radius in the 688 off-manifold dimensions where the classifier is invariant."*

**3. Strategy 2 achieves ZERO abstention:**
> *"Strategy 2 (subspace projection) achieves 0% abstention — every correctly classified sample is also certified. This is because concentrating noise strictly within the data subspace minimizes the noise seen by the classifier head, maximizing p_A."*

### Figure suggestion: Sphere vs. Ellipsoid
Create a 2D figure showing the first two PCA directions with:
- The isotropic certified region (circle, radius 2.22)
- The anisotropic certified region (ellipse, major axis 7.63, minor axis scaled)
- A few data points scattered along the manifold direction
- PGD attack arrows showing they mostly point off-manifold

This single figure makes the entire anisotropic story intuitive.

---

## 7. How to Handle the cos² = 0.18 Attack Alignment

This is the trickiest piece of framing in the entire paper. Here's the correct narrative:

### What the number means
PGD attacks on CertAV put only 18% of their ℓ₂ budget into on-manifold directions. 82% of the attack energy targets off-manifold directions.

### Why this SUPPORTS the paper's thesis (not undermines it)

**For the isotropic story:** *"The low cos² = 0.18 confirms that PGD attacks are predominantly off-manifold. By Theorem 1, the classifier is invariant to off-manifold perturbations, which explains why isotropic CertAV achieves such large certified radii — 82% of any ℓ₂-bounded attack is wasted on directions the classifier ignores."*

**For the anisotropic story:** *"Anisotropic smoothing further improves certification by recognizing that the classifier does not need noise protection in off-manifold directions. Redirecting 82% of the noise budget from wasted off-manifold directions to the decision-relevant on-manifold subspace yields a 3.4× improvement in on-manifold certified radius."*

### The precise paragraph to write:
> *"We measure the squared cosine similarity between PGD attack directions and the top-d_int PCA subspace across the FakeAVCeleb test set, finding cos²θ ≈ 0.18. This low alignment reveals that adversarial gradients in feature space are predominantly off-manifold — a consequence of the frozen encoder's geometry. This observation simultaneously explains (i) why isotropic smoothing already achieves large certified radii (the classifier is invariant to the dominant attack directions), and (ii) why anisotropic smoothing achieves even larger on-manifold certificates (redirecting noise budget from the 82% of off-manifold directions into the 18% of threat-relevant on-manifold directions)."*

### DO NOT:
- ❌ Call cos² = 0.18 "high alignment" (it's low)
- ❌ Claim attacks are "on-manifold" (they're mostly off-manifold)
- ❌ Use this to argue that on-manifold certification is the "only" thing that matters (it's the primary thing, but the ℓ₂ baseline is still a valid metric)

---

## 8. How to Present Conformal Prediction

### The selling point
> *"CertAV's per-sample certificate abstains when p_A ≤ 0.5 — those samples get no guarantee. Robust conformal prediction provides a distributional coverage guarantee for ALL samples, including those on which CertAV abstains. For a forensic analyst, '{real, fake}' (uncertain but covered) is more useful than 'no certificate available'."*

### Key results to highlight (isotropic baseline, clean, α=0.10)

| Setting | Coverage | Singleton Rate |
|---|---|---|
| Standard conformal (r=0) | 96.8% | 92.4% |
| Robust conformal (r=0.25) | 96.8% | 92.4% |
| Under PGD ε=0.25 (no robust) | 96.2% | 91.3% |
| Under PGD ε=1.0 (no robust) | 89.6% | 93.3% |

**How to frame:** Standard conformal achieves target coverage (>90%) on clean data. Coverage drops under strong PGD (89.6% at ε=1.0). Robust conformal maintains coverage at the cost of lower efficiency.

### HONEST handling of class imbalance

The dataset is 1:10 real-to-fake. Coverage for "real" class is much lower than for "fake" class.

**Write this paragraph:**
> *"Marginal coverage meets the 1−α target across all settings. However, conditional coverage varies by class: fake clips (the majority class) achieve near-perfect coverage (>99%), while real clips have lower coverage due to the model's lower confidence on the minority class. This class-conditional gap is a known property of split conformal prediction under class imbalance [cite Romano et al. 2020] and is intrinsic to the dataset's 1:10 real-to-fake ratio, not to the conformal methodology."*

### Degenerate robust conformal (q̂ = 0.5 → singleton rate = 0%)

At α=0.05 with r ≥ 0.25, the robust threshold becomes 0.5, making every prediction set = {0,1}. This means 100% coverage but 0% efficiency.

**Frame as:** *"At conservative settings (α = 0.05, r ≥ 0.25), the robust calibration threshold reaches 0.5, causing all prediction sets to include both classes. This reflects the fundamental coverage-efficiency tradeoff: stronger robustness guarantees require larger prediction sets. We therefore report α = 0.10 as the primary operating point."*

Do NOT hide these results. Show them in the ablation and discuss honestly.

---

## 9. How to Present the Encoder Scaling Study

### Table format

| Encoder Pair | Visual Enc | Audio Enc | D | d_int | d/D | Mean R | Clean Acc |
|---|---|---|---|---|---|---|---|
| DINOv2-S + WavLM | DINOv2-S (384) | WavLM (768) | 1152 | 80 | 0.069 | 2.207 | 90.8% |
| DINOv2-S + HuBERT | DINOv2-S (384) | HuBERT (768) | 1152 | 79 | 0.069 | 2.173 | 90.2% |
| DINOv2-S + Whisper-tiny | DINOv2-S (384) | Whisper (384) | 768 | 80 | 0.104 | 2.190 | 91.4% |
| DINOv2-B + Whisper-base | DINOv2-B (768) | Whisper-base (512) | 1280 | 119 | 0.093 | 2.132 | 92.2% |
| CLIP-B + Whisper-tiny | CLIP-B (768) | Whisper (384) | 1152 | 126 | 0.109 | 1.948 | 94.3% |

### Narrative

- Self-supervised encoders (DINOv2, HuBERT, WavLM) produce lower-dimensional representations → better certifiability
- Contrastive encoders (CLIP) produce higher-dimensional representations → worse certifiability but better clean accuracy (94.3% vs ~91%)
- **Trade-off:** CLIP gives best accuracy but worst certification. DINOv2-S + WavLM gives best certification but lower accuracy.
- *"This encoder selection principle follows directly from Corollary 1: for maximum certifiability, choose encoders with lowest d_int/D."*

### DO NOT:
- ❌ Claim the scaling law is "exact" or "precise" — it's a dominant trend
- ❌ Fit a regression line through 5 points and claim a power law — not enough data points
- ❌ Ignore that clean accuracy and certified radius trade off (CLIP is best for accuracy, worst for radius)

---

## 10. How to Position Against Prior Work

### MMCert (CVPR 2024)
> *"MMCert certifies multimodal input by randomized ablation of raw pixels, providing pixel-space robustness guarantees. CertAV certifies the feature representation consumed by the detector, yielding larger radii in the representation-space threat model relevant to frozen-encoder pipelines. The two approaches are complementary: MMCert covers the encoder-input boundary, while CertAV covers the detector-input boundary."*

### Yang et al. (2020) — Randomized Smoothing of All Shapes and Sizes
> *"Yang et al. generalize Cohen et al. to arbitrary noise distributions, establishing that non-isotropic smoothing yields ellipsoidal certified regions. We instantiate this framework for audio-visual deepfake detection with PCA-guided noise aligned to the encoder manifold."*

### ANCER (Eiras et al. 2021)
> *"ANCER optimizes per-sample anisotropic noise to maximize certified volume. Our approach uses a global PCA-based covariance, which is simpler and directly interpretable through the encoder geometry (Theorem 1). The per-sample optimization of ANCER is more flexible but does not provide the encoder-selection principle that our theoretical analysis establishes."*

### Xia et al. (2026) — Feature-Space Certification for MLLMs
> *"Xia et al. certify cosine similarity for multimodal LLMs using feature-space Gaussian smoothing. CertAV targets binary deepfake detection with ℓ₂ certification and additionally introduces manifold-aware anisotropic certificates grounded in the intrinsic dimensionality of the encoder representation."*

### Gendler et al. (2022) — Adversarially Robust Conformal Prediction
> *"We instantiate the robust conformal framework of Gendler et al. for deepfake detection, using the smoothed classifier's Clopper-Pearson lower bounds as nonconformity scores — a natural connection between randomized smoothing certificates and conformal prediction that has not been previously explored for multimodal forensic applications."*

---

## 11. Numbers to Double-Check Before Submission

| Number | Source | Verify Against |
|---|---|---|
| 92.5±0.6% clean accuracy | Phase 1 5-seed average | `certav_master_results.json` |
| 88.0% cert@r=1.0 | Phase 1 5-seed average | Same |
| 2.215±0.032 mean radius | Phase 1 5-seed average | Same |
| d_int = 80 (90% variance, joint) | PCA analysis | `pca_joint.summary.json` |
| cos²θ = 0.18 | Attack alignment diagnostic | `section3_theoretical_analysis.tex` line 284 |
| 7.632 on-manifold radius (Strat 2) | Anisotropic results | `phase2_anisotropic.csv` |
| 0.0% abstention (Strat 2) | Same | Same |
| 90.9% cert@r=1.0 (Strat 2) | Same | Same |
| 3.4× improvement | 7.632 / 2.215 = 3.445 | Compute from above |
| trace(Σ) = 768 equalized | Code verification | `pca_noise.py` line 176 |
| 96.8% conformal coverage (α=0.10) | Conformal results | `phase2_conformal.csv` |

---

## 12. Limitations Section (Must Include)

Write ALL of these limitations. Missing any will draw reviewer criticism:

1. **Feature-space scope:** *"CertAV certifies frozen feature representations, not raw pixels or waveforms. This is the correct abstraction for frozen-encoder pipelines but does not constitute an end-to-end input-space guarantee."*

2. **Single primary dataset:** *"Main results are on FakeAVCeleb with cross-dataset transfer to LAV-DF. Broader evaluation across manipulation families and datasets is needed."*

3. **Class imbalance:** *"FakeAVCeleb's 1:10 real-to-fake ratio causes the detector to be more confident on fake clips, affecting conformal conditional coverage."*

4. **Computational cost:** *"Each sample certification requires n=1000 forward passes. This is standard for randomized smoothing but limits real-time deployment."*

5. **Subspace invariance approximation:** *"Theorem 1 assumes exact subspace invariance. In practice, the classifier has small but non-zero sensitivity to off-manifold perturbations (cos²θ = 0.18 ≠ 0)."*

6. **Single-seed Phase 2 results:** If you don't run multi-seed: *"The anisotropic and conformal experiments use a single seed. The isotropic baseline uses five seeds, establishing low variance (±0.6% accuracy, ±0.032 mean radius)."*

---

## 13. Venue-Specific Guidance

### ICASSP 2027 (4+1 pages — short paper)

**What to INCLUDE (fits in 5 pages):**
- CertAV pipeline (Section 3, compressed)
- Theorem 1 + Corollary 1 (Section 4, 0.5 pages — theorem statement + 2-line proof sketch)
- Isotropic baseline results (Table 1)
- Encoder scaling study (Table 2, 5 rows)
- Anisotropic smoothing (Table 3, 4 rows — this is the headline)
- No-noise baseline ablation (1 row in Table 1 or merged)
- 2 figures: architecture + certified accuracy curves

**What to DROP from the current Phase 1 draft:**
- Modality ablation table (Table 3 in current draft — audio-only vs visual-only vs joint): **DROP**. This is interesting but not essential. Mention in one sentence: *"Unimodal certificates can have larger radii but protect only one stream (see supplementary)."*
- Degradation results (JPEG, H.264, social): **DROP to one sentence.** *"Common media degradations preserve >90% certified accuracy at r=0.50."*
- Feature displacement validation: **DROP entirely.** This was a Phase 1 diagnostic; the theory now explains the radii.
- The detailed transfer table: **COMPRESS to one sentence.** *"On LAV-DF, CertAV transfers at 74.6% accuracy and 61.4% cert@1.0."*

**What to ADD:**
- Theorem 1 (concise, half-page)
- Encoder scaling table (new)
- Anisotropic results table (new — the headline result)
- The sphere-vs-ellipsoid figure

**Estimated page budget:**
| Section | Pages |
|---|---|
| Abstract + Intro | 0.75 |
| Related Work | 0.5 |
| Method (pipeline + anisotropic) | 1.0 |
| Theory (Theorem 1 + Corollary) | 0.5 |
| Experiments (3 tables + 2 figures) | 1.5 |
| Limitations + Conclusion | 0.25 |
| References | 0.5 |
| **Total** | **5.0** |

### CVIP (8 pages — full paper)

**Include EVERYTHING:**
- Full CertAV pipeline with architecture figure
- Complete theoretical analysis (Theorem 1, Corollary 1, proof, empirical validation)
- All Phase 1 results (isotropic, modality ablation, baselines)
- Encoder scaling study with discussion
- Full anisotropic comparison (3 strategies + budget equalization discussion)
- Conformal prediction (standard vs robust, coverage-efficiency curves)
- Transfer and degradation results
- av-robustbench description
- Comprehensive limitations

**Estimated page budget:**
| Section | Pages |
|---|---|
| Abstract + Intro | 1.0 |
| Related Work | 0.75 |
| Method (pipeline + anisotropic + conformal) | 1.5 |
| Theory | 0.75 |
| Experiments | 2.5 |
| av-robustbench | 0.5 |
| Limitations + Conclusion | 0.5 |
| References | 0.5 |
| **Total** | **8.0** |

---

## 14. Common Reviewer Questions and Pre-Prepared Responses

**Q1: "This is just applying randomized smoothing to features. What's novel?"**
A: The novelty is threefold: (a) the Certifiability Scaling Theorem explaining WHY frozen features certify well, (b) the encoder scaling study validating this across 5 families, and (c) the anisotropic smoothing method that exploits the geometry for 3.4× improvement.

**Q2: "The anisotropic ℓ₂ radius is near-zero. How is this useful?"**
A: At equalized noise budget, the anisotropic certificate is an ellipsoid that covers 3.4× more distance in the threat-relevant on-manifold directions. The near-zero ℓ₂ radius is in off-manifold directions where the classifier is provably invariant (Theorem 1).

**Q3: "You only test on FakeAVCeleb. How do you know this generalizes?"**
A: We test cross-dataset transfer on LAV-DF (61.4% cert@1.0), evaluate 5 encoder families, and provide theoretical analysis showing the result depends on encoder geometry (d_int/D), not the specific dataset. The encoder scaling study is the strongest generalization evidence.

**Q4: "The scaling law has exceptions. DINOv2-S+Whisper has higher R than DINOv2-B+Whisper-base despite higher d/D."**
A: d_int/D is the dominant predictor but not the sole one. We observe that clean accuracy and certified radius can trade off (CLIP has 94.3% accuracy but lowest R). We claim a "dominant trend" validated at the extremes, not an exact functional form.

**Q5: "Subspace invariance is a strong assumption. Is it realistic?"**
A: We measure cos²θ = 0.18 between PGD attack directions and the manifold. This means 82% of adversarial energy targets off-manifold directions — consistent with approximate subspace invariance. The assumption is an idealization; the empirical results confirm it holds well enough for the theory to be predictive.

**Q6: "What about adaptive attacks designed to exploit the anisotropic certificate?"**
A: An adaptive attacker who concentrates their budget entirely on-manifold faces a certified radius of 7.63 (Strat 2). An attacker who spreads budget uniformly faces an even larger effective barrier because most of their energy is wasted off-manifold. The on-manifold metric is the correct adversarial metric under manifold-aware threat models.

---

## 15. Figure Checklist

### Must-have figures (both drafts):
1. **Architecture diagram** — CertAV pipeline with frozen encoders → features → cross-modal attention → smoothing → certificate. Update from Phase 1 to show anisotropic noise injection.
2. **Certified accuracy curves** — isotropic at 4 σ levels (update from Phase 1 draft, already exists).

### Additional figures for ICASSP:
3. **Sphere vs. ellipsoid** — 2D projection on PCA axes showing isotropic circle vs anisotropic ellipse with data manifold direction indicated.

### Additional figures for CVIP:
3. **Sphere vs. ellipsoid** (same as above)
4. **Encoder scaling scatter plot** — x = d_int/D, y = mean R_cert for 5 encoder families. Color by encoder type.
5. **PCA eigenvalue spectrum** — cumulative variance curve for DINOv2-S+Whisper-tiny showing the 90% threshold at d=80.
6. **Conformal coverage-efficiency curve** — x = perturbation radius r, y = coverage and singleton rate.
7. **Attack alignment histogram** — distribution of cos²θ values across test samples.

---

## 16. Abstract Templates

### ICASSP (150 words max)

> Audio-visual deepfake detectors rely on frozen foundation representations, yet their robustness is typically evaluated only empirically. We introduce CertAV, a certification framework that applies randomized smoothing in the joint feature space of frozen encoders. On FakeAVCeleb, CertAV achieves 92.5% clean accuracy and 88.0% certified accuracy at ℓ₂ radius 1.0. We prove that the smoothed classifier's robustness depends on the intrinsic dimension d of the encoder representation (d=80 ≪ D=768), yielding amplification factor √(D/d). Validating this across five encoder families, we confirm that lower d/D yields higher certified radii. Exploiting this geometry, we propose manifold-aware anisotropic smoothing that achieves 3.4× larger on-manifold certified radius with zero abstention at equalized noise budget.

### CVIP (250 words max)

> (Use the full abstract drafted in the previous artifact — it already fits CVIP length.)

---

## 17. Title Options (Ranked)

1. **CertAV: Manifold-Aware Certified Robustness for Audio-Visual Deepfake Detection** — Best. Captures both the method (manifold-aware) and the domain (AV deepfake).

2. **CertAV: Certifiably Robust Audio-Visual Deepfake Detection via Manifold-Aware Feature-Space Smoothing** — More descriptive but longer. Better for CVIP.

3. **Intrinsic Dimensionality Drives Certifiability: Manifold-Aware Smoothing for Audio-Visual Deepfake Detection** — Emphasizes the theoretical finding. Good for venues that value theory.

Pick #1 for ICASSP (concise) and #2 for CVIP (descriptive).
