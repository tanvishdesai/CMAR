# CertAV Phase 2: Implementation Reference Guide

**Project:** CertAV — Certifiably Robust Audio-Visual Deepfake Detection  
**Phase:** 2 — Beyond Isotropic Smoothing  
**Directions:** A+D (Manifold-Aware Certification) and C (Conformal Deepfake Detection)  
**Status:** Reference document for implementation, ablation design, and result interpretation

---

## Part 0: Inventory of What You Already Have

Before touching any new code, map what is reusable. Everything listed here is a hard dependency for Phase 2.

### Models

| Checkpoint | Location | Purpose in Phase 2 |
|---|---|---|
| `certav_sigma100/best.pt` | `runs/certav_sigma100/` | Primary no-noise and noise-augmented baseline |
| `certav_sigma025/best.pt` | `runs/certav_sigma025/` | Mid-range σ baseline for tradeoff curves |
| `baseline_no_noise/best.pt` | `runs/baseline_no_noise/` | Critical: shows feature geometry is the driver |
| `baseline_pgd_at/best.pt` | `runs/baseline_pgd_at/` | Comparison against adversarial training |

All of these are inputs, not starting points. You do not retrain these. They serve as baselines in your new papers.

### Cached Features

| Cache | Content | Reuse |
|---|---|---|
| `cmar_cache/visual/` | DINOv2-Small frame embeddings (16×384) | Reuse directly for all experiments on existing encoder |
| `cmar_cache/audio/` | Whisper-tiny temporal embeddings (64×384) | Reuse directly |
| `cmar_cache/manifests/` | Train/val/test splits, metadata | Reuse for every experiment |

The cache for DINOv2-Small + Whisper-tiny is your gold standard. Any new encoder family study requires new preprocessing runs (separate cache directories), but the dataset splits from the manifests are identical — the split is based on clip IDs, not features.

### Certification Infrastructure

| File | Reuse |
|---|---|
| `cmar/certification/core.py` | Clopper-Pearson bounds, certified radius formula — reuse unchanged |
| `cmar/certification/smoothing.py` | `SmoothedClassifier` — extend, do not rewrite |
| `cmar/training/noise_augmented_trainer.py` | Extend for anisotropic noise injection |
| `scripts/11_certify.py` | Extend for anisotropic certification output |
| `scripts/13_certav_figures.py` | Extend for new figure types |
| `agg-results-cmvrta/certav_aggregated/` | Your baseline result JSON files for comparison |

### Manifold Analysis Results

The PCA/intrinsic dimension analysis from the elevation experiments is the seed of the entire Phase 2 contribution. Specifically:
- Visual (DINOv2-S): d_int at 90% variance = 73 out of D = 384
- Audio (Whisper-tiny): d_int at 90% variance = 13 out of D = 384
- Joint concatenation: d_int at 90% variance = 75 out of D = 768
- Source: `agg-results-cmvrta/certav-bench-v2-results/elevation_experiments/manifold_analysis.json`

These numbers are your hypothesis engine. Everything in Direction A+D flows from them.

---

## Part 1: Direction A+D — Manifold-Aware Certified Robustness

### 1.1 The Central Claim

Standard CertAV uses isotropic Gaussian noise: every dimension of the joint feature vector gets the same noise level σ. This ignores the fact that the DINOv2 visual features live on a 73-dimensional submanifold inside a 384-dimensional space, and Whisper audio features live on a 13-dimensional submanifold inside 384 dimensions.

The Phase 2 claim is twofold:

**Claim D (Empirical Law):** Certified radius is predictable from encoder geometry. Specifically, encoders with lower intrinsic-dimension-to-ambient-dimension ratio (d_int/D) produce larger certified radii under randomized smoothing, because the Gaussian noise concentrates in the off-manifold directions where it does not affect the prediction.

**Claim A (New Method):** By designing noise that is deliberately larger in high-variance (on-manifold) directions, you certify a larger region specifically against the attacks that matter — those that operate along the data manifold where the decision boundary lies.

These two claims are designed to be mutually reinforcing in a single paper. The empirical law explains why manifold-aware smoothing works; the new method demonstrates the principle.

### 1.2 Mathematical Foundation (What You Need Without Proofs)

#### The Isotropic Certificate

Cohen et al. (2019) shows: for smoothed classifier g with base classifier f_θ and noise N(0, σ²I), if the top-class probability lower bound is p_A (computed via Clopper-Pearson at confidence 1-α), and p_A > 0.5, then the prediction g(x) is stable for all perturbations δ satisfying:

```
||δ||₂ ≤ R_iso = σ × Φ⁻¹(p_A)
```

This R_iso is a uniform ball. It is the same in every direction.

#### Why Isotropic Is Suboptimal on Low-Dimensional Manifolds

When features live on a d_int-dimensional manifold in D dimensions, a random Gaussian perturbation ε ~ N(0, σ²I) has only approximately d_int/D of its energy projected onto the manifold. The rest of the noise goes into off-manifold directions where the classifier is invariant (since the decision boundary is defined in terms of the manifold features).

The consequence: for fixed σ, the effective noise on the decision-relevant (on-manifold) part of the feature space is only σ × sqrt(d_int/D) of the full noise. This is actually good — it means the classifier is not disturbed much by the noise — but it also means you are "paying" for noise in 311 irrelevant directions (visual case: D - d_int = 384 - 73 = 311) without getting any certification benefit for them.

Intuitively, you are spreading your noise budget equally across 384 dimensions when only 73 of them matter.

#### The Anisotropic Certificate

Yang et al. (2020) generalize Cohen et al. to any noise distribution. For Gaussian noise N(0, Σ) where Σ is not necessarily σ²I, the certified region is no longer a sphere but an ellipsoid:

```
Certified ellipsoid: { δ : δᵀ Σ⁻¹ δ ≤ [Φ⁻¹(p_A)]² }
```

The certified radius in a specific direction d (unit vector) is:

```
R(d) = Φ⁻¹(p_A) × sqrt(dᵀ Σ d)
```

To maximize R(d) in on-manifold directions (where attacks live), set the diagonal entries of Σ in the PCA basis to be proportional to the eigenvalues:

```
Σ_PCA = diag(σ₁², σ₂², ..., σ_D²) where σᵢ² ∝ λᵢ
```

(λᵢ are the PCA eigenvalues of the training features). This gives more noise in high-variance directions (principal components), which directly enlarges the certificate in those attack-relevant directions.

This is equivalent to: apply isotropic smoothing in the PCA-whitened space, then transform the certified region back to the original feature space. The certified region in the original space is an ellipsoid with axes aligned to the PCA directions, where the longest axes are in the highest-variance directions.

#### Why This Matters for Your Threat Model

Real adversarial attacks on your CertAV model (PGD-based) primarily operate along the gradient of the loss function, which approximately aligns with the principal components of the feature space (since the classifier is smooth and the features are on-manifold). This is empirically testable: measure the cosine similarity between the PGD attack direction and the top PCA components. You should observe high alignment.

#### The Expected Improvement

For a fixed total noise budget (trace(Σ) = Dσ² held constant between isotropic and anisotropic), the on-manifold certified radius improves by approximately:

```
R_aniso / R_iso ≈ sqrt(D × λ_max / Σᵢλᵢ)
```

Where the sum is over the d_int dominant eigenvalues. For your audio features (d_int=13, D=384), this ratio can be substantial. For visual features (d_int=73, D=384), the improvement is more modest. This means you should lead your paper with audio-domain results.

### 1.3 Phase D: The Encoder Family Study

This is the lowest-risk, highest-citation component of Phase 2. It requires no new theory and only preprocessing + certification runs.

#### Which Encoders to Test

| Visual Encoder | Model String | Ambient Dim | Expected d_int |
|---|---|---|---|
| DINOv2-Small (current) | `facebook/dinov2-small` | 384 | 73 (known) |
| DINOv2-Base | `facebook/dinov2-base` | 768 | ~150 (estimate) |
| CLIP ViT-B/16 | `openai/clip-vit-base-patch16` (visual) | 512 | ~50 (predict lower — contrastive training is more structured) |
| MAE ViT-B | `facebook/vit-mae-base` | 768 | ~180 (estimate — MAE is less structured than DINO) |

| Audio Encoder | Model String | Ambient Dim | Expected d_int |
|---|---|---|---|
| Whisper-tiny (current) | `openai/whisper-tiny` | 384 | 13 (known) |
| Whisper-base | `openai/whisper-base` | 512 | ~20 (estimate) |
| HuBERT-base | `facebook/hubert-base-ls960` | 768 | ~40 (estimate — HuBERT is speech-focused) |
| WavLM-base | `microsoft/wavlm-base` | 768 | ~35 (estimate) |

You do not need to test all combinations. Prioritize: (DINOv2-S, Whisper-tiny) [existing], (DINOv2-B, Whisper-base), (CLIP, Whisper-tiny), (DINOv2-S, HuBERT-base). Four encoder pairs is enough for a convincing empirical law.

#### What to Run Per Encoder Pair

For each new encoder pair, in order:

1. **Feature preprocessing**: Extract and cache features using the existing preprocessing pipeline (`01_preprocess_features.py` with modified encoder arguments). Store in a separate cache directory per encoder pair.

2. **PCA analysis**: Run the existing manifold analysis script on the new cached features. Record: d_int at 80%, 90%, 95% variance thresholds; explained variance curve; spectral decay rate.

3. **No-noise baseline certification**: Train a no-noise classifier (fast, no σ hyperparameter) on the new features. Certify it using the standard smoothed certification pipeline with σ=1.00. Record: clean accuracy, mean certified radius, certified accuracy at r=0.25, 0.50, 1.00.

4. **Noise-augmented baseline**: Train with σ=1.00 joint noise. Certify. Record same metrics.

The no-noise baseline is sufficient to establish the scaling law. Noise-augmented training adds one more data point per encoder pair but doubles the compute.

#### The Scaling Law Hypothesis

At the end of Phase D, plot: x-axis = d_int/D (intrinsic dimension ratio), y-axis = mean certified radius at σ=1.00 from the no-noise baseline. If the scaling law holds, you should observe a monotonically decreasing relationship: lower d_int/D → higher mean radius.

The predicted functional form is approximately:
```
R_cert ≈ σ × Φ⁻¹(p_A_avg) × f(1 - d_int/D)
```

where f is an increasing function (likely approximately linear or square-root). The exact functional form is an empirical finding, not something to assume in advance.

Also plot: spectral decay rate (how quickly the PCA eigenvalues decay to zero) vs. certified radius. Encoders with faster spectral decay should certify better, even controlling for d_int/D.

#### Key Ablation: Does Noise Training Change d_int?

Compare d_int of no-noise trained features vs. noise-augmented trained features for the same encoder. If they are similar, it confirms that d_int is an encoder property, not a training property. If noise training substantially changes d_int, the interpretation becomes more complex and should be flagged in the paper.

### 1.4 Phase A: Anisotropic Smoothing

This is the methodological contribution. It requires modifying the training and certification pipelines.

#### Step 1: Establish the PCA Basis

From the existing manifold analysis, you already have PCA components. What you need to make explicit:
- Compute the PCA basis U (384×384 orthogonal matrix for visual) on the training set features only (never validation/test).
- Store U and the eigenvalue vector λ as fixed tensors alongside the model checkpoint.
- For the joint space: compute PCA on the concatenated [visual_pooled; audio_pooled] features (768-dimensional vectors), not on the per-frame features.

Key implementation note: The PCA must be computed once on the training set and fixed. It must not be recomputed per-batch or updated during training.

#### Step 2: Design the Noise Covariance

Three strategies to compare (all are ablations against each other):

**Strategy 1 — Eigenvalue-proportional (recommended starting point):**
```
Σ = α × U diag(λ₁, λ₂, ..., λ_D) Uᵀ   (where α normalizes total budget)
```
- Noise variance in direction uᵢ (i-th principal component) is α × λᵢ
- Σ = α × (empirical covariance matrix of training features)
- This is exactly the training feature covariance scaled by α. Implementation: just sample noise from N(0, α × C̃) where C̃ is the sample covariance.
- Budget constraint: trace(Σ) = α × trace(C̃) = Dσ² → α = Dσ²/trace(C̃)

**Strategy 2 — Subspace projection (clean theoretical story):**
```
σ_on = σ for i ∈ top-k PCA components (on-manifold)
σ_off = ε (small, near-zero) for i ∉ top-k PCA components (off-manifold)
```
- This is equivalent to projecting onto the top-k subspace and applying isotropic noise there
- Set k = d_int (the intrinsic dimension at 90% variance)
- The certified region is a cylinder aligned with the manifold: very large radius in off-manifold directions (because noise is tiny there), radius σ × Φ⁻¹(p_A) in on-manifold directions
- Clean theoretical interpretation: the classifier is smooth only within the manifold

**Strategy 3 — Inverse-eigenvalue (aggressive off-manifold robustness):**
```
σᵢ ∝ 1/sqrt(λᵢ)   (normalized to same total budget)
```
- This is the "whitening" approach: equal noise in the PCA-whitened space
- Gives uniform certified radius in the Mahalanobis sense
- The l₂ certified radius in high-variance directions is actually SMALLER than isotropic, but the Mahalanobis certified region is a perfect sphere
- Use this as an ablation to show it performs worse than Strategy 1 for the on-manifold threat model

The paper comparison: Strategy 1 (proposed) > isotropic CertAV > Strategy 3 under the on-manifold threat model. Strategy 3 > Strategy 1 under the worst-case off-manifold threat model. This is a clean story about threat model alignment.

#### Step 3: Training Modifications

Minimal changes to `noise_augmented_trainer.py`:
- Add a `PCANoise` module that takes (features, sigma, covariance_matrix, strategy) and returns noise-augmented features
- For Strategy 1: sample from N(0, α × C̃) where C̃ is pre-computed; add to features
- For Strategy 2: apply top-k PCA projection of features, add N(0, σ²I_k) in the projected space, reconstruct
- For Strategy 3: sample from N(0, C̃⁻¹ × budget) where C̃⁻¹ is the pseudo-inverse (truncated at d_int components)
- The training hyperparameters (batch size, learning rate, epochs, patience) remain identical to Phase 1 CertAV
- Keep the existing `noise_mode` flag system; add `noise_mode = "anisotropic_strat1"`, etc.

#### Step 4: Certification Modifications

The Clopper-Pearson bounds in `core.py` do not change. What changes is how the certified radius is reported.

The MC sampling procedure in `smoothing.py` stays identical — you still sample n=1000 noisy versions of the feature vector and count votes. The noise used at certification time must match the noise used at training time (same covariance Σ).

New output fields in the certification JSON:
- `certified_radius_l2`: the minimum l₂ radius over all directions (worst case)
- `certified_radius_onmanifold`: the certified l₂ radius in the average direction of the top-d_int principal components
- `certified_ellipsoid_volume`: the volume of the certified ellipsoid (proportional to |Σ|^{1/2} × (Φ⁻¹(p_A))^D)
- `alignment_attack_manifold`: cos² between the PGD attack direction and the principal component subspace (a diagnostic)

Important: `certified_radius_l2` for anisotropic certification can be LOWER than isotropic CertAV's certified radius. This is expected and should be discussed honestly. The key metric is `certified_radius_onmanifold` and `certified_ellipsoid_volume`.

#### Step 5: Matching the Certification to the Threat Model

The main paper argument requires you to empirically validate that real PGD attacks are on-manifold. Add a diagnostic experiment:

For each certified test sample, run a PGD attack in feature space. Record: the cosine similarity between the PGD attack direction (normalized δ*) and the principal component subspace (projection onto the top-d_int PCA directions). If this cosine similarity is high (> 0.7), it confirms the on-manifold threat model assumption.

This is not a new experiment in complexity — it's adding a diagnostic to the existing empirical attack comparison script (`12_empirical_attack_comparison.py`).

### 1.5 Ablation Study Design for A+D

The ablation table must answer exactly three questions. Organize it as follows:

#### Ablation 1: Does the scaling law hold across encoders? (Direction D)

Table columns: Encoder pair | D_visual + D_audio | d_int_visual | d_int_audio | d_int_joint | Mean certified radius (no-noise, σ=1.00) | Clean accuracy

Expected ordering: encoders with lower d_int/D have higher mean certified radius. If CLIP-visual has d_int ≈ 50 (lower than DINOv2-S at 73), CLIP should certify better.

Ablation within this table: separately vary visual encoder while keeping audio fixed, and vice versa. This tests whether the law is about the joint dimension ratio or the individual modality ratios.

#### Ablation 2: Which anisotropic strategy works best? (Direction A, comparison of strategies)

Table columns: Strategy | Training noise | Certification metric | Clean accuracy | R_iso_min | R_onmanifold | Certified volume

Rows: Isotropic CertAV (baseline) | Anisotropic S1 (eigenvalue-prop) | Anisotropic S2 (subspace projection) | Anisotropic S3 (inverse-eigenvalue) | No-noise baseline

Primary metric for comparison: R_onmanifold. Secondary: clean accuracy (should not degrade).

The expected result is S1 ≥ S2 > Isotropic > S3 for the R_onmanifold metric, but S3 > S1 > Isotropic > S2 for the R_iso_min metric.

#### Ablation 3: Budget equalization (confirming the improvement is not from more total noise)

This ablation is critical for review soundness. A skeptical reviewer will ask: "Your anisotropic noise has larger σ in some directions — doesn't that just mean you're using more total noise?"

Counter this with: equalize the noise budget (trace(Σ) = Dσ² for all strategies) and re-run. The improvement should persist because the budget is redistributed, not increased.

Additionally, compare: isotropic CertAV with σ_iso = sqrt(trace(Σ_aniso)/D) (inflated σ to match total budget) vs. anisotropic CertAV at the same budget. The anisotropic version should outperform the inflated-σ isotropic version for on-manifold certification.

#### Ablation 4: Does attack direction align with manifold? (Validating the threat model assumption)

Report the cosine similarity between PGD attack directions and the top-d_int principal components for a sample of certified test clips. Split by modality (visual attack direction vs. visual PCA, audio attack direction vs. audio PCA).

Expected result: cosine similarity > 0.6 for visual, > 0.7 for audio (audio is more constrained to its manifold).

### 1.6 Expected Results and Success Criteria

#### Direction D Success Criteria

**Minimum success (publishable at ICASSP/ICIP):**
- The scaling law holds for at least 3 of the 4 tested encoder pairs
- Specifically: the encoder pair with lowest d_int/D achieves the highest mean certified radius
- The no-noise baseline for each encoder confirms that certifiability is an encoder property, not a training-procedure property
- Correlation coefficient between d_int/D and mean certified radius: R² > 0.75

**Full success (CVPR/ICCV-level):**
- The scaling law holds for all 4 encoder pairs
- A quantitative relationship can be fitted: R_cert ≈ c₁ × σ × (D/d_int)^c₂ where c₂ > 0 is estimated from data
- Self-supervised encoders (DINOv2, MAE) show lower d_int than contrastive encoders (CLIP) or supervised encoders, and correspondingly higher certified radii
- The spectral decay rate predicts certified radius better than d_int/D alone (more granular law)

**What to do if the law does not hold:**
If CLIP-visual certifies WORSE than DINOv2-S despite lower d_int, this is the most interesting negative result. Investigate: is the CLIP feature distribution more heavy-tailed (non-Gaussian)? Does the certified radius depend on feature kurtosis, not just d_int/D? This pivots the finding from "law" to "factors affecting certifiability of foundation encoders," which is still highly publishable.

#### Direction A Success Criteria

**Minimum success (publishable):**
- Strategy 1 anisotropic certification achieves R_onmanifold > R_iso_min for the same clean accuracy
- The improvement is at least 10% in mean certified radius (on-manifold metric) at σ=1.00
- Clean accuracy does not degrade more than 0.5% absolute
- PGD attack alignment with manifold > 0.5 cosine similarity

**Full success (CVPR/ICCV-level):**
- Strategy 1 achieves 15–25% improvement in R_onmanifold at σ=1.00 while maintaining clean accuracy
- The certified ellipsoid volume improves by a factor of (D/d_int)^{d_int/D} or more (the theoretical maximum)
- Audio features show larger improvement than visual features (because d_int_audio/D_audio = 13/384 << d_int_visual/D_visual = 73/384)
- The scaling law from Direction D predicts the amount of improvement from anisotropic smoothing

**What to do if Direction A does not improve over isotropic:**
If anisotropic smoothing does not improve certification in practice, the explanation is likely that PGD attacks are not on-manifold (i.e., the decision boundary is not aligned with the feature manifold). In this case, the attack alignment diagnostic will show low cosine similarity. This is still a publishable negative result if it refutes the common assumption that adversarial attacks respect data manifolds. Pivot the narrative: "We show that for frozen foundation features, adversarial attacks do not operate on-manifold, which explains why isotropic smoothing already achieves high certification."

#### Combined A+D: What Constitutes the Full Paper

The paper needs the following table structure to be complete:

Table 1 (Direction D): Scaling law across encoder families — 4 rows, 6 columns
Table 2 (Direction A): Anisotropic strategy comparison — 5 rows, 6 columns  
Table 3 (Combined): For the best encoder pair, anisotropic smoothing vs. isotropic at matched budget — 3 rows
Figure 1: Scatter plot of d_int/D vs. certified radius (the scaling law visualization)
Figure 2: Certified accuracy curves for best anisotropic strategy vs. isotropic (the method improvement)
Figure 3: PCA eigenvalue spectrum + attack direction alignment diagnostic

---

## Part 2: Direction C — Conformal Prediction for Certified Deepfake Detection

### 2.1 Why Conformal Prediction and Why It Is Different

CertAV's current guarantee is a *per-sample, per-radius* statement: "For this specific sample x, the prediction is stable under all l₂ perturbations within radius R_cert(x), with probability ≥ 1-α over the randomness of the certification procedure." This is strong but binary: a sample either gets a certificate or it abstains.

Conformal prediction offers a complementary guarantee: *marginal* coverage over the data distribution. For a calibrated threshold q̂_α, the prediction set C_α(x) (which for binary detection is a subset of {real, fake}) satisfies:

```
P(y_true ∈ C_α(x_test)) ≥ 1 - α
```

with probability ≥ 1-α over the calibration set. The key word is "probability over the data distribution" — not for every individual sample, but on average over the distribution of test samples.

For adversarially robust conformal prediction, this extends to:

```
P(y_true ∈ C_α(x_test + δ)) ≥ 1 - α   for all ||δ||₂ ≤ r
```

This is a different type of guarantee from CertAV:
- CertAV: per-sample, per-radius certificate (100% guarantee for certified samples, abstain otherwise)
- Robust conformal: distributional coverage guarantee under adversarial perturbation (no per-sample guarantee, but no abstention either)

For forensic deepfake detection, the conformal prediction set has a natural interpretation: a singleton {fake} means "I am confident this is fake even under attack"; a singleton {real} means "I am confident this is real even under attack"; the full set {real, fake} means "I cannot distinguish with confidence under attack." This is more informative for a human analyst than a binary abstain/not-abstain decision.

### 2.2 Mathematical Foundation

#### Standard Split Conformal Prediction

Given: calibration set (x₁, y₁), ..., (x_n, y_n) drawn exchangeably with the test set.

Step 1 (nonconformity scores): compute a score s(xᵢ, yᵢ) for each calibration sample that measures how "nonconforming" the sample is. For a classifier with probability output p(y|x):
```
s(x, y) = 1 - p̂(y | x)   (1 minus the predicted probability for the true class)
```

For your smoothed classifier, p̂(y | x) is the fraction of n=1000 noisy samples that predicted class y.

Step 2 (calibration threshold): sort the calibration scores s₁ ≤ s₂ ≤ ... ≤ s_n. The threshold is:
```
q̂_α = the ⌈(n_cal + 1)(1 - α)⌉-th smallest score
```

Step 3 (prediction set at test time): for a new test sample x_test, compute p̂(y | x_test) for each y ∈ {0, 1}. The prediction set is:
```
C_α(x_test) = { y : 1 - p̂(y | x_test) ≤ q̂_α }
            = { y : p̂(y | x_test) ≥ 1 - q̂_α }
```

This set includes all classes whose predicted probability exceeds the calibrated threshold.

Step 4 (coverage guarantee): by Vovk's exchangeability theorem:
```
P(y_true ∈ C_α(x_test)) ≥ 1 - α
```
This holds without any distributional assumptions beyond exchangeability.

#### Smoothed Nonconformity Scores

The key design decision is what nonconformity score to use when the base classifier is a smoothed classifier. The natural choice for CertAV is to use the smoothed probability directly:
```
s(x, y) = 1 - g(x)[y]
```
where g is the smoothed classifier and g(x)[y] is the estimated probability of class y under n samples of Gaussian noise. This is already computed during certification.

**Implementation note:** You already compute these scores during certification. The certify script collects, for each sample: p̂_A (the Clopper-Pearson lower bound on the top-class probability). The nonconformity score is approximately 1 - p̂_A for the predicted class, and p̂_A for the other class. The conformal calibration just sets a threshold on these existing scores.

This means conformal prediction requires almost no new computation if you reuse the CertAV certification output. The main addition is the calibration step and the new prediction-set interpretation.

#### Adversarially Robust Conformal Prediction

The standard conformal guarantee breaks under adversarial perturbation because x_test is replaced by x_test + δ, which is not exchangeable with the calibration set. The adversarially robust extension (Gendler et al. 2022, "Adversarially Robust Conformal Prediction") modifies the calibration to account for worst-case perturbations.

The key insight: for the smoothed classifier, the probability p̂(y | x + δ) is bounded below by:
```
p̂(y | x + δ) ≥ p̂(y | x) - Δ(r, σ)
```
where Δ(r, σ) is a function of the perturbation radius r and the smoothing noise σ. For isotropic Gaussian smoothing, this bound is related to the certified radius: if R_cert > r, then Δ(r, σ) can be bounded explicitly.

The adversarially robust conformal procedure replaces the standard nonconformity score with a worst-case score:
```
s_robust(x, y, r) = 1 - min_{||δ||₂ ≤ r} p̂(y | x + δ)
                  ≈ 1 - max(0, p̂(y | x) - Δ(r, σ))
```

Calibrating with this worst-case score gives a threshold q̂_α(r) such that:
```
P(y_true ∈ C_α(x + δ)) ≥ 1 - α   for all ||δ||₂ ≤ r
```

**Practical implementation:** You do not need to compute Δ(r, σ) exactly in closed form. The approximation is: for a sample with certified radius R_cert ≥ r (from standard CertAV certification), the worst-case probability lower bound is p̂_A_lower (the Clopper-Pearson bound already computed). For a sample with R_cert < r (uncertified at radius r), use p̂ = 0.5 as the worst-case probability bound (anything below 0.5 would cause the smoothed classifier to abstain already).

This means: the adversarially robust conformal prediction set at radius r is computed using the Clopper-Pearson lower bounds from CertAV directly as the probability estimates. You are combining the two systems.

#### The Binary Classification Case

For deepfake detection with labels y ∈ {0=real, 1=fake}:
- p̂(1 | x) is the estimated fake probability under the smoothed classifier
- p̂(0 | x) = 1 - p̂(1 | x) by the binary property (approximately, for the Monte Carlo estimate)

The prediction set C_α(x) is one of:
- {1} only: "definitely fake" — p̂(1|x) ≥ 1 - q̂_α and p̂(0|x) < 1 - q̂_α
- {0} only: "definitely real" — p̂(0|x) ≥ 1 - q̂_α and p̂(1|x) < 1 - q̂_α
- {0, 1}: "uncertain" — both probabilities exceed the threshold (this happens when q̂_α is low, i.e., α is large or calibration is conservative)

Note: in binary classification, the complement of {0,1} (an empty set) should not occur if the threshold is set correctly. The calibration ensures this.

The analog to CertAV's abstention is the set {0,1}. But unlike CertAV's abstention (which occurs when p̂_A ≤ 0.5), the conformal "uncertain" set occurs when both probabilities exceed the threshold. For typical α = 0.1 and a well-calibrated model, q̂_α ≈ 0.1-0.2, so the threshold for inclusion is 1 - q̂_α ≈ 0.8-0.9. Only samples where the model is quite confident avoid the {0,1} prediction set.

### 2.3 Implementation Plan

#### Step 1: Calibration

Add a calibration module to the existing CertAV codebase. This should accept:
- An existing checkpoint (any of the trained CertAV models)
- The validation set features from the cache
- Confidence level α (default 0.1)
- Perturbation radius r (for robust conformal; 0.0 for standard)

The module outputs a threshold q̂_α(r) and a calibration curve: q̂ as a function of α and r.

**Critical design decision:** Use the validation set for calibration. Do not use the test set. The test set is for evaluation only. The validation set from `cmar_cache/manifests/val.csv` has 825 samples, which is sufficient for stable calibration (the standard recommendation is n_cal ≥ 200).

For robust conformal (non-zero r), the calibration scores come from the Clopper-Pearson lower bounds on p̂_A from a fresh certification run on the validation set. Store these bounds from the existing certification output JSON.

#### Step 2: Test-Time Prediction Sets

Modify the certification script or create a new `11b_conformal_certify.py` that:
- Loads the calibration threshold q̂_α(r) from Step 1
- For each test sample, produces the prediction set C_α(x) rather than just the top-class prediction
- Reports: set size (1 or 2), set content, coverage (whether y_true ∈ C_α(x))

#### Step 3: The Coverage-Efficiency Curve

The main new figure type for Direction C is the coverage-efficiency tradeoff curve:
- x-axis: perturbation radius r
- y-axis (left): empirical coverage P(y_true ∈ C_α(x+δ)) (should stay near 1-α)
- y-axis (right): fraction of test samples with singleton prediction sets (efficiency)

As r increases, coverage should stay constant (by design) but efficiency drops (more samples get {0,1}). The curve shows how the conformal guarantee "costs" in efficiency.

Compare this curve for: standard CertAV (treating "abstain" as the singleton set) vs. robust conformal (treating {0,1} as "abstain"). The conformal version should trade coverage guarantee for abstention rate more gracefully.

### 2.4 Ablation Study Design for Direction C

#### Ablation C1: Standard vs. Robust Conformal

Rows: standard conformal (r=0) | robust conformal (r=0.25) | robust conformal (r=0.50) | robust conformal (r=1.00) | CertAV baseline (no conformal)
Columns: α | calibration threshold q̂ | marginal coverage (clean) | marginal coverage (under PGD) | singleton rate | efficiency

Expected result: standard conformal achieves ~1-α coverage on clean data but coverage drops under PGD. Robust conformal maintains ~1-α coverage under PGD at the cost of lower singleton rate.

#### Ablation C2: Choice of Nonconformity Score

Compare three nonconformity score functions:
1. `s₁(x,y) = 1 - p̂_smooth(x, y)` (raw Monte Carlo probability — the simple approach)
2. `s₂(x,y) = 1 - p̂_CP_lower(x, y)` (Clopper-Pearson lower bound — more conservative)
3. `s₃(x,y)` based on log-probability / temperature scaling of the smoothed probability

The CP-lower-bound score (s₂) corresponds naturally to the certified robustness guarantee: a sample with high p_A (far from the decision boundary) gets a low nonconformity score and is confidently included in its singleton prediction set. This score directly connects conformal prediction to the existing certification machinery.

Expected result: s₂ gives the best coverage-efficiency tradeoff under adversarial perturbation because it is inherently conservative.

#### Ablation C3: Coverage Decomposed by Clip Type

Compute coverage separately for:
- Real clips (y=0)
- Fake clips (y=1)
- Clips that CertAV certifies at radius ≥ r (high-confidence certified)
- Clips that CertAV abstains on (low-confidence uncertified)

This answers an important question: does conformal prediction improve coverage specifically for the samples that CertAV currently fails on? Expected result: yes — the conformal guarantee specifically helps low-confidence uncertified samples by providing coverage even when CertAV abstains.

#### Ablation C4: Calibration Set Size

Vary n_cal from 50 to 825 (full validation set). Report how the calibration threshold q̂_α and empirical coverage stability change. Expected result: for α=0.1, n_cal ≥ 200 gives stable calibration (coverage within ±2% of 1-α).

#### Ablation C5: Cross-Dataset Conformal Coverage (LAV-DF)

Calibrate on FakeAVCeleb validation, test on LAV-DF. The standard conformal guarantee requires exchangeability, which breaks across datasets (distribution shift). Report the coverage drop. Expected result: coverage on LAV-DF is lower than 1-α (exchangeability is violated) but the degradation is bounded. This is an honest limitation section.

Compare: CertAV cross-dataset certification (certifies 61.4% at radius 1.00) vs. conformal coverage on LAV-DF (you report marginal coverage, not per-sample certificates). The conformal approach should show more graceful degradation because it trades per-sample guarantees for distributional coverage.

### 2.5 New Metrics Specific to Direction C

Introduce these metrics alongside the existing CertAV metrics, with explanations:

**Marginal coverage at radius r:** `P(y_true ∈ C_α(x+δ))` for δ chosen adversarially. This is the primary claim. Compute empirically by running PGD attacks at ε=r on all test samples and measuring coverage.

**Conditional coverage:** Marginal coverage conditioned on y=1 (fake), y=0 (real), and conditioned on CertAV abstaining. This tests whether the conformal guarantee is "uniformly good" or concentrated in easy samples.

**Efficiency / singleton rate:** Fraction of test samples where |C_α(x)| = 1. Higher is better. This is analogous to CertAV's (1 - abstention rate).

**Conformal efficiency under attack:** Efficiency when x is replaced by the PGD adversarial example. Expected to be lower than clean efficiency, with the gap indicating how much the guarantee "costs" under adversarial pressure.

**Threshold stability:** How much q̂_α varies when calibration is repeated on different random subsets of the validation set (bootstrap analysis). Stable calibration is a sign of sufficient calibration set size.

### 2.6 Expected Results and Success Criteria

#### Minimum Success (Publishable at ICASSP/ICIP)

- Standard conformal achieves 89–91% marginal coverage on clean FakeAVCeleb test set (target: 1-α = 90%)
- Singleton rate on clean data: ≥ 80%
- Standard conformal coverage under PGD at ε=0.1: drops to 70–80% (showing the need for robust version)
- Robust conformal at r=0.25: maintains ≥ 87% coverage under PGD at ε=0.25
- Singleton rate of robust conformal at r=0.25: still ≥ 65%

These numbers represent a clean demonstration that robust conformal adds value over standard conformal.

#### Full Success (NeurIPS/ICML-level)

- Robust conformal at r=1.00 achieves ≥ 88% coverage even under strong PGD attacks at ε=1.00 (matching CertAV's 88.0% certified accuracy at r=1.0 in a comparable metric)
- Conditional coverage gap (real vs. fake): ≤ 5% absolute (conformal is class-balanced in coverage)
- Conformal coverage for CertAV-abstaining samples: ≥ 80% even on samples CertAV cannot certify (this is the main new contribution — covering previously uncoverable samples)
- Coverage on LAV-DF (cross-dataset): ≥ 75% at r=0.5 (graceful degradation)
- The paper provides: a proof that the smoothed CP-lower-bound nonconformity score gives valid robust conformal coverage, as a formal theorem connecting CertAV's certification to conformal prediction

#### The Definitive Selling Point

The strongest argument for Direction C over CertAV alone: CertAV abstains on ~0.8% of clean samples and a higher fraction at r > 1.5. Those abstaining samples get no certificate. Robust conformal prediction provides a coverage guarantee for ALL test samples — including the ones CertAV abstains on — at the cost of providing a set {0,1} instead of a singleton. For a forensic analyst reviewing suspicious media, "I cannot determine this with confidence" ({0,1}) is far more useful than "no certificate available" (CertAV abstention).

#### What to Do If Coverage Drops Below 1-α Under Attack

If robust conformal coverage consistently falls below 1-α = 90% under strong PGD attacks, check these in order:

1. Is the calibration threshold q̂ computed using the clean validation set, but tested on perturbed samples? This would indicate the robust conformal version is not correctly implemented — you need worst-case nonconformity scores during calibration, not clean scores.

2. Is the PGD attack budget (ε) larger than the certified radius r of the robust conformal procedure? If you are using r=0.25 but testing with ε=1.0, the coverage guarantee does not apply — this is not a failure, it is a correct behavior of the guarantee.

3. Is the FakeAVCeleb dataset large enough for reliable calibration? With n_cal=825 and α=0.1, you need ≥ 8-9 miscoverage events. If coverage empirically drops by > 5%, this is a calibration instability issue.

---

## Part 3: Cross-Cutting Concerns

### 3.1 Shared Baselines

Both directions share the same set of baselines. Never recompute baselines separately for each direction; run them once and share.

| Baseline | Config | Used In |
|---|---|---|
| Isotropic CertAV σ=1.00 (5-seed avg) | `certav_sigma100` | A+D primary comparison, C's CertAV certificate input |
| No-noise baseline | `baseline_no_noise` | A+D manifold story, C's coverage comparison |
| PGD-AT baseline | `baseline_pgd_at` | A+D comparison only |
| Standard conformal (r=0) | New calibration on val set | C, as ablation against robust conformal |

The existing `certav_master_results.json` contains all the main CertAV baselines. Treat this file as immutable and reference it; never regenerate it.

### 3.2 Evaluation Protocol

Both directions must report results on the same test set: the 825 FakeAVCeleb test clips defined by `manifests/test.csv`. No exceptions. Do not report results on the validation set in the main table.

For all new certifications: use n₀=100, n=1000, α=0.001 (matching existing CertAV). This ensures comparability.

For conformal prediction: use α_conformal = 0.05, 0.10, 0.20 as the coverage target. Report all three in the ablation, use 0.10 as the primary reported value.

All new experiments: run at minimum 3 seeds (can report single-seed for ablations, 5-seed for primary claims). The existing CertAV 5-seed result can be taken as-is.

### 3.3 Paper Positioning

#### For A+D: How to Position Against Prior Work

The two papers to explicitly compare against:
1. **MMCert (CVPR 2024):** Uses randomized ablation on pixels, not Gaussian smoothing on features. Different threat model and approach. CertAV's feature-space approach certifies much larger radii (mean 2.215 vs. MMCert's pixel-space certificates which are small). The key distinction: CertAV certifies the representation used by the detector, not the raw pixels, which is the correct level for a detector that uses frozen encoders.

2. **Feature-Space Smoothing for MLLMs (arXiv 2601.16200, Jan 2026):** Certifies cosine similarity for MLLMs, not binary classification. The Gaussian Smoothness Booster they propose (a plug-in module to improve encoder robustness scores) is related but targets MLLM task performance under attack, not deepfake detection certifiability. Your contribution is the empirical certifiability law and the manifold-aware certification method.

The key phrase to use in the paper: "Our work is the first to establish that the certifiability of frozen foundation features under randomized smoothing is predictable from the intrinsic dimensionality of the learned representation, and to exploit this geometry for improved audio-visual deepfake certification."

#### For C: How to Position

Conformal prediction for deepfake detection does not have a direct prior work competitor. The nearest is:
- Standard certified deepfake detection (CertAV is the baseline, not the prior work — you are extending it)
- Conformal prediction for image classification under attack (multiple papers, none specifically for deepfakes or audio-visual)

The key framing: "We extend certified audio-visual deepfake detection from per-sample l₂ certificates (CertAV) to distributional coverage guarantees (robust conformal prediction), which provides formal guarantees for all test samples including those on which CertAV abstains."

---

## Appendix A: Key Papers to Read for Implementation

### For A+D

1. **Cohen, Rosenfeld, Kolter (2019) — ICML.** "Certified adversarial robustness via randomized smoothing." The foundation. Read sections 2 and 3.

2. **Yang et al. (2020) — ICML.** "Randomized Smoothing of All Shapes and Sizes." The generalization to non-isotropic noise. Read sections 2–4. The key theorem is Theorem 1, which gives the certified region for arbitrary noise distributions.

3. **Pfrommer, Anderson, Sojoudi (2023) — ICLR submission.** "Projected Randomized Smoothing for Certified Adversarial Robustness." The manifold projection approach that directly motivates your anisotropic strategy. Read the main theorem and the CIFAR-10 experiments.

4. **Eiras et al. (2021) — AAAI.** "ANCER: Anisotropic Certification via Sample-Wise Volume Maximization." The data-dependent anisotropic smoothing paper. Read to understand what they optimize and why you are doing something simpler (PCA-based rather than sample-wise).

5. **Xia et al. (2026) — arXiv 2601.16200.** "Feature-Space Adversarial Robustness Certification for Multimodal Large Language Models." Read to understand the closest related work and how to differentiate.

### For C

1. **Vovk, Gammerman, Shafer (2005).** "Algorithmic Learning in a Random World." Chapter 2 (conformal prediction basics). You only need sections 2.1-2.3 for the mathematics.

2. **Angelopoulos and Bates (2021).** "A gentle introduction to conformal prediction and distribution-free uncertainty quantification." arXiv 2107.07511. The best practical introduction. Read sections 1–4.

3. **Gendler et al. (2022).** "Adversarially robust conformal prediction." arXiv 2206.01367. The key paper for robust conformal. Read sections 2 and 3 carefully.

4. **Romano, Bates, Candès (2020) — NeurIPS.** "Classification with Valid and Adaptive Coverage." Introduces RAPS (regularized adaptive prediction sets), which controls prediction set sizes. Read for Section 4.

5. **Einbinder et al. (2025) — ICML.** "Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability." Published at ICML 2025. Closest recent work. Read to see what has been done at ICML and to differentiate your deepfake-specific contributions.

---

## Appendix B: Decision Points and Checkpoints

Use this section to record go/no-go decisions during implementation.

### A+D Checkpoints

**Checkpoint 1 (Phase D, ~2 months):**
- Have you preprocessed features for at least 2 new encoder pairs?
- Does the no-noise baseline certification for at least one new encoder pair fall within ±10% of what the scaling law predicts based on d_int?
- Decision: If yes, continue to all 4 encoder pairs. If no, investigate why the law fails before proceeding.

**Checkpoint 2 (Phase A, ~3 months):**
- Does the PCA analysis confirm that d_int has not changed significantly between no-noise and noise-augmented training?
- Does the PGD attack direction alignment with the PCA subspace exceed 0.5 cosine similarity?
- Decision: If cosine similarity < 0.3, reconsider the threat model assumption. The anisotropic story requires on-manifold attacks. If attacks are off-manifold, pivot to characterizing why and make that the finding.

**Checkpoint 3 (Phase A, ~4 months):**
- Does Strategy 1 anisotropic certification improve on-manifold certified radius by ≥ 10%?
- Does clean accuracy degrade by < 1%?
- Decision: If improvement < 5%, report as a null result with an analysis of why (likely: the manifold is too large relative to D, or PGD attacks are not on-manifold). If improvement ≥ 10%, this is a clear positive result.

### C Checkpoints

**Checkpoint 1 (calibration, ~1 month):**
- Does the standard conformal prediction set achieve marginal coverage ≥ 88% on the FakeAVCeleb test set with α = 0.10?
- Decision: If coverage is significantly below 88%, the nonconformity score is miscalibrated. Check that you are using the validation set (not the test set) for calibration and that the smoothed probabilities are computed consistently.

**Checkpoint 2 (robust conformal, ~2 months):**
- Does the robust conformal procedure at r=0.25 maintain ≥ 85% coverage under PGD attacks with ε=0.25?
- Is the singleton rate ≥ 55%?
- Decision: If coverage holds but singleton rate is very low (<30%), the procedure is too conservative. Investigate whether the Clopper-Pearson bound is too tight for use as the worst-case score.

---

## Appendix C: The "Honest Failure" Catalog

Research doesn't always go as planned. Here are the most likely failure modes and what to do.

| Failure Mode | Likely Cause | Pivot |
|---|---|---|
| Scaling law doesn't hold (d_int/D doesn't predict R_cert) | Feature non-Gaussianity, heavy tails | Study spectral decay rate and kurtosis as alternative predictors |
| CLIP certifies worse than DINOv2-S despite lower d_int | CLIP features are more "peaked" (non-Gaussian) | Feature distribution shape is also a factor; write "factors affecting certifiability" paper |
| Anisotropic noise hurts clean accuracy | Noise too large in high-variance directions | Reduce budget parameter α; use Strategy 2 (subspace projection) as primary |
| Conformal coverage drops below 1-α under attack | Robust conformal miscalibrated | Verify that worst-case nonconformity scores use Clopper-Pearson (not point estimates); check that calibration is done with robust scores, not clean scores |
| PGD attack direction not aligned with manifold | Decision boundary not manifold-aligned | Negative result: report that isotropic smoothing is already manifold-optimal for this model; investigate what the decision boundary looks like |
| Cross-dataset conformal coverage collapses on LAV-DF | Distribution shift breaks exchangeability | Report honestly; propose weighted conformal (importance weighting by domain shift) as future work |
