# CertAV — Project Explainer for Mentors

**Audience:** Palak Parmar, Dr. SantoshKumar Bharti, Dr. Chintan Bhatt
**Purpose:** A self-contained, plain-language brief on what this project is, why it matters, the
concepts it uses (many of which are uncommon even within mainstream deep learning), what we built,
and what we found. No prior knowledge of certified robustness or conformal prediction is assumed.

> **One-sentence summary.** We take an audio-visual deepfake detector built on frozen foundation
> encoders and, instead of just measuring its accuracy, we give every prediction a *mathematical
> guarantee* of how large a perturbation it can withstand — and we explain, prove, and then exploit
> *why* those guarantees come out unusually strong.

---

## Part 0 — How to read this document

- **Part 1** is the problem in plain English.
- **Part 2** is a concept primer — read this if terms like *certification*, *randomized smoothing*,
*intrinsic dimension*, *anisotropic*, or *conformal prediction* are unfamiliar. Each concept has an
intuition, an analogy, and only then the formula.
- **Part 3** is what we actually built and measured (the project itself).
- **Part 4** is the contributions, results, honest limitations, and submission status.
- **Part 5** is a one-page glossary you can keep open while reading the papers.

---

# Part 1 — The problem

## 1.1 The world problem

Deepfakes now manipulate **both** what a person appears to say (video) and how they sound (audio),
in a coordinated way. Detectors that look at only one stream miss forgeries that are only
inconsistent *across* streams (e.g., the lips and the voice don't match). Modern detectors therefore
fuse audio and video.

## 1.2 The technical gap

These detectors are evaluated almost entirely by **accuracy on clean test data** and, at best, by
**accuracy against a fixed list of attacks**. That is *empirical* robustness: "we tried these attacks
and it survived." It says nothing about attacks we did not try. For forensic or legal use — where a
detector's verdict may be used as evidence — "we tried a few attacks" is not a strong enough claim.

## 1.3 What we actually deliver (and a framing note)

Our stated thesis is "enhance the robustness of deepfake detection against adversarial attacks."
It is worth being precise about *what kind* of robustness we add:

- We do **not** primarily make the detector *harder to fool* by retraining it (in fact, we show that
the usual "train harder" recipe **hurts** this model — see §3.4).
- Instead, we provide **certified robustness**: for each clip, a *provable* radius such that **no**
perturbation smaller than that radius (within our threat model) can change the decision. This is the
strongest form of robustness claim available in machine learning, because it covers *all* attacks at
once, not a tested subset.

So the correct one-line framing is: **CertAV turns an existing detector into one that issues
per-sample provable robustness certificates, and explains why those certificates are large.** This is
a *certification + explainability* contribution, which is a recognized and respected category of
robustness research.

---

# Part 2 — Concept primer (the unfamiliar ideas, explained)

## 2.1 Adversarial attacks and the two kinds of "robust"

An **adversarial attack** adds a tiny, carefully chosen perturbation to an input so that a model
flips its decision, even though a human sees no meaningful change. Robustness to such attacks comes in
two flavours:


|              | **Empirical robustness**           | **Certified (provable) robustness**                 |
| ------------ | ---------------------------------- | --------------------------------------------------- |
| Claim        | "Survived the attacks we ran."     | "Provably survives *any* attack within radius *r*." |
| Coverage     | The attacks tested                 | All perturbations up to a size                      |
| Risk         | A stronger/new attack may break it | None within the certified radius                    |
| This project | Used only as a stress test         | **The main contribution**                           |


## 2.2 Randomized smoothing — the certification engine (Cohen et al., 2019)

This is the core tool. The idea is counter-intuitive but simple:

> To make a fragile classifier *certifiable*, ask it the same question many times under random noise
> and take a **majority vote**. The vote is provably stable in a ball around the input.

**Analogy.** Imagine asking a slightly unreliable witness, "is this real or fake?" once — you might get
an unlucky answer. Now ask 1,000 slightly different versions of the question (each with a bit of random
"static" added) and take the majority. If 95% of the noisy versions still say "fake," then small changes
to the scene cannot easily change that majority. The *more lopsided* the vote, the *larger* the
guaranteed-stable region.

**The formula (the only one you need).** If the top class wins a (statistically lower-bounded)
fraction `p_A` of the noisy votes under Gaussian noise of standard deviation `σ`, then the prediction is
provably constant within an L2 ball of radius

```
R = σ · Φ⁻¹(p_A)
```

where `Φ⁻¹` is the inverse standard-normal CDF (a function that grows as `p_A → 1`). Two knobs:
larger noise `σ` and a more confident vote `p_A` both give a larger certified radius `R`.

If the vote is too close (`p_A ≤ 0.5`), the method **abstains** — it refuses to certify rather than
risk being wrong. (Abstention is normally common; one of our findings is that for us it is rare.)

## 2.3 Frozen foundation encoders and "feature space"

Modern detectors don't process raw pixels/waveforms directly. They pass the video through a large
pretrained vision model (**DINOv2**) and the audio through a large pretrained speech model
(**Whisper**), and use the resulting numeric vectors — the **features** or **embeddings**. These big
encoders are **frozen**: their weights are not updated; they are used as fixed feature extractors. Only
a small detector head on top is trained.

- **Input space** = raw pixels and audio samples (millions of numbers).
- **Feature space** = the compact vector the frozen encoders produce (for us, **768 numbers**: 384
from DINOv2 + 384 from Whisper). This is the space the detector actually "sees."

## 2.4 Feature-space vs input-space certification (the key design choice — and the main critique)

We certify in **feature space**: our guarantee is "no perturbation of the 768-dim feature vector
smaller than `R` can flip the decision." We do **not** (directly) certify raw pixels/audio.

- **Why this is the right level for frozen-encoder pipelines.** The detector never sees raw input;
it only sees features. Certifying raw input would require certifying the giant frozen encoders too,
which is currently computationally intractable and would yield tiny, useless radii.
- **The honest caveat.** A real-world attacker perturbs pixels/audio, not features directly. To connect
the two you need to know how much a pixel change can move the feature vector (a "Lipschitz" bound).
We have a **practical bridge** for this (see §3.7): we measure empirically how much feature
movement a given pixel/audio attack causes, and use it to translate feature radii into approximate
input radii. It is an empirical, not a worst-case-formal, bridge — but it directly addresses the
"feature-space only" concern.

## 2.5 The data manifold and "intrinsic dimension" (the geometry that makes everything work)

This is the conceptual heart of the project.

- Our feature vectors live in a space of **D = 768** dimensions.
- But real data does **not** fill all 768 dimensions. If you measure how the features actually spread
out, **~80 directions** capture 90% of all the variation. The remaining ~688 directions are nearly
unused — the data is essentially "flat" along them.
- That ~80-dimensional region where the data actually lives is called the **data manifold**, and the
number 80 is the **intrinsic dimension** `d`. So `d ≈ 80 ≪ D = 768`.

**Analogy.** Picture a sheet of paper floating in a 3-D room. The room is 3-D (`D=3`), but the paper is
effectively 2-D (`d=2`). Anything written on the paper only depends on where you are *on the sheet*,
not on how high the sheet floats. A detector trained on data living on that "sheet" learns a decision
boundary *on the sheet* and is essentially blind to movements *off* the sheet.

## 2.6 Why the certified radii are large (our Theorem 1, in words)

Combine §2.2 and §2.5:

- Randomized smoothing adds noise in **all 768 directions**.
- But the detector only "cares about" the ~80 on-manifold directions; it is (approximately) invariant
to the other ~688.
- So most of the noise is "wasted" on directions the detector ignores. The classifier effectively only
feels the noise inside the small ~80-dim region. With less *effective* noise disturbing it, its vote
stays lopsided (`p_A` near 1), so the certified radius `R` is **large**.

**The amplification factor (Corollary 1).** An attacker who doesn't know the manifold spreads their
budget across all 768 directions, so only a `d/D` fraction lands in the directions that matter. To
actually move the decision by an amount `m` on-manifold, they must spend

```
amplification  A = √(D / d) ≈ √(768 / 80) ≈ 3.1×
```

more budget than the naive radius suggests. Lower `d/D` ⇒ bigger amplification ⇒ more certifiable.
This is a *design principle*, not just an observation (see §2.8).

## 2.7 Anisotropic smoothing — spending the noise budget wisely (sphere → ellipsoid)

Standard ("isotropic") smoothing uses the **same** amount of noise in every direction — a **sphere**.
But §2.6 says off-manifold noise is wasted. **Anisotropic** smoothing instead shapes the noise like an
**ellipsoid**: lots of noise along the ~80 directions that matter, almost none along the ~688 that
don't — *without using more noise overall* (we keep the total noise "budget" fixed).

**Analogy.** You have a fixed budget of paint to protect a fence. The fence is long and thin. Isotropic
= paint a circle (wasting paint above and below the fence). Anisotropic = paint a long thin stripe that
matches the fence — same paint, far more of the fence actually covered.

Result: the **on-manifold certified radius jumps from 2.22 to 7.63 (a 3.4× increase)** at the same
budget, and abstention drops to **zero**. (The "worst-case sphere" radius becomes tiny — that's
expected and by design, because we deliberately put almost no protection in the directions the
classifier ignores anyway.)

## 2.8 The "encoder scaling law"

Corollary 1 predicts: *encoders whose features have a smaller intrinsic-dimension ratio `d/D` give
larger certified radii.* We tested this across **five different encoder pairs** (different vision and
audio backbones). The prediction holds at the extremes: the lowest `d/D` pairs (self-supervised audio
encoders like WavLM/HuBERT) give the largest radii; the highest `d/D` pair (CLIP) gives the smallest —
*even though* CLIP has the best clean accuracy. So there is a **measurable trade-off**: pick encoders by
`d/D` if you want certifiability. We call this trend a "scaling law" (a dominant trend, not an exact
formula — five points is not enough to claim a precise law, and we say so).

## 2.9 Conformal prediction — a safety net for the abstentions (Draft 2 / 8-page only)

Randomized smoothing's weakness: when it's unsure it **abstains**, giving you nothing. **Conformal
prediction** is a different, complementary guarantee. Instead of a single label, it outputs a
**set** of labels (`{real}`, `{fake}`, or `{real, fake}`) calibrated so that the true label is inside
the set at least, say, 90% of the time — a **coverage guarantee** that holds for *every* sample,
including the ones smoothing abstained on.

**Analogy.** A weather service that must never be "wrong" can hedge: instead of "rain," it says
"rain or cloudy" when unsure. The set is wider when it's less certain, but it's calibrated to contain
the truth with a stated probability. For a forensic analyst, "{real, fake} — uncertain but covered" is
more useful than "no certificate." **Robust** conformal prediction extends this so the coverage holds
even under adversarial perturbation.

---

# Part 3 — What we built and did

## 3.1 The CertAV pipeline (architecture in plain terms)

1. **Frozen encoders.** Video → DINOv2-Small (16 frames → 384-d features). Audio → Whisper-tiny
  (≤10 s → 384-d features). Weights frozen; features cached to disk so we don't re-run the big models.
2. **Detector head (trained).** Each modality is summarized over time (a small transformer for video,
  pooling for audio), then a **cross-modal attention** module lets video and audio "look at" each
   other to spot inconsistencies, and a small classifier outputs real/fake.
3. **Certification.** At test time we inject Gaussian noise (isotropic or anisotropic) into the 768-d
  feature vector, take 1,000 noisy votes, and emit: a prediction, an abstain flag, and a certified
   radius `R`.

## 3.2 Data and protocol

- **Primary dataset:** FakeAVCeleb — 3,850 train / 825 val / 825 test clips, with a **1:10**
real:fake ratio (this imbalance matters later).
- **Transfer dataset:** LAV-DF (a different deepfake dataset) to test generalization.
- **Repeatability:** main results are averaged over **5 random seeds** (variance is small, e.g.
±0.6% accuracy), which is why the more expensive Phase-2 studies use a single seed.

## 3.3 Headline isotropic result

At noise level σ = 1.0 on FakeAVCeleb:


| Metric                          | Value     |
| ------------------------------- | --------- |
| Clean accuracy                  | **92.5%** |
| Mean certified radius           | **2.215** |
| Certified accuracy @ radius 1.0 | **88.0%** |
| Certified accuracy @ radius 1.5 | **84.9%** |
| Abstention rate                 | **< 1%**  |


Two unusual facts: bigger noise *increases* the radius **without** hurting clean accuracy, and
abstention is tiny. Both are explained by the manifold geometry (Theorem 1).

## 3.4 The diagnostic that pins down the cause (geometry, not training)

- A **no-noise** baseline (trained without noise augmentation) still certifies to radius **2.173** —
within **1.9%** of the noise-trained model. ⇒ The certifiability is **not** created by the training
trick; it's a property of the frozen representation's geometry.
- **PGD adversarial training** (the standard "train harder" defense) *raises* the radius to 2.463 but
**collapses** the model's discrimination quality (validation AUC 0.941 → 0.730). ⇒ It "wins" the
certificate by flattening the decision boundary so everything looks stable — a warning, not a
recipe. This is why we frame the work as *certification*, not *defense-by-training*.

## 3.5 The theory (proved) and its validation

- **Theorem 1 (Certifiability Scaling):** under subspace invariance, the certified radius depends only
on the ~80-dim on-manifold noise. **Corollary 1:** amplification `A = √(D/d)`.
- **Validated three ways:** (a) the 5-encoder scaling study (§2.8); (b) the no-noise vs noise-trained
near-tie (§3.4); (c) a measurement that PGD attacks put only **cos²θ ≈ 0.18** (18%) of their energy
on-manifold — i.e., 82% is wasted in directions the classifier ignores, exactly the regime the
theorem assumes.

## 3.6 The anisotropic headline (Phase 2)

PCA-aligned anisotropic smoothing (Strategy 2: concentrate noise in the manifold) at **equal total
budget**: on-manifold certified radius **7.63 vs 2.22 (3.4×)**, **0% abstention**, certified accuracy
@ r=1.0 of **90.9%** vs 88.0%. This is the strongest single result.

## 3.7 The input-space bridge (already implemented — important for reviewers)

We already have `scripts/24_compose_input_certificate.py` + `composed_input_certificate.json`: it
measures, empirically, how much a pixel/audio attack of size ε moves the feature vector (an empirical
Lipschitz constant `L`), then converts feature radii into **approximate input-space radii** (`R/L`).
It is labeled as an empirical bridge, not a formal worst-case proof — but it directly addresses the
"you only certify features" critique, and **none of the three reviewers knew it existed**.

## 3.8 Conformal prediction result (Draft 2)

Standard conformal achieves ~96.8% coverage on clean data at the 90% target and degrades gracefully
under attack. **Known soft spot:** because the dataset is 1:10 real:fake, coverage on the minority
**"real"** class is much lower (down to ~55% under strong attack) than on "fake" (>98%). This is a
property of class imbalance, not of the method, but it needs honest, prominent discussion because the
"real" class is the safety-critical one (false accusations).

## 3.9 av-robustbench

A released Python library packaging the whole evaluation (attacks, certification, degradations,
robustness "cards"), so every number in the paper is reproducible from a structured pipeline.

---

# Part 4 — Contributions, status, and honest limitations

## 4.1 The five core contributions

1. **CertAV pipeline** — feature-space randomized smoothing for AV deepfake detection.
2. **Certifiability Scaling theorem** + amplification corollary (the geometric explanation).
3. **Encoder-scaling study** validating the theory across 5 encoder families.
4. **Manifold-aware anisotropic smoothing** — 3.4× on-manifold radius at equal budget, 0% abstention.
5. **Robust conformal prediction** — a coverage guarantee for *every* sample (8-page version).

## 4.2 Honest limitations (we state these in the paper)

1. Feature-space scope (mitigated by the §3.7 bridge, but not a formal end-to-end proof).
2. Single primary dataset + one transfer set.
3. Class imbalance hurts minority-class conformal coverage.
4. Certification cost (1,000 forward passes per clip).
5. Subspace invariance is approximate (cos²θ = 0.18 ≠ 0), not exact.

## 4.3 Where it stands (external review consensus)

Three independent LLM reviews agree on the shape: **strong theory and narrative; the empirical
breadth (datasets, attack diversity) and the feature-space scope are the vulnerabilities.** All three
rate it an excellent fit for **ICASSP** (signal-processing + audio-visual + theory) and for a
**forensics/signal Q1 journal** (e.g., IEEE TIFS, Information Fusion), and a **harder sell at CVPR**,
where reviewers demand end-to-end guarantees and multi-dataset benchmarks. (See the separate strategy
note for the submission plan.)

---

# Part 5 — One-page glossary


| Term                        | Plain meaning                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------ |
| **Adversarial attack**      | A tiny crafted perturbation that flips a model's decision.                           |
| **Empirical robustness**    | "Survived the attacks we tried." No guarantee beyond them.                           |
| **Certified robustness**    | A *proof* that no perturbation below radius `R` can flip the decision.               |
| **Randomized smoothing**    | Vote over many noisy copies; the majority is provably stable in a ball.              |
| **Certified radius `R*`*    | The provable safe radius: `R = σ·Φ⁻¹(p_A)`.                                          |
| **Abstention**              | The certifier refuses to answer when the vote is too close (`p_A ≤ 0.5`).            |
| **Frozen encoder**          | A large pretrained model used as a fixed feature extractor (DINOv2, Whisper).        |
| **Feature / input space**   | The 768-d vector the detector sees / the raw pixels & audio.                         |
| **Data manifold**           | The low-dimensional region where real features actually live.                        |
| **Intrinsic dimension `d`** | How many directions the data really uses (~80 of 768 here).                          |
| **Subspace invariance**     | The detector reacts only to on-manifold movement, ignores the rest.                  |
| **Amplification `√(D/d)`**  | How much extra budget an attacker needs because most directions are ignored.         |
| **Isotropic smoothing**     | Equal noise in all directions (a sphere).                                            |
| **Anisotropic smoothing**   | Noise shaped to the manifold (an ellipsoid); same budget, bigger on-manifold radius. |
| **Encoder scaling law**     | Lower `d/D` ⇒ larger certified radius (validated across 5 encoders).                 |
| **Conformal prediction**    | Outputs a *set* of labels with a calibrated coverage guarantee.                      |
| **Robust conformal**        | Conformal coverage that holds even under bounded adversarial perturbation.           |
| **Lipschitz bound**         | How much a feature can move per unit of input change (links feature ↔ input radii).  |
| **PGD**                     | A standard strong iterative adversarial attack, used here as a stress test.          |
| **AUC / EER**               | Quality-of-ranking metrics; PGD-AT inflated the radius but wrecked these.            |


---

*Prepared as a standalone brief; the two paper drafts (`paper/certav_icassp2027.pdf` and
`paper-cvpr/main.pdf`) contain the full technical detail, tables, and proofs.*