# Reviewer Assessment: CertAV / CertAV-Bench

## Overall Verdict

This is a strong, publishable research project if the paper is framed carefully.
The core contribution is not merely "randomized smoothing for AV deepfake
detection"; the stronger story is:

> Frozen foundation-model audio-visual feature spaces can be inherently
> certifiable, and CertAV turns that geometry into practical randomized-smoothing
> certificates for deepfake detection.

My current rating depends on venue:

- ICASSP / ICIP / WACV: strong submission if written cleanly.
- ACM Multimedia: plausible if `av-robustbench` is emphasized as a community resource.
- IEEE TIFS: strong journal target after tightening experiments and threat-model discussion.
- CVPR / ICCV / ECCV main track: not yet at the bar unless input-space validation,
  cross-dataset breadth, and comparisons are expanded.

## Front-by-Front Evaluation

| Front | Rating | Assessment |
|:---|:---:|:---|
| Problem importance | 8.5/10 | Certified robustness for audio-visual deepfake detection is timely and underexplored. |
| Novelty | 7.5/10 | Feature-space smoothing itself is adjacent to prior work, but AV deepfake certification plus joint multimodal threat modeling is fresh. |
| Technical soundness | 8/10 | The Cohen-style smoothing pipeline, Clopper-Pearson bounds, joint L2 threat model, and certification outputs are sound. |
| Empirical strength | 7.5/10 | Five-seed CertAV results are solid; elevation experiments are persuasive but some are pilot-scale. |
| Baselines | 7/10 | No-noise and PGD-AT are valuable. More detector baselines would help for top CV venues. |
| Reproducibility | 8/10 | Scripts, aggregate JSONs, figures, benchmark package, and tests are now accessible. |
| Community contribution | 8/10 | `av-robustbench` is genuinely useful and can become a second contribution. |
| Paper readiness | 7/10 | Ready to draft, but claim language must be tightly regulated. |

## Strongest Evidence

- CertAV sigma=1.00 five-seed result: accuracy 0.925 +/- 0.006, mean certified radius 2.215 +/- 0.032.
- No-noise baseline certifies almost as well: accuracy 0.923 and mean radius 2.173 at sigma=1.00.
- PGD-AT baseline shows the accuracy/robustness tradeoff: accuracy 0.905, mean radius 2.463, but lower clean performance.
- LAV-DF zero-shot certification is non-trivial: accuracy 0.746 and mean radius 1.626 at sigma=1.00.
- Input-space pilot is the key credibility result: certificate hold rate is 0.91-0.96 across pixel eps values 0.002-0.020.
- Manifold analysis supports the mechanism: visual dim@90%=73/384, audio dim@90%=13/384, joint dim@90%=75/768.

## Main Weaknesses A Reviewer Will Attack

1. The certified guarantee is in feature space, not raw pixel/waveform space.
   The input-space pilot helps, but it is still a pilot. A top CV reviewer will
   ask whether certificates hold under adaptive attacks through the full encoder.

2. The no-noise baseline weakens the original training story.
   This is not fatal. It is actually interesting, but the paper must pivot from
   "noise augmentation is essential" to "foundation feature geometry is the main
   driver of certifiability."

3. Cross-dataset evidence is limited.
   LAV-DF is useful, but one external dataset is not enough for a top-tier
   generalization claim. A second external dataset or a stronger limitation
   statement would help.

4. Detector comparisons are thin.
   The project compares variants of CMAR/CertAV and PGD-AT. For CVPR/ICCV/ECCV,
   reviewers will expect more detector families or at least adapter-based
   evaluation of one outside AV detector.

5. Threat-model language can easily overclaim.
   You should not imply end-to-end raw-input certified robustness unless the
   certificate is formally composed with encoder Lipschitz bounds or validated
   much more extensively.

## Claim Regulation

Use these claims:

- "We provide feature-space L2 certificates for audio-visual deepfake detection."
- "Input-space PGD pilots show that feature-space certificates remain meaningful
  under small raw-frame perturbations."
- "Frozen DINOv2/Whisper features appear inherently certifiable because the joint
  representation is low-dimensional relative to its ambient dimension."
- "CertAV avoids the clean-accuracy penalty observed in PGD adversarial training."
- "`av-robustbench` standardizes attacks, certification, degradations, robustness
  cards, and leaderboard-format outputs for AV deepfake detectors."

Avoid these claims:

- "First provably robust deepfake detector" unless carefully scoped to
  audio-visual feature-space certificates.
- "Noise-augmented training is necessary" because your own baseline contradicts it.
- "Certified robustness to real-world pixel attacks" unless phrased as pilot
  validation rather than formal certification.
- "No accuracy-robustness tradeoff universally" because this may be specific to
  frozen foundation-feature spaces.

## Recommended Paper Shape

Title:

> CertAV: Certifiably Robust Audio-Visual Deepfake Detection via Feature-Space Randomized Smoothing

Core contributions:

1. A feature-space randomized-smoothing framework for joint audio-visual deepfake detection.
2. A multimodal threat model with joint L2 certification over visual and audio features.
3. Empirical evidence that frozen DINOv2/Whisper features are inherently certifiable.
4. Validation through PGD-AT comparison, LAV-DF transfer, input-space attack pilot, and manifold analysis.
5. `av-robustbench`, a reusable AV robustness benchmark toolkit.

Best paper narrative:

1. Deepfake detectors are brittle and empirical defenses are insufficient.
2. Certification is missing for AV deepfake detection.
3. Feature-space smoothing is practical because foundation features are compact and stable.
4. CertAV certifies joint AV features without sacrificing clean accuracy.
5. The surprising no-noise result reveals a broader phenomenon: certifiability
   may come from the representation manifold itself.

## Venue Strategy

1. ICASSP: best near-term target. The signal-processing angle is natural:
   audio-visual features, randomized smoothing, robustness, and certification.

2. ICIP: good fit if the visual/deepfake robustness story is emphasized and the
   paper stays focused rather than benchmark-heavy.

3. WACV: realistic computer-vision target. Stronger than a workshop, less brutal
   than CVPR/ICCV main, and receptive to applied robustness work.

4. ACM Multimedia: strong fit if you present CertAV plus `av-robustbench` as an
   audio-visual multimedia robustness resource.

5. IEEE TIFS: very suitable journal target after expanding cross-dataset and
   security analysis. Deepfake detection plus formal robustness fits TIFS well.

6. CVPR / ICCV / ECCV main: aspirational. To be competitive, add stronger
   end-to-end input-space attacks, more external datasets, more detector-family
   baselines, and a sharper theoretical explanation.

I would not prioritize "CVIP" as the main target if you mean the common Computer
Vision and Image Processing venues; it is usually not the same tier as CVPR,
ICCV, ECCV, ACM MM, ICASSP, or TIFS.

## Minimum Work Before Submission

1. Write the paper around feature-space certificates and representation geometry.
2. Add a clear threat-model subsection with a boxed "what is certified / what is not certified."
3. Report the no-noise baseline honestly and explain why it strengthens the geometry story.
4. Move the input-space attack pilot into a prominent validation subsection.
5. Include `av-robustbench` as a community contribution, but do not let it distract
   from the CertAV scientific claim.
6. Add one more outside detector or one more outside dataset if targeting ACM MM,
   WACV, CVPR, ICCV, or ECCV.

## Reviewer-Style Decision

If submitted today to ICASSP with a clean paper: weak accept to accept.

If submitted today to ACM MM: borderline, leaning weak accept if `av-robustbench`
is positioned well.

If submitted today to CVPR/ICCV/ECCV main: borderline to weak reject, mainly
because the feature-space threat model and limited detector/dataset breadth will
be scrutinized.

With one more cross-dataset result, stronger input-space adaptive attack
validation, and one non-CMAR detector adapter: this becomes a serious top-tier
multimedia/forensics paper.
