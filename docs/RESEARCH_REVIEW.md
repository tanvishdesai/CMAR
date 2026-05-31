# Research Proposal Review

## Overall Assessment

The proposal is promising because the central question is clean: do audio-visual detectors degrade more gracefully than unimodal detectors under single-modality attacks and realistic media degradation?

That is a real research angle. The asymmetric attack framing is stronger than simply proposing another fusion architecture.

## Strengths

- The hypothesis is testable with clear conditions: visual-only, audio-only, and both-modality attacks.
- FakeAVCeleb is an appropriate primary dataset because it contains RR, FR, RF, and FF categories.
- CMRR gives the paper a memorable quantitative hook.
- Feature caching is practical for Kaggle and makes the work reproducible.
- The ablations are well chosen: visual-only, no-consistency, CMCM depth, and TTDA.

## Main Risks

- If the final experiments only use cached-feature attacks, reviewers may reject the adversarial robustness claim as not truly input-space adversarial.
- The default cached-feature training path does not LN-tune DINOv2/Whisper, so the paper must not overclaim LN-tuning unless raw-mode runs are added.
- FakeAVCeleb is heavily imbalanced. AUC and AP are appropriate, but thresholded accuracy alone would be weak.
- Baseline execution is a practical risk because LipForensics, DeepfakeBench/Xception, and AASIST checkpoints can be brittle to adapt.
- Cross-dataset LAV-DF evaluation may underperform because LAV-DF has localized manipulations and a different data distribution.

## Paper Potential

Chance of becoming a good research paper: moderate to good, assuming the adversarial evaluation is made rigorous.

The paper is most likely to be compelling if these results hold:

- CMAR clean AUC is competitive, even if not best.
- CMAR has higher RAR than baselines on D12 and other realistic degradations.
- CMAR keeps much higher AUC under visual-only attack than the visual-only ablation.
- Both-modality attacks hurt substantially more than single-modality attacks.
- Late fusion does not get the same CMRR benefit as learned cross-modal fusion.

If those results do not hold, the project can still become a useful negative/diagnostic paper, but the ICASSP case becomes harder.

## Recommendation

Keep the project, but treat input-space adversarial evaluation as non-negotiable for the final paper. The code currently provides a fast cached-feature proxy; the next research milestone should be raw visual/audio PGD through the encoders on a small subset, then scaled once memory is understood.

## Update After V1 Results

The first CMAR run supports a safer intermediate claim, not the original strong
claim yet. Clean FakeAVCeleb AUC is just above the minimum target, and D12 social
simulation RAR is strong, so real-world degradation robustness is a promising
paper direction.

However, the feature-space adversarial proxy collapses badly under audio and
both-modality attacks. Because this attack perturbs cached features rather than
raw frames or waveforms, it should not be interpreted as final adversarial
evidence. It is still an important warning: the phrase "inherent adversarial
robustness" is too strong until input-space attacks and baselines support it.

Recommended paper posture now:

- Keep CMAR as the method.
- Lead with robust foundation-feature fusion under compression, resize, noise,
  H.264, and social-media simulation.
- Treat cross-modal adversarial robustness as a hypothesis under audit, not as a
  settled conclusion.
- If raw input-space attacks later show single-modality attacks hurt CMAR less
  than unimodal/late-fusion baselines, restore the stronger CMRR story.
- If raw attacks also collapse CMAR, pivot to a robustness-characterization
  paper: multimodal fusion helps realistic degradations but does not
  automatically confer adversarial robustness.
