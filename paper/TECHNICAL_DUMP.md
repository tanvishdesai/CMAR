# Technical Dump: CertAV / CMAR Project

## Core idea

CertAV studies certified robustness for audio-visual deepfake detection by applying randomized smoothing in the frozen feature space of a multimodal detector. The detector uses DINOv2 visual embeddings and Whisper audio embeddings, aggregates temporal evidence, fuses the modalities with cross-modal attention, and predicts whether a clip is real or fake. Certification is performed over the joint feature vector, giving per-sample L2 radii under a feature-space threat model.

The main scientific finding is that the frozen representation is already highly certifiable: a no-noise baseline certifies nearly as well as a noise-trained model, while feature-space PGD adversarial training reduces validation AUC. The paper should present this as a positive finding about foundation-feature geometry, not as a failure of the proposed method.

## Data

- Primary dataset: FakeAVCeleb.
- Deterministic split seed: 2026.
- Train: 350 real and 3500 fake clips, 3850 total.
- Validation: 75 real and 750 fake clips, 825 total.
- Test: 75 real and 750 fake clips, 825 total.
- Labels: real = 0, fake = 1.
- Cross-dataset transfer: LAV-DF, 500 test samples.

## Feature extraction

- Visual encoder: DINOv2-Small from timm (`vit_small_patch14_dinov2.lvd142m`), frozen.
- Visual input: 16 frames, 224 px preprocessing.
- Visual feature dimension: 384 per frame.
- Audio encoder: Whisper tiny (`openai/whisper-tiny`), frozen encoder.
- Audio input: 16 kHz waveform, maximum 10 seconds.
- Audio feature dimension: 384 per token; temporal length pooled to at most 64.
- Feature cache: float16 on disk, converted to float32 for training and certification.

## Detector architecture

- Visual temporal aggregator:
  - 384 -> 256 projection.
  - Learned temporal positional embedding.
  - One Transformer encoder layer.
  - 8 attention heads.
  - Feed-forward width 4x hidden dimension.
  - Dropout 0.1.
  - Layer normalization.
  - Adaptive temporal pooling to 8 segments.
- Audio temporal aggregator:
  - 384 -> 256 projection.
  - Adaptive temporal pooling to 8 segments.
  - Layer normalization.
- Cross-modal consistency module:
  - Two bidirectional attention layers.
  - Visual attends to audio and audio attends to visual.
  - 8 heads, residual connections, layer normalization, GELU feed-forward blocks.
  - Fusion projection from 512 to 256.
- Classifier:
  - LayerNorm.
  - Linear 256 -> 128.
  - ReLU.
  - Dropout 0.3.
  - Linear 128 -> 1.
  - Segment logits averaged before binary prediction.

## Training

- Objective: binary cross-entropy with logits.
- Noise augmentation: Gaussian noise injected into selected feature modalities.
- Noise modes: joint, visual-only, audio-only.
- Sigma values: 0.12, 0.25, 0.50, 1.00.
- Seeds: 42, 69, 420, 2026, 2804.
- Optimizer: AdamW.
- Learning rate: 5e-4.
- Weight decay: 0.01.
- Batch size: 8.
- Gradient accumulation: 4.
- Epochs: 30.
- Warmup: 3 epochs.
- Schedule: cosine decay.
- Gradient clipping: 1.0.
- Early stopping patience: 7 epochs.
- Validation: average predictions over 10 noise samples.

## Certification

The smoothed classifier is

`g(z) = argmax_c P(f(z + epsilon) = c)`, with `epsilon ~ N(0, sigma^2 I)`.

Certification uses a two-stage Monte Carlo procedure:

- `n0 = 100` samples for class selection.
- `n = 1000` samples for certification.
- `alpha = 0.001`, corresponding to 99.9% confidence.
- Clopper-Pearson lower confidence bound.
- Certified radius: `R = sigma * Phi^{-1}(lower_p_A)`.
- Abstain if the lower bound is not greater than 0.5.
- Certified accuracy at radius `r`: fraction of samples that are correct, non-abstained, and have radius at least `r`.

## Main results: joint noise-trained CertAV

| Sigma | Accuracy | Abstain | Mean radius | Cert@0.25 | Cert@0.50 | Cert@1.00 | Cert@1.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.12 | 91.2 +/- 1.0 | 0.3 | 0.288 +/- 0.003 | 88.8 | 0.0 | 0.0 | 0.0 |
| 0.25 | 91.2 +/- 1.1 | 0.4 | 0.588 +/- 0.012 | 89.4 | 86.6 | 0.0 | 0.0 |
| 0.50 | 91.5 +/- 1.1 | 0.4 | 1.165 +/- 0.034 | 90.5 | 89.2 | 85.4 | 0.0 |
| 1.00 | 92.5 +/- 0.6 | 0.8 | 2.215 +/- 0.032 | 91.8 | 90.7 | 88.0 | 84.9 |

## Validation AUC / EER

| Mode | Sigma | AUC | EER |
| --- | ---: | ---: | ---: |
| joint | 0.12 | 0.920 +/- 0.005 | 0.160 +/- 0.008 |
| joint | 0.25 | 0.920 +/- 0.003 | 0.173 +/- 0.014 |
| joint | 0.50 | 0.915 +/- 0.029 | 0.167 +/- 0.038 |
| joint | 1.00 | 0.941 +/- 0.007 | 0.139 +/- 0.009 |
| visual-only | 0.25 | 0.918 +/- 0.016 | 0.163 +/- 0.019 |
| audio-only | 0.25 | 0.912 +/- 0.016 | 0.176 +/- 0.022 |
| visual-only | 1.00 | 0.931 +/- 0.011 | 0.148 +/- 0.015 |
| audio-only | 1.00 | 0.909 +/- 0.007 | 0.186 +/- 0.022 |

## Modality ablation

| Mode | Sigma | Accuracy | Mean radius | Cert@0.25 | Cert@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 0.25 | 91.2 +/- 1.1 | 0.588 +/- 0.012 | 89.4 | 86.6 |
| visual-only | 0.25 | 91.0 +/- 1.1 | 0.589 +/- 0.006 | 89.3 | 86.5 |
| audio-only | 0.25 | 90.9 +/- 0.3 | 0.607 +/- 0.002 | 90.3 | 89.5 |
| joint | 1.00 | 92.5 +/- 0.6 | 2.215 +/- 0.032 | 91.8 | 90.7 |
| visual-only | 1.00 | 92.3 +/- 0.3 | 2.273 +/- 0.029 | 91.6 | 90.7 |
| audio-only | 1.00 | 91.9 +/- 0.5 | 2.404 +/- 0.034 | 91.7 | 91.5 |

Interpretation: unimodal radii can match or exceed joint radii, but the certified threat is narrower because only one modality is perturbed. The paper should not imply joint smoothing strictly dominates unimodal smoothing numerically.

## Baselines and negative results

- No-noise baseline:
  - Validation AUC: 0.8773.
  - EER: 0.194.
  - Sigma 0.25 certification: 90.7% accuracy, 0.7% abstention, mean radius 0.586, Cert@0.25 87.7%, Cert@0.50 85.3%.
  - Sigma 1.00 certification: 92.3% accuracy, 0.7% abstention, mean radius 2.173, Cert@0.25 91.3%, Cert@0.50 90.0%, Cert@1.00 86.3%, Cert@1.50 82.0%.
- Feature-space PGD adversarial training:
  - Validation AUC: 0.7298.
  - EER: 0.3347.
  - Sigma 1.00 certification: 90.5% accuracy, 0.0% abstention, mean radius 2.463, Cert@1.00 90.5%, Cert@1.50 90.5%.
  - Interpretation: PGD-AT can harden smoothed predictions but hurts ranking/separation quality substantially.

## Empirical attack stress tests

Smoothed accuracy under feature-space PGD drops with perturbation size, while larger smoothing noise improves resistance to small attacks:

- Sigma 0.25: smoothed accuracy 53.4% at eps 0.05, 18.2% at eps 0.10, 4.6% at eps 0.20.
- Sigma 0.50: smoothed accuracy 65.2% at eps 0.05, 38.5% at eps 0.10, 20.2% at eps 0.20.
- Sigma 1.00: smoothed accuracy 69.1% at eps 0.05, 32.8% at eps 0.10, 5.9% at eps 0.20.

These results are not the primary proof; the certificate is the primary robustness result. The empirical attacks are diagnostic stress tests.

## Degradation robustness

For sigma 1.00:

- Social-media-style degradation: Cert@0 91.3%, Cert@0.25 90.8%, Cert@0.50 90.5%.
- H.264 CRF 28: Cert@0 90.7%, Cert@0.25 90.3%, Cert@0.50 89.9%.
- JPEG quality 75: Cert@0 91.6%, Cert@0.25 91.0%, Cert@0.50 90.5%.

## LAV-DF transfer

- Sigma 0.25: accuracy 60.0%, abstention 1.2%, mean radius 0.527, Cert@0.25 55.0%, Cert@0.50 49.8%.
- Sigma 1.00: accuracy 74.6%, abstention 2.6%, mean radius 1.626, Cert@0.25 72.2%, Cert@0.50 69.2%, Cert@1.00 61.4%, Cert@1.50 53.2%.

Interpretation: the certificate transfers better than expected at high sigma, but cross-dataset generalization is still a limitation.

## Feature-displacement validation

The script named as an input-space attack actually uses feature-displacement proxies because raw frames are not available in the cached-feature setting. The correct paper language is "feature-displacement validation" or "input-proxy stress test."

Results:

- eps 0.002: mean feature L2 0.100, clean accuracy 92.0%, adversarial accuracy 92.0%, certificate-hold rate 96.0%.
- eps 0.005: mean feature L2 0.250, clean accuracy 92.0%, adversarial accuracy 90.0%, hold rate 96.0%.
- eps 0.010: mean feature L2 0.499, clean accuracy 92.0%, adversarial accuracy 90.0%, hold rate 94.0%.
- eps 0.020: mean feature L2 1.000, clean accuracy 92.0%, adversarial accuracy 88.0%, hold rate 91.0%.

## Manifold analysis

- Visual features:
  - 90% variance dimension: 73 / 384.
  - 95% variance dimension: 113 / 384.
- Audio features:
  - 90% variance dimension: 13 / 384.
  - 95% variance dimension: 36 / 384.
- Joint features:
  - 90% variance dimension: 75 / 768.
  - 95% variance dimension: 116 / 768.
- Prediction flip rate at sigma 1.00: 1.0%.

Interpretation: the effective data manifold occupies a small subspace of the concatenated feature space, explaining why isotropic Gaussian smoothing can produce large certificates without destroying classification.

## av-robustbench

The repository includes a Python library for evaluating audio-visual robust detection:

- Detector adapter interface.
- PGD-Linf and PGD-L2 attacks.
- Joint L2 projection across visual and audio features.
- Square Attack and AutoAttack-style wrappers.
- Randomized smoothing certification.
- Degradation battery.
- Robustness-card and leaderboard JSON outputs.

In the paper, this should be described as a reproducibility artifact and benchmark scaffold, not as a fully populated public leaderboard unless the release infrastructure is complete.

