# Literature List

This list records the papers used to shape the draft and the role each plays in the argument.

## Certified robustness

1. Cohen, Rosenfeld, and Kolter, "Certified Adversarial Robustness via Randomized Smoothing," ICML 2019.
   - Core certification theorem and radius formula used by CertAV.
2. Lecuyer et al., "Certified Robustness to Adversarial Examples with Differential Privacy," IEEE S&P 2019.
   - Early randomized-noise certification framing.
3. Salman et al., "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers," NeurIPS 2019.
   - Relationship between adversarial training and smoothed classifiers.
4. Yang et al., "Randomized Smoothing of All Shapes and Sizes," ICML 2020.
   - Context that smoothing can certify more general perturbation families, supporting the broader certification discussion.

## Audio-visual deepfake detection and datasets

5. Khalid et al., "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset," NeurIPS Datasets and Benchmarks 2021.
   - Primary dataset for training, validation, and in-domain certification.
6. Cai et al., "LAV-DF: A Large-Scale Audio-Visual Deepfake Dataset," DICTA 2022.
   - Cross-dataset transfer benchmark.
7. Chugh et al., "Not Made for Each Other: Audio-Visual Dissonance-Based Deepfake Detection and Localization," ACM MM 2020.
   - Early audio-visual inconsistency framing.
8. Mittal et al., "Emotions Don't Lie: An Audio-Visual Deepfake Detection Method Using Affective Cues," ACM MM Workshops 2020.
   - Affective-cue audio-visual detection baseline context.
9. Feng, Chen, and Owens, "Self-Supervised Video Forensics by Audio-Visual Anomaly Detection," CVPR 2023.
   - Self-supervised audio-visual anomaly detection and cross-modal forensic context.
10. Yang et al., "AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake," IEEE TIFS 2023.
    - Strong modern audio-visual joint-learning context.
11. Astrid, Ghorbel, and Aouada, "Audio-Visual Deepfake Detection with Local Temporal Inconsistencies," ICASSP 2025.
    - Recent ICASSP-relevant audio-visual temporal inconsistency detector.

## Foundation representations

12. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024.
    - Frozen visual representation used in CertAV.
13. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2023.
    - Whisper encoder and large-scale audio representation context.

## Benchmarks

14. Croce et al., "RobustBench: A Standardized Adversarial Robustness Benchmark," NeurIPS Datasets and Benchmarks 2021.
    - Benchmarking precedent for `av-robustbench`.

## Deliberately omitted or de-emphasized

- Broad deepfake-survey papers are not central enough for a short ICASSP paper.
- Raw-image adversarial-robustness papers are cited only when needed for the smoothing or adversarial-training argument.
- Recent feature-space smoothing papers should be added only after exact bibliographic verification; the current draft makes the feature-space contribution through its own method rather than relying on uncertain citations.
