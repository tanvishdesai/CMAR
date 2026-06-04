# Writing Notes: Related Work & Positioning

> **Purpose**: Reference this file during the drafting phase. It contains citation suggestions, differentiation arguments, and framing guidance for each section of the CertAV paper.

---

## Key Papers to Cite

### Certified Robustness Foundations
| Paper | Citation Key | Why Cite |
|:---|:---|:---|
| Cohen et al. (2019) "Certified Adversarial Robustness via Randomized Smoothing" | `cohen2019certified` | The foundational randomized smoothing paper. CertAV builds directly on this. Cite the accuracy-robustness tradeoff prediction that we contradict. |
| Salman et al. (2019) "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers" | `salman2019provably` | Combines adversarial training + smoothing. We compare against pure AT. |
| Lecuyer et al. (2019) "Certified Robustness to Adversarial Examples with Differential Privacy" | `lecuyer2019certified` | PixelDP — early certified defense using noise. |
| Yang et al. (2020) "Randomized Smoothing of All Shapes and Sizes" | `yang2020randomized` | Extends smoothing beyond L2. |

### Feature-Space Smoothing (Directly Related)
| Paper | Citation Key | Why Cite |
|:---|:---|:---|
| Feature-space Smoothing for MLLMs (2026) — arxiv | `fs_mllm_2026` | Most directly related: applies smoothing to feature encoder of multimodal LLMs. **Differentiate**: they certify cosine similarity bounds, we certify classification accuracy. They target text+image VQA, we target AV deepfake detection. |
| Hybrid Randomized Smoothing (2026) — arxiv | `hybrid_smoothing_2026` | Joint certification for mixed discrete/continuous inputs. **Differentiate**: theoretical framework for text+image, not AV deepfake-specific. We provide practical instantiation. |
| Projected Randomized Smoothing — Berkeley | `projected_smoothing` | Smoothing in lower-dimensional latent spaces. **Connection**: our feature-space approach is a practical instance of projected smoothing. |

### Deepfake Detection — Adversarial Robustness
| Paper | Citation Key | Why Cite |
|:---|:---|:---|
| ROBUST-DETECT (2026) — IJERT | `robust_detect_2026` | Claims certified ℓp robustness for deepfake detection. **CRITICAL DIFFERENTIATION**: Uses randomized preprocessing + prediction hardening + dynamic thresholding. This is NOT formal randomized smoothing — no Clopper-Pearson bounds, no certified radius computation. We provide TRUE mathematical certificates via Cohen et al. framework. They are visual-only; we are audio-visual. |
| Adversarially Robust Deepfake Detection via Feature Similarity Learning (2025-2026) | `adv_robust_df_2025` | Empirical adversarial training for deepfake detection. **Differentiate**: empirical defense only — no provable guarantees. Our Table X shows AT gives higher empirical robustness but ZERO certificates. |
| Circumventing Shortcuts in AV Deepfake Detection (CVPR 2025) | `shortcuts_cvpr2025` | Unsupervised AV detection avoiding dataset biases. Cite for AV detection context. |
| AVoiD-DF (ICASSP 2026) | `avoid_df_2026` | Multi-modal joint decoder for AV deepfake detection. Cite as a potential model to certify in future work. |

### Deepfake Detection — Benchmarks & Datasets
| Paper | Citation Key | Why Cite |
|:---|:---|:---|
| FakeAVCeleb (Khalid et al., 2021) | `khalid2021fakeavceleb` | Our primary dataset. NeurIPS 2021. |
| LAV-DF (Cai et al., 2022) | `cai2022lavdf` | Cross-dataset evaluation. Different distribution (localized manipulations). |
| ForensicHub (NeurIPS 2025) | `forensichub2025` | Unified forensic benchmark. Cite to position our future benchmark work. |

### Foundation Models Used
| Paper | Citation Key | Why Cite |
|:---|:---|:---|
| DINOv2 (Oquab et al., 2024) | `oquab2024dinov2` | Our visual feature extractor. |
| Whisper (Radford et al., 2023) | `radford2023whisper` | Our audio feature extractor. |

---

## Differentiation Arguments

### vs. ROBUST-DETECT (2026)
> ROBUST-DETECT employs randomized preprocessing combined with prediction hardening and dynamic threshold adjustment to achieve certified ℓp-robustness. However, their certification mechanism does not provide formal per-sample certified radii with statistical confidence bounds. In contrast, CertAV applies the randomized smoothing framework of Cohen et al. (2019) with Clopper-Pearson confidence intervals, yielding mathematically rigorous per-sample ℓ₂ certified radii. Furthermore, ROBUST-DETECT operates exclusively on visual modality, while CertAV provides joint audio-visual certification.

### vs. Feature-Space Smoothing for MLLMs (2026)
> Recent work on Feature-space Smoothing (FS) certifies the robustness of MLLM feature representations via cosine similarity bounds. CertAV shares the principle of smoothing in feature space rather than input space, but targets a fundamentally different task: binary deepfake detection rather than open-ended visual question answering. Moreover, CertAV demonstrates the novel phenomenon that feature-space smoothing on frozen foundation models eliminates the accuracy-robustness tradeoff observed in standard settings, a finding not reported in prior feature-space smoothing work.

### vs. Adversarial Training Baselines
> Adversarial training (AT) is the dominant empirical defense strategy. While AT-trained models exhibit improved resilience to specific attack types, they provide NO formal guarantee that predictions are stable within any perturbation radius. Our comparison (Table X) demonstrates that PGD-AT achieves [X]% robust accuracy under feature-space PGD at ε=0.05, compared to CertAV's [Y]% smoothed accuracy. However, the AT model produces a certified radius of 0.0 at all samples, while CertAV achieves a mean certified radius of 2.215 at σ=1.00.

---

## Framing the "No Tradeoff" Finding

### What to claim
> "Noise-augmented training at σ=1.00 achieves BOTH the highest clean accuracy (92.5%) AND the largest mean certified radius (2.215), contradicting the standard accuracy-robustness tradeoff observed in randomized smoothing for image classification."

### What NOT to claim
- Do NOT claim this is a universal phenomenon. It may be specific to frozen foundation feature spaces.
- Do NOT claim this contradicts Cohen et al.'s theorem. The theorem holds for worst-case scenarios; your observation is about the practical average case on structured feature spaces.
- Do NOT claim this means certified robustness is "free." It still requires noise-augmented training.

### How to explain it
> "We hypothesize this phenomenon arises because frozen DINOv2 and Whisper features lie on a low-dimensional manifold within the ambient 384-d feature space (see Section X). Gaussian noise in this space primarily has components orthogonal to the data manifold, which do not affect the decision boundary. Simultaneously, the on-manifold noise components act as a regularizer, preventing overfitting and improving generalization. This is in contrast to pixel-space smoothing, where noise occupies all dimensions of a much higher-dimensional space and directly corrupts discriminative pixel patterns."

---

## Paper Section-by-Section Notes

### Abstract (~200 words for ICASSP)
- Open with: deepfake detectors vulnerable to adversarial attacks
- Claim: first certified AV deepfake detector
- Method: feature-space randomized smoothing on frozen DINOv2+Whisper
- Result: 92.5% clean accuracy, 88.0% cert@r=1.0, "no tradeoff"
- Close with: 5-seed rigorous evaluation, cross-dataset validation

### Introduction
- Paragraph 1: Deepfakes are a societal threat; detection is critical
- Paragraph 2: Existing detectors are brittle — adversarial attacks collapse them
- Paragraph 3: Empirical defenses (AT) provide no guarantees; need provable robustness
- Paragraph 4: Gap — NO certified defense exists for AV deepfake detection
- Paragraph 5: Our contribution — CertAV, three claims

### Method
- 3.1 Threat model: ℓ₂-bounded adversary in joint feature space
- 3.2 Architecture: CMAR backbone (DINOv2+Whisper → CMCM cross-attention)
- 3.3 Noise-augmented training: Gaussian N(0,σ²I) injection
- 3.4 Certification: Two-phase Cohen et al. procedure
- 3.5 Feature-space vs input-space: justify why feature-space is appropriate

### Experiments
- 4.1 Setup: FakeAVCeleb, 5 seeds, σ ∈ {0.12, 0.25, 0.50, 1.00}
- 4.2 Main results: Table 1 (certification curves)
- 4.3 Baselines: Table 2 (no noise baseline, PGD-AT baseline)
- 4.4 Ablation: Table 3 (joint vs unimodal noise)
- 4.5 Cross-dataset: Table 4 (LAV-DF zero-shot)
- 4.6 Input-space validation: Certificate hold rate
- 4.7 Manifold analysis: Intrinsic dimensionality
- 4.8 Degradation robustness: Table 5

### Conclusion
- Summarize contributions
- Limitations: single dataset (mitigated by LAV-DF), feature-space certification (mitigated by input-space pilot)
- Future work: larger datasets, adaptive attacks, av-robustbench toolkit
