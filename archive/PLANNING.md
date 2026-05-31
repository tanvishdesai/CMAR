# PLANNING.md — Cross-Modal Adversarial Robustness (CMAR)
## Audio-Visual Deepfake Detection via Foundation Model Feature Fusion

> **AI Agent Instruction**: This is the master reference document for the CMAR research project. Read this file completely at the start of every session before taking any action. Do not modify this file unless explicitly instructed by the researcher. All session-level state, discoveries, and progress are tracked in `CONTEXT.md`. Task status is tracked in `TASK.md`.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | CMAR — Cross-Modal Adversarial Robustness |
| **Full Paper Title** | *Cross-Modal Adversarial Robustness for Audio-Visual Deepfake Detection via Foundation Model Feature Fusion* |
| **Target Venue** | ICASSP 2027 — IEEE International Conference on Acoustics, Speech, and Signal Processing |
| **Conference Location** | Toronto, Canada |
| **Submission Deadline** | **September 16, 2026** |
| **Paper Format** | 5 pages + references (standard track) OR 8+1 pages (IEEE OJSP extended track — decide by Week 6) |
| **Primary Platform** | Kaggle (T4 GPU, 16GB VRAM, 12-hour session limit) |
| **Researcher** | [Your name] |

---

## 2. Research Domain and Motivation

### 2.1 Domain Statement

> **To enhance the robustness of deepfake detection models against adversarial attacks.**

This is the ground truth constraint. Every architectural decision, experiment, and contribution in this paper must map directly back to this statement.

### 2.2 The Problem

State-of-the-art deepfake detectors achieve high accuracy on clean benchmarks but are brittle under adversarial conditions. This matters for two reasons:

1. **Real-world deployment**: Deepfakes shared on social media undergo compression, resolution changes, and re-encoding pipelines that degrade detection accuracy.
2. **Active adversaries**: A motivated attacker who knows a detector is deployed can craft adversarial perturbations — imperceptible to the human eye but sufficient to fool the detector — using gradient-based attacks.

The critical gap is in the **audio-visual (AV) domain**: all prior adversarial robustness work on deepfake detection targets unimodal systems (image-only or audio-only). Nobody has asked: *what happens when you attack a bimodal detector, and does cross-modal fusion provide inherent adversarial protection?*

### 2.3 The Core Claim (Central Hypothesis)

> **Attacking an audio-visual deepfake detector is fundamentally harder than attacking a unimodal detector, because an adversary must corrupt both modalities simultaneously and consistently to fool the system. Cross-modal fusion using frozen foundation model features provides inherent adversarial robustness by creating redundancy across modalities — a single-modality attack leaves the other modality's forgery signal intact.**

This is the paper's thesis. Every experiment must either support, quantify, or stress-test this claim.

### 2.4 Research Gap (Why This Paper Exists)

The following combination has not appeared in the literature as of May 2026:

| Component | Status in Literature |
|---|---|
| LN-tuning of frozen DINOv2 for deepfake detection | Published (GenD WACV 2026; "Generalizes Across Benchmarks" Aug 2025) |
| Audio-visual deepfake detection (FakeAVCeleb benchmark) | Active area (AV-LMMDetect ICASSP 2026, BA-TFD, etc.) |
| Adversarial robustness of audio deepfake detectors | Emerging (ICASSP 2024, arxiv 2025) — audio-only |
| **Adversarial robustness of AV detectors** | **NOT PUBLISHED — our gap** |
| **Asymmetric modality attacks (attack one vs both)** | **NOT PUBLISHED — our contribution** |
| **Cross-modal redundancy as adversarial defense** | **NOT PUBLISHED — our contribution** |

---

## 3. Proposed Method: CMAR Architecture

### 3.1 High-Level Design Philosophy

- **Frozen foundation model encoders**: Do not train from scratch. Use DINOv2-Small and Whisper-Tiny as fixed feature extractors. Only LayerNorm (LN) parameters inside the encoders are tunable. This inherits powerful, generator-agnostic representations without overfitting to the small FakeAVCeleb dataset.
- **Lightweight cross-modal fusion**: A 2-layer cross-attention module (CMCM) learns the consistency relationship between audio and visual features. Authentic AV content has natural synchronization; deepfakes disrupt it.
- **Parameter efficiency**: Only ~4M parameters are trainable out of ~60M total. This prevents overfitting, speeds training, and makes the system deployable.
- **Adversarial robustness by design**: The frozen foundation features are naturally harder to attack (adversarial perturbations tuned for one encoder's feature space may not generalize). Cross-modal redundancy means a visual-only attack cannot suppress the audio forgery signal.

### 3.2 Component Specifications

#### Visual Encoder — DINOv2-Small (ViT-S/14)

| Property | Value | Rationale |
|---|---|---|
| Architecture | ViT-S/14 (Vision Transformer, 14×14 patches) | Proven best-in-class for deepfake detection |
| Total parameters | ~22M | Frozen — fits T4 easily |
| Tunable parameters | ~18K (LayerNorm only) | 0.08% of backbone — prevents overfitting |
| Feature dimension | 384 | Rich enough for cross-modal fusion |
| Input | 16 frames × 224×224×3, uniformly sampled from each clip | Temporal coverage without memory overflow |
| Pre-training | LVD-142M (self-supervised, generator-agnostic) | No deepfake-specific bias |
| Source | `timm`: `vit_small_patch14_dinov2.lvd142m` | |

**Why not DINOv2-Base or DINOv2-Large**: DINOv2-B (86M) and DINOv2-L (300M) exceed T4 memory budget when combined with Whisper-Tiny and a batch of 16-frame clips. DINOv2-Small is the correct choice for this hardware.

**Why not CLIP**: CLIP's contrastive objective creates text-aligned features that are less sensitive to fine-grained local artifacts than DINOv2's self-supervised patch-level features.

#### Audio Encoder — Whisper-Tiny (Encoder Only)

| Property | Value | Rationale |
|---|---|---|
| Architecture | Transformer encoder (4 layers, 6 heads) | Lightweight, proven audio features |
| Total parameters | ~37M (encoder + decoder; we use encoder only: ~15M) | Fits T4 comfortably with DINOv2-Small |
| Tunable parameters | ~7K (LayerNorm only) | Same LN-tuning strategy as visual |
| Feature dimension | 384 | Projected to 256 for fusion |
| Input | 16kHz waveform → 80-channel log-mel spectrogram (Whisper handles internally) | Standard preprocessing |
| Pre-training | 680K hours of multilingual speech (diverse, noisy) | Robust to compression artifacts |
| Source | `openai/whisper-tiny` via HuggingFace Transformers | |

**Why Whisper-Tiny over Whisper-Small**: Whisper-Small encoder is 244M parameters. Even frozen, its memory footprint combined with DINOv2-Small and video batches causes OOM on T4. Whisper-Tiny (encoder ~15M frozen) eliminates this risk entirely while still providing vastly richer audio features than mel-spectrogram CNNs trained on ASVspoof. This was the execution problem in the previous RobustAV plan.

**Why encoder only**: We need frame-level audio features, not transcriptions. The decoder adds parameters and latency with no benefit for our task.

#### Cross-Modal Consistency Module (CMCM)

```
Inputs:
  V_feat: (N_seg × 256)  — pooled visual features after temporal aggregation
  A_feat: (N_seg × 256)  — projected audio features after temporal pooling

Architecture:
  Layer 1: CrossAttention(Q=V, K=A, V=A) + LayerNorm + FFN  → V_enhanced
  Layer 2: CrossAttention(Q=A, K=V, V=V) + LayerNorm + FFN  → A_enhanced
  Fusion: Concat([V_enhanced, A_enhanced]) → Linear(512→256) → fused (N_seg × 256)

Trainable parameters: ~5M (all of CMCM is randomly initialized and fully trained)
```

**Why bidirectional cross-attention**: Video attending to audio captures lip-sync alignment; audio attending to video captures emotional/spectral consistency. Unidirectional attention misses half the forgery signal.

**Why 2 layers**: Foundation model features are already high quality. The CMCM needs only to learn the consistency mapping, not rebuild representations. 2 layers (~5M params) is sufficient and avoids overfitting on FakeAVCeleb's ~3,850 training clips.

#### Temporal Aggregation

- **Visual**: Learnable [CLS] token + 1-layer Transformer over 16 per-frame features → (N_seg=8 × 256)
- **Audio**: Linear projection (384→256) + mean pooling over Whisper's temporal output → (N_seg=8 × 256)
- **Segment alignment**: Both streams are aligned to N_seg=8 temporal segments. Visual frames are grouped 2 per segment; audio frames are chunked proportionally.

#### Classification Head

```
Global Average Pool over N_seg dimension → (256,)
Linear(256, 128) → ReLU → Dropout(0.3)
Linear(128, 1) → Sigmoid
Output: P(fake) ∈ [0, 1]
```

### 3.3 Full Parameter Count

| Component | Total Params | Trainable Params | % Trainable |
|---|---|---|---|
| DINOv2-Small | 22M | ~18K (LN only) | 0.08% |
| Whisper-Tiny encoder | ~15M | ~7K (LN only) | 0.05% |
| Visual temporal aggregation | ~0.4M | 0.4M | 100% |
| Audio linear projection | ~0.1M | 0.1M | 100% |
| CMCM (2-layer bidirectional) | ~5M | 5M | 100% |
| Classification head | ~0.04M | 0.04M | 100% |
| **Total** | **~42M** | **~5.56M** | **~13.2%** |

### 3.4 Training Procedure

#### Loss Function

```
L_total = L_BCE + λ_con · L_consistency

Where:
  L_BCE = BinaryCrossEntropy(P(fake), y_label)           weight = 1.0

  L_consistency = KL(P_clean || P_degraded) + KL(P_degraded || P_clean)
                  (symmetric KL divergence between predictions on
                   clean and randomly degraded versions of the same clip)
                  weight λ_con = 0.3

  P_degraded is computed on the fly during training by randomly applying
  one of: JPEG(QF=75), resize(0.75×), Gaussian noise(σ=0.01), MP3(128kbps)
```

**Why no contrastive loss**: Contrastive loss requires careful negative pair construction and is sensitive to batch size. With batch size 8 (T4 constraint), a contrastive loss would need at least 64+ samples for stable gradient estimates. The consistency loss achieves a similar regularization effect (model predicts consistently across clean/degraded versions) without these requirements.

#### Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for transformer fine-tuning |
| LR (LN params in encoders) | 1e-4 | Small LR for pre-trained parameters |
| LR (CMCM + head) | 5e-4 | Larger LR for randomly initialized modules |
| Weight decay | 0.01 | Prevent overfitting |
| Batch size | 8 (physical) | T4 memory constraint |
| Gradient accumulation | 4 steps → effective batch = 32 | Stable gradients |
| Epochs | 30 with early stopping (patience = 5) | Sufficient for convergence |
| Scheduler | Cosine annealing with 3-epoch warmup | Standard practice |
| Video frames per clip | 16 uniformly sampled | Temporal coverage vs memory |
| Audio length | 10s (pad/trim to fixed length) | Covers FakeAVCeleb clips |

### 3.5 Test-Time Consistency (Simplified TTDA)

At inference, the model runs the sample through the pipeline once, then generates K=3 lightly degraded copies (no gradient computation — pure forward pass ensemble) and takes the mean prediction:

```
P_final = mean([P(original), P(jpeg75_copy), P(resize75_copy), P(noise001_copy)])
```

This is a zero-gradient, zero-risk inference-time ensemble. It is simpler and safer than the original RobustAV TTDA (which required a gradient step on LN parameters during inference, risking OOM and instability). The simplified version achieves the same robustness-improving effect and is presented as an optional ablation in the paper.

---

## 4. Datasets

### 4.1 Primary Training and Evaluation Dataset: FakeAVCeleb

| Property | Detail |
|---|---|
| Full name | FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset |
| Paper | Khalid et al., NeurIPS 2021 Datasets Track |
| Total videos | ~500 real + ~19,500 fake |
| Manipulation types | FaceSwap, FSGAN (face swap); Wav2Lip, SV2TTS (lip-sync + TTS audio) |
| AV categories | RealVideo-RealAudio (RR), FakeVideo-RealAudio (FR), RealVideo-FakeAudio (RF), FakeVideo-FakeAudio (FF) |
| Resolution | 224×224 face-cropped |
| Duration per clip | 3–10 seconds |
| Kaggle availability | Available as Kaggle dataset |
| Usage | Training, validation, in-domain test |

**Data split (follow standard race/gender-balanced protocol):**

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 350 | 3,500 | 3,850 |
| Validation | 75 | 750 | 825 |
| Test (clean) | 75 | 750 | 825 |
| Test (degraded variants) | Generated from test split | | |

**Why FakeAVCeleb as primary**: It is the de facto standard benchmark for AV deepfake detection (used in every major AV paper since 2022). It covers all four AV manipulation categories essential for testing cross-modal consistency. Its pre-cropped face format eliminates the need for a face detection preprocessing step that would consume additional Kaggle session time.

### 4.2 Cross-Dataset Generalization: LAV-DF

| Property | Detail |
|---|---|
| Full name | Localized Audio-Visual DeepFake Dataset |
| Paper | Cai et al., 2022 |
| Total video segments | ~36K |
| Manipulation | Localized AV manipulations with temporal boundary annotations |
| Key feature | Per-segment fake/real labels and start/end timestamps |
| Kaggle availability | Available as Kaggle dataset |
| Usage | **Test only** — zero-shot generalization evaluation |

**Critical rule**: CMAR is never trained or fine-tuned on LAV-DF. All LAV-DF results reflect pure cross-dataset generalization from FakeAVCeleb training.

### 4.3 Degraded Test Sets (Generated from FakeAVCeleb Test Split)

These are generated once during Kaggle Session 1 and cached. They are applied to both visual and audio streams as appropriate.

| Degradation ID | Type | Parameters | Stream | Simulates |
|---|---|---|---|---|
| D1 | JPEG compression | QF = 75 | Visual | Standard web upload |
| D2 | JPEG compression | QF = 50 | Visual | Heavy social media compression |
| D3 | Resolution reduction | 0.75× bicubic down/up | Visual | Forwarded media quality loss |
| D4 | Resolution reduction | 0.50× bicubic down/up | Visual | Severe resolution loss |
| D5 | Gaussian noise | σ = 0.01, additive | Visual | Sensor/re-capture noise |
| D6 | Gaussian noise | σ = 0.02, additive | Visual | Stronger noise |
| D7 | MP3 compression | 128 kbps encode/decode | Audio | Social media audio codec |
| D8 | MP3 compression | 64 kbps encode/decode | Audio | Low-quality audio compression |
| D9 | Gaussian audio noise | SNR = 30 dB | Audio | Background/channel noise |
| D10 | Gaussian audio noise | SNR = 20 dB | Audio | Severe background noise |
| D11 | H.264 re-encoding | CRF = 28 | Visual + Audio | Platform video re-encoding |
| D12 | Social media simulation | JPEG75 + resize0.75 + MP3-128k sequential | Both | Realistic sharing pipeline |

**Total degraded conditions**: 12 + 1 clean baseline = 13 test conditions

### 4.4 Adversarial Attack Test Sets

Generated during Kaggle Session 5.

| Attack ID | Attack Type | Target Stream | Budget | Purpose |
|---|---|---|---|---|
| A1 | PGD-20, L∞ | Visual only | ε = 2/255 | Asymmetric: visual attack on AV system |
| A2 | PGD-20, L∞ | Visual only | ε = 4/255 | Stronger visual attack |
| A3 | PGD-20, L∞ | Visual only | ε = 8/255 | Severe visual attack |
| A4 | PGD-20, L∞ | Audio only | SNR equiv = 30 dB | Asymmetric: audio attack on AV system |
| A5 | PGD-20, L∞ | Audio only | SNR equiv = 20 dB | Stronger audio attack |
| A6 | PGD-20, L∞ | Both modalities | ε = 4/255, SNR = 30 dB | Symmetric attack (both streams) |
| A7 | FGSM | Visual only | ε = 4/255 | Baseline single-step visual attack |
| A8 | FGSM | Audio only | SNR = 30 dB | Baseline single-step audio attack |

**Imperceptibility constraints** (attacks that violate these are invalid and must be re-run with smaller budget):
- Visual: SSIM(original frame, attacked frame) ≥ 0.92
- Audio: PESQ(original waveform, attacked waveform) ≥ 3.0

**White-box attack setup**: Gradients flow end-to-end through the frozen encoders (which are differentiable) to the input. Frozen does not mean non-differentiable. This is standard white-box attack procedure.

---

## 5. Baseline Models

All baselines are evaluated on the same test sets as CMAR.

### Baseline 1: LipForensics (Video + Lip Motion)
- **Type**: Video-only, semantic lip-motion temporal detector
- **Source**: Pre-trained checkpoint from official LipForensics repository
- **Why**: Widely cited video-only baseline that uses temporal semantics, bridging the visual and AV domains
- **Evaluation**: All 13 degraded conditions + all 8 adversarial conditions

### Baseline 2: Late-Fusion (XceptionNet + AASIST)
- **Type**: Score-level fusion of unimodal detectors (image CNN + audio graph network)
- **Construction**: `P_final = 0.5 × P_XceptionNet + 0.5 × P_AASIST`
- **Source**: XceptionNet from DeepfakeBench weights; AASIST from official ASVspoof repository
- **Why**: Tests whether naive score-level fusion of strong unimodal detectors is more or less adversarially robust than learned cross-modal fusion
- **Evaluation**: All conditions

### Ablation Baseline 3: CMAR-VisualOnly
- **Type**: DINOv2-Small (LN-tuned) + temporal aggregation + classification head, no audio
- **Construction**: Remove audio branch and CMCM, train with BCE loss only
- **Why**: Isolates the contribution of audio-visual fusion to robustness. The key comparison: does audio + cross-modal consistency improve robustness beyond what the visual foundation model alone provides?

### Ablation Baseline 4: CMAR-NoConsistency
- **Type**: Full CMAR architecture but trained with BCE loss only (no L_consistency)
- **Construction**: Same model, L_con = 0
- **Why**: Isolates the contribution of the consistency loss to robustness under degradation

### Ablation Baseline 5: CMAR-NoTTDA
- **Type**: Full CMAR at test time but without the inference-time ensemble
- **Construction**: Single forward pass, no degraded copies
- **Why**: Isolates the benefit of the simplified TTDA at inference

---

## 6. Evaluation Metrics

### 6.1 Primary Detection Metrics

| Metric | Formula | Interpretation | Range | Better |
|---|---|---|---|---|
| **AUC-ROC** | Area under ROC curve | Overall discrimination ability regardless of threshold | [0, 1] | ↑ |
| **EER** | FPR at the point FPR = FNR | Operating point where false accepts = false rejects; lower is better | [0, 1] | ↓ |
| **AP** | Area under Precision-Recall curve | Detection under class imbalance (important: FakeAVCeleb is 10:1 fake/real) | [0, 1] | ↑ |

**Reporting standard**: All metrics are reported as `mean ± 1.96σ/√3` across 3 runs with different random seeds (95% CI). Any reported number without confidence intervals will be rejected by top venue reviewers.

### 6.2 Robustness Metrics (Novel Contributions of this Paper)

#### RAR — Robustness-Accuracy Ratio

```
RAR(model, condition) = AUC_condition(model) / AUC_clean(model)
```

**Interpretation**:
- RAR = 1.00: Perfectly robust — performance is identical under degradation/attack
- RAR = 0.90: 10% relative performance degradation (acceptable)
- RAR = 0.75: 25% relative degradation (significant fragility)
- RAR < 0.60: Model has collapsed under this condition (critical fragility)

**Reporting**: Plot RAR curves across degradation severity (x-axis) for all models (multiple lines). The **slope of the RAR curve** is the finding — a model that drops steeply is brittle; one that degrades gracefully is robust.

#### Δ-AUC — Absolute Performance Drop

```
ΔAUC(model, condition) = AUC_clean(model) - AUC_condition(model)
```

**Interpretation**: Intuitive absolute drop. Use alongside RAR to distinguish between a model that starts low and stays low (high RAR but low clean AUC) versus one that starts high and stays high (our target).

#### TTDA-Gain

```
TTDA_Gain(model, condition) = AUC_with_TTDA(model, condition) - AUC_without_TTDA(model, condition)
```

**Interpretation**: Per-condition benefit of the test-time ensemble. Expected: near-zero on clean data, positive and growing on degraded/attacked data. If TTDA-Gain is negative on any condition, investigate why.

#### Cross-Modal Robustness Ratio (CMRR) — The Paper's Key Novel Metric

```
CMRR(model) = [AUC_visual_attack(model) / AUC_clean(model)] + [AUC_audio_attack(model) / AUC_clean(model)]
              ────────────────────────────────────────────────────────────────────────────────────────────
                              AUC_both_attack(model) / AUC_clean(model)
```

**Interpretation**: Measures how much protection cross-modal redundancy provides. For a perfectly multimodal-robust model, attacking one modality barely degrades performance (numerator stays high) while attacking both degrades more (denominator lower) → CMRR > 2.0. For a unimodal-equivalent model, the protection is absent → CMRR ≈ 2.0. For a model with strong cross-modal coupling (our hypothesis), CMRR > 2.5.

**Why this metric**: This is the metric that quantifies the paper's central claim. No prior paper has measured this because no prior paper studied asymmetric modality attacks on AV detectors.

### 6.3 How to Interpret Results: Decision Table

| Outcome | Interpretation | Paper Action |
|---|---|---|
| CMAR has highest clean AUC AND highest RAR across degradations | Strong result — both claims hold | Present as primary finding |
| CMAR has comparable clean AUC but significantly higher RAR | Robustness claim holds, clean accuracy parity | Lead with robustness, note parity on accuracy |
| CMAR-VisualOnly shows lower RAR than full CMAR on visual attacks | Audio modality protects against visual attacks | Strong evidence for central hypothesis |
| TTDA-Gain > 0 for all degradation conditions | TTDA is a valid inference-time defense | Keep as contribution |
| TTDA-Gain < 0 on clean data | Expected — TTDA shouldn't hurt clean performance; if it does, reduce to K=2 copies |  |
| CMRR > 2.0 for CMAR, ~2.0 for Late-Fusion | Cross-modal fusion outperforms score fusion for robustness | Supports learned vs naive fusion argument |
| Clean AUC is below BA-TFD or LipForensics baseline | Still acceptable if RAR is higher — the paper is about robustness, not clean accuracy | Reframe story as "comparable accuracy, superior robustness" |

### 6.4 Per-Category Analysis

FakeAVCeleb's four manipulation categories each test different aspects of the CMCM:

| Category | What CMCM Should Detect | Expected Performance |
|---|---|---|
| FakeVideo-RealAudio (FR) | Visual-audio sync mismatch — video is fake but audio is real | High AUC; CMCM attends to lip-audio mismatch |
| RealVideo-FakeAudio (RF) | Audio forgery against real video — spectral inconsistency | High AUC; CMCM attends to audio anomalies |
| FakeVideo-FakeAudio (FF) | Both are fake but from the same generation pipeline — hardest | Moderate-lower AUC; minimal cross-modal inconsistency signal |
| RealVideo-RealAudio (RR) | All real — should be classified as real (false positive rate) | Measure specificity; ideally < 5% false positive rate |

---

## 7. Contributions (Paper's Claim List)

The paper will make exactly four contributions. Do not expand beyond four — ICASSP reviewers expect focus.

**C1 — The asymmetric attack finding** *(empirical finding, highest novelty)*
> First systematic study of asymmetric modality attacks (visual-only, audio-only, and both-modality PGD) on an audio-visual deepfake detector. Demonstrates that cross-modal redundancy significantly reduces the effectiveness of single-modality adversarial attacks.

**C2 — CMAR framework** *(method contribution)*
> A parameter-efficient audio-visual deepfake detection framework combining frozen foundation model feature extraction (DINOv2-Small + Whisper-Tiny) with a lightweight bidirectional cross-attention module. Only 5.56M parameters are trained, achieving competitive detection accuracy with strong robustness properties.

**C3 — Robustness under real-world degradations** *(robustness evaluation)*
> Systematic evaluation across 12 degradation conditions including social media simulation pipelines, with the Cross-Modal Robustness Ratio (CMRR) metric that quantifies the protection afforded by cross-modal fusion.

**C4 — Reproducible evaluation protocol** *(community contribution)*
> All code, feature caches, and evaluation splits released as public Kaggle datasets and notebooks for full reproducibility.

---

## 8. Technology Stack and Environment

### 8.1 Hardware

| Resource | Specification | Notes |
|---|---|---|
| GPU | Kaggle T4, 16GB VRAM | Primary compute; 12-hour session limit |
| RAM | 13GB (Kaggle default) | Feature caching reduces RAM pressure |
| Disk | ~50GB per session (20GB features + models) | Upload features as Kaggle datasets |
| Sessions target | 7 sessions × avg 6h = ~42 GPU-hours total | Well within Kaggle limits |

### 8.2 Software Versions

```
Python         3.10
torch          >= 2.1
torchvision    >= 0.16
torchaudio     >= 2.1
transformers   >= 4.37    # Whisper-Tiny via HuggingFace
timm           >= 0.9.12  # DINOv2-Small via timm
torchattacks   >= 3.5     # PGD, FGSM
librosa        >= 0.10
soundfile      >= 0.12
pydub          >= 0.25    # MP3 encode/decode for audio degradation
opencv-python  >= 4.9
Pillow         >= 10.2
ffmpeg-python  >= 0.2     # H.264 re-encoding
scikit-learn   >= 1.4     # AUC, EER, metrics
scipy          >= 1.12    # confidence intervals
pandas         >= 2.1
numpy          >= 1.26
tqdm
matplotlib     >= 3.8
seaborn        >= 0.13
pesq           >= 0.0.4   # PESQ score for audio imperceptibility check
```

### 8.3 Models and Checkpoints

| Model | Source | Download Size | Notes |
|---|---|---|---|
| DINOv2-Small | `timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)` | ~90MB | Auto-downloaded |
| Whisper-Tiny | `whisper.load_model('tiny')` or HuggingFace | ~150MB | Encoder only |
| XceptionNet (for baseline) | DeepfakeBench repository | ~88MB | FaceForensics++ weights |
| AASIST (for baseline) | Official AASIST repository | ~84MB | ASVspoof 2021 weights |
| LipForensics (for baseline) | Official LipForensics repository | ~85MB | FaceForensics++ weights |

### 8.4 Feature Caching Strategy

This is the most critical engineering decision for Kaggle feasibility. Feature extraction through frozen encoders is deterministic — run it once, cache, reuse across all training and evaluation sessions.

```
/kaggle/working/cmar_cache/
├── features/
│   ├── visual/
│   │   ├── train/{clip_id}.pt        # (16, 384) — 16 frame features
│   │   ├── val/{clip_id}.pt
│   │   └── test/{clip_id}.pt
│   ├── audio/
│   │   ├── train/{clip_id}.pt        # (T_a, 384) — Whisper-Tiny output
│   │   ├── val/{clip_id}.pt
│   │   └── test/{clip_id}.pt
│   └── degraded_test/
│       ├── d1_jpeg75/visual/{clip_id}.pt
│       ├── d7_mp3_128k/audio/{clip_id}.pt
│       └── ... (one subdirectory per degradation condition)
├── manifests/
│   ├── train.csv         # clip_id, label, av_category, video_path, audio_path
│   ├── val.csv
│   └── test.csv
└── metadata.json         # extraction config, model versions, commit hash, timestamp
```

**Upload as private Kaggle dataset** after Session 1. All subsequent sessions attach this dataset as input. No re-extraction ever.

**Estimated cache size**: ~3,500 train + 825 val + 825 test × 2 modalities × ~0.1MB per .pt file = ~1GB for clean features + ~12GB for all degraded test variants.

---

## 9. Kaggle Session Plan

### Session 1 — Feature Extraction and Caching (~4 hours)
- Notebook: `01_feature_extraction.ipynb`
- Load DINOv2-Small and Whisper-Tiny
- Build FakeAVCeleb manifest (train/val/test CSVs)
- Extract and save clean visual + audio features for all 5,500 clips
- Generate all 12 degraded test variants
- Extract features for all degraded variants
- Save metadata.json
- **Output**: Upload cache as private Kaggle dataset `cmar-features-v1`

### Session 2 — Model Training (~7 hours)
- Notebook: `02_train_cmar.ipynb`
- Load `cmar-features-v1` as input dataset
- Build `CachedAVDataset` (loads .pt files, applies on-the-fly degradation for consistency loss)
- Initialize CMCM, temporal aggregation, classification head
- Enable LN parameter gradients on DINOv2-Small and Whisper-Tiny
- Train for 30 epochs with early stopping, gradient accumulation
- Log train/val AUC and loss per epoch (save to CSV for plotting)
- Save best checkpoint every 5 epochs
- **Output**: Upload checkpoint as `cmar-checkpoint-v1`

### Session 3 — Main Evaluation: Clean + Degraded (~5 hours)
- Notebook: `03_evaluate_clean_degraded.ipynb`
- Load CMAR checkpoint
- Evaluate on FakeAVCeleb clean test (E1)
- Evaluate on all 12 degraded conditions, with and without TTDA (E2–E12)
- Compute AUC, EER, AP, RAR, TTDA-Gain per condition
- Evaluate on LAV-DF test split clean (E13: cross-dataset)
- Save all results as structured JSON
- **Output**: `cmar-results-clean-degraded.json`

### Session 4 — Baseline Evaluation (~5 hours)
- Notebook: `04_evaluate_baselines.ipynb`
- Set up LipForensics with pre-trained checkpoint
- Set up Late-Fusion (XceptionNet + AASIST, score averaging)
- Evaluate both on all clean + degraded conditions
- **Output**: `baseline-results.json`

### Session 5 — Adversarial Evaluation (~6 hours)
- Notebook: `05_adversarial_evaluation.ipynb`
- Implement PGD-20 for visual stream (L∞, end-to-end through frozen DINOv2)
- Implement PGD-20 for audio stream (L∞ on waveform, end-to-end through frozen Whisper)
- Generate adversarial examples: A1–A8 conditions
- Verify imperceptibility (SSIM ≥ 0.92 visual; PESQ ≥ 3.0 audio)
- Evaluate CMAR + all baselines on adversarial test sets
- Compute CMRR for CMAR and Late-Fusion
- **Output**: `adversarial-results.json`

### Session 6 — Ablation Studies (~6 hours)
- Notebook: `06_ablations.ipynb`
- Train CMAR-VisualOnly (no audio branch) from cached visual features
- Train CMAR-NoConsistency (BCE only, same architecture)
- Evaluate all ablations on: clean, D12 (social media sim), A2 (PGD visual ε=4), A5 (PGD audio strong)
- Test TTDA with K ∈ {1, 2, 3, 5} copies (ablation A3_ttda)
- Test CMCM with 1 layer and 4 layers
- **Output**: `ablation-results.json`

### Session 7 — Visualizations and Qualitative Analysis (~3 hours)
- Notebook: `07_analysis_figures.ipynb`
- Generate all paper figures (see Section 10)
- Cross-modal attention weight visualizations (heatmaps over temporal segments)
- t-SNE visualization of CMAR features: real vs fake clustering, before/after degradation
- TTDA prediction consistency examples (scatter: P_clean vs P_degraded)
- RAR curves across degradation severity (the paper's main robustness figure)
- Asymmetric attack bar chart (the paper's main adversarial figure — highest priority)
- **Output**: All figures as 300 DPI PNG + PDF

---

## 10. Paper Figures (Required)

| Figure | Content | Location in Paper | Priority |
|---|---|---|---|
| Fig. 1 | CMAR architecture diagram (TikZ or draw.io) | Section 3: Method | Required |
| Fig. 2 | **Asymmetric attack bar chart**: AUC under visual-only / audio-only / both-modality PGD, grouped by model | Section 4: Experiments | **Highest — this is the paper's main finding** |
| Fig. 3 | RAR curves across degradation severity for all models (x = severity, y = RAR, multiple lines) | Section 4 | Required |
| Fig. 4 | Per-category AUC breakdown (FR, RF, FF, RR) for CMAR and best baseline | Section 4 | Required |
| Fig. 5 | Cross-modal attention maps on example clips (FR vs RF vs FF categories) | Section 5: Analysis | Required |
| Fig. 6 | Ablation bar chart (AUC clean vs degraded for each ablation variant) | Section 4: Ablations | Required |

---

## 11. Expected Results and Success Criteria

### 11.1 Minimum Viable Results (Paper is submittable if ALL of these hold)

| Metric | Minimum Threshold |
|---|---|
| CMAR clean AUC on FakeAVCeleb | ≥ 0.88 |
| CMAR RAR under D12 (social media sim) | ≥ 0.82 (vs expected 0.60–0.72 for baselines) |
| CMAR AUC under A1 (visual-only PGD ε=2/255) | Significantly higher than CMAR-VisualOnly under same attack |
| CMRR for CMAR | > CMRR for Late-Fusion (cross-modal fusion outperforms score fusion) |
| CMAR cross-dataset AUC on LAV-DF | ≥ 0.75 |

### 11.2 Strong Results (Paper is competitive for acceptance)

- CMAR RAR ≥ 0.90 under moderate degradation (D1, D3, D7)
- CMRR ≥ 2.5 (strong cross-modal protection demonstrated)
- TTDA-Gain > 0.05 AUC under at least 5 degradation conditions

### 11.3 Fallback Plan if Clean AUC is Below Baseline

If CMAR's clean FakeAVCeleb AUC is lower than LipForensics or Late-Fusion, the story shifts to: *"CMAR trades a small amount of clean accuracy for large gains in adversarial robustness"*. This is still a valid paper — ICASSP reviewers understand the accuracy-robustness tradeoff. The asymmetric attack finding remains valid regardless.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VRAM OOM during training (Whisper-Tiny + DINOv2-S + batch) | Low (Whisper-Tiny is 15M params) | High | Reduce batch to 4; use gradient checkpointing; split visual/audio forward passes |
| VRAM OOM during adversarial attack generation (PGD through frozen encoders) | Medium | High | Generate attacks per-sample with batch=1; cache attack results |
| CMAR clean AUC below LipForensics baseline | Medium | Medium | Reframe as robustness-accuracy tradeoff; fallback story is still valid |
| LipForensics checkpoint does not generalize to FakeAVCeleb | Medium | Low | Use as-is; generalization gap is expected and actually supports our story |
| PGD generates invalid adversarial examples (SSIM < 0.92) | Low | Low | Re-run with smaller ε; adjust step size; document budget |
| FakeAVCeleb dataset unavailable on Kaggle | Low | High | Use direct download in notebook from official source |
| Whisper-Tiny features insufficient for audio forgery detection | Low | High | Fallback: Wav2Vec2-Base (HuggingFace, similar footprint) |
| Session timeout mid-training | Medium | Low | Checkpoint every 5 epochs; all data cached; restart is fast |

---

## 13. File and Notebook Naming Convention

```
Notebooks (Kaggle):
  01_feature_extraction.ipynb
  02_train_cmar.ipynb
  03_evaluate_clean_degraded.ipynb
  04_evaluate_baselines.ipynb
  05_adversarial_evaluation.ipynb
  06_ablations.ipynb
  07_analysis_figures.ipynb

Model code (local/GitHub):
  cmar/
  ├── models/
  │   ├── visual_encoder.py       # DINOv2-Small LN-tuning wrapper
  │   ├── audio_encoder.py        # Whisper-Tiny encoder LN-tuning wrapper
  │   ├── cmcm.py                 # Cross-Modal Consistency Module
  │   ├── temporal_aggregation.py # Visual temporal pooling + Audio projection
  │   ├── classifier.py           # Classification head
  │   └── cmar.py                 # Full model assembly
  ├── training/
  │   ├── dataset.py              # CachedAVDataset
  │   ├── losses.py               # BCE + consistency loss
  │   └── trainer.py              # Training loop with gradient accumulation
  ├── evaluation/
  │   ├── metrics.py              # AUC, EER, AP, RAR, CMRR, TTDA-Gain
  │   ├── degradations.py         # All 12 degradation transforms
  │   └── attacks.py              # PGD-20 visual + audio
  └── utils/
      ├── cache.py                # Feature extraction and caching utilities
      └── visualization.py        # All paper figures

Results (Kaggle outputs):
  cmar-results-clean-degraded.json
  baseline-results.json
  adversarial-results.json
  ablation-results.json

Paper (LaTeX):
  paper/
  ├── main.tex
  ├── figures/
  └── references.bib
```

---

## 14. Writing Schedule

| Week | Dates (approx) | Activity |
|---|---|---|
| 1 | June 2 – June 8 | Sessions 1–2: Feature extraction + model training |
| 2 | June 9 – June 15 | Sessions 3–4: Main evaluation + baselines |
| 3 | June 16 – June 22 | Sessions 5–6: Adversarial + ablations |
| 4 | June 23 – June 29 | Session 7: Figures + qualitative analysis |
| 5–6 | June 30 – July 13 | Paper writing: Introduction, Method, Experiments |
| 7 | July 14 – July 20 | Paper writing: Analysis, Conclusion, Related Work |
| 8–9 | July 21 – Aug 3 | Internal revision; re-run any failed experiments |
| 10–12 | Aug 4 – Aug 25 | External feedback; polish; supplementary material |
| 13–14 | Aug 26 – Sep 7 | Final revision; camera-ready formatting |
| 15 | Sep 8 – Sep 16 | Buffer + submission |

---

*End of PLANNING.md — Do not modify without researcher instruction. Version: 1.0 | Last updated: May 2026*
