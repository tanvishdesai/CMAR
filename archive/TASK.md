# TASK.md — CMAR Project Task Tracker

> **AI Agent Instruction**: This file is your live task board. At the start of every session, read PLANNING.md first, then CONTEXT.md, then this file. After completing any task or sub-task, update its status here immediately. Use the status codes defined below. Add new tasks to the Backlog section as they are discovered. Never delete completed tasks — mark them DONE with a completion note. This file is a living record of all work done.

---

## Status Codes

| Code | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress (currently being worked on) |
| `[!]` | Blocked — needs researcher input or external dependency |
| `[?]` | Needs investigation / unclear how to proceed |
| `[x]` | Completed |
| `[s]` | Skipped — decision made not to do this; reason logged |

---

## Current Sprint

> **Update this section at the start of each Kaggle session.**
> Format: `Active Session: [session number] | Date: [date] | Goal: [one sentence]`

**Active Session**: V2 results audit | Date: May 29, 2026 | Goal: Interpret second-version diagnostics and choose the next research direction  
**Current Phase**: Phase 3/5/6 diagnostic bridge - degradation robustness supported; final adversarial claim still unproven  
**Next Action**: Rerun ablation test evaluation with the correct checkpoint root, then implement a small raw visual input-space PGD pilot with SSIM checks

---

## Phase 0 — Environment Validation
> **Goal**: Confirm hardware feasibility BEFORE investing in feature extraction. This phase takes 2 hours and eliminates the single biggest execution risk.
> **Success criterion**: Both models load, forward pass completes, VRAM < 12GB with batch of 8 clips.

- [ ] **P0.1** — Create Kaggle notebook `00_environment_check.ipynb`
  - [ ] P0.1.1 — Install all required packages (see PLANNING.md §8.2 for versions)
  - [ ] P0.1.2 — Load DINOv2-Small via `timm`: `vit_small_patch14_dinov2.lvd142m`
  - [ ] P0.1.3 — Load Whisper-Tiny encoder via `openai-whisper` or HuggingFace
  - [ ] P0.1.4 — Load both models simultaneously and check `torch.cuda.memory_allocated()`
  - [ ] P0.1.5 — Run a dummy forward pass: batch of 8 clips × 16 frames × 224×224×3 through DINOv2; batch of 8 × 10s audio through Whisper-Tiny encoder
  - [ ] P0.1.6 — Record peak VRAM usage; log to CONTEXT.md
  - [ ] P0.1.7 — If VRAM > 14GB: reduce to batch=4; if still OOM, use gradient checkpointing
  - [ ] P0.1.8 — Verify FakeAVCeleb dataset is accessible as Kaggle dataset (search Kaggle datasets: "FakeAVCeleb")
  - [ ] P0.1.9 — Verify LAV-DF dataset is accessible as Kaggle dataset
  - [ ] P0.1.10 — Test all degradation transforms (JPEG, resize, noise, MP3, H264) on a single dummy clip
  - [ ] P0.1.11 — Test `torchattacks` PGD-20 on a small dummy tensor
  - [ ] P0.1.12 — Log all library versions actually installed to CONTEXT.md

**P0 Exit Criteria**: VRAM check passed + FakeAVCeleb accessible + all transforms work

---

## Phase 1 — Feature Extraction and Caching
> **Kaggle Session 1** | Estimated: 3–4 hours
> **Goal**: Extract DINOv2-Small visual features and Whisper-Tiny audio features for all FakeAVCeleb splits + generate all 12 degraded test variants.
> **Output**: Private Kaggle dataset `vasuaashadesai/cmar-features-clean-v1` (display name: "CMAR Clean Features V1"; contains clean and degraded cache)

- [x] **P1.1** — Create Kaggle/Colab preprocessing workflow `01_feature_extraction.ipynb`/single-cell drivers

- [x] **P1.2** — Build dataset manifests
  - [x] P1.2.1 — Scan FakeAVCeleb directory structure; understand folder layout
  - [x] P1.2.2 — Build `train.csv` with columns: `clip_id, label, av_category, video_path, audio_path`
  - [x] P1.2.3 — Build `val.csv` (same columns)
  - [x] P1.2.4 — Build `test.csv` (same columns)
  - [x] P1.2.5 — Verify split sizes match PLANNING.md §4.1 (confirmed: train 3,850; val 825; test 825)
  - [x] P1.2.6 — Confirm all four AV categories (RR, FR, RF, FF) are represented in test split

- [x] **P1.3** — Visual feature extraction (DINOv2-Small)
  - [x] P1.3.1 — Write `extract_visual_features(video_path, n_frames=16)` function
    - Load video with OpenCV; sample 16 frames uniformly
    - Preprocess: resize 224×224, normalize with ImageNet mean/std
    - Forward pass through frozen DINOv2-Small; extract [CLS] token per frame
    - Output: tensor (16, 384)
  - [x] P1.3.2 — Run extraction for all train clips; save to `cache/features/visual/train/{clip_id}.pt`
  - [x] P1.3.3 — Run for val and test clips
  - [x] P1.3.4 — Verify: spot-check 10 random clips; confirm tensor shapes are (16, 384)
  - [x] P1.3.5 — Log total extraction time and average time per clip to CONTEXT.md

- [x] **P1.4** — Audio feature extraction (Whisper-Tiny)
  - [x] P1.4.1 — Write `extract_audio_features(audio_path, sr=16000, max_len=10)` function
    - Load audio with librosa at 16kHz; trim/pad to 10s
    - Convert to log-mel spectrogram (80 channels, Whisper's internal preprocessing)
    - Forward pass through frozen Whisper-Tiny encoder (encoder only; discard decoder)
    - Output: tensor (T_a, 384) where T_a is Whisper's temporal output length
  - [x] P1.4.2 — Run extraction for all train, val, test clips
  - [x] P1.4.3 — Verify tensor shapes; log average T_a to CONTEXT.md (pooled to <=64 tokens)
  - [x] P1.4.4 — Handle edge cases: very short clips (<1s), missing audio tracks

- [x] **P1.5** — Generate degraded test sets
  - [x] P1.5.1 — Implement all 12 degradation transforms (reference PLANNING.md §4.3)
    - D1: JPEG QF=75 per frame (save → reload with PIL)
    - D2: JPEG QF=50
    - D3: Resize 0.75× bicubic down then up to 224×224
    - D4: Resize 0.50×
    - D5: Gaussian noise σ=0.01 additive to frames
    - D6: Gaussian noise σ=0.02
    - D7: MP3 128kbps encode/decode via pydub
    - D8: MP3 64kbps
    - D9: White Gaussian noise SNR=30dB on waveform
    - D10: White Gaussian noise SNR=20dB
    - D11: H.264 CRF=28 re-encoding via ffmpeg-python (applies to both visual+audio)
    - D12: Sequential pipeline D1 → D3 → D7 (social media simulation)
  - [x] P1.5.2 — Extract DINOv2 features for visually-degraded test clips (D1–D6, D11, D12)
  - [x] P1.5.3 — Extract Whisper features for audio-degraded test clips (D7–D10, D11, D12)
  - [x] P1.5.4 — Verify all 12 degraded feature sets are saved with correct directory structure

- [x] **P1.6** — Finalize and upload cache
  - [x] P1.6.1 — Create `metadata.json` with: extraction date, model versions, split sizes, avg T_a, notes
  - [x] P1.6.2 — Verify total cache size is accepted by Kaggle dataset upload
  - [x] P1.6.3 — Upload entire cache directory as private Kaggle dataset `vasuaashadesai/cmar-features-clean-v1`
  - [x] P1.6.4 — Confirm dataset is accessible in a new notebook; log dataset ID to CONTEXT.md

**P1 Exit Criteria**: All features extracted, degraded variants ready, cache uploaded and verified accessible

---

## Phase 2 — Model Training
> **Kaggle Session 2** | Estimated: 6–7 hours
> **Goal**: Train CMAR model to convergence on FakeAVCeleb training split.
> **Output**: Private Kaggle dataset `cmar-checkpoint-v1` with best model checkpoint

- [x] **P2.1** — Implement model components
  - [x] P2.1.1 — `visual_encoder.py`: DINOv2-Small wrapper that freezes all params except LayerNorm
  - [x] P2.1.2 — `audio_encoder.py`: Whisper-Tiny encoder wrapper that freezes all params except LayerNorm
  - [x] P2.1.3 — `temporal_aggregation.py`:
    - Visual: learnable [CLS] token + 1-layer Transformer over (16, 384) → (8, 256) segments
    - Audio: linear projection 384→256 + temporal chunking to 8 segments (mean pooling per chunk)
  - [x] P2.1.4 — `cmcm.py`: 2-layer bidirectional cross-attention
    - Layer 1: CrossAttention(Q=V, K=A, V=A) → LayerNorm → FFN → V_enhanced
    - Layer 2: CrossAttention(Q=A, K=V, V=V) → LayerNorm → FFN → A_enhanced
    - Fusion: Concat [V_enhanced, A_enhanced] (8, 512) → Linear(512→256) → (8, 256)
  - [x] P2.1.5 — `classifier.py`: Global avg pool → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,1) → Sigmoid
  - [x] P2.1.6 — `cmar.py`: Full model; confirmed trainable parameter count is 4,316,929 in local dummy check
  - [x] P2.1.7 — Quick sanity check: forward pass on dummy batch; verify output shape is (batch, 1)

- [x] **P2.2** — Implement training infrastructure
  - [x] P2.2.1 — `dataset.py`: `CachedAVDataset`
    - Load .pt files from cache by clip_id
    - On-the-fly degradation augmentation during training (randomly apply one of: JPEG75, resize0.75, noise001, MP3-128k to loaded features — re-apply degradation transform, not reload from degraded cache)
    - Return: (visual_feat, audio_feat, visual_feat_degraded, audio_feat_degraded, label)
  - [x] P2.2.2 — `losses.py`: `CMARLoss`
    - L_BCE on clean predictions
    - L_consistency = symmetric KL(P_clean, P_degraded)
    - L_total = L_BCE + 0.3 × L_consistency
  - [x] P2.2.3 — `trainer.py`: Training loop
    - AdamW with two parameter groups (LN params lr=1e-4; CMCM+head lr=5e-4)
    - Gradient accumulation every 4 steps
    - Cosine annealing with 3-epoch warmup
    - Early stopping patience=5 on validation AUC
    - Save checkpoint every 5 epochs AND whenever val AUC improves

- [ ] **P2.3** — Notebook `02_train_cmar.ipynb`
  - [x] P2.3.1 — Mount `CMAR Clean Features V1` / `vasuaashadesai/cmar-features-clean-v1` as input dataset
  - [x] P2.3.2 — Initialize model; print trainable parameter count
  - [x] P2.3.3 — Run training for 30 epochs (or until early stopping)
  - [x] P2.3.4 — Log per-epoch: train_loss, val_loss, val_AUC, val_EER to CSV
  - [ ] P2.3.5 — Generate training curve plots (loss + AUC vs epoch)
  - [x] P2.3.6 — Report: best val AUC 0.9253688889 at epoch 10; early stopped after epoch 15
  - [ ] P2.3.7 — Upload best checkpoint as `cmar-checkpoint-v1`
  - [x] P2.3.8 — Log training metrics to CONTEXT.md

**P2 Exit Criteria**: Training converged, val AUC ≥ 0.85, checkpoint uploaded

---

## Phase 3 — Main Evaluation (Clean + Degraded)
> **Kaggle Session 3** | Estimated: 4–5 hours
> **Output**: `cmar-results-clean-degraded.json`

- [ ] **P3.1** — Notebook `03_evaluate_clean_degraded.ipynb`

- [ ] **P3.2** — Evaluation setup
  - [ ] P3.2.1 — Load CMAR checkpoint from `cmar-checkpoint-v1`
  - [ ] P3.2.2 — Implement `evaluate_model(model, dataloader)` → returns AUC, EER, AP with 95% CI over 3 bootstrap resamples
  - [ ] P3.2.3 — Implement `compute_RAR(auc_degraded, auc_clean)` and `compute_delta_auc`
  - [ ] P3.2.4 — Implement simplified TTDA: forward pass original + K=3 degraded copies; mean prediction

- [ ] **P3.3** — In-domain evaluations
  - [ ] P3.3.1 — E1: FakeAVCeleb clean test (no TTDA)
  - [ ] P3.3.2 — E1-TTDA: FakeAVCeleb clean test (with TTDA) — verify TTDA-Gain ≈ 0 on clean
  - [ ] P3.3.3 — E2–E12: Each of the 12 degraded conditions, without TTDA
  - [ ] P3.3.4 — E2T–E12T: Each of the 12 degraded conditions, with TTDA
  - [ ] P3.3.5 — Compute RAR, Δ-AUC, TTDA-Gain for each condition

- [ ] **P3.4** — Cross-dataset evaluation
  - [ ] P3.4.1 — E13: LAV-DF test split, clean (zero-shot generalization)
  - [ ] P3.4.2 — E14: LAV-DF test split + D12 (social media sim) — cross-dataset + degradation

- [ ] **P3.5** — Per-category analysis (FakeAVCeleb test split)
  - [ ] P3.5.1 — E15: AUC per AV category (FR, RF, FF, RR) on clean test set

- [ ] **P3.6** — Save results
  - [ ] P3.6.1 — Compile all metrics into `cmar-results-clean-degraded.json`
  - [ ] P3.6.2 — Log key results to CONTEXT.md: E1 clean AUC, RAR on D12, cross-dataset AUC

**P3 Exit Criteria**: All 14 conditions evaluated; results JSON saved

---

## Phase 4 — Baseline Evaluation
> **Kaggle Session 4** | Estimated: 4–5 hours
> **Output**: `baseline-results.json`

- [ ] **P4.1** — Set up LipForensics baseline
  - [ ] P4.1.1 — Download LipForensics pre-trained checkpoint (FaceForensics++ weights)
  - [ ] P4.1.2 — Adapt inference script to accept FakeAVCeleb video format
  - [ ] P4.1.3 — Verify LipForensics produces meaningful output on FakeAVCeleb clean test (AUC > 0.60 expected; log actual value)
  - [ ] P4.1.4 — If LipForensics fails to generalize: mark as [?] and alert researcher; note this in CONTEXT.md

- [ ] **P4.2** — Set up Late-Fusion baseline (XceptionNet + AASIST)
  - [ ] P4.2.1 — Load XceptionNet with DeepfakeBench FaceForensics++ checkpoint
  - [ ] P4.2.2 — Load AASIST with ASVspoof 2021 checkpoint
  - [ ] P4.2.3 — Implement score fusion: `P_final = 0.5 × P_XceptionNet + 0.5 × P_AASIST`
  - [ ] P4.2.4 — Run on FakeAVCeleb clean test; verify AUC is reasonable

- [ ] **P4.3** — Evaluate both baselines on all conditions
  - [ ] P4.3.1 — LipForensics: E1, E2–E12 (degraded), E13 (LAV-DF)
  - [ ] P4.3.2 — Late-Fusion: E1, E2–E12, E13
  - [ ] P4.3.3 — Compute RAR and Δ-AUC for all conditions for both baselines

- [ ] **P4.4** — Save baseline results
  - [ ] P4.4.1 — Save `baseline-results.json` with same structure as CMAR results

**P4 Exit Criteria**: Both baselines evaluated; results JSON saved

---

## Phase 5 — Adversarial Evaluation
> **Kaggle Session 5** | Estimated: 5–6 hours (most compute-intensive session)
> **Output**: `adversarial-results.json`

- [ ] **P5.1** — Implement adversarial attack infrastructure
  - [ ] P5.1.1 — Implement `pgd_attack_visual(model, frames, audio_feat, label, eps, n_iter=20, step_size=eps/10)` using `torchattacks` or manual PGD
  - [ ] P5.1.2 — Implement `pgd_attack_audio(model, visual_feat, waveform, label, eps, n_iter=20)`
  - [ ] P5.1.3 — Implement `pgd_attack_both(model, frames, waveform, label, eps_v, eps_a)`
  - [ ] P5.1.4 — Implement `fgsm_attack_visual` and `fgsm_attack_audio`
  - [ ] P5.1.5 — Implement SSIM checker (verify attacked frames still look similar to originals)
  - [ ] P5.1.6 — Implement PESQ checker for audio attacks
  - [ ] P5.1.7 — **Critical**: Test gradient flow through frozen DINOv2 and Whisper — verify gradients reach input
  - [ ] P5.1.8 — If gradient flow is broken (e.g. no grad on frozen params): enable `.requires_grad_(True)` on input tensor, not model params

- [ ] **P5.2** — Generate adversarial examples
  - [ ] P5.2.1 — A1: PGD visual ε=2/255; verify SSIM ≥ 0.92
  - [ ] P5.2.2 — A2: PGD visual ε=4/255; verify SSIM ≥ 0.92
  - [ ] P5.2.3 — A3: PGD visual ε=8/255; verify SSIM ≥ 0.92 (may fail — log and report)
  - [ ] P5.2.4 — A4: PGD audio SNR=30dB; verify PESQ ≥ 3.0
  - [ ] P5.2.5 — A5: PGD audio SNR=20dB; verify PESQ ≥ 3.0
  - [ ] P5.2.6 — A6: PGD both modalities simultaneously (ε=4/255 visual, SNR=30dB audio)
  - [ ] P5.2.7 — A7: FGSM visual ε=4/255 (quick single-step baseline)
  - [ ] P5.2.8 — A8: FGSM audio SNR=30dB
  - [ ] P5.2.9 — Log: average SSIM, average PESQ, avg attack success rate (% of samples where model changes prediction) to CONTEXT.md

- [ ] **P5.3** — Evaluate CMAR on all adversarial conditions
  - [ ] P5.3.1 — CMAR on A1–A8
  - [ ] P5.3.2 — CMAR-VisualOnly on A1–A3 (visual attacks on visual-only model — the comparison point)
  - [ ] P5.3.3 — Compute CMRR for CMAR and Late-Fusion

- [ ] **P5.4** — Evaluate baselines on adversarial conditions
  - [ ] P5.4.1 — LipForensics on A1–A3 (it is video-only so audio attacks don't apply)
  - [ ] P5.4.2 — Late-Fusion on A1–A8 (generates the CMRR comparison)

- [ ] **P5.5** — Save adversarial results
  - [ ] P5.5.1 — Save `adversarial-results.json` including CMRR values
  - [ ] P5.5.2 — Log key adversarial metrics to CONTEXT.md: CMAR vs baseline AUC on A2, CMRR values

**P5 Exit Criteria**: All 8 attack conditions evaluated for CMAR and applicable baselines; CMRR computed

---

## Phase 6 — Ablation Studies
> **Kaggle Session 6** | Estimated: 5–6 hours
> **Output**: `ablation-results.json`

- [ ] **P6.1** — Train ablation models
  - [ ] P6.1.1 — **AB1: CMAR-VisualOnly** — remove audio branch and CMCM; train with BCE only on visual features
  - [ ] P6.1.2 — **AB2: CMAR-NoConsistency** — full CMAR architecture, BCE loss only (λ_con = 0)
  - [ ] P6.1.3 — **AB3: CMAR-1LayerCMCM** — replace 2-layer CMCM with 1-layer cross-attention
  - [ ] P6.1.4 — **AB4: CMAR-4LayerCMCM** — replace 2-layer CMCM with 4-layer cross-attention
  - [ ] P6.1.5 — All ablations use identical hyperparameters; only the specified component changes

- [ ] **P6.2** — Evaluate ablation models
  - [ ] P6.2.1 — Evaluate all 4 ablations on: E1 (clean), D12 (social media sim), A2 (PGD visual ε=4), A5 (PGD audio strong)
  - [ ] P6.2.2 — Evaluate TTDA with K ∈ {1, 2, 3, 5} degraded copies (on D12 condition only)

- [ ] **P6.3** — Cross-dataset ablation
  - [ ] P6.3.1 — Evaluate full CMAR vs AB1-CMAR-VisualOnly on LAV-DF clean (E13)
  - [ ] P6.3.2 — This tests whether audio features + cross-modal fusion helps generalization

- [ ] **P6.4** — Save ablation results
  - [ ] P6.4.1 — Compile into `ablation-results.json`
  - [ ] P6.4.2 — Log key finding: which component contributes most to robustness vs clean accuracy

**P6 Exit Criteria**: All ablations trained and evaluated; results JSON saved

---

## Phase 7 — Visualizations and Qualitative Analysis
> **Kaggle Session 7** | Estimated: 2–3 hours
> **Output**: All paper figures as 300 DPI PNG + PDF

- [ ] **P7.1** — Figure 2 (HIGHEST PRIORITY): Asymmetric attack bar chart
  - [ ] P7.1.1 — Grouped bar chart: x-axis = attack condition (A1 visual-only, A4 audio-only, A6 both), groups = models (CMAR, Late-Fusion, LipForensics, CMAR-VisualOnly)
  - [ ] P7.1.2 — Show AUC with error bars (95% CI)
  - [ ] P7.1.3 — Annotate CMRR value for CMAR and Late-Fusion as text on figure

- [ ] **P7.2** — Figure 3: RAR curves
  - [ ] P7.2.1 — Line chart: x-axis = degradation severity (group D1/D2 for JPEG; D3/D4 for resize; etc.), y-axis = RAR
  - [ ] P7.2.2 — One line per model (CMAR, Late-Fusion, LipForensics, CMAR-VisualOnly)
  - [ ] P7.2.3 — Optional: shaded 95% CI bands around each line

- [ ] **P7.3** — Figure 4: Per-category AUC
  - [ ] P7.3.1 — Horizontal bar chart: AUC per category (FR, RF, FF, RR) for CMAR and best baseline
  - [ ] P7.3.2 — Annotate which categories benefit most from cross-modal fusion

- [ ] **P7.4** — Figure 5: Cross-modal attention maps
  - [ ] P7.4.1 — Extract CMCM attention weights for 3 example clips (FR, RF, FF categories)
  - [ ] P7.4.2 — Visualize as heatmap over temporal segments (x-axis) and attention heads (y-axis)
  - [ ] P7.4.3 — Annotate what the model is "attending to" in each case

- [ ] **P7.5** — Figure 6: Ablation bar chart
  - [ ] P7.5.1 — Grouped bars: y-axis = AUC, groups = (clean, D12-degraded, A2-adversarial)
  - [ ] P7.5.2 — One bar per model variant (Full CMAR, AB1-VisualOnly, AB2-NoConsistency, AB3-1Layer, AB4-4Layer)

- [ ] **P7.6** — Supplementary: t-SNE feature space
  - [ ] P7.6.1 — t-SNE on CMAR fusion features: color by real/fake, marker by AV category
  - [ ] P7.6.2 — Compare: features on clean samples vs degraded samples

- [ ] **P7.7** — Export all figures
  - [ ] P7.7.1 — Save all figures as 300 DPI PNG and vector PDF
  - [ ] P7.7.2 — Verify all text in figures is ≥ 8pt when scaled to ICASSP two-column format

**P7 Exit Criteria**: All 6 main figures generated and exported

---

## Phase 8 — Paper Writing

- [ ] **P8.1** — Set up LaTeX environment
  - [ ] P8.1.1 — Download ICASSP 2027 LaTeX template from IEEE website
  - [ ] P8.1.2 — Create paper/ directory structure (see PLANNING.md §13)
  - [ ] P8.1.3 — Set up references.bib with all key citations

- [ ] **P8.2** — Write paper sections (suggested order)
  - [ ] P8.2.1 — **Method section (Section 3)**: Architecture diagram + component descriptions. Write first — clearest section to write after implementation.
  - [ ] P8.2.2 — **Experiments section (Section 4)**: Paste in tables/figures from Phase 7; write surrounding text.
  - [ ] P8.2.3 — **Abstract**: Write after Method + Experiments are drafted. 200 words max. Must mention: adversarial robustness, audio-visual, foundation models, cross-modal redundancy, CMRR metric.
  - [ ] P8.2.4 — **Introduction**: Background → problem → gap → contributions (C1–C4). Final version written last.
  - [ ] P8.2.5 — **Related Work**: 3 paragraphs — (1) deepfake detection, (2) adversarial robustness for forgery detection, (3) multimodal learning. Be specific; cite recent 2024–2026 papers.
  - [ ] P8.2.6 — **Analysis / Discussion section (Section 5)**: Attention map interpretation, CMRR analysis, failure cases.
  - [ ] P8.2.7 — **Conclusion**: 150 words; restate contributions; one sentence future work.

- [ ] **P8.3** — Internal review
  - [ ] P8.3.1 — Check page count: must be exactly 5 pages with references on page 5 (or 8+1 for OJSP track)
  - [ ] P8.3.2 — Verify all figures are legible at print size
  - [ ] P8.3.3 — Verify all claims in text are backed by a table/figure result
  - [ ] P8.3.4 — IEEE check: confirm no author names in paper (double-blind)

- [ ] **P8.4** — External feedback
  - [ ] P8.4.1 — Share draft with advisor/colleague for feedback
  - [ ] P8.4.2 — Address all feedback; re-run any experiments flagged as weak

- [ ] **P8.5** — Final submission
  - [ ] P8.5.1 — Register on ICASSP 2027 submission system
  - [ ] P8.5.2 — Upload PDF + supplementary material
  - [ ] P8.5.3 — Confirm receipt email
  - [ ] P8.5.4 — Log submission ID and timestamp to CONTEXT.md

---

## Discovered Tasks / Backlog

> Tasks added during the project that were not in the original plan.
> Format: `[status] Task description | Discovered: [date] | Reason: [why it was added]`

- [x] Implement Kaggle-ready project codebase with reusable feature cache scripts | Discovered: May 26, 2026 | Reason: Researcher requested complete implementation before Kaggle upload
- [ ] Add true input-space PGD through raw DINOv2/Whisper encoders | Discovered: May 26, 2026 | Reason: Current adversarial script attacks cached features as a fast proxy; final paper robustness claim needs input-space attacks
- [ ] Decide whether to run optional raw-mode LayerNorm tuning experiment | Discovered: May 26, 2026 | Reason: Default cached-feature training freezes foundation encoders, so LN tuning is not part of the fast path
- [x] Add claim-audit workflow to prevent overclaiming from V1 proxy attacks | Discovered: May 29, 2026 | Reason: V1 degradation results are promising, but feature-space adversarial results do not support the original strong claim yet
- [x] Replace invalid per-category AUC with fake-category-vs-RR contrasts | Discovered: May 29, 2026 | Reason: FakeAVCeleb categories are label-pure, so AUC within FR/RF/FF/RR is undefined
- [x] Add test-set ablation evaluator | Discovered: May 29, 2026 | Reason: Ablation validation AUC alone does not answer the robustness claim
- [ ] Run raw visual input-space PGD pilot on a small subset | Discovered: May 29, 2026 | Reason: Needed to decide whether the original cross-modal adversarial claim survives
- [x] Run modality masking diagnostics on clean and D12 | Discovered: May 29, 2026 | Reason: Needed to measure whether CMAR relies on both modalities or collapses to one stream; V2 shows full CMAR beats either masked-modality probe on clean and D12
- [ ] Rerun ablation test evaluation with correct `--ablation-root` | Discovered: May 29, 2026 | Reason: V2 `ablation-results.json` only contains `full_cmar` even though ablation checkpoints exist
- [ ] Add threshold calibration / EER operating point report | Discovered: May 29, 2026 | Reason: V2 default 0.5 threshold falsely flags 57.33% of RR real clips as fake

---

## Decisions Log

> Record all non-trivial decisions made during execution.
> Format: `[date] Decision: [what was decided] | Reason: [why] | Alternatives considered: [what was ruled out]`

**[Pre-execution]**
- Decision: Use Whisper-Tiny instead of Whisper-Small | Reason: Whisper-Small (244M params) causes OOM on T4 when combined with DINOv2-Small and a batch of video clips. Whisper-Tiny (encoder ~15M) eliminates this risk with minimal quality trade-off. | Alternatives: Whisper-Small with gradient checkpointing (adds complexity); Wav2Vec2-Base (different feature space, less established for deepfake detection)

**[Pre-execution]**
- Decision: Simplified TTDA (forward-pass ensemble, no gradient step) instead of gradient-based TTDA | Reason: Gradient-step TTA during inference is an OOM risk on T4 and adds implementation complexity. The ensemble approach achieves similar robustness benefits. | Alternatives: Original 1-step LN gradient adaptation (from RobustAV plan — rejected for hardware reasons)

**[Pre-execution]**
- Decision: Drop BA-TFD as baseline | Reason: BA-TFD checkpoints trained on FakeAVCeleb are not publicly maintained; reproducing from scratch adds ~8 GPU hours with high failure risk. LipForensics is a better-maintained, more widely cited baseline. | Alternatives: Training BA-TFD from scratch (too time-consuming); using a different AV detector

**[May 26, 2026]**
- Decision: Make cached-feature training the default Kaggle path | Reason: Researcher explicitly requested preprocessing once so model/config reruns do not reprocess samples one at a time. | Alternatives: End-to-end LN tuning every run (closer to original plan but much slower); caching raw frames/audio only (saves decode time but still reruns foundation encoders)

---

## Milestones

| Milestone | Target Date | Status | Notes |
|---|---|---|---|
| M0: Environment validated, both models load on T4 | Week 1 | `[ ]` | |
| M1: Feature cache uploaded as Kaggle dataset | Week 1 | `[x]` | Uploaded as `vasuaashadesai/cmar-features-clean-v1`; includes clean and degraded-test cache |
| M2: CMAR training converged, val AUC ≥ 0.85 | Week 2 | `[ ]` | |
| M3: All clean + degraded results generated | Week 2 | `[ ]` | |
| M4: All baseline results generated | Week 2–3 | `[ ]` | |
| M5: All adversarial results generated + CMRR computed | Week 3 | `[ ]` | |
| M6: All ablations complete | Week 3 | `[ ]` | |
| M7: All paper figures generated | Week 4 | `[ ]` | |
| M8: Full paper draft complete | Week 7 | `[ ]` | |
| M9: External feedback incorporated | Week 10 | `[ ]` | |
| M10: Paper submitted to ICASSP 2027 | Sep 16, 2026 | `[ ]` | **Hard deadline** |

---

## Known Issues

> Issues currently open that need resolution.
> Format: `[issue-id] Description | Severity: HIGH/MEDIUM/LOW | Status: open/investigating/resolved`

*(None yet — populate as project progresses)*

---

*End of TASK.md — Maintained by AI agent; reviewed by researcher each session. Version: 1.0 | Created: May 2026*
