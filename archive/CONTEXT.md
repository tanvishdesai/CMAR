# CONTEXT.md — Session State and Running Project Memory

> **AI Agent Instruction — READ THIS FIRST**:
>
> This file is your working memory for the CMAR project. It is the ONLY file you modify freely during a session. PLANNING.md is read-only. TASK.md is updated only to change task statuses and add discovered tasks.
>
> **At the start of every session**:
> 1. Read PLANNING.md in full
> 2. Read this file (CONTEXT.md) in full
> 3. Read TASK.md and identify the current active phase and open tasks
> 4. Confirm with researcher what the session goal is before writing any code
>
> **During every session**:
> - Update the "Current Session" section below immediately
> - Log every meaningful action, result, decision, or error in the "Session Log" for the active session
> - If you discover a problem or unexpected result, add it to the "Issues" section immediately
> - If a task is completed, mark it `[x]` in TASK.md with a one-line completion note
>
> **At the end of every session**:
> - Fill in the "End of Session Summary" for the completed session
> - Update the "Persistent State" section with any values that future sessions need
> - Update the "Next Session" section so the next agent instance knows exactly what to do
> - Commit this file's changes before closing the session
>
> **CRITICAL RULES**:
> - Never assume a previous result is correct without re-reading it from this file
> - Never re-run a completed phase unless the researcher explicitly instructs it
> - If you are uncertain about a decision, log it as [?] and ask the researcher rather than assuming
> - All numerical results go in the "Metrics Ledger" section below, not just in session logs

---

## Project Metadata

| Field | Value |
|---|---|
| Project | CMAR — Cross-Modal Adversarial Robustness |
| Started | May 2026 |
| Current Phase | V2 audit complete; pivot toward adversarial robustness characterization |
| Total GPU hours used | Not centrally logged |
| Kaggle sessions completed | Preprocessing completed across Kaggle/Colab runs |
| Last updated | May 29, 2026 |

---

## Persistent State
> Stable facts discovered during the project that all future sessions need.
> Update these when values are first confirmed; they do not change unless re-measured.

### Hardware confirmed
- T4 VRAM available: *(fill in after P0)*
- Peak VRAM with DINOv2-Small + Whisper-Tiny + batch=8: *(fill in after P0.1)*
- Safe batch size: *(fill in after P0.1)*
- Gradient checkpointing required: *(Yes/No — fill in after P0.1)*

### Dataset paths (Kaggle)
- FakeAVCeleb dataset path: `/kaggle/input/datasets/shreyaty08/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2` during preprocessing; some Colab/Kaggle runs used the nested `frames` directory for frame-backed variants.
- LAV-DF dataset path: `/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF`
- Feature cache dataset ID: `vasuaashadesai/cmar-features-clean-v1` (display name: "CMAR Clean Features V1"; now contains clean and all degraded-test features)
- Feature cache mounted path: usually `/kaggle/input/cmar-features-clean-v1/cmar_cache`; if different, run `find /kaggle/input -maxdepth 5 -type d -name cmar_cache`
- Checkpoint dataset ID (`cmar-checkpoint-v1`): *(fill in after P2.3.7)*

### Model architecture confirmed
- DINOv2-Small feature dimension: 384 ✓ (from PLANNING.md)
- Whisper-Tiny output feature dimension: 384
- Average Whisper-Tiny temporal output length T_a for 10s audio: pooled to `<=64` tokens in cached features
- Confirmed trainable parameter count: 4,316,929 in local dummy check

### FakeAVCeleb dataset structure
- Directory layout: Observed from `dataset info/` screenshots and documented in `docs/DATASET_LAYOUT.md`; scanner expects the four category folders under `FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/`.
- Confirmed split sizes: train 3,850; val 825; test 825
- AV categories present in test split: RR, FR, RF, FF expected from manifest scanner and completed cache

### Software versions actually installed
*(fill in after P0.1.12)*
```
torch:
torchvision:
torchaudio:
transformers:
timm:
torchattacks:
librosa:
soundfile:
pydub:
```

---

## Metrics Ledger
> Ground truth for all numerical results. One row per experiment condition per model.
> This is the authoritative source. Paper tables are generated from this data.
> Format: model | condition | AUC | EER | AP | RAR | notes

### Training Metrics
| Metric | Value | Session | Notes |
|---|---|---|---|
| Best validation AUC | 0.9253688889 | V1 | Full CMAR, epoch 10 |
| Epoch of best val AUC | 10 | V1 | Early stopped after epoch 15 |
| Final train loss | 0.0456398231 | V1 | Epoch 15, overfitting visible |
| Final val loss | 0.4427891037 | V1 | Epoch 15, best checkpoint remains epoch 10 |

### Clean Evaluation (E1: FakeAVCeleb test, no TTDA)
| Model | AUC ± CI | EER ± CI | AP ± CI | Session |
|---|---|---|---|---|
| CMAR (ours) | 0.8814577778 ± 0.0353484145 | 0.2106666667 ± 0.0624312102 | 0.9862074794 ± 0.0052551748 | V1 |
| CMAR + TTDA | TBD | TBD | TBD | S3 |
| LipForensics | TBD | TBD | TBD | S4 |
| Late-Fusion | TBD | TBD | TBD | S4 |
| CMAR-VisualOnly | TBD | TBD | TBD | S6 |
| CMAR-NoConsistency | TBD | TBD | TBD | S6 |

### Degraded Evaluation — RAR by Condition
| Condition | CMAR RAR | Late-Fusion RAR | LipForensics RAR | Session |
|---|---|---|---|---|
| D1 (JPEG QF=75) | TBD | TBD | TBD | S3/S4 |
| D2 (JPEG QF=50) | TBD | TBD | TBD | S3/S4 |
| D3 (resize 0.75×) | TBD | TBD | TBD | S3/S4 |
| D4 (resize 0.50×) | TBD | TBD | TBD | S3/S4 |
| D5 (vis noise σ=0.01) | TBD | TBD | TBD | S3/S4 |
| D6 (vis noise σ=0.02) | TBD | TBD | TBD | S3/S4 |
| D7 (MP3 128k) | TBD | TBD | TBD | S3/S4 |
| D8 (MP3 64k) | TBD | TBD | TBD | S3/S4 |
| D9 (audio SNR=30dB) | TBD | TBD | TBD | S3/S4 |
| D10 (audio SNR=20dB) | TBD | TBD | TBD | S3/S4 |
| D11 (H.264 CRF=28) | 0.8766689524 | TBD | TBD | V1 |
| D12 (social media sim) | 0.9204348352 | TBD | TBD | V1 |
| E13 (LAV-DF clean) | TBD | TBD | TBD | S3/S4 |

### Adversarial Evaluation
| Attack | CMAR AUC | CMAR-VisOnly AUC | Late-Fusion AUC | LipForensics AUC | Session |
|---|---|---|---|---|---|
| A1 PGD visual ε=2 | TBD | TBD | TBD | TBD | S5 |
| A2 PGD visual ε=4 | TBD | TBD | TBD | TBD | S5 |
| A3 PGD visual ε=8 | TBD | TBD | TBD | N/A | S5 |
| A4 PGD audio 30dB | TBD | N/A | TBD | N/A | S5 |
| A5 PGD audio 20dB | TBD | N/A | TBD | N/A | S5 |
| A6 PGD both | TBD | N/A | TBD | N/A | S5 |
| A7 FGSM visual ε=4 | TBD | TBD | TBD | TBD | S5 |
| A8 FGSM audio 30dB | TBD | N/A | TBD | N/A | S5 |

### CMRR (Cross-Modal Robustness Ratio)
| Model | CMRR value | Interpretation | Session |
|---|---|---|---|
| CMAR (ours) | TBD | >2.5 = strong protection | S5 |
| Late-Fusion | TBD | ~2.0 = no cross-modal protection | S5 |

### Ablation Results (key conditions only: E1 clean, D12, A2, A5)
| Model | Clean AUC | D12 RAR | A2 AUC | A5 AUC | Session |
|---|---|---|---|---|---|
| Full CMAR | TBD | TBD | TBD | TBD | S3/S6 |
| AB1 VisualOnly | TBD | TBD | TBD | N/A | S6 |
| AB2 NoConsistency | TBD | TBD | TBD | TBD | S6 |
| AB3 1-Layer CMCM | TBD | TBD | TBD | TBD | S6 |
| AB4 4-Layer CMCM | TBD | TBD | TBD | TBD | S6 |
| TTDA K=1 copy | TBD | TBD | N/A | N/A | S6 |
| TTDA K=2 copies | TBD | TBD | N/A | N/A | S6 |
| TTDA K=3 copies | TBD | TBD | N/A | N/A | S6 |
| TTDA K=5 copies | TBD | TBD | N/A | N/A | S6 |

---

## Issues Tracker
> Format: `[ISSUE-ID] Status | Description | Severity | Discovered | Resolution`
> Status: OPEN / INVESTIGATING / RESOLVED / WONT-FIX

[ISSUE-001] OPEN | Current adversarial results are cached-feature proxy attacks, not raw input-space attacks; they cannot support the final adversarial robustness claim. | Severity: HIGH | Discovered: May 29, 2026 | Resolution: Added explicit protocol metadata to `scripts/05_adversarial_evaluation.py`; raw input-space PGD remains required.

[ISSUE-002] OPEN | Per-category AUC within FR/RF/FF/RR is invalid because categories are label-pure; old result JSON contains NaN category AUCs. | Severity: MEDIUM | Discovered: May 29, 2026 | Resolution: Added category-vs-RR contrasts and threshold operating points to `scripts/03_evaluate_clean_degraded.py`.

[ISSUE-003] OPEN | TTDA-style cached ensemble hurt clean AUC in V1 and was mislabeled as TTDA. | Severity: MEDIUM | Discovered: May 29, 2026 | Resolution: Deprecated `clean_ttda` output and renamed the audit-only result to `clean_cached_ensemble`; true runtime TTDA is not claimed.

[ISSUE-004] OPEN | V2 ablation test evaluation only contains `full_cmar`; trained ablation checkpoints exist, but `ablation-results.json` did not include them, likely due to an ablation-root path mismatch. | Severity: MEDIUM | Discovered: May 29, 2026 | Resolution: Rerun `scripts/09_evaluate_ablations.py` with the correct `--ablation-root`.

[ISSUE-005] OPEN | Clean RR false-positive rate is high at the default 0.5 threshold: 57.33% of real clips are predicted fake. AUC remains useful, but thresholded deployment claims are weak until calibration is addressed. | Severity: MEDIUM | Discovered: May 29, 2026 | Resolution: Add threshold calibration / report EER-threshold operating point.

---

## Session Logs

### Session 0 — Pre-execution Planning
**Date**: May 2026  
**Goal**: Planning and document creation  
**Completed**: PLANNING.md and TASK.md created  
**Key decisions**: See Decisions Log in TASK.md  
**Next session goal**: Phase 0 environment validation  

### Local Implementation Session — May 26, 2026
**Kaggle notebook**: N/A (local code implementation)  
**Goal**: Implement the CMAR project codebase and reusable preprocessing cache workflow before Kaggle execution  

**Actions log**:
- [x] Implemented `cmar/` package with cached-feature CMAR, visual-only ablation, CMCM fusion, temporal aggregation, classifier, encoder wrappers, datasets, losses, trainer, metrics, degradations, attacks, checkpoint loading, and plotting helpers.
- [x] Implemented Kaggle-facing scripts `scripts/00_environment_check.py` through `scripts/07_analysis_figures.py`.
- [x] Added reusable preprocessing script `scripts/01_preprocess_features.py` that builds manifests, extracts DINOv2/Whisper features, and writes clean plus degraded test caches.
- [x] Added configs, requirements, README, Kaggle runbook, dataset layout notes, preprocessing cache documentation, and research proposal review.
- [x] Ran `python -m compileall cmar scripts` successfully.
- [x] Ran dummy CMAR forward pass on tensors shaped like cached DINOv2 and Whisper features successfully.
- [x] Ran one-epoch synthetic cached-feature training smoke test; checkpoint save/load worked.
- [x] Ran CLI help checks for all scripts successfully.

**Important implementation decision**:
- Default training consumes cached DINOv2/Whisper features for fast Kaggle reruns. This freezes foundation features at preprocessing time; LayerNorm tuning inside DINOv2/Whisper is not active in the default cached-feature training path. The paper should describe this accurately unless a later raw-mode LN-tuning experiment is added.

**Results**:
- CMAR cached-feature model parameter count: 4,316,929 trainable parameters in the local dummy check.
- Local environment check ran on CPU; CUDA was not available locally, so GPU memory validation remains a Kaggle Session 0 task.

**Issues discovered**:
- `torchattacks` was missing locally and `ffmpeg` was not found by pydub on this Windows environment. Kaggle should be checked separately with `scripts/00_environment_check.py --load-models`.

**End of session summary**:
- Code implementation was ready for Kaggle validation at this point. This note is superseded by the May 29 feature-cache completion session below.

### Kaggle Fix Session — May 26, 2026
**Kaggle notebook**: User-reported Session 0/1 errors  
**Goal**: Fix DINOv2 input-size mismatch during environment check and preprocessing  

**Actions log**:
- [x] Patched `cmar/models/visual_encoder.py` so `DINOv2FeatureExtractor` creates timm DINOv2 with `img_size=224` and `dynamic_img_size=True` when supported.
- [x] Patched `scripts/00_environment_check.py` to report the DINOv2 image size and patch embedding image size.
- [x] Patched `cmar/utils/cache.py` and `scripts/01_preprocess_features.py` so preprocessing passes the configured image size into extractor construction.
- [x] Added troubleshooting notes to preprocessing and Kaggle run docs.
- [x] Ran `python -m compileall cmar scripts` successfully after the patch.

**Issue resolved**:
- Kaggle `AssertionError: Input height (224) doesn't match model (518)` was caused by recent `timm` DINOv2 pretrained configs defaulting to 518px inputs unless `img_size=224` is explicitly supplied.

### Kaggle Stability Fix Session — May 26, 2026
**Kaggle notebook**: User-reported preprocessing crash near 278/3850 rows  
**Goal**: Harden preprocessing against backend error 137 / OS-level container kill  

**Actions log**:
- [x] Inspected prior `ROBUSTAV/02_extract_features.py` crash-safe implementation.
- [x] Added resumable preprocessing guards to CMAR: `--max-new-rows`, `--max-runtime-seconds`, `--chunk-size`, `--splits`, `--conditions`, and `--degraded-only`.
- [x] Added chunk-level garbage collection and CUDA cache cleanup.
- [x] Added RSS memory progress reporting during preprocessing chunks.
- [x] Changed saved Whisper features to pooled `<=64` tokens by default via `max_audio_tokens`.
- [x] Replaced pydub MP3 roundtrip with direct ffmpeg subprocess calls using `-threads 1`.
- [x] Changed H.264 degraded visual preprocessing to sampled-frame roundtrip compression instead of full-video re-encoding.
- [x] Added AAC roundtrip for D11 audio without full-video re-encoding.
- [x] Updated preprocessing docs with crash-safe Kaggle commands.

**Important operational note**:
- Because audio cache shape changed from raw Whisper length to pooled `<=64` tokens, delete any partial `/kaggle/working/cmar_cache` created before this patch and rerun clean preprocessing in sliced mode.

### Training Cache Preflight Fix — May 26, 2026
**Kaggle notebook**: User-reported training FileNotFoundError after sliced preprocessing  
**Goal**: Prevent DataLoader worker crashes when feature cache is incomplete  

**Actions log**:
- [x] Added `cache_coverage_report` to `cmar/training/dataset.py`.
- [x] Added `--cache-report-only` and `--allow-partial-cache` to `scripts/02_train_cmar.py`.
- [x] Training now prints train/val feature coverage before constructing DataLoader workers.
- [x] Default training stops with a clear resume-preprocessing message if cache is incomplete.
- [x] Partial-cache training is available only through an explicit smoke-test flag.

**Issue resolved**:
- The manifest lists all split rows even when preprocessing was intentionally stopped after `--max-new-rows 200`; training must wait until clean train/val features are complete or explicitly filter to available rows for smoke tests.

### Feature Cache Completion Session - May 29, 2026
**Kaggle/Colab notebooks**: clean preprocessing plus degraded parallel/condition runs  
**Goal**: Finish reusable feature cache and upload it as a Kaggle dataset  

**Actions log**:
- [x] Completed clean cached features for FakeAVCeleb train/val/test: train 3,850, val 825, test 825.
- [x] Completed all 12 degraded test feature folders under `features/degraded_test/`: D1 through D12 are present in the Kaggle dataset screenshot.
- [x] Uploaded the combined cache as Kaggle dataset `vasuaashadesai/cmar-features-clean-v1` with display name "CMAR Clean Features V1".
- [x] Updated runbook and cache documentation so future Kaggle notebooks train/evaluate from the completed cache instead of rerunning preprocessing.

**Important operational note**:
- The dataset name still says "Clean Features", but the mounted `cmar_cache/` now includes both clean features and all degraded-test folders. The next step is training CMAR, not preprocessing.

**Next action**:
- In Kaggle, attach the CMAR code dataset and `CMAR Clean Features V1`, run the cache coverage report, then train with `scripts/02_train_cmar.py`.

### Kaggle Training/Evaluation Session - May 29, 2026
**Kaggle notebook**: CMAR training/evaluation run  
**Goal**: Train CMAR and begin downstream evaluations from the completed feature cache  

**Actions log**:
- [x] Confirmed train cache coverage: 3,850/3,850 rows available.
- [x] Confirmed val cache coverage: 825/825 rows available.
- [x] Trained full CMAR with `--no-amp --lr 0.0002`; early stopped after epoch 15.
- [x] Best full CMAR validation metrics: epoch 10, val AUC 0.9253688889, val EER 0.1573333333, val AP 0.9918310648.
- [x] Clean/degraded evaluation produced metrics in stdout through all 12 degraded conditions, but crashed before writing JSON because `lavdf_test.csv` existed without LAV-DF feature tensors.
- [x] Feature-space adversarial evaluation completed and wrote `/kaggle/working/adversarial-results.json`.
- [x] Ablation training completed for `visual_only`, `no_consistency`, `cmcm_1`, and `cmcm_4`.

**Key partial results from stdout**:
- Clean FakeAVCeleb test: AUC 0.8814577778, EER 0.2106666667, AP 0.9862074794.
- D12 social: AUC 0.8113244444, RAR 0.9204348352, delta AUC 0.0701333333.
- D11 H.264 CRF28 was the hardest degradation: AUC 0.7727466667, RAR 0.8766689524.
- Feature-space adversarial AUCs: A2 visual PGD 0.4034666667, A5 audio PGD 0.0128177778, A6 both 0.0.
- Ablation best validation AUCs: visual_only 0.8661511111, no_consistency 0.9359822222, cmcm_1 0.9358577778, cmcm_4 0.9372266667.

**Fix applied after this run**:
- `scripts/03_evaluate_clean_degraded.py` now supports `--skip-lavdf`, checks LAV-DF cache coverage before optional evaluation, and writes partial results after each condition.
- `requirements.txt` no longer installs `torchattacks` by default because it downgrades `requests` on Kaggle; CMAR uses manual feature-space PGD/FGSM.

**Next action**:
- Rerun `scripts/03_evaluate_clean_degraded.py` with `--skip-lavdf` to write `cmar-results-clean-degraded.json`, then rerun `scripts/07_analysis_figures.py`.

### Claim Audit Direction Update - May 29, 2026
**Kaggle notebook**: N/A (local code/research direction update)  
**Goal**: Decide how to proceed after V1 results without forcing the original claim if evidence does not support it  

**Interpretation of V1**:
- Clean/degraded evidence is promising: clean test AUC 0.8814577778 meets the minimum target narrowly; D12 social RAR 0.9204348352 is strong.
- The original strong claim, "cross-modal fusion provides inherent adversarial robustness," is not supported yet. The only completed adversarial evaluation is cached-feature PGD/FGSM, and it collapses badly under feature-space audio/both attacks.
- The better current paper posture is conditional: CMAR is a robust cached foundation-feature fusion model under realistic media degradations; adversarial robustness remains a hypothesis under audit.
- If future raw input-space attacks show CMAR resists single-modality attacks better than unimodal/late-fusion baselines, the stronger CMRR story can be restored. If not, pivot to a rigorous robustness-characterization paper showing multimodal fusion helps degradation robustness but does not automatically solve adversarial robustness.

**Code changes made**:
- Updated `scripts/03_evaluate_clean_degraded.py` to enforce strict degraded-cache coverage by default, report valid category-vs-RR contrasts, report category operating points, rename TTDA-like cached ensembling as `clean_cached_ensemble`, and optionally run modality-masking probes.
- Updated `scripts/05_adversarial_evaluation.py` so output JSON explicitly marks attacks as cached-feature proxy diagnostics and `valid_for_final_adversarial_claim=false`.
- Added `scripts/08_claim_audit.py` to generate `claim-audit.json` and `claim-audit.md` with a recommended paper direction based on current metrics.
- Added `scripts/09_evaluate_ablations.py` to evaluate trained ablation checkpoints on clean/D12/D11 test conditions instead of relying only on validation AUC.
- Updated figure generation to use `category_contrasts` instead of invalid per-category AUC and to label adversarial proxy plots clearly.
- Updated README, Kaggle runbook, and research review notes with the revised claim-audit workflow.

**Next action**:
- Run the updated sequence: strict clean/degraded evaluation with `--include-cached-ensemble --include-modality-masking`, rerun feature-space adversarial proxy, evaluate ablations on test/degraded conditions, run `scripts/08_claim_audit.py`, then generate figures.

### V2 Results Audit - May 29, 2026
**Kaggle notebook**: V2 result folder  
**Goal**: Analyze the second-version audit workflow results and decide the next research direction  

**Key results**:
- Strict cache coverage passed for all 12 degraded conditions: 825/825 rows available per condition.
- Clean AUC remained 0.8814577778; D12 social AUC 0.8113244444 with RAR 0.9204348352; D11 H.264 AUC 0.7727466667 with RAR 0.8766689524.
- Cached ensemble hurt clean AUC: 0.8586133333 vs clean 0.8814577778, gain -0.0228444444. Do not use TTDA/cached ensemble as a contribution.
- Category contrasts: FF_vs_RR AUC 0.9900709220 is excellent; FR_vs_RR AUC 0.7681523810 is weak; RF_vs_RR AUC 0.8322222222 is moderate but based on only 24 fake RF samples.
- Default 0.5-threshold RR false-positive rate is 0.5733333333, which is too high for deployment-style claims even though AUC is acceptable.
- Modality masking shows fusion helps: clean full 0.8814577778 vs visual-only probe 0.82032 and audio-only probe 0.7463377778; D12 full 0.8113244444 vs visual-only probe 0.6890666667 and audio-only probe 0.7428355556.
- Feature-space adversarial proxy still collapses: A2 visual PGD AUC 0.4034666667, A5 audio PGD AUC 0.0128355556, A6 both AUC 0.0. This is not final input-space evidence, but it warns against claiming inherent adversarial robustness.
- Ablation training summary exists and shows visual_only val AUC 0.8661511111, full CMAR val AUC 0.9253688889, no_consistency val AUC 0.9359822222, cmcm_1 val AUC 0.9358577778, cmcm_4 val AUC 0.9372266667.
- `ablation-results.json` only contains `full_cmar`, so test-condition ablation evaluation still needs to be rerun with the correct checkpoint root.

**Interpretation**:
- V2 aligns with the post-V1 expectation: degradation robustness is promising and multimodal fusion helps, but the original strong adversarial robustness claim is not yet supported.
- The domain should remain adversarial robustness for deepfake detection, but the paper narrative should become a rigorous robustness characterization: cross-modal fusion improves robustness to realistic degradations and provides a framework for testing adversarial robustness, while raw input-space attacks will decide whether stronger CMRR claims survive.

**Next action**:
- Rerun ablation test evaluation with the correct ablation root, calibrate/evaluate threshold behavior, then implement a raw input-space visual PGD pilot with SSIM checks before scaling to full adversarial evaluation.

---

### Session 1 — [Date: TBD]
**Kaggle notebook**: `00_environment_check.ipynb`  
**Goal**: Environment validation (Phase 0) → Feature extraction start (Phase 1)  
**Session start VRAM**: TBD  
**Packages installed**: See Persistent State → Software versions  

**Actions log** *(agent fills this in during the session)*:
- [ ] *(to be filled during session)*

**Results**:
- Peak VRAM (DINOv2 + Whisper-Tiny + batch=8): TBD
- FakeAVCeleb accessible: TBD
- LAV-DF accessible: TBD
- All transforms working: TBD

**Issues discovered**: *(none yet)*

**End of session summary**: *(fill in at session end)*

**Tasks completed this session**: *(list P0.x and P1.x task IDs)*

**GPU hours used this session**: TBD  
**Total GPU hours to date**: TBD  

---

### Session 2 — [Date: TBD]
**Kaggle notebook**: `01_feature_extraction.ipynb`  
**Goal**: Complete Phase 1 — extract all features, upload cache  

**Actions log**: *(to be filled during session)*

**Results**:
- Total clips processed: TBD
- Average visual extraction time per clip: TBD
- Average audio extraction time per clip: TBD
- Whisper-Tiny T_a for 10s audio: TBD
- Cache total size: TBD
- Cache dataset ID (cmar-features-v1): TBD

**Issues discovered**: *(none yet)*

**End of session summary**: *(fill in at session end)*

**Tasks completed**: *(list task IDs)*  
**GPU hours this session**: TBD  
**Total GPU hours to date**: TBD  

---

### Session 3 — [Date: TBD]
**Kaggle notebook**: `02_train_cmar.ipynb`  
**Goal**: Phase 2 — train CMAR to convergence  

**Actions log**: *(to be filled)*

**Results**:
- Confirmed trainable parameter count: TBD
- Training converged at epoch: TBD
- Best val AUC: TBD (at epoch TBD)
- Final val loss: TBD
- Checkpoint saved as: TBD

**End of session summary**: *(fill in at session end)*

**GPU hours this session**: TBD  
**Total GPU hours to date**: TBD  

---

### Session 4 — [Date: TBD]
**Kaggle notebook**: `03_evaluate_clean_degraded.ipynb`  
**Goal**: Phase 3 — clean + degraded evaluation  

*(to be filled)*

---

### Session 5 — [Date: TBD]
**Kaggle notebook**: `04_evaluate_baselines.ipynb`  
**Goal**: Phase 4 — baseline evaluation  

*(to be filled)*

---

### Session 6 — [Date: TBD]
**Kaggle notebook**: `05_adversarial_evaluation.ipynb`  
**Goal**: Phase 5 — adversarial evaluation + CMRR  

*(to be filled)*

---

### Session 7 — [Date: TBD]
**Kaggle notebook**: `06_ablations.ipynb`  
**Goal**: Phase 6 — ablation studies  

*(to be filled)*

---

### Session 8 — [Date: TBD]
**Kaggle notebook**: `07_analysis_figures.ipynb`  
**Goal**: Phase 7 — all paper figures  

*(to be filled)*

---

## Flags for Researcher Attention
> Items that need a human decision. Agent cannot proceed on these without explicit instruction.
> Format: `[FLAG-ID] Waiting for: description | Added: date | Blocking: which task`

*(None yet)*

---

## Quick Reference: Key Numbers
> Filled in progressively as experiments complete. Useful for sanity-checking new results.

| Number | Value | Source |
|---|---|---|
| FakeAVCeleb train size | 3,850 clips | PLANNING.md |
| FakeAVCeleb test size | 825 clips | PLANNING.md |
| DINOv2-Small feature dim | 384 | PLANNING.md |
| Whisper-Tiny feature dim | 384 | Cached feature contract |
| Total trainable params | 4,316,929 | Local dummy CMAR check |
| Target clean AUC (minimum) | 0.88 | PLANNING.md §11.1 |
| Target D12 RAR (minimum) | 0.82 | PLANNING.md §11.1 |
| Target CMRR (minimum) | > CMRR of Late-Fusion | PLANNING.md §11.1 |
| ICASSP 2027 deadline | September 16, 2026 | Confirmed |
| Kaggle session limit | 12 hours | Hard constraint |

---

*This file is maintained by the AI agent. Researcher should review the "Flags for Researcher Attention" section at the start of each session. Version: 1.0 | Template created: May 2026*
