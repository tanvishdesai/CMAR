# av-robustbench — Task Tracker

> **Status**: Initial package implementation complete  
> **Last Updated**: 2026-05-31  
> **Reference**: [PLANNING.md](./PLANNING.md) for architecture & design decisions

---

## Milestone 1: Package Skeleton & Core Abstractions (Week 1)

### 1.1 Project Setup
- [ ] Initialize Python package structure with `pyproject.toml`
  - Package name: `av-robustbench`
  - Python ≥ 3.9
  - Optional dependency groups: `[models]`, `[degradations]`, `[all]`
  - Entry point: `av-robustbench` CLI
- [ ] Create directory structure per PLANNING.md §3.1
- [ ] Set up `ruff` linting config
- [ ] Set up `mypy` type checking config
- [ ] Create `pytest` configuration and test directory
- [ ] Write `README.md` (project overview, installation, quickstart)
- [ ] Create `LICENSE` (MIT or Apache 2.0)
- [ ] Set up GitHub repository structure

### 1.2 Core Abstractions (`av_robustbench/__init__.py` + core modules)
- [ ] Implement `AVDetector` abstract base class
  - `predict(visual, audio) -> dict`
  - `input_type` property: `"features"` or `"raw"`
  - `feature_dims` property: `{"visual": (T_v, D_v), "audio": (T_a, D_a)}`
  - `name` property
  - `from_checkpoint(path)` classmethod
- [ ] Implement `CertificationResult` dataclass
  - Fields: `predicted_class`, `certified_radius`, `correct`, `pA_lower`, `abstained`
  - `certified_correct` property
- [ ] Implement `AttackResult` dataclass
  - Fields: `clean_logits`, `adversarial_logits`, `labels`, `eps`, `threat_model`
  - Methods: `clean_accuracy()`, `adversarial_accuracy()`, `attack_success_rate()`
- [ ] Implement `RobustnessCard` dataclass
  - Fields: `model_name`, `dataset`, `clean_metrics`, `certification`, `attacks`, `degradations`
  - Methods: `to_json()`, `to_latex()`, `to_markdown()`
- [ ] Write unit tests for all core dataclasses

---

## Milestone 2: Certification Engine (Week 2)

### 2.1 Smoothing Core (`av_robustbench/certification/`)
- [ ] Port and generalize `cmar/certification/core.py`
  - `lower_confidence_bound_exact(nA, n, alpha)` — Clopper-Pearson
  - `certified_radius(sigma, pA_lower)` — Cohen et al. formula
  - Unit tests with known values from Cohen et al. paper
- [ ] Port and generalize `cmar/certification/smoothing.py` → `SmoothedAVClassifier`
  - Accept any `AVDetector` instance (not just CMAR)
  - Support noise modes: `"joint"`, `"visual_only"`, `"audio_only"`
  - `predict(visual, audio, n_samples)` → predicted class
  - `certify(visual, audio, label, n0, n, alpha)` → `CertificationResult`
  - `certify_dataset(dataset, ...)` → `list[CertificationResult]`
  - Progress callback support (for CLI progress bars)
- [ ] Implement `certified_accuracy_curve(results, radii)` utility
- [ ] Implement `certified_accuracy_at_radius(results, r)` utility
- [ ] Write comprehensive tests
  - Test with a mock AVDetector that always returns class 1
  - Test with a mock that flips under noise (should have small radius)
  - Test boundary: pA_lower exactly 0.5 → radius 0 (abstain boundary)

### 2.2 Multi-Sigma Certification Runner
- [ ] Implement `certify_multi_sigma(model, dataset, sigmas=[0.12, 0.25, 0.50, 1.00])`
  - Returns a dict mapping sigma → certification results
  - Generates certified accuracy curves for each sigma
  - Computes comparative metrics (optimal sigma per sample)
- [ ] Add `--sigma` list support to CLI

---

## Milestone 3: Attack Suite (Week 3)

### 3.1 Attack Framework (`av_robustbench/attacks/`)
- [ ] Implement `BaseAttack` abstract class
  - `attack(model, visual, audio, label) -> (adv_visual, adv_audio)`
  - `threat_model` property: `"Linf"`, `"L2"`, `"feature_L2"`
  - `eps` property
  - Supports both feature-space and input-space modes
- [ ] Implement `PGDAttack` (L∞ in feature space)
  - Port from `scripts/12_empirical_attack_comparison.py`
  - Configurable: `eps`, `n_steps`, `step_size`, `random_start`
  - `attack_target`: `"visual"`, `"audio"`, `"both"`
  - Supports batch processing
- [ ] Implement `PGDAttackL2` (L₂ in feature space)
  - L₂ projected gradient descent
  - Same interface as PGDAttack
- [ ] Implement `PGDInputSpace` (L∞ through frozen encoders)
  - Port from `scripts/17_input_space_attack.py`
  - Loads DINOv2/Whisper with gradient tracking
  - Measures feature displacement
  - Returns both adversarial inputs AND feature displacement metrics
- [ ] Write tests
  - Test PGD on a linear model (analytical solution known)
  - Test L2 projection correctness
  - Test attack_target modes

### 3.2 Advanced Attacks (v1.1 — defer)
- [ ] Implement `SquareAttack` (black-box, score-based)
  - Adapted for binary AV detection
  - Random square perturbations in feature space
- [ ] Implement `AutoAttackAV` (ensemble)
  - APGD-CE + APGD-DLR + Square Attack
  - Binary detection variant (no DLR for 2-class)
  - Automatic detection of gradient masking
- [ ] Implement `CrossModalTransferAttack`
  - Train a surrogate visual-only model
  - Generate adversarial examples using surrogate
  - Test transferability to the target AV model

### 3.3 Attack Evaluation Pipeline
- [ ] Implement `evaluate_under_attack(model, dataset, attacks, max_samples=None)`
  - Runs all specified attacks on the dataset
  - Returns `AttackResult` for each attack
  - Computes clean AUC, adversarial AUC, attack success rate
- [ ] Support parallel evaluation (multiple attacks on same data)

---

## Milestone 4: Model Zoo & Adapters (Week 4)

### 4.1 Model Registry (`av_robustbench/models/registry.py`)
- [ ] Implement model registry
  - `register_model(name, adapter_class, checkpoint_url, metadata)`
  - `load_model(name, dataset=None, threat_model=None)` → `AVDetector`
  - Auto-download from HuggingFace Hub
  - Local cache management
  - `list_models()` → table of available models
- [ ] Create model metadata JSON format
  ```json
  {
    "name": "certav_sigma100",
    "paper": "CertAV: Certified Robustness for AV Deepfake Detection",
    "paper_url": "https://arxiv.org/...",
    "architecture": "CMAR + Randomized Smoothing",
    "training_sigma": 1.0,
    "clean_auc": 0.985,
    "certified_radius_mean": 2.215,
    "checkpoint_url": "https://huggingface.co/.../certav_sigma100/best.pt",
    "feature_extractors": ["dinov2-small", "whisper-tiny"]
  }
  ```

### 4.2 Model Adapters (`av_robustbench/models/adapters/`)
- [ ] Implement `CMARadapter(AVDetector)` — wraps CMAR model
  - Loads checkpoint, creates CMAR instance
  - Implements predict() with proper input handling
  - Returns logits + probs
- [ ] Implement `CertAVAdapter(AVDetector)` — wraps CertAV (noise-augmented CMAR)
  - Same as CMARadapter but with sigma metadata
  - Pre-configured for smoothed evaluation
- [ ] Implement `GenericFeatureAdapter(AVDetector)` — wraps any model using cached features
  - For models that also use DINOv2+Whisper features
  - Only requires implementing a forward function
- [ ] Register initial models in the zoo
  - `cmar_baseline` — CMAR without noise augmentation
  - `certav_sigma025` — CertAV σ=0.25
  - `certav_sigma100` — CertAV σ=1.00
  - `cmar_pgd_at` — PGD adversarially trained CMAR

### 4.3 Upload Checkpoints
- [ ] Upload CertAV checkpoints to HuggingFace Hub
  - Create `av-robustbench` organization on HuggingFace
  - Upload best.pt for each sigma configuration
  - Write model cards for each checkpoint
- [ ] Test `load_model()` end-to-end from HuggingFace

---

## Milestone 5: Degradation Battery (Week 5)

### 5.1 Degradation Specs (`av_robustbench/degradations/specs.py`)
- [ ] Define `DegradationSpec` dataclass
  - `name`, `description`, `visual` (bool), `audio` (bool)
  - `params` dict
  - `requires_ffmpeg` flag
- [ ] Implement 12 standard degradation conditions
  - Port from `cmar/evaluation/degradations.py`
  - `d1_jpeg75`, `d2_jpeg50`: JPEG compression
  - `d3_resize075`, `d4_resize050`: Spatial downsampling
  - `d5_vnoise001`, `d6_vnoise002`: Gaussian video noise
  - `d7_mp3_128k`, `d8_mp3_64k`: MP3 audio compression
  - `d9_anoise_30db`, `d10_anoise_20db`: Audio noise injection
  - `d11_h264_crf28`: H.264 video codec
  - `d12_social`: Social media simulation (combined)

### 5.2 Degradation Runners
- [ ] Implement `apply_degradation(video_path, audio_path, spec) -> (degraded_video, degraded_audio)`
  - Uses ffmpeg for codec operations
  - Uses OpenCV for frame-level operations
  - Uses soundfile for audio operations
- [ ] Implement `DegradationBattery`
  - Runs all 12 conditions on a dataset
  - Extracts features after degradation
  - Computes AUC under each condition
  - Returns degradation robustness report
- [ ] Implement RAR (Robustness-Accuracy Ratio) metric
  - `RAR = mean_degradation_auc / clean_auc`

### 5.3 Pre-computed Degraded Features (Optional)
- [ ] Create pre-degraded feature caches for FakeAVCeleb test set
  - Upload as HuggingFace datasets
  - Fast evaluation path (no ffmpeg needed)

---

## Milestone 6: Evaluation Orchestrator & CLI (Week 6)

### 6.1 Benchmark Function (`av_robustbench/evaluate/benchmark.py`)
- [ ] Implement `benchmark(model, dataset, attacks, certify, degrade, output_dir)`
  - Orchestrates: clean eval → attacks → certification → degradation
  - Returns `RobustnessCard`
  - Saves all intermediate results as JSON
  - Progress tracking with tqdm
- [ ] Support resume from partial results (if evaluation is interrupted)
- [ ] Support multi-GPU evaluation (DataParallel for certification)

### 6.2 Robustness Card (`av_robustbench/evaluate/robustness_card.py`)
- [ ] Implement robustness card generation
  - Input: all evaluation results (clean, attacks, certification, degradation)
  - Output: structured JSON
  - Fields:
    ```json
    {
      "model": {...},
      "clean_evaluation": {"auc": ..., "accuracy": ..., "eer": ...},
      "adversarial_evaluation": {"pgd_linf": {...}, ...},
      "certified_evaluation": {"sigma_0.25": {...}, "sigma_1.00": {...}},
      "degradation_evaluation": {"d1_jpeg75": {...}, ...},
      "cross_dataset": {"lavdf": {...}},
      "overall_score": ...
    }
    ```
  - Generate Markdown summary
  - Generate LaTeX table for paper
  - Generate PDF report (optional, using matplotlib + reportlab)

### 6.3 CLI (`av_robustbench/cli.py`)
- [ ] Implement `evaluate` command
  - `--model`: model name from zoo or path to checkpoint
  - `--dataset`: dataset name
  - `--attacks`: list of attack names
  - `--certify`: enable certification (with --sigma)
  - `--degrade`: enable degradation battery
  - `--output`: output directory
  - `--max-samples`: limit samples for speed
- [ ] Implement `certify` command
  - Focused certification only
- [ ] Implement `attack` command
  - Focused attack evaluation only
- [ ] Implement `card` command
  - Generate robustness card from results
- [ ] Implement `submit` command
  - Format results for leaderboard
- [ ] Implement `list-models` command
  - Print available models
- [ ] Use `click` or `typer` for CLI framework

---

## Milestone 7: Datasets (Week 6-7)

### 7.1 Dataset Abstraction (`av_robustbench/datasets/base.py`)
- [ ] Implement `AVDataset` abstract class
  - `__getitem__` returns `{"visual": ..., "audio": ..., "label": ..., "clip_id": ...}`
  - `split` property
  - `n_samples` property
  - `class_distribution` property
- [ ] Implement `FakeAVCelebDataset`
  - Loads from feature cache (fast) or raw videos (slow)
  - Train/val/test splits with deterministic seeding
  - Port manifest building from `cmar/utils/manifest.py`
- [ ] Implement `LAVDFDataset`
  - Loads from feature cache or raw videos
  - Test split only (for cross-dataset evaluation)
  - Port manifest building from `cmar/utils/manifest.py`
- [ ] Implement `FeatureCacheDataset` (generic)
  - Loads pre-extracted features from any cache directory
  - Compatible with CMAR cache format

---

## Milestone 8: Metrics & Visualization (Week 7)

### 8.1 Metrics (`av_robustbench/metrics/`)
- [ ] Implement `binary_metrics(labels, scores)` → AUC, EER, AP, balanced accuracy
  - Port from `cmar/evaluation/metrics.py`
- [ ] Implement certification metrics
  - `certified_accuracy(results, r)` — fraction certified at radius ≥ r
  - `mean_certified_radius(results)` — average radius of non-abstaining samples
  - `abstention_rate(results)` — fraction of abstaining samples
- [ ] Implement RAR metric
  - `robustness_accuracy_ratio(clean_acc, degradation_accs, cert_radius)`

### 8.2 Visualization (`av_robustbench/utils/visualization.py`)
- [ ] Certified accuracy curve plotter
  - Multiple sigmas with confidence bands
  - Paper-quality figures (matplotlib, publication fonts)
- [ ] Attack comparison bar chart
  - Base vs smoothed under different ε values
- [ ] Accuracy-radius tradeoff scatter plot
- [ ] Degradation heatmap
- [ ] Robustness card visual summary
- [ ] All plots exportable as PDF + PNG

---

## Milestone 9: Leaderboard (Week 8)

### 9.1 Leaderboard Data Format
- [ ] Define leaderboard JSON schema
  ```json
  {
    "models": [
      {
        "name": "CertAV (σ=1.00)",
        "paper_url": "...",
        "clean_auc": 0.985,
        "clean_accuracy": 0.925,
        "certified_radius_mean": 2.215,
        "cert_acc_r025": 0.910,
        "cert_acc_r100": 0.880,
        "robust_acc_eps010": 0.920,
        "robust_acc_eps020": 0.915,
        "degradation_avg_auc": 0.950,
        "cross_dataset_acc": 0.82,
        "overall_score": 0.91,
        "submission_date": "2026-10-01"
      }
    ]
  }
  ```
- [ ] Implement leaderboard update script
- [ ] Implement submission validation

### 9.2 Leaderboard Website
- [ ] Create GitHub Pages site
  - Sortable leaderboard table
  - Per-model detail pages
  - Submission instructions
  - Interactive certified accuracy curve viewer
- [ ] Or: HuggingFace Spaces app (Gradio/Streamlit)
  - Upload model → get robustness card
  - Compare models interactively

---

## Milestone 10: Documentation & Testing (Week 8-9)

### 10.1 Documentation
- [ ] API reference (auto-generated from docstrings)
- [ ] Tutorial: "Evaluate your AV detector in 5 minutes"
- [ ] Tutorial: "Add certification to your detector"
- [ ] Tutorial: "Submit to the leaderboard"
- [ ] Tutorial: "Implement a custom attack"
- [ ] Architecture overview with diagrams

### 10.2 Testing
- [ ] Unit tests for all modules (target >90% coverage)
- [ ] Integration tests: full pipeline from model loading → evaluation → robustness card
- [ ] Regression tests: ensure leaderboard numbers don't change
- [ ] CI/CD: GitHub Actions for test + lint + type check on every PR

### 10.3 Release
- [ ] Publish to PyPI (test first on TestPyPI)
- [ ] Create GitHub release with changelog
- [ ] Announce on Twitter/X, Reddit, Hugging Face

---

## Backlog (v2.0+)

- [ ] Add DFDC dataset support
- [ ] Add Celeb-DF dataset support
- [ ] Add DeepfakeBench model adapters
- [ ] Add temporal attacks (attack specific frames)
- [ ] Add audio-only attack pipeline
- [ ] Support video-level (not clip-level) evaluation
- [ ] Add DP-based certification (PixelDP alternative)
- [ ] Multi-GPU distributed certification
- [ ] Web API for remote evaluation
- [ ] Integration with Weights & Biases for experiment tracking

---

## Current Sprint

> **Sprint 0** (Pre-implementation): Documentation & planning ✅

### Completed Sprint: Sprint 1+ Integrated Package Build
Focus: publishable package skeleton, certification, attacks, metrics, datasets, CLI, degradation battery, cards, leaderboard format.
- [x] Set up repository and package structure
- [x] Implement core abstractions (AVDetector, CertificationResult, AttackResult, RobustnessCard)
- [x] Port and generalize certification engine from CertAV
- [x] Implement attack suite (PGD-Linf, PGD-L2, input-space PGD, Square, AutoAttack-style, transfer)
- [x] Implement dataset/cache loaders, binary metrics, certification metrics, RAR metrics
- [x] Implement degradation specs, file-level degradation helpers, and feature-cache degradation evaluation
- [x] Implement CLI commands: evaluate, certify, attack, card, submit, list-models
- [x] Implement model registry and CMAR/CertAV adapters
- [x] Write initial tests
- [x] Create runnable example: load CMAR-compatible checkpoint → benchmark → robustness card

### Discovered Mid-Process
*(Items discovered during implementation — add here as they come up)*
