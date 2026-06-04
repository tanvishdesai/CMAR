# av-robustbench — Planning Document

> **Status**: Planning Phase  
> **Author**: CertAV Research Team  
> **Created**: 2026-05-31  
> **Target**: NeurIPS 2027 Evaluations & Datasets Track / Standalone open-source tool

---

## 1. Vision

### 1.1 Problem Statement

The deepfake detection community currently lacks a standardized way to evaluate adversarial robustness. The situation:

- **Image classification** has RobustBench: a unified leaderboard with AutoAttack-based evaluation, a model zoo with 120+ pre-trained robust models, and one-line model loading. Published at NeurIPS 2021.
- **Image forensics** (detection + localization) has ForensicHub: 23 datasets, 42 baselines, unified YAML-driven pipelines, cross-domain evaluation. Published at NeurIPS 2025.
- **Audio-visual deepfake detection** has **nothing**. Every group reimplements PGD attacks ad-hoc, uses different ε budgets, different metrics, and there is no certified defense baseline to compare against.

### 1.2 What av-robustbench Is

A **Python package** that provides:
1. **Standardized adversarial evaluation** for AV deepfake detection models
2. **Certified robustness evaluation** (randomized smoothing for any AV model)
3. **Real-world degradation battery** (codec, compression, social media simulation)
4. **Model zoo** with pre-trained robust and non-robust detectors
5. **Public leaderboard** tracking both empirical and certified robustness

### 1.3 What av-robustbench Is NOT

- NOT a copy of RobustBench ported to deepfake detection
- NOT a copy of ForensicHub with an attack module bolted on
- NOT a collection of pre-extracted features
- NOT a paper-specific artifact — it's a community tool

---

## 2. Differentiation from Prior Work

### 2.1 vs. RobustBench (NeurIPS 2021)

| Aspect | RobustBench | av-robustbench |
|:---|:---|:---|
| **Domain** | Image classification (CIFAR-10, ImageNet) | AV deepfake detection |
| **Modalities** | Single (images) | Multi-modal (video frames + audio) |
| **Threat models** | L∞, L₂, common corruptions | L₂ feature-space, L∞ input-space, degradation |
| **Certification** | Not supported (empirical only) | **First-class citizen**: randomized smoothing engine wraps any model |
| **Attack protocol** | AutoAttack (fixed ensemble) | Extended: AutoAttack + AV-specific attacks (cross-modal transfer, temporal) |
| **Evaluation** | Robust accuracy on clean test set | Certified accuracy curves, RAR (Robustness-Accuracy Ratio), cross-dataset |
| **Models** | Image classifiers (ResNet, WideResNet) | AV detectors (CMAR, AVoiD-DF, LipForensics, AASIST, Xception) |

**Key novelty vs RobustBench**: 
- Certification as a first-class metric (not just empirical attacks)
- Multi-modal threat models (attack one modality, certify the other anchors)
- AV-specific degradation battery (codec transcoding, social media simulation)
- Cross-dataset evaluation protocol

### 2.2 vs. ForensicHub (NeurIPS 2025)

| Aspect | ForensicHub | av-robustbench |
|:---|:---|:---|
| **Focus** | Detection + localization (all-domain FIDL) | Adversarial robustness evaluation only |
| **Media types** | Static images only | Video + audio (temporal, multi-modal) |
| **Attacks** | None (clean evaluation only) | PGD, AutoAttack, certified defense, cross-modal |
| **Certification** | None | Randomized smoothing engine |
| **Degradation** | Image augmentation transforms | AV-specific: H.264 CRF, MP3 compression, social media chains |
| **Config** | YAML-driven codeless workflow | Programmatic Python API + CLI + optional YAML |
| **Output** | IFF-Protocol metrics | Robustness cards, leaderboard-compatible JSON |

**Key novelty vs ForensicHub**:
- Adversarial robustness (attacks + defenses) — ForensicHub has ZERO adversarial evaluation
- Video + audio support — ForensicHub is image-only
- Certified defense evaluation — completely new capability
- Robustness cards (analogous to model cards but for adversarial resilience)

### 2.3 Unique Contributions

1. **First adversarial robustness benchmark for multimodal deepfake detection**
2. **Certified robustness engine** that wraps ANY AV detector in a smoothed classifier
3. **Cross-modal threat model**: attack one modality, evaluate if the other modality provides sufficient anchor
4. **AV degradation battery**: standardized real-world corruption pipeline for video+audio
5. **Robustness card**: a structured report format (JSON/PDF) that any model can generate

---

## 3. Architecture

### 3.1 Package Structure

```
av_robustbench/
├── __init__.py                     # Version, public API surface
├── attacks/                        # Adversarial attack implementations
│   ├── __init__.py
│   ├── base.py                     # Abstract BaseAttack class
│   ├── pgd.py                      # PGD (L∞ and L₂) in feature space
│   ├── pgd_input.py                # PGD through frozen encoders (input space)
│   ├── autoattack_av.py            # AutoAttack adapted for AV binary detection
│   ├── square_attack.py            # Black-box Square Attack for AV
│   └── cross_modal.py              # Cross-modal transfer attacks
│
├── certification/                  # Certified defense evaluation
│   ├── __init__.py
│   ├── smoothing.py                # SmoothedAVClassifier (generic, wraps any model)
│   ├── core.py                     # Clopper-Pearson bounds, radius computation
│   └── curves.py                   # Certified accuracy curve utilities
│
├── degradations/                   # Real-world degradation battery
│   ├── __init__.py
│   ├── specs.py                    # Degradation specifications
│   ├── visual.py                   # JPEG, resize, H.264, noise
│   ├── audio.py                    # MP3, Opus, noise injection
│   ├── chains.py                   # Social media simulation chains
│   └── battery.py                  # Full degradation battery runner
│
├── models/                         # Model zoo & adapters
│   ├── __init__.py
│   ├── registry.py                 # Model registry & loading
│   ├── adapters/                   # Adapter interfaces for each detector
│   │   ├── base.py                 # Abstract AVDetector interface
│   │   ├── cmar.py                 # CMAR adapter
│   │   ├── certav.py               # CertAV adapter
│   │   ├── avoid_df.py             # AVoiD-DF adapter
│   │   ├── lipforensics.py         # LipForensics adapter
│   │   ├── aasist.py               # AASIST adapter (audio-only)
│   │   └── xception.py             # XceptionNet adapter (visual-only)
│   └── zoo/                        # Pre-trained checkpoint metadata
│       ├── certav_sigma025.json    # Checkpoint URLs, metrics
│       ├── certav_sigma100.json
│       └── ...
│
├── datasets/                       # Dataset loading & preprocessing
│   ├── __init__.py
│   ├── base.py                     # Abstract AVDataset interface
│   ├── fakeavceleb.py              # FakeAVCeleb loader
│   ├── lavdf.py                    # LAV-DF loader
│   ├── celebdf.py                  # Celeb-DF loader (future)
│   └── feature_cache.py            # Cached feature loading (for fast evaluation)
│
├── evaluate/                       # High-level evaluation orchestrator
│   ├── __init__.py
│   ├── benchmark.py                # Main benchmark() function
│   ├── robustness_card.py          # Robustness card generation
│   └── report.py                   # LaTeX/PDF report generation
│
├── metrics/                        # Evaluation metrics
│   ├── __init__.py
│   ├── binary.py                   # AUC, EER, AP, balanced accuracy
│   ├── certification.py            # Certified accuracy, mean radius, cert@r
│   └── rar.py                      # Robustness-Accuracy Ratio (RAR)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── io.py                       # JSON, tensor I/O
│   ├── seed.py                     # Reproducibility
│   └── visualization.py            # Plotting utilities
│
├── cli.py                          # CLI entry point (av-robustbench evaluate ...)
└── leaderboard/                    # Leaderboard data & submission tools
    ├── __init__.py
    ├── submit.py                   # Format results for leaderboard submission
    └── data/                       # Current leaderboard data (JSON)
        └── leaderboard.json
```

### 3.2 Core Abstractions

#### AVDetector (Abstract Base Class)
```python
class AVDetector(ABC):
    """Interface that any AV deepfake detection model must implement."""

    @abstractmethod
    def predict(self, visual: Tensor, audio: Tensor) -> dict:
        """Forward pass.
        
        Args:
            visual: (B, T_v, D_v) visual features OR (B, N, C, H, W) raw frames
            audio: (B, T_a, D_a) audio features OR (B, 1, T) raw waveform
        
        Returns:
            {"logits": Tensor(B,), "probs": Tensor(B,)}
        """
        ...
    
    @property
    @abstractmethod
    def input_type(self) -> str:
        """'features' or 'raw' — determines which attack pipeline to use."""
        ...
    
    @property
    @abstractmethod
    def feature_dims(self) -> dict:
        """{"visual": (T_v, D_v), "audio": (T_a, D_a)} for feature-space models."""
        ...
```

#### SmoothedAVClassifier (Generic Wrapper)
```python
class SmoothedAVClassifier:
    """Wraps ANY AVDetector in a randomized smoothing classifier.
    
    Usage:
        model = load_model("certav_sigma100")
        smoothed = SmoothedAVClassifier(model, sigma=1.0, noise_mode="joint")
        result = smoothed.certify(visual, audio, label=1)
        print(f"Certified radius: {result.certified_radius:.3f}")
    """
    
    def __init__(self, base: AVDetector, sigma: float, noise_mode: str = "joint"):
        ...
    
    def predict(self, visual, audio, n_samples=100) -> int:
        ...
    
    def certify(self, visual, audio, label, n0=100, n=1000, alpha=0.001) -> CertificationResult:
        ...
    
    def certify_dataset(self, dataset, **kwargs) -> list[CertificationResult]:
        ...
```

#### BaseAttack (Abstract Base Class)
```python
class BaseAttack(ABC):
    """Interface for adversarial attacks."""
    
    @abstractmethod
    def attack(self, model: AVDetector, visual: Tensor, audio: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        """Generate adversarial examples.
        
        Returns:
            (adversarial_visual, adversarial_audio)
        """
        ...
    
    @property
    @abstractmethod
    def threat_model(self) -> str:
        """'Linf', 'L2', 'feature_L2', etc."""
        ...
```

---

## 4. Tech Stack

### 4.1 Core Dependencies

| Package | Version | Purpose |
|:---|:---|:---|
| Python | ≥ 3.9 | Language |
| PyTorch | ≥ 2.0 | Deep learning framework |
| numpy | ≥ 1.24 | Numerical computation |
| scipy | ≥ 1.10 | Statistics (Clopper-Pearson bounds) |
| scikit-learn | ≥ 1.2 | Metrics (AUC, etc.) |
| pandas | ≥ 2.0 | Data loading |
| matplotlib | ≥ 3.7 | Plotting |
| tqdm | ≥ 4.0 | Progress bars |

### 4.2 Optional Dependencies (for specific features)

| Package | Purpose | Install Group |
|:---|:---|:---|
| transformers | HuggingFace model loading (DINOv2, Whisper) | `models` |
| timm | Additional visual backbones | `models` |
| ffmpeg-python | Video/audio degradation | `degradations` |
| opencv-python | Frame extraction | `degradations` |
| librosa / soundfile | Audio processing | `degradations` |
| huggingface_hub | Model downloading | `zoo` |

### 4.3 Installation

```bash
# Core (attacks + certification + metrics)
pip install av-robustbench

# With model zoo
pip install av-robustbench[models]

# With degradation battery
pip install av-robustbench[degradations]

# Everything
pip install av-robustbench[all]
```

### 4.4 Build & Distribution

| Tool | Purpose |
|:---|:---|
| `pyproject.toml` | Package metadata (PEP 621) |
| `setuptools` / `hatchling` | Build backend |
| `pytest` | Testing |
| `ruff` | Linting |
| `mypy` | Type checking |
| `mkdocs` / `sphinx` | Documentation |
| PyPI | Distribution |
| GitHub Actions | CI/CD |

---

## 5. Design Principles

### 5.1 Inspired by RobustBench
- **One-line model loading**: `model = load_model("certav_sigma100", dataset="fakeavceleb")`
- **Standardized evaluation**: `clean_acc, robust_acc = benchmark(model, attacks=["pgd", "autoattack"])`
- **Community contributions**: anyone can submit a model via GitHub issue
- **Model differentiability requirement**: only models with non-zero gradients accepted for standard eval

### 5.2 Inspired by ForensicHub
- **Dataset abstraction**: unified interface across FakeAVCeleb, LAV-DF, DFDC
- **Modular transforms**: degradation conditions are composable
- **Cross-domain evaluation**: train on FakeAVCeleb, test on LAV-DF, measure generalization

### 5.3 Novel to av-robustbench
- **Certification as first-class metric**: every model gets a certified accuracy curve, not just empirical attack accuracy
- **Multi-modal threat model awareness**: specify which modalities are under attack
- **Robustness cards**: auto-generated structured reports (JSON → LaTeX/PDF)
- **Feature-space AND input-space attacks**: unified pipeline that handles both

---

## 6. CLI Design

```bash
# One-command full evaluation
av-robustbench evaluate \
    --model certav_sigma100 \
    --dataset fakeavceleb \
    --attacks pgd autoattack \
    --certify --sigma 1.0 \
    --degradations all \
    --output results/

# Just certification
av-robustbench certify \
    --model certav_sigma100 \
    --dataset fakeavceleb \
    --sigma 0.25 0.50 1.00 \
    --output cert_results.json

# Just attacks
av-robustbench attack \
    --model cmar_baseline \
    --dataset fakeavceleb \
    --attacks pgd --eps 0.05 0.10 0.20 \
    --output attack_results.json

# Generate robustness card
av-robustbench card \
    --results results/ \
    --output robustness_card.pdf

# Submit to leaderboard
av-robustbench submit \
    --results results/ \
    --model-name "MyCoolDetector" \
    --paper-url "https://arxiv.org/..."
```

---

## 7. Leaderboard Design

### 7.1 Metrics Tracked

| Metric | Description |
|:---|:---|
| Clean AUC | Standard detection AUC without attacks |
| Clean Accuracy | Binary accuracy on clean test set |
| Robust Accuracy @ ε=0.1 | Accuracy under PGD L∞ ε=0.1 |
| Robust Accuracy @ ε=0.2 | Accuracy under PGD L∞ ε=0.2 |
| Certified Accuracy @ r=0.25 | % of samples certified at radius ≥ 0.25 |
| Certified Accuracy @ r=1.00 | % of samples certified at radius ≥ 1.00 |
| Mean Certified Radius | Average certified L₂ radius |
| RAR (Robustness-Accuracy Ratio) | Mean radius / (1 - accuracy) |
| Cross-Dataset Accuracy | Clean accuracy on LAV-DF (zero-shot) |
| Degradation AUC | Average AUC across 12 degradation conditions |

### 7.2 Leaderboard Categories

1. **Certified Robustness** (sorted by Mean Certified Radius)
2. **Empirical Robustness** (sorted by Robust Accuracy @ ε=0.2)
3. **Overall** (composite score: clean + certified + degradation)

### 7.3 Hosting

- **GitHub Pages** for the leaderboard website (like RobustBench)
- **HuggingFace Spaces** as alternative / interactive viewer
- **HuggingFace Hub** for model checkpoints
- JSON files in the repository for leaderboard data

---

## 8. Constraints & Decisions

### 8.1 Scope Constraints (v1.0)

- **Datasets**: FakeAVCeleb + LAV-DF only (v1.0). Celeb-DF, DFDC in v2.0.
- **Models**: 3 models in zoo (CertAV, CMAR baseline, one third-party). More via community.
- **Attacks**: PGD (L∞, L₂) + feature-space PGD. AutoAttack in v1.1.
- **No training**: The package evaluates models, it does NOT train them.

### 8.2 Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| Configuration | Python API-first, YAML optional | RobustBench is API-first; researchers prefer programmatic control |
| Model loading | HuggingFace Hub for checkpoints | Standard, community-familiar, free hosting |
| Attack default | PGD-L∞ ε=0.1 (feature space) | Matches existing CertAV evaluation protocol |
| Certification default | σ=0.25, n=1000, α=0.001 | Matches Cohen et al. (2019) standard settings |
| Degradation | ffmpeg-based | Reproducible, cross-platform, industry standard |
| Python version | ≥ 3.9 | Type hints (Union → |), dataclasses |

### 8.3 What We Explicitly Do NOT Build

- A training framework (use PyTorch Lightning / the model's own training code)
- A data preprocessing pipeline (use the model's own preprocessing)
- A general-purpose adversarial robustness library (that's ART/foolbox)
- A deepfake generation tool

---

## 9. Publication Strategy

### 9.1 Paper Structure (NeurIPS E&D Track)

1. **Abstract**: Gap (no AV robustness benchmark) → Solution (av-robustbench) → Claims
2. **Introduction**: Motivation + related work comparison table
3. **System Design**: Architecture, abstractions, CLI
4. **Evaluation Protocol**: Attack suite, certification engine, degradation battery
5. **Baseline Results**: CertAV vs CMAR vs PGD-AT vs (third-party model)
6. **Leaderboard**: Design, submission process, initial entries
7. **Community Impact**: How researchers would use this

### 9.2 Timeline

| Milestone | Target Date | Notes |
|:---|:---|:---|
| CertAV paper submission (ICASSP) | Sep 16, 2026 | Primary paper — finish this first |
| av-robustbench v0.1 alpha | Oct 2026 | Core package: attacks + certification |
| av-robustbench v0.5 beta | Dec 2026 | Model zoo + degradations + CLI |
| av-robustbench v1.0 release | Jan 2027 | Leaderboard + docs + paper draft |
| Paper submission (NeurIPS E&D) | May 2027 | Full paper with baseline results |

---

## 10. Dependencies on CertAV Paper

av-robustbench directly reuses and generalizes code from CertAV:

| CertAV Component | av-robustbench Generalization |
|:---|:---|
| `cmar/certification/smoothing.py` | → `av_robustbench/certification/smoothing.py` (generalized to any AVDetector) |
| `cmar/certification/core.py` | → `av_robustbench/certification/core.py` (unchanged math) |
| `scripts/12_empirical_attack_comparison.py` | → `av_robustbench/attacks/pgd.py` (generalized) |
| `cmar/evaluation/degradations.py` | → `av_robustbench/degradations/` (expanded) |
| `cmar/evaluation/metrics.py` | → `av_robustbench/metrics/binary.py` (reused) |
| `cmar/training/dataset.py` | → `av_robustbench/datasets/feature_cache.py` (generalized) |

---

## 11. Risk Assessment

| Risk | Impact | Mitigation |
|:---|:---|:---|
| No third-party model adapters | Weak model zoo | Start with CertAV + CMAR; reach out to AVoiD-DF authors |
| AutoAttack adaptation too complex | Missing key feature | Start with PGD-only; AutoAttack in v1.1 |
| ForensicHub releases AV support | Reduced novelty | Our certification engine is unique; focus on that |
| Insufficient community adoption | Low impact | Release alongside CertAV paper; write clear docs |
| ffmpeg dependency issues | Degradation battery breaks | Make degradations optional; provide pre-degraded test sets |
