# av-robustbench

`av-robustbench` is a Python package for standardized adversarial and certified robustness evaluation of audio-visual deepfake detectors.

The package is built around four reusable pieces:

- A small `AVDetector` adapter contract for arbitrary audio-visual detectors.
- Feature-space and input-space attack implementations with consistent threat-model metadata.
- A randomized-smoothing certification engine that wraps any `AVDetector`.
- Robustness cards and leaderboard-compatible JSON output.

## Installation

```bash
pip install av-robustbench
```

For optional model-zoo and degradation tooling:

```bash
pip install "av-robustbench[all]"
```

From this repository:

```bash
pip install -e av-robustbench
```

## Quick Start

```python
from av_robustbench.certification import SmoothedAVClassifier
from av_robustbench.models.adapters import TorchFeatureDetector

detector = TorchFeatureDetector(
    module=my_torch_model,
    name="my-av-detector",
    feature_dims={"visual": (16, 384), "audio": (64, 384)},
)

smoothed = SmoothedAVClassifier(detector, sigma=0.25, noise_mode="joint")
result = smoothed.certify(visual_features, audio_features, true_label=1)
print(result.certified_radius)
```

Run a single AutoAttack-style feature-space evaluation:

```python
from av_robustbench.attacks import AutoAttackAV, evaluate_under_attack

attack = AutoAttackAV(eps_value=0.10)
results = evaluate_under_attack(
    detector,
    dataset,
    [attack],
    max_samples=200,
    device="cuda",
)
print(results["autoattack_av"].adversarial_accuracy())
```

Evaluate a cached-feature model:

```bash
av-robustbench evaluate \
  --model certav_sigma100 \
  --checkpoint /path/to/best.pt \
  --dataset feature_cache \
  --cache-dir /path/to/cmar_cache \
  --attacks pgd_linf pgd_l2 \
  --certify \
  --sigma 0.25 1.00 \
  --output results/
```

Run a benchmark card with attacks, certification, and degradation metrics:

```bash
av-robustbench evaluate \
  --model certav_sigma100 \
  --checkpoint /path/to/best.pt \
  --dataset feature_cache \
  --cache-dir /path/to/cmar_cache \
  --attacks pgd_linf pgd_l2 autoattack_av \
  --certify \
  --sigma 0.25 1.00 \
  --degrade \
  --max-samples 300 \
  --output results/certav_sigma100_card
```

Generate Markdown and LaTeX card outputs from an existing JSON card:

```bash
av-robustbench card \
  --results results/certav_sigma100_card/robustness_card.json \
  --output results/certav_sigma100_card
```

## Design Notes

The package evaluates models; it does not train them. Raw preprocessing, detector training, and checkpoint hosting stay with the detector authors. `av-robustbench` standardizes the evaluation interface, attack budgets, certification outputs, degradation protocols, and leaderboard submission format.

When public checkpoints are available, register them with:

```python
from av_robustbench.models import register_model
from av_robustbench.models.adapters import CertAVAdapter

register_model(
    name="certav_sigma100",
    adapter_class=CertAVAdapter,
    metadata={
        "dataset": "fakeavceleb",
        "training_sigma": 1.0,
        "hf_repo_id": "your-org/certav-sigma100",
        "hf_filename": "best.pt",
    },
)
```

## Benchmark Components

### Attacks

- `PGDAttack`: feature-space Linf PGD for visual, audio, or joint attacks.
- `PGDAttackL2`: feature-space L2 PGD with joint projection across modalities.
- `SquareAttack`: score-based black-box attack over feature blocks.
- `AutoAttackAV`: deterministic ensemble of Linf PGD, L2 PGD, and Square Attack.

```python
from av_robustbench.attacks import PGDAttackL2

attack = PGDAttackL2(eps_value=1.0, n_steps=40, attack_target="both")
adv_visual, adv_audio = attack.attack(detector, visual, audio, labels)
```

### Certification

```python
from av_robustbench.certification import certify_multi_sigma

certification = certify_multi_sigma(
    detector,
    dataset,
    sigmas=[0.25, 0.50, 1.00],
    n0=100,
    n=1000,
    alpha=0.001,
)
```

### Degradations

The degradation battery includes 12 visual/audio conditions: JPEG, resizing,
visual noise, MP3/AAC codec roundtrips, audio SNR noise, H.264, and a social
media style chain.

```python
from av_robustbench.degradations import DegradationBattery

battery = DegradationBattery()
results = battery.run_feature_caches(detector, "/path/to/cmar_cache", max_samples=300)
```

### Leaderboard JSON

```bash
av-robustbench submit \
  --card results/certav_sigma100_card/robustness_card.json \
  --leaderboard leaderboard.json \
  --model-name certav_sigma100 \
  --paper-url https://example.org/paper \
  --code-url https://github.com/example/certav
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT.

## Publishing

See `docs/PUBLISHING.md` for the local verification and PyPI release checklist.
