# CertAV Phase 2 Final Kaggle Cells

This is the single start-to-finish Kaggle cell sequence for Phase 2.

It covers:

- Direction A: PCA geometry, anisotropic smoothing, attack-manifold alignment, and the existing-cache encoder baseline.
- Optional A+B bridge: empirical input-to-feature certificate composition.
- Direction C: conformal calibration and robust conformal evaluation.

The cells assume you attached these Kaggle datasets:

- `cmar-code`, mounted as `/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR`
- `cmar-features-clean-v1`, mounted as `/kaggle/input/datasets/vasuaashadesai/cmar-features-clean-v1/cmar_cache`
- seed checkpoint datasets for seeds `2026`, `42`, `69`, `420`, and `2804`

Run the cells in order. For a quick smoke test, set `SMOKE = True` in Cell 2.
For paper-quality results, keep `SMOKE = False`.

---

## Cell 1: Configure Fixed Kaggle Paths, Install Requirements, Define Helpers

```python
from pathlib import Path
import json
import os
import subprocess
import shutil
import sys
from typing import Iterable

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

CODE_ROOT = Path("/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR")
CACHE_DIR = Path("/kaggle/input/datasets/vasuaashadesai/cmar-features-clean-v1/cmar_cache")
SEED_2026_DIR = Path("/kaggle/input/datasets/vasuaashadesai/cvrta-2026-data/certav_seed_2026")

SEED_ROOTS = {
    2026: SEED_2026_DIR,
    42: Path("/kaggle/input/datasets/vasuaashadesai/cvrta-42-data/certav_seed_42"),
    69: Path("/kaggle/input/datasets/riyabaddiepatel/cvrta-69-data/certav_seed_69"),
    420: Path("/kaggle/input/datasets/riyabaddiepatel/cvrta-420-data/certav_seed_420"),
    2804: Path("/kaggle/input/datasets/shilpavdesai/cvrta-2804-data/certav_seed_2804"),
}

PROJECT_DIR = CODE_ROOT
os.chdir(PROJECT_DIR)
print("PROJECT_DIR =", PROJECT_DIR)
print("CACHE_DIR =", CACHE_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

PHASE2 = KAGGLE_WORKING / "phase2"
PHASE2.mkdir(parents=True, exist_ok=True)
print("PHASE2 =", PHASE2)

def run(args: Iterable[object], cwd: Path = PROJECT_DIR) -> None:
    cmd = [str(x) for x in args]
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)

def add_if_not_none(args: list[str], flag: str, value) -> list[str]:
    if value is not None:
        args.extend([flag, str(value)])
    return args

def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)
```

---

## Cell 2: Configure Phase 2 Run Size And Checkpoints

```python
# Set True only for debugging the notebook wiring. Keep False for real Phase 2 results.
SMOKE = False

if SMOKE:
    N0 = 20
    N = 100
    MAX_SAMPLES = 30
    EPOCHS = 2
    ATTACK_MAX_SAMPLES = 30
    CONFORMAL_N = 100
else:
    N0 = 100
    N = 1000
    MAX_SAMPLES = None
    EPOCHS = 30
    ATTACK_MAX_SAMPLES = 200
    CONFORMAL_N = 1000

SEEDS = [2026, 2804, 42, 420, 69]
SIGMAS = ["0.12", "0.25", "0.50", "1.00"]
PRIMARY_SEED = 2026
PRIMARY_SIGMA = "1.00"

BASE_CACHE = CACHE_DIR
if not (BASE_CACHE / "features").exists() or not (BASE_CACHE / "manifests").exists():
    raise FileNotFoundError(f"CMAR feature cache is incomplete or missing: {BASE_CACHE}")

def seed_checkpoint(seed: int, sigma: str) -> Path:
    sigma_dir = f"sigma_{sigma}"
    candidates = [
        SEED_ROOTS[seed] / "certav" / sigma_dir / "best.pt",
        SEED_ROOTS[seed] / sigma_dir / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing checkpoint for seed={seed}, sigma={sigma}: tried {candidates}")

print("BASE_CACHE =", BASE_CACHE)

CHECKPOINTS = {}
for seed in SEEDS:
    for sigma in SIGMAS:
        ckpt = seed_checkpoint(seed, sigma)
        CHECKPOINTS[(seed, sigma)] = ckpt
        print(f"checkpoint seed={seed} sigma={sigma}: {ckpt}")

SIGMA100_CKPT = CHECKPOINTS.get((PRIMARY_SEED, PRIMARY_SIGMA))
if SIGMA100_CKPT is None:
    raise FileNotFoundError("Could not find primary seed 2026 sigma_1.00 checkpoint.")

print("SIGMA100_CKPT =", SIGMA100_CKPT)
```

---

## Cell 3: Sanity Check Cache And Existing Checkpoints

```python
run([
    sys.executable, "scripts/10_train_certav.py",
    "--sigma", "1.00",
    "--cache-dir", BASE_CACHE,
    "--output-dir", PHASE2 / "cache_probe",
    "--cache-report-only",
])

print("\nFound checkpoints:")
for key, value in sorted(CHECKPOINTS.items()):
    print(key, "->", value)
```

---

## Cell 4: Certify Existing Isotropic Baselines

This gives you comparable isotropic CertAV results from the existing seed
checkpoints. These files are also useful for Direction C.

```python
BASELINES = PHASE2 / "baselines"
BASELINES.mkdir(parents=True, exist_ok=True)

for (seed, sigma), ckpt in sorted(CHECKPOINTS.items()):
    out = BASELINES / f"seed_{seed}_sigma_{sigma}_cert.json"
    if out.exists():
        print("skip existing", out)
        continue

    args = [
        sys.executable, "scripts/11_certify.py",
        "--checkpoint", ckpt,
        "--sigma", sigma,
        "--noise-mode", "joint",
        "--cache-dir", BASE_CACHE,
        "--output", out,
        "--n0", N0,
        "--n", N,
        "--alpha", "0.001",
        "--seed", seed,
    ]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", MAX_SAMPLES]
    run(args)
```

---

## Cell 5: Direction A+D Step 1 - Fit PCA On Existing Feature Cache

```python
ANISO = PHASE2 / "anisotropic"
ANISO.mkdir(parents=True, exist_ok=True)
PCA_JOINT = ANISO / "pca_joint.pt"

if not PCA_JOINT.exists():
    run([
        sys.executable, "scripts/20_fit_pca_noise.py",
        "--cache-dir", BASE_CACHE,
        "--feature-space", "joint",
        "--output", PCA_JOINT,
        "--summary-output", ANISO / "pca_joint.summary.json",
    ])
else:
    print("skip existing", PCA_JOINT)

print(read_json(ANISO / "pca_joint.summary.json"))
```

---

## Cell 6: Direction A+D Step 2 - Train And Certify Anisotropic Strategies

Strategies:

- `anisotropic_strat1`: eigenvalue-proportional PCA covariance
- `anisotropic_strat2`: top-k subspace projection
- `anisotropic_strat3`: inverse-eigenvalue covariance

```python
ANISO_STRATEGIES = ["anisotropic_strat1", "anisotropic_strat2", "anisotropic_strat3"]

for strat in ANISO_STRATEGIES:
    run_dir = ANISO / strat
    ckpt = run_dir / "best.pt"
    cert = run_dir / "certification.json"

    if not ckpt.exists():
        run([
            sys.executable, "scripts/10_train_certav.py",
            "--sigma", "1.00",
            "--noise-mode", strat,
            "--pca-noise-path", PCA_JOINT,
            "--cache-dir", BASE_CACHE,
            "--output-dir", run_dir,
            "--epochs", EPOCHS,
            "--batch-size", "8",
            "--grad-accum", "4",
            "--patience", "7",
            "--seed", PRIMARY_SEED,
        ])
    else:
        print("skip existing", ckpt)

    if not cert.exists():
        args = [
            sys.executable, "scripts/11_certify.py",
            "--checkpoint", ckpt,
            "--sigma", "1.00",
            "--noise-mode", strat,
            "--pca-noise-path", PCA_JOINT,
            "--cache-dir", BASE_CACHE,
            "--output", cert,
            "--n0", N0,
            "--n", N,
            "--alpha", "0.001",
            "--seed", PRIMARY_SEED,
        ]
        if MAX_SAMPLES is not None:
            args += ["--max-samples", MAX_SAMPLES]
        run(args)
    else:
        print("skip existing", cert)
```

---

## Cell 7: Direction A+D Step 3 - PGD Attack-Manifold Alignment Diagnostic

```python
ALIGNMENT_OUT = ANISO / "anisotropic_strat1" / "attack_alignment.json"

if not ALIGNMENT_OUT.exists():
    run([
        sys.executable, "scripts/12_empirical_attack_comparison.py",
        "--checkpoint", ANISO / "anisotropic_strat1" / "best.pt",
        "--sigma", "1.00",
        "--noise-mode", "anisotropic_strat1",
        "--pca-noise-path", PCA_JOINT,
        "--cache-dir", BASE_CACHE,
        "--output", ALIGNMENT_OUT,
        "--eps-values", "0.05", "0.10", "0.20",
        "--max-samples", ATTACK_MAX_SAMPLES,
        "--n-smoothing-samples", "100",
        "--seed", PRIMARY_SEED,
    ])
else:
    print("skip existing", ALIGNMENT_OUT)
```

---

## Cell 8: Optional A+B Bridge - Empirical Input-Space Composition

This is optional. It is useful if you want to pursue the A+B framing from the
reviews. It composes feature certificates with empirical input-to-feature
displacement estimates.

```python
	
```

---

## Cell 9: Direction D - Existing Cache No-Noise Scaling-Law Baseline

This establishes the no-noise point for your current DINOv2-Small +
Whisper-tiny encoder pair.

```python
ENCODER_STUDY = PHASE2 / "encoder_study"
BASE_PAIR = ENCODER_STUDY / "dinov2small_whispertiny_existing_cache"
BASE_PAIR.mkdir(parents=True, exist_ok=True)

base_no_noise_dir = BASE_PAIR / "baseline_no_noise"
base_no_noise_ckpt = base_no_noise_dir / "best.pt"
base_no_noise_cert = BASE_PAIR / "baseline_no_noise_cert.json"

if not (BASE_PAIR / "pca_joint.pt").exists():
    run([
        sys.executable, "scripts/20_fit_pca_noise.py",
        "--cache-dir", BASE_CACHE,
        "--feature-space", "joint",
        "--output", BASE_PAIR / "pca_joint.pt",
        "--summary-output", BASE_PAIR / "pca_joint.summary.json",
    ])

if not base_no_noise_ckpt.exists():
    run([
        sys.executable, "scripts/14_train_baseline_no_noise.py",
        "--cache-dir", BASE_CACHE,
        "--output-dir", base_no_noise_dir,
        "--epochs", EPOCHS,
        "--batch-size", "8",
        "--grad-accum", "4",
        "--patience", "7",
        "--seed", PRIMARY_SEED,
    ])

if not base_no_noise_cert.exists():
    args = [
        sys.executable, "scripts/11_certify.py",
        "--checkpoint", base_no_noise_ckpt,
        "--sigma", "1.00",
        "--noise-mode", "joint",
        "--cache-dir", BASE_CACHE,
        "--output", base_no_noise_cert,
        "--n0", N0,
        "--n", N,
        "--alpha", "0.001",
        "--seed", PRIMARY_SEED,
    ]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", MAX_SAMPLES]
    run(args)
```

---

## Cell 10: Direction C - Calibrate Conformal CertAV

This calibrates on the validation split only. Do not calibrate on test.

```python
CONFORMAL = PHASE2 / "conformal"
CONFORMAL.mkdir(parents=True, exist_ok=True)

CONFORMAL_RUNS = [
    {
        "name": "isotropic_sigma100_seed2026",
        "checkpoint": SIGMA100_CKPT,
        "noise_mode": "joint",
        "pca_noise_path": None,
    }
]

aniso_s1_ckpt = ANISO / "anisotropic_strat1" / "best.pt"
if aniso_s1_ckpt.exists():
    CONFORMAL_RUNS.append({
        "name": "anisotropic_strat1_seed2026",
        "checkpoint": aniso_s1_ckpt,
        "noise_mode": "anisotropic_strat1",
        "pca_noise_path": PCA_JOINT,
    })

for cfg in CONFORMAL_RUNS:
    calibration_out = CONFORMAL / f"{cfg['name']}_calibration.json"
    if calibration_out.exists():
        print("skip existing", calibration_out)
        continue

    args = [
        sys.executable, "scripts/21_conformal_calibrate.py",
        "--checkpoint", cfg["checkpoint"],
        "--cache-dir", BASE_CACHE,
        "--sigma", "1.00",
        "--noise-mode", cfg["noise_mode"],
        "--output", calibration_out,
        "--alphas", "0.05", "0.10", "0.20",
        "--radii", "0.00", "0.25", "0.50", "1.00",
        "--score-types", "raw", "cp", "log",
        "--n", CONFORMAL_N,
        "--cp-alpha", "0.001",
        "--split", "val",
        "--seed", PRIMARY_SEED,
    ]
    if cfg["pca_noise_path"] is not None:
        args += ["--pca-noise-path", cfg["pca_noise_path"]]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", MAX_SAMPLES]
    run(args)
```

---

## Cell 11: Direction C - Evaluate Clean And Robust Conformal Coverage

This evaluates clean test coverage and coverage under feature-space PGD.

```python
for cfg in CONFORMAL_RUNS:
    calibration_out = CONFORMAL / f"{cfg['name']}_calibration.json"
    eval_out = CONFORMAL / f"{cfg['name']}_test_eval.json"
    if eval_out.exists():
        print("skip existing", eval_out)
        continue

    args = [
        sys.executable, "scripts/22_conformal_evaluate.py",
        "--checkpoint", cfg["checkpoint"],
        "--cache-dir", BASE_CACHE,
        "--calibration", calibration_out,
        "--sigma", "1.00",
        "--noise-mode", cfg["noise_mode"],
        "--output", eval_out,
        "--split", "test",
        "--condition", "clean",
        "--n", CONFORMAL_N,
        "--attack-eps-values", "0.25", "0.50", "1.00",
        "--attack-steps", "20",
        "--seed", PRIMARY_SEED,
    ]
    if cfg["pca_noise_path"] is not None:
        args += ["--pca-noise-path", cfg["pca_noise_path"]]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", MAX_SAMPLES]
    run(args)
```

---

## Cell 12: Aggregate Phase 2 Tables And Figures

```python
SUMMARY = PHASE2 / "summary"
run([
    sys.executable, "scripts/23_phase2_summarize.py",
    "--phase2-dir", PHASE2,
    "--output-dir", SUMMARY,
])

print("\nSummary files:")
for path in sorted(SUMMARY.glob("*")):
    print(path)
```

---

## Cell 13: Display Main Summary Tables

```python
import pandas as pd

for name in [
    "phase2_encoder_scaling.csv",
    "phase2_anisotropic.csv",
    "phase2_conformal.csv",
]:
    path = SUMMARY / name
    print("\n==", name, "==")
    if path.exists() and path.stat().st_size > 0:
        display(pd.read_csv(path))
    else:
        print("No rows yet:", path)
```

---

## Cell 14: Package Outputs For Kaggle Dataset Upload

```python
ARCHIVE_BASE = KAGGLE_WORKING / "certav_phase2_results"
if ARCHIVE_BASE.exists():
    shutil.rmtree(ARCHIVE_BASE)
shutil.copytree(PHASE2, ARCHIVE_BASE)
print("Upload this folder as a Kaggle dataset:")
print(ARCHIVE_BASE)
```
