# CertAV — Elevation Experiments Notebook (Priority 1 + 2)

> **How to use:** Copy each fenced code block into a **Kaggle notebook cell** (top to bottom).
>
> **Kaggle datasets to attach:**
> - `vasuaashadesai/cmar-code` — the CMAR codebase
> - `vasuaashadesai/cmar-features-clean-v1` — cached FakeAVCeleb features
> - `elin75/localized-audio-visual-deepfake-dataset-lav-df` — LAV-DF raw dataset
> - One of your seed runs (e.g., `cvrta-2026-data`) — for the σ=1.00 checkpoint
>
> **GPU**: T4 or P100 required. Total runtime: ~4-6 hours.

---

## Cell 1 — Configuration (Python)

```python
# ============================================================
#  CertAV Elevation Experiments — Configuration
# ============================================================
SEED = 2026

# Paths — UPDATE THESE to match your Kaggle dataset mounts
CODE_ROOT = "/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR"
CACHE_DIR = "/kaggle/input/datasets/vasuaashadesai/cmar-features-clean-v1/cmar_cache"
LAVDF_ROOT = "/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF"

# Path to existing CertAV σ=1.00 checkpoint (from your 5-seed runs)
# UPDATE THIS to your actual mount path
CERTAV_SIGMA100_CKPT = "/kaggle/input/cvrta-2026-data/certav_seed_2026/sigma_1.00/best.pt"
CERTAV_SIGMA025_CKPT = "/kaggle/input/cvrta-2026-data/certav_seed_2026/sigma_0.25/best.pt"
CERTAV_SIGMA050_CKPT = "/kaggle/input/cvrta-2026-data/certav_seed_2026/sigma_0.50/best.pt"

OUTPUT_ROOT = "/kaggle/working/elevation_experiments"

import os, sys
os.makedirs(OUTPUT_ROOT, exist_ok=True)
sys.path.insert(0, CODE_ROOT)

print(f"SEED = {SEED}")
print(f"CODE_ROOT = {CODE_ROOT}")
print(f"CACHE_DIR = {CACHE_DIR}")
print(f"LAVDF_ROOT = {LAVDF_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")

# Verify paths
for name, path in [
    ("Code", CODE_ROOT),
    ("Cache", CACHE_DIR),
    ("LAV-DF", LAVDF_ROOT),
    ("CertAV σ=1.00", CERTAV_SIGMA100_CKPT),
]:
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'} {name}: {path}")
```

---

## Cell 2 — Priority 1.1: Train Baseline (No Noise) (~30 min)

```python
import subprocess

baseline_dir = f"{OUTPUT_ROOT}/baseline_no_noise"
print("=" * 60)
print("  TRAINING BASELINE: σ=0 (no noise augmentation)")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/14_train_baseline_no_noise.py",
    "--cache-dir", CACHE_DIR,
    "--output-dir", baseline_dir,
    "--epochs", "30",
    "--batch-size", "8",
    "--grad-accum", "4",
    "--patience", "7",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ Baseline training complete → {baseline_dir}/best.pt")
```

---

## Cell 3 — Priority 1.1: Certify Baseline at σ=0.25 and σ=1.00 (~10 min)

```python
# Certify the baseline (no noise) model using smoothing
# This should show VERY poor certification (high abstention, tiny radii)
import subprocess

for sigma in [0.25, 1.00]:
    out_json = f"{OUTPUT_ROOT}/baseline_cert_{sigma:.2f}.json"
    print(f"\n--- Certifying baseline at σ={sigma} ---")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/11_certify.py",
        "--checkpoint", f"{baseline_dir}/best.pt",
        "--sigma", str(sigma),
        "--noise-mode", "joint",
        "--cache-dir", CACHE_DIR,
        "--output", out_json,
        "--n0", "100",
        "--n", "1000",
        "--alpha", "0.001",
        "--batch-size", "64",
        "--seed", str(SEED),
    ], check=True)
    print(f"  ✓ Baseline cert σ={sigma} → {out_json}")

print("\n✓ Baseline certification complete")
print("  Expected: high abstention rate and/or very small certified radii")
```

---

## Cell 4 — Priority 1.2: Train PGD-AT Baseline (~45 min)

```python
import subprocess

pgd_at_dir = f"{OUTPUT_ROOT}/baseline_pgd_at"
print("=" * 60)
print("  TRAINING PGD-AT BASELINE: ε=0.1, steps=7")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/15_train_pgd_at.py",
    "--cache-dir", CACHE_DIR,
    "--output-dir", pgd_at_dir,
    "--at-eps", "0.1",
    "--at-steps", "7",
    "--epochs", "30",
    "--batch-size", "8",
    "--grad-accum", "4",
    "--patience", "7",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ PGD-AT training complete → {pgd_at_dir}/best.pt")
```

---

## Cell 5 — Priority 1.2: Evaluate PGD-AT Under Attack (~20 min)

```python
# Evaluate the PGD-AT model: clean AUC + attacked AUC
# Compare with CertAV's empirical attack results
import subprocess

pgd_at_eval = f"{OUTPUT_ROOT}/pgd_at_empirical.json"
print("=" * 60)
print("  EVALUATING PGD-AT UNDER ATTACK")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/12_empirical_attack_comparison.py",
    "--checkpoint", f"{pgd_at_dir}/best.pt",
    "--sigma", "0.25",  # For smoothed comparison
    "--noise-mode", "joint",
    "--cache-dir", CACHE_DIR,
    "--output", pgd_at_eval,
    "--eps-values", "0.05", "0.10", "0.20",
    "--max-samples", "200",
    "--n-smoothing-samples", "100",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ PGD-AT evaluation complete → {pgd_at_eval}")
```

---

## Cell 6 — Priority 1.2: Certify PGD-AT at σ=1.00 (~5 min)

```python
# Try to certify the PGD-AT model — should produce small/zero radii
import subprocess

pgd_at_cert = f"{OUTPUT_ROOT}/pgd_at_cert_1.00.json"
print("--- Certifying PGD-AT at σ=1.00 ---")
subprocess.run([
    "python", f"{CODE_ROOT}/scripts/11_certify.py",
    "--checkpoint", f"{pgd_at_dir}/best.pt",
    "--sigma", "1.00",
    "--noise-mode", "joint",
    "--cache-dir", CACHE_DIR,
    "--output", pgd_at_cert,
    "--n0", "100",
    "--n", "1000",
    "--alpha", "0.001",
    "--batch-size", "64",
    "--max-samples", "200",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ PGD-AT certification complete → {pgd_at_cert}")
print("  Expected: very high abstention or very small radii")
print("  (AT trains for empirical robustness, not certified robustness)")
```

---

## Cell 7 — Priority 1.3: Preprocess LAV-DF Features (~1-2 hours)

```python
# Extract DINOv2 + Whisper features from LAV-DF test videos
# This creates a LAV-DF feature cache compatible with our certification scripts
import subprocess, os

lavdf_cache_dir = f"{OUTPUT_ROOT}/lavdf_cache"
os.makedirs(lavdf_cache_dir, exist_ok=True)

print("=" * 60)
print("  PREPROCESSING LAV-DF FEATURES")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/01_preprocess_features.py",
    "--config", f"{CODE_ROOT}/configs/preprocess_fakeavceleb.json",
    "--lavdf-root", LAVDF_ROOT,
    "--output-dir", lavdf_cache_dir,
    "--no-degraded",
    "--splits",  # empty = skip FakeAVCeleb splits (we just want LAV-DF)
    "--max-new-rows", "500",  # Process up to 500 LAV-DF samples
    "--max-runtime-seconds", "5400",  # 90 minutes max
], check=True)

# Check what was produced
lavdf_manifest = f"{lavdf_cache_dir}/manifests/lavdf_test.csv"
if os.path.exists(lavdf_manifest):
    import pandas as pd
    df = pd.read_csv(lavdf_manifest)
    print(f"\n✓ LAV-DF manifest: {len(df)} samples")
    print(f"  Real: {(df['label']==0).sum()}, Fake: {(df['label']==1).sum()}")
else:
    print(f"\n[WARN] LAV-DF manifest not found at {lavdf_manifest}")
    print("  You may need to adjust the preprocessing command.")
```

---

## Cell 8 — Priority 1.3: Certify on LAV-DF (Cross-Dataset) (~15 min)

```python
# Zero-shot certification: FakeAVCeleb-trained → LAV-DF test
import subprocess

for sigma, ckpt in [(0.25, CERTAV_SIGMA025_CKPT), (1.00, CERTAV_SIGMA100_CKPT)]:
    out_json = f"{OUTPUT_ROOT}/cert_lavdf_{sigma:.2f}.json"
    print(f"\n--- Cross-dataset certification: σ={sigma} on LAV-DF ---")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/16_certify_cross_dataset.py",
        "--checkpoint", ckpt,
        "--sigma", str(sigma),
        "--lavdf-cache-dir", lavdf_cache_dir,
        "--output", out_json,
        "--n0", "100",
        "--n", "1000",
        "--alpha", "0.001",
        "--batch-size", "64",
        "--seed", str(SEED),
    ], check=True)
    print(f"  ✓ LAV-DF cert σ={sigma} → {out_json}")

print("\n✓ Cross-dataset certification complete")
```

---

## Cell 9 — Priority 2.1: Input-Space PGD Attack Pilot (~1-2 hours)

```python
# Input-space PGD through feature space with controlled L2 budgets
# Tests whether certified radii are meaningful in practice
import subprocess

input_space_result = f"{OUTPUT_ROOT}/input_space_attack.json"
print("=" * 60)
print("  INPUT-SPACE PGD ATTACK PILOT")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/17_input_space_attack.py",
    "--checkpoint", CERTAV_SIGMA100_CKPT,
    "--sigma", "1.00",
    "--cache-dir", CACHE_DIR,
    "--output", input_space_result,
    "--eps-values", "0.002", "0.005", "0.01", "0.02",
    "--n-steps", "20",
    "--max-samples", "100",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ Input-space attack complete → {input_space_result}")
```

---

## Cell 10 — Priority 2.2: Manifold Analysis (~30 min)

```python
# Measure intrinsic dimensionality and noise-manifold alignment
import subprocess

manifold_result = f"{OUTPUT_ROOT}/manifold_analysis.json"
print("=" * 60)
print("  MANIFOLD ANALYSIS")
print("=" * 60)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/18_manifold_analysis.py",
    "--cache-dir", CACHE_DIR,
    "--checkpoint", CERTAV_SIGMA100_CKPT,
    "--output", manifold_result,
    "--max-samples", "1000",
    "--seed", str(SEED),
], check=True)

print(f"\n✓ Manifold analysis complete → {manifold_result}")
```

---

## Cell 11 — Aggregate All Results

```python
# Collect all elevation experiment results into a summary
import json, os

results = {}

# Baseline (no noise) certification
for sigma in [0.25, 1.00]:
    path = f"{OUTPUT_ROOT}/baseline_cert_{sigma:.2f}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            results[f"baseline_no_noise_cert_{sigma:.2f}"] = data.get("summary", {})
            results[f"baseline_no_noise_cert_{sigma:.2f}"]["cert_at_radii"] = data.get("certified_accuracy_at_radii", {})

# PGD-AT results
if os.path.exists(f"{OUTPUT_ROOT}/pgd_at_empirical.json"):
    with open(f"{OUTPUT_ROOT}/pgd_at_empirical.json") as f:
        results["pgd_at_empirical"] = json.load(f)

if os.path.exists(f"{OUTPUT_ROOT}/pgd_at_cert_1.00.json"):
    with open(f"{OUTPUT_ROOT}/pgd_at_cert_1.00.json") as f:
        data = json.load(f)
        results["pgd_at_cert_1.00"] = data.get("summary", {})
        results["pgd_at_cert_1.00"]["cert_at_radii"] = data.get("certified_accuracy_at_radii", {})

# LAV-DF cross-dataset
for sigma in [0.25, 1.00]:
    path = f"{OUTPUT_ROOT}/cert_lavdf_{sigma:.2f}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            results[f"lavdf_cert_{sigma:.2f}"] = data.get("summary", {})
            results[f"lavdf_cert_{sigma:.2f}"]["cert_at_radii"] = data.get("certified_accuracy_at_radii", {})

# Input-space attack
if os.path.exists(f"{OUTPUT_ROOT}/input_space_attack.json"):
    with open(f"{OUTPUT_ROOT}/input_space_attack.json") as f:
        data = json.load(f)
        results["input_space_attack"] = data.get("per_eps_summary", {})

# Manifold analysis
if os.path.exists(f"{OUTPUT_ROOT}/manifold_analysis.json"):
    with open(f"{OUTPUT_ROOT}/manifold_analysis.json") as f:
        data = json.load(f)
        results["manifold_analysis"] = {
            "visual_dim_95pct": data.get("intrinsic_dim_visual", {}).get("dim_at_95pct", "?"),
            "audio_dim_95pct": data.get("intrinsic_dim_audio", {}).get("dim_at_95pct", "?"),
            "joint_dim_95pct": data.get("intrinsic_dim_joint", {}).get("dim_at_95pct", "?"),
            "prediction_stability": data.get("prediction_stability", {}),
        }

# Save aggregate
with open(f"{OUTPUT_ROOT}/elevation_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("=" * 70)
print("  ELEVATION EXPERIMENTS — SUMMARY")
print("=" * 70)

# Print key comparisons
print("\n--- Baseline (No Noise) vs CertAV at σ=1.00 ---")
b = results.get("baseline_no_noise_cert_1.00", {})
print(f"  Baseline: accuracy={b.get('accuracy', '?')}, "
      f"abstain_rate={b.get('abstain_rate', '?')}, "
      f"mean_radius={b.get('mean_certified_radius', '?')}")
print(f"  CertAV:   accuracy=0.925, abstain_rate=0.008, mean_radius=2.215")

print("\n--- PGD-AT Certification at σ=1.00 ---")
p = results.get("pgd_at_cert_1.00", {})
print(f"  PGD-AT:  accuracy={p.get('accuracy', '?')}, "
      f"abstain_rate={p.get('abstain_rate', '?')}, "
      f"mean_radius={p.get('mean_certified_radius', '?')}")
print(f"  CertAV:  accuracy=0.925, abstain_rate=0.008, mean_radius=2.215")

print("\n--- Cross-Dataset (LAV-DF) ---")
for sigma in [0.25, 1.00]:
    l = results.get(f"lavdf_cert_{sigma:.2f}", {})
    print(f"  σ={sigma}: accuracy={l.get('accuracy', '?')}, "
          f"mean_radius={l.get('mean_certified_radius', '?')}")

print("\n--- Manifold Analysis ---")
m = results.get("manifold_analysis", {})
print(f"  Visual intrinsic dim (95%): {m.get('visual_dim_95pct', '?')} / 384")
print(f"  Audio intrinsic dim (95%):  {m.get('audio_dim_95pct', '?')} / 384")
print(f"  Joint intrinsic dim (95%):  {m.get('joint_dim_95pct', '?')} / 768")

print(f"\n✓ All results saved to {OUTPUT_ROOT}/elevation_summary.json")
```

---

## Cell 12 — Save and Package

```python
import os

print(f"Output directory: {OUTPUT_ROOT}")
print("=" * 60)
total_size = 0
for root, dirs, files in os.walk(OUTPUT_ROOT):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath)
        total_size += size
        rel = os.path.relpath(fpath, OUTPUT_ROOT)
        print(f"  {rel:<55} {size/1024:>8.1f} KB")

print("=" * 60)
print(f"Total: {total_size/1024/1024:.1f} MB")
print(f"\n>>> Save {OUTPUT_ROOT} as a Kaggle dataset <<<")
print(f">>> Dataset name suggestion: certav-elevation-results <<<")
```
