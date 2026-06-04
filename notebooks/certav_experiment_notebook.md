# CertAV — Complete Experiment Notebook

> **How to use:** Copy each fenced code block below into a **separate Kaggle notebook cell** (top to bottom). Change only the `SEED` value in Cell 1 for parallel runs.

> **Run 5 parallel notebooks** with seeds: `2026`, `42`, `123`, `7`, `2025`. After each finishes, save `/kaggle/working/certav_seed_{SEED}/` as a Kaggle dataset.

---

## Cell 1 — Configuration (Python)

```python
# ============================================================
#  CertAV Experiment — CHANGE SEED HERE FOR PARALLEL RUNS
# ============================================================
SEED = 2026  # Use: 2026, 42, 123, 7, 2025

# Paths
CODE_ROOT = "/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR"
CACHE_DIR = "/kaggle/input/datasets/vasuaashadesai/cmar-features-clean-v1/cmar_cache"
OUTPUT_ROOT = f"/kaggle/working/certav_seed_{SEED}"

import os
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Save seed info
with open(f"{OUTPUT_ROOT}/seed.txt", "w") as f:
    f.write(str(SEED))

print(f"SEED = {SEED}")
print(f"CODE_ROOT = {CODE_ROOT}")
print(f"CACHE_DIR = {CACHE_DIR}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
```

---

## Cell 2 — Train all σ models (joint noise) (Python)

```python
# Train 4 joint-noise models at σ ∈ {0.12, 0.25, 0.50, 1.00}
import subprocess, os

sigmas = [0.12, 0.25, 0.50, 1.00]

for sigma in sigmas:
    tag = f"sigma_{sigma:.2f}"
    out_dir = f"{OUTPUT_ROOT}/{tag}"
    print(f"\n{'='*60}")
    print(f"  TRAINING: σ={sigma}, noise_mode=joint, seed={SEED}")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/10_train_certav.py",
        "--sigma", str(sigma),
        "--noise-mode", "joint",
        "--cache-dir", CACHE_DIR,
        "--output-dir", out_dir,
        "--epochs", "30",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--patience", "7",
        "--seed", str(SEED),
    ], check=True)
    print(f"\n✓ {tag} training complete → {out_dir}/best.pt")
```

---

## Cell 3 — Train ablation models (unimodal noise) (Python)

```python
# Train ablation models: visual_only and audio_only noise at σ ∈ {0.25, 1.00}
import subprocess

ablation_configs = [
    (0.25, "visual_only"),
    (0.25, "audio_only"),
    (1.00, "visual_only"),
    (1.00, "audio_only"),
]

for sigma, noise_mode in ablation_configs:
    tag = f"ablation_{noise_mode}_{sigma:.2f}"
    out_dir = f"{OUTPUT_ROOT}/{tag}"
    print(f"\n{'='*60}")
    print(f"  TRAINING ABLATION: σ={sigma}, noise_mode={noise_mode}, seed={SEED}")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/10_train_certav.py",
        "--sigma", str(sigma),
        "--noise-mode", noise_mode,
        "--cache-dir", CACHE_DIR,
        "--output-dir", out_dir,
        "--epochs", "30",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--patience", "7",
        "--seed", str(SEED),
    ], check=True)
    print(f"\n✓ {tag} training complete → {out_dir}/best.pt")
```

---

## Cell 4 — Certify all joint models (Python)

```python
# Certify all 4 joint-noise models with matching σ
import subprocess

sigmas = [0.12, 0.25, 0.50, 1.00]

for sigma in sigmas:
    tag = f"sigma_{sigma:.2f}"
    ckpt = f"{OUTPUT_ROOT}/{tag}/best.pt"
    out_json = f"{OUTPUT_ROOT}/cert_joint_{sigma:.2f}.json"
    print(f"\n{'='*60}")
    print(f"  CERTIFYING: σ={sigma}, mode=joint")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/11_certify.py",
        "--checkpoint", ckpt,
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
    print(f"\n✓ cert_joint_{sigma:.2f} → {out_json}")
```

---

## Cell 5 — Certify all ablation models (Python)

```python
# Certify all 4 ablation models with matching σ and noise mode
import subprocess

ablation_configs = [
    (0.25, "visual_only"),
    (0.25, "audio_only"),
    (1.00, "visual_only"),
    (1.00, "audio_only"),
]

for sigma, noise_mode in ablation_configs:
    tag = f"ablation_{noise_mode}_{sigma:.2f}"
    ckpt = f"{OUTPUT_ROOT}/{tag}/best.pt"
    out_json = f"{OUTPUT_ROOT}/cert_{noise_mode}_{sigma:.2f}.json"
    print(f"\n{'='*60}")
    print(f"  CERTIFYING ABLATION: σ={sigma}, mode={noise_mode}")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/11_certify.py",
        "--checkpoint", ckpt,
        "--sigma", str(sigma),
        "--noise-mode", noise_mode,
        "--cache-dir", CACHE_DIR,
        "--output", out_json,
        "--n0", "100",
        "--n", "1000",
        "--alpha", "0.001",
        "--batch-size", "64",
        "--seed", str(SEED),
    ], check=True)
    print(f"\n✓ cert_{noise_mode}_{sigma:.2f} → {out_json}")
```

---

## Cell 6 — Empirical PGD attack comparison (Python)

```python
# Run empirical PGD attacks on σ ∈ {0.25, 0.50, 1.00} with MATCHING σ smoothing
import subprocess

attack_sigmas = [0.25, 0.50, 1.00]

for sigma in attack_sigmas:
    tag = f"sigma_{sigma:.2f}"
    ckpt = f"{OUTPUT_ROOT}/{tag}/best.pt"
    out_json = f"{OUTPUT_ROOT}/empirical_attack_{sigma:.2f}.json"
    print(f"\n{'='*60}")
    print(f"  EMPIRICAL ATTACK: σ={sigma} (matching training and smoothing)")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", f"{CODE_ROOT}/scripts/12_empirical_attack_comparison.py",
        "--checkpoint", ckpt,
        "--sigma", str(sigma),
        "--noise-mode", "joint",
        "--cache-dir", CACHE_DIR,
        "--output", out_json,
        "--eps-values", "0.05", "0.10", "0.20",
        "--max-samples", "200",
        "--n-smoothing-samples", "100",
        "--seed", str(SEED),
    ], check=True)
    print(f"\n✓ empirical_attack_{sigma:.2f} → {out_json}")
```

---

## Cell 7 — Degradation robustness under certification (Python)

```python
# Certify on degraded test conditions for σ ∈ {0.25, 0.50, 1.00}
import subprocess

deg_sigmas = [0.25, 0.50, 1.00]
conditions = ["d12_social", "d11_h264_crf28", "d1_jpeg75"]

for sigma in deg_sigmas:
    tag = f"sigma_{sigma:.2f}"
    ckpt = f"{OUTPUT_ROOT}/{tag}/best.pt"
    for cond in conditions:
        out_json = f"{OUTPUT_ROOT}/cert_degraded_{sigma:.2f}_{cond}.json"
        print(f"\n--- Certifying: σ={sigma}, condition={cond} ---")
        subprocess.run([
            "python", f"{CODE_ROOT}/scripts/11_certify.py",
            "--checkpoint", ckpt,
            "--sigma", str(sigma),
            "--noise-mode", "joint",
            "--cache-dir", CACHE_DIR,
            "--condition", cond,
            "--output", out_json,
            "--n0", "100",
            "--n", "1000",
            "--alpha", "0.001",
            "--batch-size", "64",
            "--seed", str(SEED),
        ], check=True)
        print(f"  ✓ {out_json}")

print("\n✓ All degradation certifications complete")
```

---

## Cell 8 — Generate figures for this seed (Python)

```python
# Generate figures from this seed's results
# We create a compatible directory structure for the figure script
import subprocess, json, os, shutil

fig_dir = f"{OUTPUT_ROOT}/figures"
os.makedirs(fig_dir, exist_ok=True)

# The figure script expects certain file naming. Let's create symlinks or copies
# with the expected names in the OUTPUT_ROOT
for sigma in [0.12, 0.25, 0.50, 1.00]:
    src = f"{OUTPUT_ROOT}/cert_joint_{sigma:.2f}.json"
    dst = f"{OUTPUT_ROOT}/certav_cert_{sigma:.2f}.json"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

# Copy empirical attack result (use σ=0.25 as the default for figure 4)
for sigma in [0.25, 0.50, 1.00]:
    src = f"{OUTPUT_ROOT}/empirical_attack_{sigma:.2f}.json"
    dst = f"{OUTPUT_ROOT}/empirical_comparison.json"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

subprocess.run([
    "python", f"{CODE_ROOT}/scripts/13_certav_figures.py",
    "--results-dir", OUTPUT_ROOT,
    "--output-dir", fig_dir,
], check=True)

print(f"\n✓ Figures saved to {fig_dir}")
```

---

## Cell 9 — Generate summary JSON (Python)

```python
# Collect all results into a single summary JSON for this seed
import json, os, glob

summary = {
    "seed": SEED,
    "training": {},
    "certification_joint": {},
    "certification_ablation": {},
    "empirical_attacks": {},
    "degradation": {},
}

# Training results
for sigma in [0.12, 0.25, 0.50, 1.00]:
    best_path = f"{OUTPUT_ROOT}/sigma_{sigma:.2f}/best_metrics.json"
    if os.path.exists(best_path):
        with open(best_path) as f:
            summary["training"][f"joint_{sigma:.2f}"] = json.load(f)

for sigma in [0.25, 1.00]:
    for mode in ["visual_only", "audio_only"]:
        best_path = f"{OUTPUT_ROOT}/ablation_{mode}_{sigma:.2f}/best_metrics.json"
        if os.path.exists(best_path):
            with open(best_path) as f:
                summary["training"][f"{mode}_{sigma:.2f}"] = json.load(f)

# Certification (joint)
for sigma in [0.12, 0.25, 0.50, 1.00]:
    cert_path = f"{OUTPUT_ROOT}/cert_joint_{sigma:.2f}.json"
    if os.path.exists(cert_path):
        with open(cert_path) as f:
            data = json.load(f)
            summary["certification_joint"][f"{sigma:.2f}"] = {
                "summary": data.get("summary", {}),
                "certified_accuracy_at_radii": data.get("certified_accuracy_at_radii", {}),
            }

# Certification (ablation)
for sigma in [0.25, 1.00]:
    for mode in ["visual_only", "audio_only"]:
        cert_path = f"{OUTPUT_ROOT}/cert_{mode}_{sigma:.2f}.json"
        if os.path.exists(cert_path):
            with open(cert_path) as f:
                data = json.load(f)
                summary["certification_ablation"][f"{mode}_{sigma:.2f}"] = {
                    "summary": data.get("summary", {}),
                    "certified_accuracy_at_radii": data.get("certified_accuracy_at_radii", {}),
                }

# Empirical attacks
for sigma in [0.25, 0.50, 1.00]:
    atk_path = f"{OUTPUT_ROOT}/empirical_attack_{sigma:.2f}.json"
    if os.path.exists(atk_path):
        with open(atk_path) as f:
            summary["empirical_attacks"][f"{sigma:.2f}"] = json.load(f)

# Degradation
for sigma in [0.25, 0.50, 1.00]:
    for cond in ["d12_social", "d11_h264_crf28", "d1_jpeg75"]:
        deg_path = f"{OUTPUT_ROOT}/cert_degraded_{sigma:.2f}_{cond}.json"
        if os.path.exists(deg_path):
            with open(deg_path) as f:
                data = json.load(f)
                summary["degradation"][f"{sigma:.2f}_{cond}"] = {
                    "summary": data.get("summary", {}),
                    "certified_accuracy_at_radii": data.get("certified_accuracy_at_radii", {}),
                }

# Save
summary_path = f"{OUTPUT_ROOT}/seed_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved to {summary_path}")
print(f"\nTraining results: {len(summary['training'])} models")
print(f"Joint certifications: {len(summary['certification_joint'])} models")
print(f"Ablation certifications: {len(summary['certification_ablation'])} models")
print(f"Empirical attacks: {len(summary['empirical_attacks'])} configs")
print(f"Degradation certifications: {len(summary['degradation'])} conditions")

# Print key table
print(f"\n{'='*70}")
print(f"  SEED {SEED} — KEY RESULTS")
print(f"{'='*70}")
print(f"{'σ':>6} | {'Val AUC':>9} | {'Cert@0.00':>10} | {'Cert@0.25':>10} | {'Cert@0.50':>10} | {'Cert@1.00':>10} | {'Mean R':>8}")
print("-" * 70)
for sigma in [0.12, 0.25, 0.50, 1.00]:
    sk = f"joint_{sigma:.2f}"
    ck = f"{sigma:.2f}"
    val_auc = summary["training"].get(sk, {}).get("val_auc", "N/A")
    cert = summary["certification_joint"].get(ck, {}).get("certified_accuracy_at_radii", {})
    cert_s = summary["certification_joint"].get(ck, {}).get("summary", {})
    c00 = cert.get("r_0.00", "N/A")
    c25 = cert.get("r_0.25", "N/A")
    c50 = cert.get("r_0.50", "N/A")
    c100 = cert.get("r_1.00", "N/A")
    mr = cert_s.get("mean_certified_radius", "N/A")
    print(f"{sigma:>6.2f} | {val_auc:>9} | {c00:>10} | {c25:>10} | {c50:>10} | {c100:>10} | {mr:>8}")
```

---

## Cell 10 — Final listing and save confirmation (Python)

```python
# List all output files
import os

print(f"Output directory: {OUTPUT_ROOT}")
print(f"{'='*60}")
total_size = 0
for root, dirs, files in os.walk(OUTPUT_ROOT):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath)
        total_size += size
        rel = os.path.relpath(fpath, OUTPUT_ROOT)
        print(f"  {rel:<55} {size/1024:>8.1f} KB")

print(f"{'='*60}")
print(f"Total: {total_size/1024/1024:.1f} MB")
print(f"\n>>> Now save {OUTPUT_ROOT} as a Kaggle dataset <<<")
print(f">>> Dataset name suggestion: certav-seed-{SEED} <<<")
```
