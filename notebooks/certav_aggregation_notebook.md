# CertAV — Aggregation Notebook

> **Prerequisites:** Upload the 5 seed-run datasets to Kaggle. Each should contain a `seed_summary.json` and the full certification JSONs.
>
> Update the `SEED_DATASET_PATHS` dict in Cell 1 with the actual Kaggle input paths for each seed's dataset.

---

## Cell 1 — Configuration (Python)

```python
# ============================================================
#  CertAV Aggregation — Point these to your 5 uploaded datasets
# ============================================================

# UPDATE THESE PATHS to where Kaggle mounts each seed's dataset
SEED_DATASET_PATHS = {
    2026: "/kaggle/input/certav-seed-2026/certav_seed_2026",
    42:   "/kaggle/input/certav-seed-42/certav_seed_42",
    123:  "/kaggle/input/certav-seed-123/certav_seed_123",
    7:    "/kaggle/input/certav-seed-7/certav_seed_7",
    2025: "/kaggle/input/certav-seed-2025/certav_seed_2025",
}

OUTPUT_DIR = "/kaggle/working/certav_aggregated"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

# Verify all datasets exist
for seed, path in SEED_DATASET_PATHS.items():
    summary_path = os.path.join(path, "seed_summary.json")
    if os.path.exists(summary_path):
        print(f"  ✓ Seed {seed}: {path}")
    else:
        print(f"  ✗ Seed {seed}: MISSING {summary_path}")

print(f"\nOutput → {OUTPUT_DIR}")
```

---

## Cell 2 — Load all seed summaries (Python)

```python
import json
import numpy as np

# Load all seed summaries
seed_data = {}
for seed, path in SEED_DATASET_PATHS.items():
    summary_path = os.path.join(path, "seed_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            seed_data[seed] = json.load(f)
        print(f"Loaded seed {seed}")
    else:
        print(f"[WARN] Missing seed {seed}")

seeds = sorted(seed_data.keys())
n_seeds = len(seeds)
print(f"\nLoaded {n_seeds} seed runs: {seeds}")
```

---

## Cell 3 — Aggregate training results (Python)

```python
# Aggregate training metrics across seeds
print("="*80)
print("  AGGREGATED TRAINING RESULTS")
print("="*80)

model_keys = ["joint_0.12", "joint_0.25", "joint_0.50", "joint_1.00",
              "visual_only_0.25", "audio_only_0.25",
              "visual_only_1.00", "audio_only_1.00"]

training_agg = {}
print(f"\n{'Model':<22} | {'Val AUC':>18} | {'Val EER':>18}")
print("-" * 62)

for mk in model_keys:
    aucs = []
    eers = []
    for seed in seeds:
        t = seed_data[seed].get("training", {}).get(mk, {})
        if "val_auc" in t:
            aucs.append(t["val_auc"])
        if "val_eer" in t:
            eers.append(t["val_eer"])

    if aucs:
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)
        eer_mean, eer_std = np.mean(eers), np.std(eers)
        training_agg[mk] = {
            "val_auc_mean": float(auc_mean), "val_auc_std": float(auc_std),
            "val_eer_mean": float(eer_mean), "val_eer_std": float(eer_std),
            "n_seeds": len(aucs),
        }
        print(f"{mk:<22} | {auc_mean:.4f} ± {auc_std:.4f}  | {eer_mean:.4f} ± {eer_std:.4f}")
    else:
        print(f"{mk:<22} | {'N/A':>18} | {'N/A':>18}")

# Save
with open(f"{OUTPUT_DIR}/training_aggregated.json", "w") as f:
    json.dump(training_agg, f, indent=2)
```

---

## Cell 4 — Aggregate certification results (joint) (Python)

```python
# Aggregate joint certification across seeds
print("="*80)
print("  AGGREGATED CERTIFICATION — JOINT NOISE")
print("="*80)

sigmas = ["0.12", "0.25", "0.50", "1.00"]
radii_keys = ["r_0.00", "r_0.10", "r_0.25", "r_0.50", "r_0.75", "r_1.00", "r_1.50"]

cert_joint_agg = {}

print(f"\n{'σ':>6} | {'Accuracy':>16} | {'Abstain%':>14} | {'Mean Radius':>16}")
print("-" * 60)

for sk in sigmas:
    accs, abstains, mean_radii = [], [], []
    cert_at_radii = {rk: [] for rk in radii_keys}

    for seed in seeds:
        c = seed_data[seed].get("certification_joint", {}).get(sk, {})
        s = c.get("summary", {})
        car = c.get("certified_accuracy_at_radii", {})

        if s:
            accs.append(s.get("accuracy", 0))
            abstains.append(s.get("abstain_rate", 0))
            mean_radii.append(s.get("mean_certified_radius", 0))
            for rk in radii_keys:
                cert_at_radii[rk].append(car.get(rk, 0))

    if accs:
        cert_joint_agg[sk] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "abstain_rate_mean": float(np.mean(abstains)),
            "abstain_rate_std": float(np.std(abstains)),
            "mean_radius_mean": float(np.mean(mean_radii)),
            "mean_radius_std": float(np.std(mean_radii)),
            "n_seeds": len(accs),
        }
        for rk in radii_keys:
            vals = cert_at_radii[rk]
            cert_joint_agg[sk][f"{rk}_mean"] = float(np.mean(vals))
            cert_joint_agg[sk][f"{rk}_std"] = float(np.std(vals))

        print(f"{sk:>6} | {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"{np.mean(abstains):.4f} ± {np.std(abstains):.4f} | "
              f"{np.mean(mean_radii):.4f} ± {np.std(mean_radii):.4f}")

# Print certified accuracy table
print(f"\n{'σ':>6}", end="")
for rk in radii_keys:
    print(f" | {rk:>14}", end="")
print()
print("-" * (8 + 17 * len(radii_keys)))
for sk in sigmas:
    agg = cert_joint_agg.get(sk, {})
    print(f"{sk:>6}", end="")
    for rk in radii_keys:
        m = agg.get(f"{rk}_mean", 0)
        s = agg.get(f"{rk}_std", 0)
        print(f" | {m:.3f} ± {s:.3f}", end="")
    print()

with open(f"{OUTPUT_DIR}/certification_joint_aggregated.json", "w") as f:
    json.dump(cert_joint_agg, f, indent=2)
```

---

## Cell 5 — Aggregate ablation certification results (Python)

```python
# Aggregate ablation (unimodal noise) certification
print("="*80)
print("  AGGREGATED CERTIFICATION — ABLATION (UNIMODAL vs JOINT)")
print("="*80)

ablation_keys = ["visual_only_0.25", "audio_only_0.25",
                 "visual_only_1.00", "audio_only_1.00"]

cert_ablation_agg = {}

print(f"\n{'Config':<22} | {'Accuracy':>16} | {'Mean Radius':>16} | {'Cert@0.25':>14} | {'Cert@0.50':>14}")
print("-" * 90)

for ak in ablation_keys:
    accs, mean_radii, c25s, c50s = [], [], [], []
    for seed in seeds:
        c = seed_data[seed].get("certification_ablation", {}).get(ak, {})
        s = c.get("summary", {})
        car = c.get("certified_accuracy_at_radii", {})
        if s:
            accs.append(s.get("accuracy", 0))
            mean_radii.append(s.get("mean_certified_radius", 0))
            c25s.append(car.get("r_0.25", 0))
            c50s.append(car.get("r_0.50", 0))

    if accs:
        cert_ablation_agg[ak] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "mean_radius_mean": float(np.mean(mean_radii)), "mean_radius_std": float(np.std(mean_radii)),
            "cert_0.25_mean": float(np.mean(c25s)), "cert_0.25_std": float(np.std(c25s)),
            "cert_0.50_mean": float(np.mean(c50s)), "cert_0.50_std": float(np.std(c50s)),
            "n_seeds": len(accs),
        }
        print(f"{ak:<22} | {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"{np.mean(mean_radii):.4f} ± {np.std(mean_radii):.4f} | "
              f"{np.mean(c25s):.4f} ± {np.std(c25s):.4f} | "
              f"{np.mean(c50s):.4f} ± {np.std(c50s):.4f}")

# Compare joint vs unimodal at same σ
print("\n--- Joint vs Unimodal Comparison ---")
for sigma in ["0.25", "1.00"]:
    j = cert_joint_agg.get(sigma, {})
    v = cert_ablation_agg.get(f"visual_only_{sigma}", {})
    a = cert_ablation_agg.get(f"audio_only_{sigma}", {})
    print(f"\n  σ = {sigma}:")
    print(f"    Joint:       acc = {j.get('accuracy_mean',0):.4f} ± {j.get('accuracy_std',0):.4f}  "
          f"radius = {j.get('mean_radius_mean',0):.4f} ± {j.get('mean_radius_std',0):.4f}")
    print(f"    Visual-only: acc = {v.get('accuracy_mean',0):.4f} ± {v.get('accuracy_std',0):.4f}  "
          f"radius = {v.get('mean_radius_mean',0):.4f} ± {v.get('mean_radius_std',0):.4f}")
    print(f"    Audio-only:  acc = {a.get('accuracy_mean',0):.4f} ± {a.get('accuracy_std',0):.4f}  "
          f"radius = {a.get('mean_radius_mean',0):.4f} ± {a.get('mean_radius_std',0):.4f}")

with open(f"{OUTPUT_DIR}/certification_ablation_aggregated.json", "w") as f:
    json.dump(cert_ablation_agg, f, indent=2)
```

---

## Cell 6 — Aggregate empirical attack results (Python)

```python
# Aggregate empirical PGD attack results
print("="*80)
print("  AGGREGATED EMPIRICAL ATTACK RESULTS")
print("="*80)

attack_sigmas = ["0.25", "0.50", "1.00"]
eps_values = ["eps_0.05", "eps_0.10", "eps_0.20"]

attack_agg = {}

for sigma in attack_sigmas:
    attack_agg[sigma] = {}
    print(f"\n  σ = {sigma}:")
    print(f"    {'ε':>8} | {'Base AUC (adv)':>20} | {'Smoothed Acc (adv)':>22}")
    print("    " + "-" * 55)

    for eps_key in eps_values:
        base_aucs, smooth_accs = [], []
        for seed in seeds:
            atk = seed_data[seed].get("empirical_attacks", {}).get(sigma, {})
            atk_data = atk.get("attacks", {}).get(eps_key, {})
            base_r = atk_data.get("base_classifier", {})
            smooth_r = atk_data.get("smoothed_classifier", {})
            if base_r:
                base_aucs.append(base_r.get("adversarial", {}).get("auc", 0))
            if smooth_r:
                smooth_accs.append(smooth_r.get("adversarial_accuracy", 0))

        if base_aucs:
            attack_agg[sigma][eps_key] = {
                "base_auc_mean": float(np.mean(base_aucs)),
                "base_auc_std": float(np.std(base_aucs)),
                "smooth_acc_mean": float(np.mean(smooth_accs)),
                "smooth_acc_std": float(np.std(smooth_accs)),
                "n_seeds": len(base_aucs),
            }
            print(f"    {eps_key:>8} | {np.mean(base_aucs):.4f} ± {np.std(base_aucs):.4f}      | "
                  f"{np.mean(smooth_accs):.4f} ± {np.std(smooth_accs):.4f}")

with open(f"{OUTPUT_DIR}/empirical_attacks_aggregated.json", "w") as f:
    json.dump(attack_agg, f, indent=2)
```

---

## Cell 7 — Aggregate degradation results (Python)

```python
# Aggregate degradation robustness results
print("="*80)
print("  AGGREGATED DEGRADATION ROBUSTNESS")
print("="*80)

deg_sigmas = ["0.25", "0.50", "1.00"]
conditions = ["d12_social", "d11_h264_crf28", "d1_jpeg75"]

deg_agg = {}

for sigma in deg_sigmas:
    print(f"\n  σ = {sigma}:")
    print(f"    {'Condition':<18} | {'Cert@0.00':>16} | {'Cert@0.25':>16} | {'Cert@0.50':>16}")
    print("    " + "-" * 70)

    for cond in conditions:
        key = f"{sigma}_{cond}"
        accs, c25s, c50s = [], [], []
        for seed in seeds:
            d = seed_data[seed].get("degradation", {}).get(key, {})
            car = d.get("certified_accuracy_at_radii", {})
            s = d.get("summary", {})
            if s:
                accs.append(car.get("r_0.00", 0))
                c25s.append(car.get("r_0.25", 0))
                c50s.append(car.get("r_0.50", 0))

        if accs:
            deg_agg[key] = {
                "cert_0.00_mean": float(np.mean(accs)), "cert_0.00_std": float(np.std(accs)),
                "cert_0.25_mean": float(np.mean(c25s)), "cert_0.25_std": float(np.std(c25s)),
                "cert_0.50_mean": float(np.mean(c50s)), "cert_0.50_std": float(np.std(c50s)),
                "n_seeds": len(accs),
            }
            print(f"    {cond:<18} | {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
                  f"{np.mean(c25s):.4f} ± {np.std(c25s):.4f} | "
                  f"{np.mean(c50s):.4f} ± {np.std(c50s):.4f}")

    # Also show clean for comparison
    j = cert_joint_agg.get(sigma, {})
    c00 = j.get("r_0.00_mean", 0)
    c25 = j.get("r_0.25_mean", 0)
    c50 = j.get("r_0.50_mean", 0)
    print(f"    {'clean (ref)':<18} | {c00:.4f}            | {c25:.4f}            | {c50:.4f}")

with open(f"{OUTPUT_DIR}/degradation_aggregated.json", "w") as f:
    json.dump(deg_agg, f, indent=2)
```

---

## Cell 8 — Generate aggregated figures with confidence bands (Python)

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

# ── Fig 1: Certified accuracy curves with confidence bands ──
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
colors = {"0.12": "#2196F3", "0.25": "#4CAF50", "0.50": "#FF9800", "1.00": "#F44336"}

for sigma_key in ["0.12", "0.25", "0.50", "1.00"]:
    # Collect full curves from each seed
    all_curves = []
    for seed in seeds:
        cert_path = os.path.join(SEED_DATASET_PATHS[seed], f"cert_joint_{sigma_key}.json")
        if os.path.exists(cert_path):
            with open(cert_path) as f:
                data = json.load(f)
                curve = data.get("certified_accuracy_curve", {})
                if curve.get("radii") and curve.get("certified_accuracy"):
                    all_curves.append(curve)

    if not all_curves:
        continue

    radii = all_curves[0]["radii"]
    acc_matrix = np.array([c["certified_accuracy"] for c in all_curves])
    mean_acc = np.mean(acc_matrix, axis=0)
    std_acc = np.std(acc_matrix, axis=0)

    color = colors[sigma_key]
    ax.plot(radii, mean_acc, color=color, linewidth=2.5, label=f"σ = {sigma_key}")
    ax.fill_between(radii, mean_acc - std_acc, mean_acc + std_acc,
                    color=color, alpha=0.15)

ax.set_xlabel("Certified L₂ Radius", fontsize=14)
ax.set_ylabel("Certified Accuracy", fontsize=14)
ax.set_title("CertAV: Certified Accuracy vs Robustness Radius\n"
             f"(mean ± std over {n_seeds} seeds)", fontsize=14, fontweight="bold")
ax.legend(fontsize=12, loc="upper right")
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig1_certified_curves_aggregated.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/figures/fig1_certified_curves_aggregated.pdf", bbox_inches="tight")
plt.show()
print("✓ Fig 1")

# ── Fig 2: Joint vs unimodal (σ=0.25 and σ=1.00 side by side) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, sigma_key in enumerate(["0.25", "1.00"]):
    ax = axes[idx]
    configs = [
        (f"cert_joint_{sigma_key}.json", "joint", f"Joint (σ={sigma_key})", "#2196F3", "-"),
        (f"cert_visual_only_{sigma_key}.json", "visual_only", f"Visual-only (σ={sigma_key})", "#4CAF50", "--"),
        (f"cert_audio_only_{sigma_key}.json", "audio_only", f"Audio-only (σ={sigma_key})", "#FF9800", "-."),
    ]

    for fname, mode, label, color, ls in configs:
        all_curves = []
        for seed in seeds:
            cert_path = os.path.join(SEED_DATASET_PATHS[seed], fname)
            if os.path.exists(cert_path):
                with open(cert_path) as f:
                    data = json.load(f)
                    curve = data.get("certified_accuracy_curve", {})
                    if curve.get("radii") and curve.get("certified_accuracy"):
                        all_curves.append(curve)

        if not all_curves:
            continue

        radii = all_curves[0]["radii"]
        acc_matrix = np.array([c["certified_accuracy"] for c in all_curves])
        mean_acc = np.mean(acc_matrix, axis=0)
        std_acc = np.std(acc_matrix, axis=0)

        ax.plot(radii, mean_acc, color=color, linestyle=ls, linewidth=2.5, label=label)
        ax.fill_between(radii, mean_acc - std_acc, mean_acc + std_acc,
                        color=color, alpha=0.12)

    ax.set_xlabel("Certified L₂ Radius", fontsize=13)
    ax.set_ylabel("Certified Accuracy", fontsize=13)
    ax.set_title(f"σ = {sigma_key}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 2.0 if sigma_key == "0.25" else 3.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(f"Multimodal vs Unimodal Certification (mean ± std, {n_seeds} seeds)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig2_multimodal_vs_unimodal_aggregated.png",
            dpi=300, bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/figures/fig2_multimodal_vs_unimodal_aggregated.pdf",
            bbox_inches="tight")
plt.show()
print("✓ Fig 2")

# ── Fig 3: Accuracy-radius tradeoff ──
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sigma_labels = ["0.12", "0.25", "0.50", "1.00"]
color_list = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for i, sk in enumerate(sigma_labels):
    agg = cert_joint_agg.get(sk, {})
    acc_m = agg.get("accuracy_mean", 0)
    acc_s = agg.get("accuracy_std", 0)
    rad_m = agg.get("mean_radius_mean", 0)
    rad_s = agg.get("mean_radius_std", 0)
    if acc_m > 0:
        ax.errorbar(rad_m, acc_m, xerr=rad_s, yerr=acc_s,
                    fmt='o', markersize=12, color=color_list[i],
                    capsize=5, capthick=2, elinewidth=2, markeredgecolor='black')
        ax.annotate(f"σ={sk}", (rad_m, acc_m), textcoords="offset points",
                    xytext=(12, 5), fontsize=12)

ax.set_xlabel("Mean Certified L₂ Radius", fontsize=14)
ax.set_ylabel("Clean Accuracy (Smoothed)", fontsize=14)
ax.set_title(f"Accuracy–Robustness Tradeoff\n(mean ± std, {n_seeds} seeds)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig3_tradeoff_aggregated.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/figures/fig3_tradeoff_aggregated.pdf", bbox_inches="tight")
plt.show()
print("✓ Fig 3")

# ── Fig 4: Empirical attack bar chart with error bars ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, sigma_key in enumerate(["0.25", "0.50", "1.00"]):
    ax = axes[idx]
    eps_labels = ["ε=0.05", "ε=0.10", "ε=0.20"]
    atk = attack_agg.get(sigma_key, {})

    base_means = [atk.get(ek, {}).get("base_auc_mean", 0) for ek in eps_values]
    base_stds = [atk.get(ek, {}).get("base_auc_std", 0) for ek in eps_values]
    smooth_means = [atk.get(ek, {}).get("smooth_acc_mean", 0) for ek in eps_values]
    smooth_stds = [atk.get(ek, {}).get("smooth_acc_std", 0) for ek in eps_values]

    x = np.arange(len(eps_labels))
    width = 0.35
    ax.bar(x - width/2, base_means, width, yerr=base_stds,
           label="Base (AUC)", color="#F44336", alpha=0.8, capsize=4, edgecolor="black")
    ax.bar(x + width/2, smooth_means, width, yerr=smooth_stds,
           label="Smoothed (Acc)", color="#2196F3", alpha=0.8, capsize=4, edgecolor="black")

    ax.set_xlabel("PGD Budget", fontsize=12)
    ax.set_title(f"σ = {sigma_key}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx == 0:
        ax.set_ylabel("Performance", fontsize=13)
        ax.legend(fontsize=10)

fig.suptitle(f"Base vs Smoothed Under PGD Attack (mean ± std, {n_seeds} seeds)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig4_attack_aggregated.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/figures/fig4_attack_aggregated.pdf", bbox_inches="tight")
plt.show()
print("✓ Fig 4")
```

---

## Cell 9 — Generate LaTeX tables (Python)

```python
# Generate publication-ready LaTeX tables
latex_tables = {}

# ── Table 1: Main certification results ──
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Certified accuracy of CertAV at different noise levels $\sigma$ (mean $\pm$ std over " + str(n_seeds) + r" seeds).}")
lines.append(r"\label{tab:main_cert}")
lines.append(r"\begin{tabular}{c|ccc|cccc}")
lines.append(r"\toprule")
lines.append(r"$\sigma$ & Clean Acc & Abstain\% & Mean $R$ & Cert@0.00 & Cert@0.25 & Cert@0.50 & Cert@1.00 \\")
lines.append(r"\midrule")

for sk in ["0.12", "0.25", "0.50", "1.00"]:
    a = cert_joint_agg.get(sk, {})
    def fmt(key):
        m = a.get(f"{key}_mean", 0)
        s = a.get(f"{key}_std", 0)
        return f"{m:.3f}\\tiny{{$\\pm${s:.3f}}}"

    acc = fmt("accuracy")
    abst = fmt("abstain_rate")
    mr = fmt("mean_radius")
    c00 = fmt("r_0.00")
    c25 = fmt("r_0.25")
    c50 = fmt("r_0.50")
    c100 = fmt("r_1.00")
    lines.append(f"{sk} & {acc} & {abst} & {mr} & {c00} & {c25} & {c50} & {c100} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
latex_tables["main_cert"] = "\n".join(lines)

# ── Table 2: Joint vs unimodal ablation ──
lines2 = []
lines2.append(r"\begin{table}[t]")
lines2.append(r"\centering")
lines2.append(r"\caption{Multimodal vs unimodal noise certification comparison.}")
lines2.append(r"\label{tab:ablation}")
lines2.append(r"\begin{tabular}{cc|ccc}")
lines2.append(r"\toprule")
lines2.append(r"$\sigma$ & Noise Mode & Clean Acc & Mean $R$ & Cert@0.25 \\")
lines2.append(r"\midrule")

for sigma in ["0.25", "1.00"]:
    for mode, label in [("joint", "Joint"), ("visual_only", "Visual"), ("audio_only", "Audio")]:
        if mode == "joint":
            a = cert_joint_agg.get(sigma, {})
            acc_m = a.get("accuracy_mean", 0)
            acc_s = a.get("accuracy_std", 0)
            mr_m = a.get("mean_radius_mean", 0)
            mr_s = a.get("mean_radius_std", 0)
            c25_m = a.get("r_0.25_mean", 0)
            c25_s = a.get("r_0.25_std", 0)
        else:
            key = f"{mode}_{sigma}"
            a = cert_ablation_agg.get(key, {})
            acc_m = a.get("accuracy_mean", 0)
            acc_s = a.get("accuracy_std", 0)
            mr_m = a.get("mean_radius_mean", 0)
            mr_s = a.get("mean_radius_std", 0)
            c25_m = a.get("cert_0.25_mean", 0)
            c25_s = a.get("cert_0.25_std", 0)

        acc_str = f"{acc_m:.3f}\\tiny{{$\\pm${acc_s:.3f}}}"
        mr_str = f"{mr_m:.3f}\\tiny{{$\\pm${mr_s:.3f}}}"
        c25_str = f"{c25_m:.3f}\\tiny{{$\\pm${c25_s:.3f}}}"
        lines2.append(f"{sigma} & {label} & {acc_str} & {mr_str} & {c25_str} \\\\")
    lines2.append(r"\midrule")

lines2[-1] = r"\bottomrule"
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table}")
latex_tables["ablation"] = "\n".join(lines2)

# Save all LaTeX tables
for name, latex in latex_tables.items():
    path = f"{OUTPUT_DIR}/tables/{name}.tex"
    with open(path, "w") as f:
        f.write(latex)
    print(f"✓ {path}")
    print(latex)
    print()
```

---

## Cell 10 — Save master aggregated JSON (Python)

```python
# Save everything into one master JSON
master = {
    "n_seeds": n_seeds,
    "seeds": seeds,
    "training": training_agg,
    "certification_joint": cert_joint_agg,
    "certification_ablation": cert_ablation_agg,
    "empirical_attacks": attack_agg,
    "degradation": deg_agg,
}

master_path = f"{OUTPUT_DIR}/certav_master_results.json"
with open(master_path, "w") as f:
    json.dump(master, f, indent=2)

print(f"✓ Master results saved to {master_path}")
print(f"\nFinal output directory contents:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        rel = os.path.relpath(fpath, OUTPUT_DIR)
        size = os.path.getsize(fpath)
        print(f"  {rel:<50} {size/1024:>8.1f} KB")

print(f"\n{'='*60}")
print(f"  ALL DONE — {n_seeds}-seed aggregated results ready")
print(f"{'='*60}")
```
