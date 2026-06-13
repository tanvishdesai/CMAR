import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

OUT_DIRS = [
    os.path.join(os.path.dirname(__file__), "figures"),
    os.path.join(os.path.dirname(__file__), "..", "paper", "figures"),
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----- Figure: encoder scaling scatter -----
csv_path = os.path.join(os.path.dirname(__file__), "..", "results-consolidated",
                        "phase2", "encoder-scaling", "encoder_scaling_all_5families.csv")
rows = list(csv.DictReader(open(csv_path)))

labels = {
    "dinov2small_wavlm": ("DINOv2-S+WavLM", "tab:green"),
    "dinov2small_hubert": ("DINOv2-S+HuBERT", "tab:green"),
    "dinov2small_whispertiny_existing_cache": ("DINOv2-S+Whisper-tiny", "tab:blue"),
    "dinov2base_whisperbase": ("DINOv2-B+Whisper-base", "tab:blue"),
    "clip_whispertiny": ("CLIP-B+Whisper-tiny", "tab:red"),
}

fig, ax = plt.subplots(figsize=(5.2, 3.6))
xs, ys = [], []
for r in rows:
    ep = r["encoder_pair"]
    x = float(r["d_int_ratio_90"])
    y = float(r["mean_certified_radius"])
    name, color = labels.get(ep, (ep, "gray"))
    xs.append(x); ys.append(y)
    ax.scatter(x, y, s=70, color=color, zorder=3, edgecolor="k", linewidth=0.5)
    dy = 0.012 if name != "DINOv2-S+Whisper-tiny" else -0.028
    ha = "left"
    ax.annotate(name, (x, y), textcoords="offset points",
                xytext=(6, 6 if dy > 0 else -12), fontsize=8, ha=ha)

# trend line
xs = np.array(xs); ys = np.array(ys)
order = np.argsort(xs)
m, b = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min() - 0.005, xs.max() + 0.005, 50)
ax.plot(xx, m * xx + b, "--", color="gray", lw=1.2, zorder=1,
        label=f"linear trend (slope {m:.1f})")
ax.set_xlabel(r"intrinsic-dimension ratio  $d_{\mathrm{int}}/D$  (90% variance)")
ax.set_ylabel(r"mean certified radius  $\bar{R}$  ($\sigma{=}1.0$)")
ax.set_title("Lower intrinsic dimension ratio yields larger certified radius")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
save(fig, "encoder_scaling_scatter.png")

# ----- Figure: sphere vs ellipsoid certified region -----
fig, ax = plt.subplots(figsize=(5.0, 3.8))
# data manifold direction = horizontal (on-manifold PC1), off-manifold = vertical
# isotropic circle radius ~2.22; anisotropic ellipse on-manifold 7.63, off-manifold ~tiny
R_iso = 2.22
R_on = 7.63
R_off = 0.6  # visually small (true ~0.002, exaggerated for visibility)

ax.add_patch(Ellipse((0, 0), 2 * R_on, 2 * R_off, facecolor="tab:orange",
                      alpha=0.30, edgecolor="tab:orange", lw=2,
                      label="anisotropic (Strat 2)"))
ax.add_patch(Circle((0, 0), R_iso, facecolor="tab:blue", alpha=0.25,
                    edgecolor="tab:blue", lw=2, label="isotropic"))

# data points along manifold
rng = np.random.default_rng(0)
mx = rng.normal(0, 3.0, 40)
my = rng.normal(0, 0.25, 40)
ax.scatter(mx, my, s=10, color="k", alpha=0.6, zorder=4, label="features (data manifold)")

# PGD attack arrows mostly off-manifold (vertical)
for x0 in (-1.2, 0.6, 1.8):
    ax.annotate("", xy=(x0 + 0.2, 2.6), xytext=(x0, 0.0),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3))
ax.text(2.0, 2.7, "PGD\n(82% off-manifold)", color="tab:red", fontsize=8)

ax.set_xlim(-8.5, 8.5)
ax.set_ylim(-4.2, 4.2)
ax.set_xlabel("on-manifold direction (PC1)")
ax.set_ylabel("off-manifold direction")
ax.set_aspect("equal", adjustable="box")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title("Certified region: sphere vs. manifold-aligned ellipsoid")
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
save(fig, "sphere_vs_ellipsoid.png")

print("figures written to:", [os.path.abspath(d) for d in OUT_DIRS])
