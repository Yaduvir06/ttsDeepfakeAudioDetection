"""
generate_figures.py
===================
Generates all paper figures from actual recorded results in all_results.csv.
Saves to ./figures/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "aasist_base":   "#6c757d",   # grey
    "aasist_adv":    "#0077b6",   # deep blue
    "lcnn_base":     "#e07a5f",   # terracotta
    "lcnn_adv":      "#f2cc8f",   # amber
    "resnet_base":   "#81b29a",   # sage
    "resnet_adv":    "#3d405b",   # dark indigo
    "ensemble":      "#e63946",   # vivid red
}

EPS   = [0.0005, 0.001, 0.002, 0.005, 0.01]
EPS_L = ["0.0005", "0.001", "0.002", "0.005", "0.01"]

# ── Ground-truth data (sourced directly from all_results.csv) ───────────────
# AASIST adversarial (ts: 2026-04-23 22:36)
AASIST_ADV_FGSM  = [99.85, 99.85, 99.71, 99.35, 99.35]
AASIST_ADV_PGD   = [99.78, 99.71, 99.27, 98.91, 98.25]

# AASIST baseline (ts: 2026-04-24 20:14)
AASIST_BASE_FGSM = [99.85, 99.93, 99.93, 99.85, 99.78]
AASIST_BASE_PGD  = [99.93, 99.93, 99.71, 98.11, 99.27]

# LFCC+LCNN baseline (ts: 2026-04-26 23:29)
LCNN_BASE_FGSM   = [71.85, 55.49, 51.56, 51.71, 51.71]
LCNN_BASE_PGD    = [30.69, 22.04, 20.07, 17.45, 16.22]

# LFCC+LCNN adversarial (ts: 2026-04-28 00:23)
LCNN_ADV_FGSM    = [99.64, 99.56, 98.98, 94.69, 75.20]
LCNN_ADV_PGD     = [97.89, 94.55, 85.38, 50.47, 36.00]

# Mel+ResNet18 baseline (ts: 2026-04-27 17:27)
RESNET_BASE_FGSM = [57.38, 50.69, 50.25, 50.25, 50.25]
RESNET_BASE_PGD  = [0.22,  0.07,  0.22,  0.51,  1.09]

# Mel+ResNet18 adversarial (ts: 2026-04-28 23:16)
RESNET_ADV_FGSM  = [99.78, 99.78, 99.78, 99.05, 95.85]
RESNET_ADV_PGD   = [99.78, 99.78, 99.56, 95.64, 45.75]

# Ensemble (ts: 2026-04-29 20:24)
ENS_FGSM = [99.93, 99.93, 99.93, 100.00, 99.85]
ENS_PGD  = [99.93, 99.93, 99.93, 99.93,  99.49]

# Post-processing & cross-deepfake
POST = {
    "AASIST-Adv":  {"MP3-64k": 99.85, "MP3-128k": 99.85, "Resample-8k": 96.87, "Cross-deepfake": 100.00},
    "AASIST-Base": {"MP3-64k": 99.85, "MP3-128k": 99.93, "Resample-8k": 98.25, "Cross-deepfake": 99.10},
    "LCNN-Base":   {"MP3-64k": 96.87, "MP3-128k": 97.38, "Resample-8k": 96.80, "Cross-deepfake": 100.00},
    "LCNN-Adv":    {"MP3-64k": 97.24, "MP3-128k": 97.24, "Resample-8k": 89.82, "Cross-deepfake": 99.10},
    "ResNet-Base": {"MP3-64k": 99.71, "MP3-128k": 99.78, "Resample-8k": 98.11, "Cross-deepfake": 100.00},
    "ResNet-Adv":  {"MP3-64k": 99.85, "MP3-128k": 99.85, "Resample-8k": 99.64, "Cross-deepfake": 99.55},
    "Ensemble":    {"MP3-64k": 100.0, "MP3-128k": 100.0, "Resample-8k": 99.64, "Cross-deepfake": 99.55},
}

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "legend.framealpha": 0.85,
})

# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — FGSM Robustness Curves (all streams)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 1: FGSM robustness curves...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Figure 1 — FGSM Robustness: Baseline vs. Adversarially Trained Streams", fontsize=14, fontweight="bold")

for ax, title, data_pairs in [
    (axes[0], "FGSM Attack — Accuracy (%)", [
        (EPS, AASIST_BASE_FGSM, C["aasist_base"], "--", "AASIST (baseline)"),
        (EPS, AASIST_ADV_FGSM,  C["aasist_adv"],  "-",  "AASIST (adversarial)"),
        (EPS, LCNN_BASE_FGSM,   C["lcnn_base"],   "--", "LCNN (baseline)"),
        (EPS, LCNN_ADV_FGSM,    C["lcnn_adv"],    "-",  "LCNN (adversarial)"),
        (EPS, RESNET_BASE_FGSM, C["resnet_base"], "--", "ResNet18 (baseline)"),
        (EPS, RESNET_ADV_FGSM,  C["resnet_adv"],  "-",  "ResNet18 (adversarial)"),
        (EPS, ENS_FGSM,         C["ensemble"],    "-",  "Ensemble (adv)"),
    ]),
    (axes[1], "FGSM — Zoom: Streams ≥ 70%", [
        (EPS, AASIST_BASE_FGSM, C["aasist_base"], "--", "AASIST (baseline)"),
        (EPS, AASIST_ADV_FGSM,  C["aasist_adv"],  "-",  "AASIST (adversarial)"),
        (EPS, LCNN_ADV_FGSM,    C["lcnn_adv"],    "-",  "LCNN (adversarial)"),
        (EPS, RESNET_ADV_FGSM,  C["resnet_adv"],  "-",  "ResNet18 (adversarial)"),
        (EPS, ENS_FGSM,         C["ensemble"],    "-",  "Ensemble (adv)"),
    ]),
]:
    for xs, ys, col, ls, lbl in data_pairs:
        lw = 2.5 if "Ensemble" in lbl else 1.8
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw,
                marker="o", markersize=5, label=lbl)
    ax.set_xlabel("Perturbation ε", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title)
    ax.set_xticks(EPS)
    ax.set_xticklabels(EPS_L, rotation=30)
    ax.legend(fontsize=8, loc="lower left")

axes[1].set_ylim([70, 101])
plt.tight_layout()
savefig("fig1_fgsm_curves.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — PGD-20 Robustness Curves
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 2: PGD-20 robustness curves...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Figure 2 — PGD-20 Robustness: Baseline vs. Adversarially Trained Streams", fontsize=14, fontweight="bold")

for ax, title, ylim, data_pairs in [
    (axes[0], "PGD-20 — All Streams", [0, 102], [
        (EPS, AASIST_BASE_PGD,  C["aasist_base"], "--", "AASIST (baseline)"),
        (EPS, AASIST_ADV_PGD,   C["aasist_adv"],  "-",  "AASIST (adversarial)"),
        (EPS, LCNN_BASE_PGD,    C["lcnn_base"],   "--", "LCNN (baseline)"),
        (EPS, LCNN_ADV_PGD,     C["lcnn_adv"],    "-",  "LCNN (adversarial)"),
        (EPS, RESNET_BASE_PGD,  C["resnet_base"], "--", "ResNet18 (baseline)"),
        (EPS, RESNET_ADV_PGD,   C["resnet_adv"],  "-",  "ResNet18 (adversarial)"),
        (EPS, ENS_PGD,          C["ensemble"],    "-",  "Ensemble (adv)"),
    ]),
    (axes[1], "PGD-20 — Adversarially Trained Only", [30, 101], [
        (EPS, AASIST_BASE_PGD,  C["aasist_base"], "--", "AASIST (baseline)"),
        (EPS, AASIST_ADV_PGD,   C["aasist_adv"],  "-",  "AASIST (adversarial)"),
        (EPS, LCNN_ADV_PGD,     C["lcnn_adv"],    "-",  "LCNN (adversarial)"),
        (EPS, RESNET_ADV_PGD,   C["resnet_adv"],  "-",  "ResNet18 (adversarial)"),
        (EPS, ENS_PGD,          C["ensemble"],    "-",  "Ensemble (adv)"),
    ]),
]:
    for xs, ys, col, ls, lbl in data_pairs:
        lw = 2.5 if "Ensemble" in lbl else 1.8
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw,
                marker="s", markersize=5, label=lbl)
    ax.set_xlabel("Perturbation ε", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title)
    ax.set_xticks(EPS)
    ax.set_xticklabels(EPS_L, rotation=30)
    ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc="lower left")

plt.tight_layout()
savefig("fig2_pgd_curves.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Adversarial Training Gain (bar chart, PGD-20 ε=0.005)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 3: Adversarial training gain bar chart...")
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Figure 3 — Adversarial Training Gain at PGD-20 ε=0.005", fontsize=14, fontweight="bold")

models   = ["AASIST", "LFCC+LCNN", "Mel+ResNet18"]
baseline = [98.11, 17.45, 0.51]
adv      = [98.91, 50.47, 95.64]

x  = np.arange(len(models))
w  = 0.32
b1 = ax.bar(x - w/2, baseline, w, label="Baseline", color=[C["aasist_base"], C["lcnn_base"], C["resnet_base"]], edgecolor="white")
b2 = ax.bar(x + w/2, adv,      w, label="Adversarial", color=[C["aasist_adv"], C["lcnn_adv"], C["resnet_adv"]], edgecolor="white")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

gains = [v2 - v1 for v1, v2 in zip(baseline, adv)]
for i, g in enumerate(gains):
    ax.annotate(f"Δ+{g:.1f}%", xy=(x[i], max(baseline[i], adv[i]) + 5),
                ha="center", color="#e63946", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.axhline(50, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.text(2.55, 51.5, "random-guess baseline (50%)", fontsize=8, color="grey")
ax.legend(fontsize=11)
plt.tight_layout()
savefig("fig3_training_gain_bar.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Post-Processing & Cross-Deepfake Grouped Bar
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 4: Post-processing and cross-deepfake bar chart...")
attacks = ["MP3-64k", "MP3-128k", "Resample-8k", "Cross-deepfake"]
models_sel = ["AASIST-Adv", "LCNN-Adv", "ResNet-Adv", "Ensemble"]
colors_sel = [C["aasist_adv"], C["lcnn_adv"], C["resnet_adv"], C["ensemble"]]

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Figure 4 — Post-Processing & Cross-Deepfake Robustness (Adversarial Models)", fontsize=13, fontweight="bold")

x  = np.arange(len(attacks))
w  = 0.18
for i, (m, col) in enumerate(zip(models_sel, colors_sel)):
    vals = [POST[m][a] for a in attacks]
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, vals, w, label=m, color=col, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(attacks, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(82, 103)
ax.legend(fontsize=10)
plt.tight_layout()
savefig("fig4_postprocessing_bar.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Ensemble vs Individual Streams Heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 5: Heatmap comparison...")
import matplotlib.colors as mcolors

rows = ["AASIST-Adv", "LCNN-Adv", "ResNet-Adv", "Ensemble"]
cols_hm = ["Clean", "FGSM\n0.001", "FGSM\n0.01", "PGD\n0.001", "PGD\n0.005", "PGD\n0.01",
           "MP3\n64k", "Resample\n8k", "Cross\nDeepfake"]

data_hm = np.array([
    # Clean  F001   F01    P001   P005   P01    MP3    Res    Cross
    [99.93, 99.85, 99.35, 99.71, 98.91, 98.25, 99.85, 96.87, 100.00],  # AASIST-Adv
    [99.85, 99.56, 75.20, 94.55, 50.47, 36.00, 97.24, 89.82, 99.10],   # LCNN-Adv
    [99.85, 99.78, 95.85, 99.78, 95.64, 45.75, 99.85, 99.64, 99.55],   # ResNet-Adv
    [99.93, 99.93, 99.85, 99.93, 99.93, 99.49, 100.0, 99.64, 99.55],   # Ensemble
])

fig, ax = plt.subplots(figsize=(13, 4))
fig.suptitle("Figure 5 — Accuracy Heatmap: All Attacks × All Adversarial Models", fontsize=13, fontweight="bold")

cmap = plt.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=30, vcenter=90, vmax=100)
im = ax.imshow(data_hm, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(cols_hm)))
ax.set_xticklabels(cols_hm, fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=11)

for i in range(len(rows)):
    for j in range(len(cols_hm)):
        v = data_hm[i, j]
        color = "white" if v < 70 else "black"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.025, pad=0.02)
plt.tight_layout()
savefig("fig5_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — EER Comparison (adversarial streams, PGD sweep)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 6: EER comparison...")
# EER data (from CSV, adversarial models only, PGD-20)
EER_AASIST_ADV_PGD  = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]  # 22:36 run
EER_LCNN_ADV_PGD    = [0.0132, 0.0351, 0.0854, 0.3705, 0.6498]
EER_RESNET_ADV_PGD  = [0.0015, 0.0029, 0.0043, 0.0333, 0.4848]
EER_ENS_PGD         = [0.0000, 0.0000, 0.0000, 0.0000, 0.0014]

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Figure 6 — EER under PGD-20: Adversarially Trained Models", fontsize=13, fontweight="bold")

ax.plot(EPS, EER_AASIST_ADV_PGD,  color=C["aasist_adv"],  marker="o",  linewidth=2, label="AASIST (adversarial)")
ax.plot(EPS, EER_LCNN_ADV_PGD,    color=C["lcnn_adv"],    marker="s",  linewidth=2, label="LCNN (adversarial)")
ax.plot(EPS, EER_RESNET_ADV_PGD,  color=C["resnet_adv"],  marker="^",  linewidth=2, label="ResNet18 (adversarial)")
ax.plot(EPS, EER_ENS_PGD,         color=C["ensemble"],    marker="D",  linewidth=2.5, label="Ensemble")

ax.set_xlabel("Perturbation ε", fontsize=11)
ax.set_ylabel("Equal Error Rate (EER) ↓", fontsize=11)
ax.set_xticks(EPS)
ax.set_xticklabels(EPS_L, rotation=30)
ax.set_ylim(-0.02, 0.70)
ax.legend(fontsize=10)
ax.annotate("Lower is better", xy=(0.98, 0.05), xycoords="axes fraction",
            ha="right", fontsize=9, color="grey")
plt.tight_layout()
savefig("fig6_eer_pgd.png")

print(f"\nAll figures saved to: {OUT_DIR}")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
