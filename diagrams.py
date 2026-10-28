"""Protocol diagrams for Figures 8-10 (method schematics; no measured values)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
BG = "#EAF1FA"; NAVY = "#1F3B63"; BLUE = "#2E6DA4"; GREEN = "#3F9E7E"; PURPLE = "#6B3FA0"; ORANGE = "#D9692B"; PALE = "#FFFFFF"; TXT = "#1F3B63"


def box(ax, x, y, w, h, title, lines, color, fs=7.2, tfs=8.4, body_bg=PALE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=body_bg, ec=color, lw=1.4, zorder=2))
    th = 0.62
    ax.add_patch(FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=0.12", fc=color, ec=color, lw=1.4, zorder=3))
    ax.add_patch(plt.Rectangle((x + 0.01, y + h - th), w - 0.02, th * 0.45, fc=color, ec=color, zorder=3))
    ax.text(x + w / 2, y + h - th / 2, title, ha="center", va="center", color="white", fontsize=tfs, fontweight="bold", zorder=4)
    n = len(lines); avail = h - th; step = min(0.36, (avail - 0.1) / max(n, 1)); cy = y + avail / 2; top = cy + step * (n - 1) / 2 + step / 2 - step / 2
    for i, l in enumerate(lines):
        bold = l.startswith("**"); l = l.strip("*")
        ax.text(x + w / 2, top - step * i, l, ha="center", va="center", color=TXT, fontsize=fs, fontweight="bold" if bold else "normal", zorder=4, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color="#4A5568"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=color, zorder=1, shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig = plt.figure(figsize=(7.2, 7.2 * h / w), facecolor=BG); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    fig.patch.set_facecolor(BG); return fig, ax


def save(fig, name):
    fig.savefig(name + ".png", dpi=300, facecolor=BG); plt.close(fig)


# ---------------- Figure 8: resampling sensitivity (Table 21)
fig, ax = canvas(15.0, 5.0)
box(ax, 0.2, 0.35, 2.5, 4.3, "Training partition", ["**72% of blocks", "genuine records", "", "Slowloris and", "Botnet-ARES are", "the rare classes", "", "Windows built", "inside blocks"], BLUE)
box(ax, 3.35, 0.35, 3.3, 4.3, "Resampling arm", ["**four strategies, fitting", "partition only", "", "(1) Class weights only", "(2) Random oversampling", "(3) SMOTE, k = 5 (proposed)", "(4) Borderline-SMOTE", "", "validation and test", "records never resampled"], PURPLE)
box(ax, 7.3, 0.35, 2.6, 4.3, "Models", ["**identical across arms", "", "CNN", "BiLSTM (T = 20)", "LightGBM", "", "same seeds, same", "tuning budgets"], NAVY)
box(ax, 10.55, 0.35, 1.95, 4.3, "DWF fusion", ["**entropy-indexed", "5 regimes", "", "tuned on the", "validation", "partition only"], ORANGE)
box(ax, 13.0, 0.35, 1.8, 4.3, "Genuine test", ["**20% held-out", "", "class-wise F1", "with 95% CI", "", "macro-F1, FPR", "", "10 seeds"], GREEN)
for x0, x1 in [(2.7, 3.35), (6.65, 7.3), (9.9, 10.55), (12.5, 13.0)]:
    arrow(ax, x0, 2.5, x1, 2.5)
save(fig, "fig8_resampling_protocol")

# ---------------- Figure 9: attribution (Table 22)
fig, ax = canvas(15.0, 6.2)
box(ax, 0.2, 2.0, 2.2, 2.2, "Test windows", ["**28 x 20 HFR", "windows", "", "fixed seed set", "(10 seeds)"], BLUE)
rows = [("CNN", ["Integrated Gradients", "last-step input"], NAVY, 4.65), ("BiLSTM", ["Integrated Gradients", "summed over T = 20"], NAVY, 3.35), ("LightGBM", ["TreeSHAP", "predicted-class log-odds"], NAVY, 2.05),
        ("DWF gate", ["permutation of H(x)", "change in macro-F1"], ORANGE, 0.75)]
for t, l, c, y in rows:
    box(ax, 3.25, y, 3.6, 1.2, t, ["**" + l[0], l[1]], c, fs=7.2, tfs=8)
    arrow(ax, 2.4, 3.1, 3.25, y + 0.57)
box(ax, 7.6, 2.35, 3.0, 2.0, "Ensemble attribution", ["**regime-weighted sum", "", "per-sample component", "attributions normalised", "(L1) then weighted by w(H)"], PURPLE)
for y, ye in ((5.25, 4.05), (3.95, 3.7), (2.65, 3.1)):
    arrow(ax, 6.85, y, 7.6, ye)
arrow(ax, 6.85, 1.35, 7.6, 2.75)
box(ax, 11.3, 1.5, 3.5, 3.6, "Stability checks", ["**reported per component", "", "Kendall tau across", "10 seeds (rank stability)", "", "Kendall tau across baselines", "(zero, benign mean, noise)", "", "IG completeness error"], GREEN)
arrow(ax, 10.6, 3.35, 11.3, 3.3)
save(fig, "fig9_attribution_protocol")

# ---------------- Figure 10: WSN controls (Table 23)
fig, ax = canvas(15.0, 5.6)
box(ax, 0.2, 1.4, 2.6, 2.8, "Reference", ["**full HFR vector", "28 components", "", "full HAI+DWF", "same dataset,", "same seed"], BLUE)
ctr = [("Raw-only HFR", "no engineered components", "NF-ToN-IoT-v2, WSN-DS", NAVY, 4.45), ("No WSN-specific components", "LQI, hop count, routing changes removed", "WSN-DS", NAVY, 3.35),
       ("Tier 1 constrained", "GBDT only, int8, <= 64 KB SRAM", "WSN-DS", ORANGE, 2.25), ("Simulated LEACH traffic", "cluster-head rotation per round", "WSN-DS", ORANGE, 1.15)]
for t, a, b, c, y in ctr:
    box(ax, 3.6, y - 0.05, 5.4, 0.98, t, [a, b], c, fs=6.8, tfs=7.8)
    arrow(ax, 2.8, 2.8, 3.6, y + 0.42)
box(ax, 9.8, 1.4, 2.4, 2.8, "Metrics", ["**macro-F1", "FPR", "", "delta macro-F1", "vs. reference", "(paired, same seed)"], PURPLE)
for y, ye in ((4.9, 3.9), (3.8, 3.35), (2.7, 2.65), (1.6, 2.1)):
    arrow(ax, 9.0, y - 0.05, 9.8, ye)
box(ax, 12.8, 1.4, 2.0, 2.8, "Reporting", ["**WSN-DS and", "Tier 1 results", "separated from", "general IoT", "datasets"], GREEN)
arrow(ax, 12.2, 2.8, 12.8, 2.8)
save(fig, "fig10_wsn_controls_protocol")
print("ok")
