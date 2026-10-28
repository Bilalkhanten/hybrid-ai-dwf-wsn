#!/usr/bin/env python3
"""Figures for Tables 21-23 from YOUR measured results.
    python make_figures.py results.json [outdir]
Writes fig_t21_resampling, fig_t22_attribution, fig_t23_wsn_controls as PNG (300 dpi) and PDF."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"blue": "#2F5D8A", "orange": "#D9822B", "green": "#4C8C6B", "grey": "#8A8F98", "red": "#B5483C"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150})


def need(x, path):
    if x is None or (isinstance(x, list) and any(v is None for v in x)):
        raise SystemExit(f"missing value at {path}: fill results.json first")
    return np.array(x, float)


def get(d, key, n, optional=False):
    out = []
    for i, v in enumerate(d[key]):
        if v is None and optional:
            out.append(None)
        else:
            out.append(need(v, f"{key}[{i}]"))
    return out


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig21(t, out):
    S = t["strategies"]; n = len(S); x = np.arange(n); w = 0.36
    sl, bo = np.array(get(t, "slowloris", n)), np.array(get(t, "botnet_ares", n))
    mf, fp = np.array(get(t, "macro_f1", n)), np.array(get(t, "fpr", n))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.7, 1]})
    a = ax[0]
    for k, (d, lab, c) in enumerate([(sl, "Slowloris", PAL["blue"]), (bo, "Botnet-ARES", PAL["orange"])]):
        xs = x + (k - 0.5) * w
        a.bar(xs, d[:, 0], w, color=c, label=lab, zorder=3)
        a.errorbar(xs, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], fmt="none", ecolor="#222", elinewidth=0.9, capsize=2.5, zorder=4)
    best = int(np.argmax(sl[:, 0] + bo[:, 0])); a.axvspan(best - 0.5, best + 0.5, color=PAL["grey"], alpha=0.12, zorder=0)
    a.set_xticks(x); a.set_xticklabels([s.replace(" (", "\n(").replace(" only", "\nonly").replace(" oversampling", "\noversampling") for s in S], fontsize=7.5)
    a.set_ylabel("Class F1 on genuine held-out records (%)"); a.set_ylim(0, None); h, l = a.get_legend_handles_labels()
    a.set_title("(a) Minority-class F1, mean and 95% CI", loc="left", fontsize=9, fontweight="bold")
    b = ax[1]
    b.errorbar(mf[:, 0], x, xerr=mf[:, 1], fmt="o", color=PAL["blue"], capsize=2.5, zorder=3)
    for i in range(n):
        b.annotate(f"FPR {fp[i, 0]:.2f} ± {fp[i, 1]:.2f}", (mf[i, 0], i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=PAL["red"])
    b.set_yticks(x); b.set_yticklabels([]); b.set_ylim(n - 0.4, -0.7); b.set_xlabel("Macro-F1 (%), mean ± SD")
    b.set_title("(b) Overall (FPR in %)", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h, l, loc="lower left", ncol=2, fontsize=8, bbox_to_anchor=(0.06, 0.0)); save(fig, out, "fig_t21_resampling")


def fig22(t, out):
    C = t["components"]; n = len(C); x = np.arange(n)
    rs = np.array(get(t, "rank_stability_tau", n))
    bs = get(t, "baseline_sensitivity_tau", n, True); ce = get(t, "completeness_error_pct", n, True)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a = ax[0]
    a.bar(x, rs[:, 0], 0.6, yerr=rs[:, 1], color=PAL["blue"], capsize=3, ecolor="#222", error_kw={"elinewidth": 0.9}, zorder=3)
    a.axhline(0, color="#222", lw=0.8); a.axhline(0.7, color=PAL["green"], lw=1, ls="--"); a.text(n - 0.5, 0.71, "τ = 0.7", color=PAL["green"], ha="right", va="bottom", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels(C, fontsize=7.5, rotation=25, ha="right"); a.set_ylim(-1, 1)
    a.set_ylabel("Kendall τ across seeds"); a.set_title("(a) Rank stability", loc="left", fontsize=9, fontweight="bold")
    b = ax[1]; idx = [i for i in range(n) if bs[i] is not None]; xi = np.arange(len(idx)); w = 0.36
    b.bar(xi - w / 2, [bs[i][0] for i in idx], w, yerr=[bs[i][1] for i in idx], color=PAL["orange"], capsize=3, label="Baseline sensitivity (τ)", ecolor="#222", zorder=3)
    b.set_ylim(0, 1.05); b.set_ylabel("Kendall τ across baselines", color=PAL["orange"])
    b3 = b.twinx(); jj = [k for k, i in enumerate(idx) if ce[i] is not None]
    b3.bar([xi[k] + w / 2 for k in jj], [ce[idx[k]][0] for k in jj], w, yerr=[ce[idx[k]][1] for k in jj], color=PAL["grey"], capsize=3, label="Completeness error (%)", ecolor="#222", zorder=3)
    b3.set_ylabel("IG completeness error (%)", color=PAL["grey"]); b3.grid(False); b3.spines["right"].set_visible(True)
    b.set_xticks(xi); b.set_xticklabels([C[i] for i in idx], fontsize=7.5, rotation=25, ha="right")
    b.set_title("(b) Baseline sensitivity and completeness", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout(); save(fig, out, "fig_t22_attribution")


def fig23(t, out):
    n = len(t["configs"]); mf = np.array(get(t, "macro_f1", n)); dl = np.array(get(t, "delta_macro_f1_pp", n))
    lab = [f"{c}\n[{d}]" for c, d in zip(t["configs"], t["datasets"])]; y = np.arange(n)[::-1]
    col = [PAL["green"] if d == "WSN-DS" else PAL["blue"] for d in t["datasets"]]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for yi, m, c in zip(y, mf, col):
        ax[0].errorbar(m[0], yi, xerr=m[1], fmt="o", color=c, ecolor="#222", capsize=3, ms=6, elinewidth=0.9, zorder=3)
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=7.5); ax[0].set_xlabel("Macro-F1 (%), mean ± SD")
    ax[0].set_xlim(max(0, (mf[:, 0] - mf[:, 1]).min() - 5), min(100, (mf[:, 0] + mf[:, 1]).max() + 3)); ax[0].set_title("(a) Performance", loc="left", fontsize=9, fontweight="bold")
    ax[1].barh(y, dl[:, 0], 0.6, xerr=dl[:, 1], color=col, capsize=3, ecolor="#222", error_kw={"elinewidth": 0.9}, zorder=3)
    ax[1].axvline(0, color="#222", lw=0.8); ax[1].set_xlabel("Δ macro-F1 vs. full model (pp), mean ± SD")
    ax[1].set_title("(b) Change relative to the full model", loc="left", fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=[Patch(color=PAL["green"], label="WSN-DS (WSN-specific)"), Patch(color=PAL["blue"], label="General IoT dataset")], loc="lower center", ncol=2, fontsize=8)
    save(fig, out, "fig_t23_wsn_controls")


if __name__ == "__main__":
    R = json.load(open(sys.argv[1])); out = Path(sys.argv[2] if len(sys.argv) > 2 else "."); out.mkdir(exist_ok=True, parents=True)
    fig21(R["table21"], out); fig22(R["table22"], out); fig23(R["table23"], out); print("written to", out)
