"""
plot_tuning.py — 2×2 grid of plots: one per pen_scale (2, 3, 5, 7).
Each subplot shows weighted_macro_F1 vs min_effect_size
with two lines: cv_perpair and mode_window.
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV = "tune_summary_overall.csv"
df = pd.read_csv(CSV)

scales = [2.0, 3.0, 5.0, 7.0]
strategies = ["cv_perpair", "mode_window"]
colors = {"cv_perpair": "#1f77b4", "mode_window": "#d62728"}
markers = {"cv_perpair": "o", "mode_window": "s"}
labels = {"cv_perpair": "CV Per-Pair", "mode_window": "Mode Window"}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle("Micro F1 vs Min Effect Size by Penalty Scale", fontsize=14, y=0.98)

for ax, scale in zip(axes.flat, scales):
    for strat in strategies:
        subset = df[(df["strategy"] == strat) & (df["pen_scale"] == scale)].sort_values("min_effect_size")
        ax.plot(
            subset["min_effect_size"],
            subset["micro_F1"],
            color=colors[strat],
            marker=markers[strat],
            linewidth=2,
            markersize=7,
            label=labels[strat],
        )
    ax.set_title(f"pen_scale = {scale:.0f}", fontsize=12)
    ax.set_xlabel("min_effect_size")
    ax.set_ylabel("Micro F1")
    ax.set_xticks([0.0, 0.10, 0.15, 0.30])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

# Single shared legend
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=2, fontsize=11, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
out = "tuning_plots.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
