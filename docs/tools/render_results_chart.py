"""Bar chart of the error ratio to Uniform per PDE and error statistic (OGAS-Loss, paper Table 8).

    python render_results_chart.py ../assets/figures/results_bars.svg
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
args = ap.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
                     "mathtext.fontset": "stix", "svg.fonttype": "path", "font.size": 11})

# Uniform error / OGAS-Loss error, averaged over 3 architectures x 3 seeds (paper, Appendix D, Table 8).
METRICS = ["Max error", "P99 error", "Error std", "Mean error"]
PDES = {
    "Gray-Scott": [1.68, 1.25, 1.40, 0.96],
    "Kuramoto-Sivashinsky": [1.74, 1.66, 1.92, 0.89],
    "Kolmogorov flow": [1.51, 1.37, 1.67, 1.05],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, validated on light surfaces
INK, INK2, MUTED, GRID, BASE, SURFACE = "#1e293b", "#475569", "#94a3b8", "#e2e8f0", "#64748b", "#ffffff"

W, H, DPI = 800, 360, 100
AX = [0.085, 0.12, 0.82, 0.66]
BAR, GAP = 0.22, 0.02
YMIN, YMAX = 0.8, 2.05

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_alpha(0)
fig.patches.append(FancyBboxPatch((0.002, 0.003), 0.996, 0.994, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, fc=SURFACE, ec="none", zorder=-10))
ax = fig.add_axes(AX)
ax.patch.set_alpha(0)


def bar(x0, x1, y0, y1, color, r_px=4):
    """Bar from the baseline y0 to y1, rounded at the data end only."""
    s = 1 if y1 >= y0 else -1
    rx = min(r_px / (W * AX[2]) * (ax.get_xlim()[1] - ax.get_xlim()[0]), (x1 - x0) / 2)
    ry = min(r_px / (H * AX[3]) * (YMAX - YMIN), abs(y1 - y0))
    verts = [(x0, y0), (x0, y1 - s * ry), (x0, y1), (x0 + rx, y1), (x1 - rx, y1), (x1, y1), (x1, y1 - s * ry),
             (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", zorder=3))


ax.set_xlim(-0.55, len(METRICS) - 0.45)
ax.set_ylim(YMIN, YMAX)
for k in (0.8, 1.2, 1.4, 1.6, 1.8, 2.0):
    ax.axhline(k, color=GRID, lw=0.8, zorder=0)
ax.axhline(1.0, color=BASE, lw=1.2, zorder=4)
ax.text(len(METRICS) - 0.45 + 0.04, 1.0, "Uniform", ha="left", va="center", fontsize=10.5, color=INK2,
        clip_on=False)
width = len(PDES) * BAR + (len(PDES) - 1) * GAP
for m in range(len(METRICS)):
    for p, vals in enumerate(PDES.values()):
        x0 = m - width / 2 + p * (BAR + GAP)
        v = vals[m]
        bar(x0, x0 + BAR, 1.0, v, COLORS[p])
        ax.text(x0 + BAR / 2, v + (0.025 if v >= 1 else -0.025), f"{v:.2f}", ha="center",
                va="bottom" if v >= 1 else "top", fontsize=10, color=INK2, zorder=5)
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(METRICS, fontsize=11.5, color=INK)
ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
ax.set_yticklabels(["0.8×", "1×", "1.2×", "1.4×", "1.6×", "1.8×", "2×"], fontsize=10, color=MUTED)
ax.tick_params(length=0, pad=6)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylabel("× lower error than Uniform", fontsize=10.5, color=INK2, labelpad=8)

# title, then the legend in one row below it (swatch + text in ink)
fig.text(AX[0], 0.935, "OGAS-Loss vs. uniform sampling", fontsize=12.5, color=INK, va="center", weight="bold")
lx = AX[0]
for p, name in enumerate(PDES):
    fig.patches.append(FancyBboxPatch((lx, 0.851), 0.015, 0.034, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fc=COLORS[p], ec="none"))
    t = fig.text(lx + 0.022, 0.868, name, fontsize=10.5, color=INK2, va="center")
    fig.canvas.draw()
    lx += 0.022 + t.get_window_extent().width / W + 0.035
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out)
print("wrote", args.out)
