import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from gifsave import check_gif, save_gif  # noqa: E402

BG, FG, DIM, GRID = "#0d1117", "#c9d1d9", "#8b949e", "#21262d"
GENS_PER_FRAME = 3
REWIND_FRAMES = 16
FPS = 12
HOLD_FIRST, HOLD_LAST = 600, 2200

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("panels", nargs="+", help="CSV:XCOL:YCOL:TITLE[:XLABEL[:YLABEL]]")
ap.add_argument("--dpi", type=int, default=100, help="resolution (100: 345 px per panel)")
args = ap.parse_args()

panels = []
for spec in args.panels:
    csv, xcol, ycol, title, *labels = spec.split(":")
    d = np.genfromtxt(csv, delimiter=",", names=True)
    labels += [xcol.replace("_", " "), ycol.replace("_", " ")][len(labels):]
    panels.append(dict(gen=d["generation"].astype(int), x=d[xcol], y=d[ycol], title=title,
                       xlabel=labels[0], ylabel=labels[1]))
g_max = max(int(p["gen"].max()) for p in panels)

cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["viridis"](np.linspace(0.18, 1.0, 256)))
norm = matplotlib.colors.Normalize(0, g_max)
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": GRID, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})
n = len(panels)
fig = plt.figure(figsize=(3.45 * n + 0.75, 3.9), dpi=args.dpi, facecolor=BG)
left, right, gap = 0.62 / fig.get_figwidth(), 0.93, 0.62 / fig.get_figwidth()
w = (right - left - (n - 1) * gap) / n
for i, p in enumerate(panels):
    ax = fig.add_axes([left + i * (w + gap), 0.14, w, 0.70], facecolor=BG)
    for v, lim in (("x", ax.set_xlim), ("y", ax.set_ylim)):
        lo, hi = p[v].min(), p[v].max()
        lim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    ax.set_xlabel(p["xlabel"], labelpad=2)
    ax.set_ylabel(p["ylabel"], labelpad=2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(p["title"], color=FG, fontsize=10.5, loc="left", pad=5)
    p["old"] = ax.scatter([], [], c=[], s=4, alpha=0.5, edgecolors="none", cmap=cmap, norm=norm)
    p["new"] = ax.scatter([], [], c=[], s=14, edgecolors="white", linewidths=0.5, cmap=cmap, norm=norm)
cax = fig.add_axes([right + 0.012, 0.14, 0.009, 0.70])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("generation", color=FG, labelpad=3)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(color=DIM, labelcolor=DIM, labelsize=8.5)
fig.text(left, 0.935, "OGAS-Loss: parameters sampled per generation (56 concurrent simulations each)",
         color=FG, fontsize=10.5, va="center")
counter = fig.text(right, 0.935, "", color=DIM, fontsize=10.5, ha="right", va="center",
                   family="Helvetica Neue")


def draw(g, highlight=True):
    cut = g - GENS_PER_FRAME + 1 if highlight else g + 1
    for p in panels:
        m_old = p["gen"] < cut
        m_new = (p["gen"] <= g) & ~m_old
        for key, m in (("old", m_old), ("new", m_new)):
            p[key].set_offsets(np.c_[p["x"][m], p["y"][m]])
            p[key].set_array(p["gen"][m])
    n_sims = int((panels[0]["gen"] <= g).sum())
    counter.set_text(f"generation {g:3d} / {g_max}   ·   {n_sims:5,d} simulations per run")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return Image.open(buf).convert("RGB")


forward = list(range(0, g_max + 1, GENS_PER_FRAME))
if forward[-1] != g_max:
    forward.append(g_max)
rgb = [draw(g) for g in forward]
rgb.append(draw(g_max, highlight=False))
rewind = (g_max * (1 - np.arange(1, REWIND_FRAMES + 1) / (REWIND_FRAMES + 1)) ** 2).round().astype(int)
rgb += [draw(int(g), highlight=False) for g in rewind]

durations = [int(1000 / FPS)] * len(rgb)
durations[0], durations[len(forward)] = HOLD_FIRST, HOLD_LAST
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
lut = np.round(cmap(np.linspace(0, 1, 40))[:, :3] * 255)
kept = save_gif(args.out, rgb, durations, colors=254, reference=(0, len(forward) - 1, len(forward)), fixed=lut)
n_frames, err = check_gif(args.out, [rgb[k] for k in kept])
print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB, {n_frames} frames, {rgb[0].size}, "
      f"loop=0, mean abs decode error {err:.2f}")
