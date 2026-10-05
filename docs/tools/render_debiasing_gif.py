"""Step-by-step animation of the density-ratio debiasing of the OGAS generator (docs/assets/ogas_debiasing.gif).

    python render_debiasing_gif.py ../assets/ogas_debiasing.gif [--stills DIR]

Synthetic 2D data: the over-sampled region is a crescent, the weights are the exact ratio uniform / history density.
"""
import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from gifsave import check_gif, save_gif  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--stills", help="also write one PNG per step to this directory")
args = ap.parse_args()

# same look as render_pipeline_gif.py and the paper figures
BG, INNER, BORDER, FAINT = "#ffffff", "#f7f7f7", "#333333", "#9a9a9a"
FG, DIM = "#000000", "#555555"
UNI, GEN = "#1f77b4", "#ff7f0e"
STEP_FILL, STEP_EDGE = "#e1d5e7", "#9673a6"
NODE_FILL, NODE_EDGE, EDGE = "#dae8fc", "#6c8ebf", "#c3d3ea"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 12, "text.color": FG})

W, H, FPS = 1000, 420, 12
STEPS = [
    ("OGAS learns from the simulations it chose itself",
     "its history piles up where it already sampled (orange), on top of the uniform draws (blue)"),
    ("Trained as is, the generator copies this bias",
     "it keeps proposing the over-sampled region, even once it is no longer hard"),
    ("A classifier compares the history with uniform draws",
     "it gives each point a weight $w(\\lambda) \\approx$ uniform density / history density (dot size)"),
    ("With these weights, the history looks uniform again",
     "the generator now follows the difficulty only, not its own past choices"),
]
T = [0.0, 4.4, 8.8, 13.2, 18.0]
ACTIVE = [{"L"}, {"L", "R"}, {"L", "M"}, {"M", "R"}]
FADE_S = 0.6

PANELS = {"L": (60, 118, 330, 388), "M": (390, 118, 610, 388), "R": (670, 118, 940, 388)}
TITLES = {"L": "Training history of the generator", "M": "Density-ratio classifier",
          "R": "What the generator learns"}
PAD = 10
rng = np.random.default_rng(3)


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def crescent(n):
    out = []
    while len(out) < n:
        p = rng.random(2)
        if np.sum((p - (0.45, 0.5)) ** 2) < 0.32 ** 2 and np.sum((p - (0.6, 0.56)) ** 2) > 0.25 ** 2:
            out.append(p)
    return np.array(out)


# history: 6 generations of 45 simulations, the OGAS share ramps up to 70 %
pts, kind, born = [], [], []
for g, share in enumerate([0.0, 0.25, 0.5, 0.7, 0.7, 0.7]):
    n_gen = int(round(45 * share))
    pts += list(rng.random((45 - n_gen, 2))) + list(crescent(n_gen))
    kind += [0] * (45 - n_gen) + [1] * n_gen
    born += list(0.3 + g * 0.6 + rng.uniform(0, 0.55, 45))
pts, kind, born = np.array(pts), np.array(kind), np.array(born)

n_grid = 90
gx, gy = np.meshgrid((np.arange(n_grid) + 0.5) / n_grid, (np.arange(n_grid) + 0.5) / n_grid)
grid = np.c_[gx.ravel(), gy.ravel()]


def kde(at, w, bw=0.07):
    d2 = ((at[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / (2 * bw ** 2)) * w[None, :]
    return k.sum(1) / w.sum()


dens_pts = kde(pts, np.ones(len(pts)))
weight = np.clip(dens_pts.mean() / dens_pts, 0.1, 10.0)        # uniform / history density
plain = kde(grid, np.ones(len(pts))).reshape(n_grid, n_grid)
weighted = kde(grid, weight).reshape(n_grid, n_grid)
vmax = plain.max()
size_w = np.clip(np.sqrt(weight), 0.35, 2.2)

n_frames = int(round(T[-1] * FPS))
times = np.arange(n_frames) / FPS
step_of = np.searchsorted(T, times, side="right") - 1

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def rbox(rect, fc=BG, ec=BORDER, r=8, lw=1.1, z=0, **kw):
    x0, y0, x1, y1 = rect
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, **kw)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=FAINT, z=1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.3, zorder=z)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]])
    ax.add_patch(plt.Polygon([p1, p1 - 8 * d + 4 * n, p1 - 8 * d - 4 * n], color=color, zorder=z))


def square(key):
    x0, y0, x1, y1 = PANELS[key]
    return x0 + PAD, y0 + PAD, x1 - PAD, y1 - PAD


for key, rect in PANELS.items():
    rbox(rect)
    ax.text(rect[0] + 2, rect[1] - 13, TITLES[key], fontsize=14, weight="bold", va="center")
SL, SR = square("L"), square("R")
rbox(SL, fc=INNER, ec="none", r=4, z=1)
rbox(SR, fc=INNER, ec="none", r=4, z=1)
arrow((PANELS["L"][2] + 6, 253), (PANELS["M"][0] - 6, 253))
arrow((PANELS["M"][2] + 6, 253), (PANELS["R"][0] - 6, 253))

# classifier: buffer points (y = 1) against uniform draws (y = 0)
mx = (PANELS["M"][0] + PANELS["M"][2]) / 2
layers = [[(mx - 45, 185 + 22 * j) for j in range(3)], [(mx, 174 + 22 * j) for j in range(4)], [(mx + 45, 207)]]
cls_art = []
for la, lb in zip(layers[:-1], layers[1:]):
    for a in la:
        for b in lb:
            cls_art.append(ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.9, color=EDGE, zorder=2)[0])
for layer in layers:
    for p in layer:
        cls_art.append(ax.add_patch(Circle(p, 5, fc=NODE_FILL, ec=NODE_EDGE, lw=1.1, zorder=3)))
cls_art.append(ax.text(mx - 62, 160, "history  vs.  uniform", fontsize=11.5, color=DIM, va="center"))
cls_art.append(ax.text(mx + 56, 207, "$D_\\psi(\\lambda)$", fontsize=13, va="center"))
cls_art.append(ax.text(mx, 300, "$w(\\lambda) = \\dfrac{D_\\psi(\\lambda)}{1 - D_\\psi(\\lambda)}$", fontsize=16,
                       ha="center", va="center"))
cls_art.append(ax.text(mx, 352, "large where the history is sparse,\nsmall where it is dense", fontsize=11, color=DIM,
                       ha="center", va="center", linespacing=1.3))

# legend
lx = 60
for col, label in ((UNI, "uniform draw"), (GEN, "generated by OGAS")):
    ax.add_patch(Circle((lx + 5, 404), 4.5, fc=col, ec="white", lw=0.8, zorder=3))
    t = ax.text(lx + 16, 404, label, fontsize=11.5, color=DIM, va="center")
    fig.canvas.draw()
    lx += 16 + t.get_window_extent().width + 30

# dynamic artists
dot_sc = ax.scatter([], [], edgecolors="white", linewidths=0.6, zorder=5)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gen", [INNER, "#fdd0a2", GEN, "#a63603"])
img = ax.imshow(np.zeros((n_grid, n_grid)), extent=(SR[0], SR[2], SR[3], SR[1]), origin="lower", cmap=cmap,
                vmin=0, vmax=vmax, interpolation="bilinear", zorder=2, alpha=0)
r_label = ax.text((PANELS["R"][0] + PANELS["R"][2]) / 2, 404, "", fontsize=12.5, style="italic", ha="center",
                  va="center", zorder=6)
dim = {k: ax.add_patch(FancyBboxPatch((r[0] - 6, r[1] - 26), r[2] - r[0] + 12, r[3] - r[1] + 32,
                                      boxstyle="round,pad=0,rounding_size=8", fc=BG, ec="none", alpha=0, zorder=20))
       for k, r in PANELS.items()}
badge = ax.add_patch(Circle((58, 46), 17, fc=STEP_FILL, ec=STEP_EDGE, lw=1.5, zorder=40))
badge_txt = ax.text(58, 47, "", fontsize=17, weight="bold", ha="center", va="center", zorder=41)
cap = ax.text(88, 36, "", fontsize=19, weight="bold", va="center", zorder=41)
sub = ax.text(88, 62, "", fontsize=14, color=DIM, va="center", style="italic", zorder=41)
prog = [ax.add_patch(Circle((890 + 24 * i, 46), 6, fc="none", ec=FAINT, lw=1.5, zorder=41)) for i in range(len(STEPS))]


def to_l(p):
    return np.c_[SL[0] + p[:, 0] * (SL[2] - SL[0]), SL[3] - p[:, 1] * (SL[3] - SL[1])]


def render(k):
    t, s = times[k], int(step_of[k])
    a = float(np.clip(min((t - T[s]) / 0.35, (T[s + 1] - t) / 0.25 if s < len(STEPS) - 1 else 9), 0, 1))
    cap.set_text(STEPS[s][0])
    sub.set_text(STEPS[s][1])
    cap.set_alpha(a)
    sub.set_alpha(float(np.clip((t - T[s] - 0.3) / 0.4, 0, 1)) * a)
    badge_txt.set_text(str(s + 1))
    for i, c in enumerate(prog):
        c.set_facecolor(STEP_EDGE if i == s else (STEP_FILL if i < s else "none"))
        c.set_edgecolor(STEP_EDGE if i <= s else FAINT)
    u = float(ease((t - T[s]) / 0.4))
    for key, p in dim.items():
        now = key in ACTIVE[s]
        prev = key in ACTIVE[s - 1] if s > 0 else now
        p.set_alpha(0.7 * ((1 - now) * u + (1 - prev) * (1 - u)))
    for art in cls_art:
        art.set_alpha(float(ease((t - T[2]) / 0.6)))
    # history points, resized by their weight from step 3 on
    vis = born <= t
    pop = ease((t - born[vis]) / 0.25)
    grow = float(ease((t - T[2] - 0.8) / 1.6))
    sizes = 26 * (1 + (size_w[vis] ** 2 - 1) * grow) * pop
    dot_sc.set_offsets(to_l(pts[vis]) if vis.any() else np.empty((0, 2)))
    dot_sc.set_sizes(sizes if vis.any() else [0])
    dot_sc.set_facecolors([UNI if c == 0 else GEN for c in kind[vis]] if vis.any() else [(0, 0, 0, 0)])
    # learned density: plain, then reweighted
    blend = float(ease((t - T[3] - 0.4) / 1.6))
    img.set_data(plain * (1 - blend) + weighted * blend)
    img.set_alpha(float(ease((t - T[1] - 0.3) / 1.2)))
    if t < T[1] + 0.8:
        r_label.set_text("")
        r_label.set_visible(False)
    else:
        r_label.set_visible(True)
        r_label.set_text("without weights: the past dominates" if blend < 0.5 else "with weights: ≈ uniform prior")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return Image.open(buf).convert("RGB")


rgb = [render(k) for k in range(n_frames)]
last, first = np.asarray(rgb[-1], float), np.asarray(rgb[0], float)
n_fade = int(round(FADE_S * FPS))
for a in np.arange(1, n_fade + 1) / (n_fade + 1):
    rgb.append(Image.fromarray(np.round((1 - a) * last + a * first).astype(np.uint8)))
durations = [int(round(1000 / FPS))] * len(rgb)
for i in range(len(STEPS)):
    durations[int(round(T[i + 1] * FPS)) - 1] += 600
if args.stills:
    os.makedirs(args.stills, exist_ok=True)
    for i in range(len(STEPS)):
        rgb[int(round(T[i + 1] * FPS)) - 2].save(os.path.join(args.stills, f"step{i + 1}.png"))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
kept = save_gif(args.out, rgb, durations, colors=254, reference=range(0, len(rgb), max(1, len(rgb) // 10)))
n_saved, err = check_gif(args.out, [rgb[k] for k in kept])
print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB, {n_saved} frames, {rgb[0].size}, "
      f"{sum(durations) / 1000:.1f} s, mean abs decode error {err:.2f}")
