"""Step-by-step animation of the OGAS online loop (docs/assets/ogas_pipeline.gif).

    python render_pipeline_gif.py ../assets/ogas_pipeline.gif [--stills DIR]

Solver thumbnails come from ../assets/ks_trajectories.gif; the parameter space and its difficulty landscape
are illustrative.
"""
import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from gifsave import check_gif, save_gif  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--stills", help="also write one PNG per step to this directory")
args = ap.parse_args()

# same look as the paper figures: white ground, Times-like serif, matplotlib tab colors, draw.io accents
BG, PANEL, INNER, BORDER, FAINT = "#ffffff", "#ffffff", "#f7f7f7", "#333333", "#9a9a9a"
FG, DIM = "#000000", "#555555"
UNI, GEN, HOT = "#1f77b4", "#ff7f0e", "#9467bd"           # uniform draw, OGAS draw, difficulty
OGAS_BLUE, STEP_FILL, STEP_EDGE = "#1a7fe0", "#e1d5e7", "#9673a6"
NODE_FILL, NODE_EDGE, EDGE, PULSE = "#dae8fc", "#6c8ebf", "#c3d3ea", "#3f6fb5"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 12, "text.color": FG})

W, H, FPS = 1000, 450, 12
STEPS = [
    ("Solvers run simulations from uniform parameters",
     "each point λ is one simulation: a physical coefficient and an initial condition"),
    ("They stream every time step to the surrogate",
     "the surrogate trains on the fly; nothing is stored on disk"),
    ("The training loss shows which simulations are hard",
     "a bright halo marks a simulation that the surrogate predicts badly"),
    ("A diffusion model learns where the loss is high",
     "it trains next to the surrogate, on (parameters, loss) pairs"),
    ("It generates the next parameters in hard regions",
     "70 % come from the diffusion model, 30 % stay uniform to cover the whole space"),
    ("Everything runs in parallel, in a closed loop",
     "same simulation budget, much lower worst-case error"),
]
T = [0.0, 4.4, 8.8, 13.2, 17.6, 22.0, 29.0]
ACTIVE = [{"A", "B"}, {"B", "C"}, {"C", "A"}, {"C", "D"}, {"D", "A"}, {"A", "B", "C", "D"}]
FADE_S = 0.6

PANELS = {"A": (40, 118, 280, 358), "B": (340, 118, 510, 358), "C": (570, 118, 740, 358), "D": (800, 118, 970, 358)}
TITLES = {"A": ("Parameter space $\\Lambda$", ""), "B": ("Solvers $S(\\lambda)$", " (CPU)"),
          "C": ("Deep Surrogate", " (GPU)"), "D": ("OGAS", " (GPU)")}
AI = (50, 128, 270, 348)          # unit square of the parameter space inside A
DM = (825, 150, 945, 270)         # its twin inside D
MID = 238                         # height of the arrows between panels
LANES_Y = [146 + 46 * i for i in range(5)]
TH_X, TH = 352, 34                # lane thumbnail
BAR_X0, BAR_X1 = 396, 498         # lane progress bar
NET_X, NET_N = [604, 637, 670, 703], [3, 5, 5, 3]
RET_Y, RET_X_A, RET_X_D = 392, 240, 885

rng = np.random.default_rng(5)


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def hard(x, y):
    """Illustrative difficulty landscape on the unit square, in [0, 1]."""
    a = 0.7
    u, v = (x - 0.70) * np.cos(a) + (y - 0.66) * np.sin(a), -(x - 0.70) * np.sin(a) + (y - 0.66) * np.cos(a)
    g = np.exp(-u ** 2 / (2 * 0.16 ** 2) - v ** 2 / (2 * 0.075 ** 2))
    g += 0.6 * np.exp(-((x - 0.24) ** 2 + (y - 0.27) ** 2) / (2 * 0.065 ** 2))
    return np.clip(g, 0.0, 1.0)


def sample_hard(n):
    out = []
    while len(out) < n:
        x, y = rng.random(2)
        if rng.random() < hard(x, y) ** 3:
            out.append((x, y))
    return np.array(out)


def to_a(x, y):
    return AI[0] + x * (AI[2] - AI[0]), AI[3] - y * (AI[3] - AI[1])


def to_d(x, y):
    return DM[0] + x * (DM[2] - DM[0]), DM[3] - y * (DM[3] - DM[1])


class Path:
    def __init__(self, pts):
        self.p = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)

    def at(self, u):
        u = float(np.clip(u, 0, 1))
        return np.array([np.interp(u, self.s, self.p[:, 0]), np.interp(u, self.s, self.p[:, 1])])


def bezier(p0, p1, p2, n=24):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * np.array(p0) + 2 * (1 - s) * s * np.array(p1) + s ** 2 * np.array(p2)


# solver thumbnails: the three Kuramoto-Sivashinsky domain sizes of ks_trajectories.gif
gif = Image.open(os.path.join(HERE, "..", "assets", "ks_trajectories.gif"))
canvas, thumbs = None, [[], [], []]
for i in range(gif.n_frames):
    gif.seek(i)
    fr = gif.convert("RGBA")
    canvas = fr if canvas is None else Image.alpha_composite(canvas, fr)
    for j, x0 in enumerate((14, 218, 422)):
        thumbs[j].append(np.asarray(canvas.crop((x0, 26, x0 + 192, 218)).convert("RGB").resize((TH, TH), Image.LANCZOS)))

# ---------------------------------------------------------------- simulate the loop, frame by frame
n_frames = int(round(T[-1] * FPS))
times = np.arange(n_frames) / FPS
step_of = np.searchsorted(T, times, side="right") - 1

dots = []          # parameter-space points: x, y, kind, t_in (visible in A), t_score, d
queue, lanes = [], [dict(dot=None, t0=0.0, dur=1.0) for _ in LANES_Y]
pending = []       # (time, x, y, kind): generated parameters still travelling back to the parameter space
lane_state = []    # per frame and lane: (progress, thumbnail, kind) or None
caught_up = False
packets = []       # moving glyphs: kind, t0, t1, path, color, extra
batches = []       # OGAS generations: t0, targets, starts
pulses, outs = [], []
spawn_t = [0.35 + 0.27 * i for i in range(12)]
next_spawn = 0.0
gen_t = [T[4] + 0.4, T[5] + 0.3, T[5] + 2.2, T[5] + 4.1]


def add_dot(x, y, kind, t):
    dots.append(dict(x=x, y=y, kind=kind, t_in=t, t_score=None, d=0.0))
    queue.append(len(dots) - 1)


def score(i, t):
    d = dots[i]
    improve = 1.0 - 0.45 * float(ease((t - T[5]) / (T[6] - T[5])))
    d["d"] = float(np.clip(hard(d["x"], d["y"]) * improve + rng.normal(0, 0.06), 0.03, 1.0))
    d["t_score"] = t + 0.25
    outs.append(t + 0.25)
    packets.append(dict(kind="bar", t0=t + 0.25, t1=t + 0.95, path=Path([(712, MID), (DM[0] - 6, MID)]),
                        color=HOT, extra=d["d"]))


for k, t in enumerate(times):
    s = step_of[k]
    while spawn_t and spawn_t[0] <= t:
        spawn_t.pop(0)
        add_dot(*rng.random(2), "uni", t)
    if T[1] <= t < T[4] + 2.4 and len(queue) < 2 and t >= next_spawn:
        add_dot(*rng.random(2), "uni", t)
        next_spawn = t + 0.5
    while gen_t and gen_t[0] <= t:
        t0 = gen_t.pop(0)
        n = 7 if t0 < T[5] else 5
        targets = sample_hard(n)
        batches.append(dict(t0=t0, targets=targets, starts=rng.random((n, 2))))
        pending += [(t0 + 2.45 + 0.06 * j, x, y, "gen") for j, (x, y) in enumerate(targets)]
        pending += [(t0 + 2.45, x, y, "uni") for x, y in rng.random((3 if n == 7 else 2, 2))]
        pending.sort()
    while pending and pending[0][0] <= t:
        _, x, y, kind = pending.pop(0)
        add_dot(x, y, kind, t)
    if s >= 2 and not caught_up:
        caught_up = True
        done = [i for i, d in enumerate(dots) if d.get("t_done") is not None]
        for j, i in enumerate(done):
            score(i, t + 0.12 * j)
    for li, ln in enumerate(lanes):
        if ln["dot"] is not None and t >= ln["t0"] + ln["dur"]:
            i = ln["dot"]
            dots[i]["t_done"] = t
            ln["dot"] = None
            if s >= 2:
                score(i, t)
        if ln["dot"] is None and ln.get("t_free", 0) <= t and queue:
            i = queue.pop(0)
            d = dots[i]
            p0 = to_a(d["x"], d["y"])
            p1 = (TH_X - 4, LANES_Y[li])
            packets.append(dict(kind="dot", t0=t, t1=t + 0.5, path=Path(bezier(p0, (300, p0[1]), p1)),
                                color=UNI if d["kind"] == "uni" else GEN, extra=None))
            ln.update(dot=i, t0=t + 0.5, dur=float(rng.uniform(2.3, 3.1)), panel=min(int(d["x"] * 3), 2),
                      next_emit=t + 0.9, t_free=t + 0.5)
        if ln["dot"] is not None and s >= 1 and t >= ln["t0"] and t >= ln["next_emit"]:
            ln["next_emit"] = t + 0.45
            y0 = LANES_Y[li]
            packets.append(dict(kind="sq", t0=t, t1=t + 0.55, path=Path(bezier((BAR_X1 + 4, y0), (540, y0), (NET_X[0] - 8, MID))),
                                color=UNI if dots[ln["dot"]]["kind"] == "uni" else GEN, extra=None))
            pulses.append(t + 0.55)
    lane_state.append([None if ln["dot"] is None or t < ln["t0"] else
                       (min((t - ln["t0"]) / ln["dur"], 1.0), ln["panel"], dots[ln["dot"]]["kind"]) for ln in lanes])

# ---------------------------------------------------------------- static drawing
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def rbox(rect, fc=PANEL, ec=BORDER, r=8, lw=1.0, z=0, **kw):
    x0, y0, x1, y1 = rect
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, **kw)
    ax.add_patch(p)
    return p


def arrow(pts, color=FAINT, lw=1.3, z=1):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
    d = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    n = np.array([-d[1], d[0]])
    ax.add_patch(plt.Polygon([pts[-1], pts[-1] - 8 * d + 4 * n, pts[-1] - 8 * d - 4 * n], color=color, zorder=z))


for key, rect in PANELS.items():
    if key == "D":  # the OGAS box of the paper's Figure 1
        rbox(rect, ec=OGAS_BLUE, lw=1.6, ls=(0, (4, 3)))
    else:
        rbox(rect, lw=1.1)
    name, hw = TITLES[key]
    t_ = ax.text(rect[0] + 2, rect[1] - 13, name, fontsize=14, weight="bold", va="center", color=FG)
    fig.canvas.draw()
    ax.text(rect[0] + 2 + t_.get_window_extent().width, rect[1] - 13, hw, fontsize=13, va="center", color=DIM)
# parameter space
rbox(AI, fc=INNER, ec="none", r=4, z=1)
ax.text(AI[0], PANELS["A"][3] + 13, "physical coefficient →", fontsize=10.5, color=DIM, va="center")
ax.text(PANELS["A"][0] - 13, AI[3], "initial condition →", fontsize=10.5, color=DIM, va="bottom", ha="center",
        rotation=90)
# solvers
for y in LANES_Y:
    rbox((TH_X, y - TH / 2, TH_X + TH, y + TH / 2), fc=INNER, ec=FAINT, r=3, lw=0.8, z=2)
    rbox((BAR_X0, y - 4, BAR_X1, y + 4), fc="#eeeeee", ec="none", r=4, z=2)
# surrogate network
nodes = [[(x, MID + (j - (n - 1) / 2) * 24) for j in range(n)] for x, n in zip(NET_X, NET_N)]
edges = []
for la, lb in zip(nodes[:-1], nodes[1:]):
    for a in la:
        for b in lb:
            edges.append(ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.9, color=EDGE, zorder=2)[0])
node_art = [ax.add_patch(Circle(p, 5, fc=NODE_FILL, ec=NODE_EDGE, lw=1.1, zorder=3)) for layer in nodes for p in layer]
ax.text((PANELS["C"][0] + PANELS["C"][2]) / 2, PANELS["C"][3] - 22, "learns to predict the next step",
        fontsize=10.5, color=DIM, ha="center", va="center")
# generator
rbox(DM, fc=INNER, ec="none", r=4, z=1)
ax.text((DM[0] + DM[2]) / 2, DM[3] + 22, "diffusion model:", fontsize=10.5, color=DIM, ha="center", va="center")
ax.text((DM[0] + DM[2]) / 2, DM[3] + 40, "where is the loss high?", fontsize=10.5, color=DIM, ha="center",
        va="center")
gx, gy = np.meshgrid(np.linspace(0, 1, 96), np.linspace(1, 0, 96))
field = hard(gx, gy)
field = field / field.max()
glow_rgba = np.zeros((96, 96, 4))
glow_rgba[..., :3] = matplotlib.colors.to_rgb(HOT)
glow_rgba[..., 3] = 0.75 * field ** 1.4
glow = ax.imshow(glow_rgba, extent=(DM[0], DM[2], DM[3], DM[1]), zorder=2, interpolation="bilinear", alpha=0)
# arrows between panels and the return path
arrow([(PANELS["A"][2] + 6, MID), (PANELS["B"][0] - 6, MID)])
arrow([(PANELS["B"][2] + 6, MID), (PANELS["C"][0] - 6, MID)])
arrow([(PANELS["C"][2] + 6, MID), (PANELS["D"][0] - 6, MID)])
ax.text((PANELS["B"][2] + PANELS["C"][0]) / 2, MID - 14, "data", fontsize=10.5, color=DIM, ha="center")
ax.text((PANELS["C"][2] + PANELS["D"][0]) / 2, MID - 14, "loss", fontsize=10.5, color=DIM, ha="center")
RET = [(RET_X_D, PANELS["D"][3] + 4), (RET_X_D, RET_Y), (RET_X_A, RET_Y), (RET_X_A, PANELS["A"][3] + 6)]
arrow(RET)
ax.text((RET_X_A + RET_X_D) / 2, RET_Y - 9, "next parameters", fontsize=10.5, color=DIM, ha="center")
# legend
lx = 40
for kind, label in (("uni", "uniform draw"), ("gen", "generated by OGAS"), ("hot", "hard simulation (high loss)")):
    if kind == "hot":
        ax.add_patch(Circle((lx + 5, 428), 10, fc=HOT, ec="none", alpha=0.45, zorder=2))
        ax.add_patch(Circle((lx + 5, 428), 4.5, fc=UNI, ec=BG, lw=1.2, zorder=3))
    else:
        ax.add_patch(Circle((lx + 5, 428), 4.5, fc=UNI if kind == "uni" else GEN, ec=BG, lw=1.2, zorder=3))
    t = ax.text(lx + 18, 428, label, fontsize=11, color=DIM, va="center")
    fig.canvas.draw()
    lx += 18 + t.get_window_extent().width + 34

# ---------------------------------------------------------------- dynamic artists
dim = {k: ax.add_patch(FancyBboxPatch((r[0] - 6, r[1] - 26), r[2] - r[0] + 12, r[3] - r[1] + 32,
                                      boxstyle="round,pad=0,rounding_size=8", fc=BG, ec="none", alpha=0, zorder=20))
       for k, r in PANELS.items()}
halo_sc = ax.scatter([], [], edgecolors="none", zorder=4)
dot_sc = ax.scatter([], [], edgecolors="white", linewidths=0.9, zorder=5)
lane_img = [ax.imshow(np.zeros((TH, TH, 3), np.uint8), extent=(TH_X, TH_X + TH, y + TH / 2, y - TH / 2), zorder=3)
            for y in LANES_Y]
lane_bar = [ax.add_patch(Rectangle((BAR_X0, y - 4), 0, 8, fc=UNI, ec="none", zorder=3)) for y in LANES_Y]
gen_sc = ax.scatter([], [], edgecolors="white", linewidths=0.8, zorder=6)
pk_dot = ax.scatter([], [], edgecolors="white", linewidths=0.8, zorder=30)
pk_sq = ax.scatter([], [], marker="s", edgecolors="none", zorder=30)
bar_pool = [ax.add_patch(Rectangle((0, 0), 0, 0, fc=HOT, ec="none", zorder=30)) for _ in range(12)]
badge = ax.add_patch(Circle((58, 46), 17, fc=STEP_FILL, ec=STEP_EDGE, lw=1.5, zorder=40))
badge_txt = ax.text(58, 47, "", fontsize=17, weight="bold", color=FG, ha="center", va="center", zorder=41)
cap = ax.text(88, 36, "", fontsize=19, weight="bold", color=FG, va="center", zorder=41)
sub = ax.text(88, 62, "", fontsize=14, color=DIM, va="center", style="italic", zorder=41)
prog = [ax.add_patch(Circle((842 + 24 * i, 46), 6, fc="none", ec=FAINT, lw=1.5, zorder=41)) for i in range(len(STEPS))]


def mix(c1, c2, a):
    c1, c2 = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple(c1 + (c2 - c1) * float(np.clip(a, 0, 1)))


def render(k):
    t, s = times[k], int(step_of[k])
    # captions and step indicator
    u_in, u_out = (t - T[s]) / 0.35, (T[s + 1] - t) / 0.25
    a = float(np.clip(min(u_in, u_out if s < len(STEPS) - 1 else 9), 0, 1))
    cap.set_text(STEPS[s][0])
    sub.set_text(STEPS[s][1])
    cap.set_alpha(a)
    sub.set_alpha(float(np.clip((t - T[s] - 0.3) / 0.4, 0, 1)) * a)
    badge_txt.set_text(str(s + 1))
    for i, c in enumerate(prog):
        c.set_facecolor(STEP_EDGE if i == s else (STEP_FILL if i < s else "none"))
        c.set_edgecolor(STEP_EDGE if i <= s else FAINT)
    # focus: dim the panels that are not part of this step
    for key, p in dim.items():
        now = key in ACTIVE[s]
        prev = key in ACTIVE[s - 1] if s > 0 else now
        u = float(ease((t - T[s]) / 0.4))
        p.set_alpha(0.7 * ((1 - now) * u + (1 - prev) * (1 - u)))
    # parameter space
    xy, cols, hxy, hs, hc = [], [], [], [], []
    for d in dots:
        if d["t_in"] > t:
            continue
        p = to_a(d["x"], d["y"])
        pop = float(ease((t - d["t_in"]) / 0.25))
        xy.append(p)
        cols.append(UNI if d["kind"] == "uni" else GEN)
        if d["t_score"] is not None and t >= d["t_score"]:
            hv = float(ease((t - d["t_score"]) / 0.5))
            hxy.append(p)
            hs.append((7 + 15 * d["d"]) ** 2 * hv)
            hc.append(matplotlib.colors.to_rgba(HOT, (0.08 + 0.5 * d["d"]) * hv))
        cols[-1] = matplotlib.colors.to_rgba(cols[-1], pop)
    dot_sc.set_offsets(np.array(xy) if xy else np.empty((0, 2)))
    dot_sc.set_facecolors(cols if cols else [(0, 0, 0, 0)])
    dot_sc.set_sizes([36] * len(xy) if xy else [0])
    halo_sc.set_offsets(np.array(hxy) if hxy else np.empty((0, 2)))
    halo_sc.set_sizes(hs if hs else [0])
    halo_sc.set_facecolors(hc if hc else [(0, 0, 0, 0)])
    # solvers
    for li, (img, bar) in enumerate(zip(lane_img, lane_bar)):
        st = lane_state[k][li]
        if st is None:
            img.set_visible(False)
            bar.set_width(0)
            continue
        prog_, panel, kind = st
        fi = int(prog_ * (len(thumbs[panel]) - 1)) // 2 * 2  # the fields change at half the frame rate
        img.set_data(thumbs[panel][fi])
        img.set_visible(True)
        bar.set_width((BAR_X1 - BAR_X0) * prog_)
        bar.set_facecolor(UNI if kind == "uni" else GEN)
    # surrogate activity
    hit = max([p for p in pulses if p <= t], default=-9.0)
    fl = max(0.0, 1.0 - (t - hit) / 0.35)
    for e in edges:
        e.set_color(mix(EDGE, PULSE, 0.8 * fl))
    out = max([o for o in outs if o <= t], default=-9.0)
    fo = max(0.0, 1.0 - (t - out) / 0.4)
    for i, c in enumerate(node_art):
        out_node = i >= len(node_art) - NET_N[-1]
        c.set_facecolor(mix(NODE_FILL, HOT, fo) if out_node else mix(NODE_FILL, PULSE, 0.6 * fl))
    # generator: learned map, then generation by denoising
    glow.set_alpha(float(ease((t - T[3] - 0.4) / 2.6)))
    gxy, gcol = [], []
    pxy, pcol, psz = [], [], []
    for b in batches:
        u = (t - b["t0"]) / 1.1
        if 0 <= u < 1:
            noise = 0.18 * (1 - ease(u)) * np.sin(7 * u + np.arange(len(b["targets"]))[:, None] * 1.7)
            pos = b["starts"] + (b["targets"] - b["starts"]) * ease(u) + noise
            gxy += [to_d(*p) for p in np.clip(pos, 0, 1)]
            gcol += [mix(DIM, GEN, ease(u))] * len(pos)
        elif 1 <= u < 1.25:
            gxy += [to_d(*p) for p in b["targets"]]
            gcol += [GEN] * len(b["targets"])
        for j, (x, y) in enumerate(b["targets"]):
            v = (t - b["t0"] - 1.25 - 0.06 * j) / 1.2
            if 0 <= v < 1:
                path = Path([to_d(x, y), (RET_X_D, PANELS["D"][3] + 4)] + RET[1:-1] + [(RET_X_A, PANELS["A"][3]), to_a(x, y)])
                pxy.append(path.at(ease(v)))
                pcol.append(GEN)
                psz.append(40)
    gen_sc.set_offsets(np.array(gxy) if gxy else np.empty((0, 2)))
    gen_sc.set_facecolors(gcol if gcol else [(0, 0, 0, 0)])
    gen_sc.set_sizes([34] * len(gxy) if gxy else [0])
    # moving packets
    sq, sqc = [], []
    bars = []
    for p in packets:
        if not p["t0"] <= t < p["t1"]:
            continue
        pos = p["path"].at(ease((t - p["t0"]) / (p["t1"] - p["t0"])))
        if p["kind"] == "dot":
            pxy.append(pos)
            pcol.append(p["color"])
            psz.append(40)
        elif p["kind"] == "sq":
            sq.append(pos)
            sqc.append(p["color"])
        else:
            bars.append((pos, p["extra"]))
    pk_dot.set_offsets(np.array(pxy) if pxy else np.empty((0, 2)))
    pk_dot.set_facecolors(pcol if pcol else [(0, 0, 0, 0)])
    pk_dot.set_sizes(psz if psz else [0])
    pk_sq.set_offsets(np.array(sq) if sq else np.empty((0, 2)))
    pk_sq.set_facecolors(sqc if sqc else [(0, 0, 0, 0)])
    pk_sq.set_sizes([30] * len(sq) if sq else [0])
    for j, r in enumerate(bar_pool):
        if j < len(bars):
            (x, y), d = bars[j]
            h = 6 + 22 * d
            r.set_bounds(x - 3, y - h / 2, 6, h)
        else:
            r.set_bounds(0, 0, 0, 0)
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
    durations[int(round(T[i + 1] * FPS)) - 1] += 400  # short pause at the end of each step
if args.stills:
    os.makedirs(args.stills, exist_ok=True)
    for i in range(len(STEPS)):
        rgb[int(round(T[i + 1] * FPS)) - 2].save(os.path.join(args.stills, f"step{i + 1}.png"))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
kept = save_gif(args.out, rgb, durations, colors=254, reference=range(0, len(rgb), max(1, len(rgb) // 10)))
n_saved, err = check_gif(args.out, [rgb[k] for k in kept])
print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB, {n_saved} frames, {rgb[0].size}, "
      f"{sum(durations) / 1000:.1f} s, mean abs decode error {err:.2f}")
