import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from gifsave import check_gif, save_gif  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("csv")
ap.add_argument("npz")
ap.add_argument("--stills", help="also write a few PNG stills to this directory")
args = ap.parse_args()

BG, PANEL, INNER, BORDER, FAINT = "#0d1117", "#161b22", "#10151d", "#30363d", "#484f58"
FG, DIM, HL = "#c9d1d9", "#8b949e", "#e6edf3"
RUN, FIN, WAIT = "#58a6ff", "#f85149", "#6e7681"
GEN, UNI, SIG, TEAL = "#f0883e", "#3fb950", "#bc8cff", "#39c5cf"
W, H, FPS = 1000, 392, 12
N_CLIENTS, N_BUDGET, G_MAX, STAGED = 56, 10_000, 178, 112
RAMP = np.linspace(0.0, 0.7, 4)
T_SIM = 4.5
T_WARM, T_NORMAL, T_FF, T_HOLD = 4.5, 13.5, 18.5, 20.0
FADE_S = 0.5
WATERMARK = 1500 / 4163
N_REV = 5
rng = np.random.default_rng(3)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "text.color": FG,
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
                     "mathtext.it": "Helvetica:italic", "mathtext.bf": "Helvetica:bold"})

n_frames = int(round(T_HOLD * FPS))
times = np.arange(n_frames) / FPS
rate = np.select([times < T_WARM, times < T_NORMAL, times < T_FF],
                 [2.2, 1.0, np.minimum(1.0 + 3.0 * (times - T_NORMAL) / 1.5, 4.0)], 0.0)
is_ff = (times >= T_NORMAL) & (times < T_FF)


def share(g):
    return 0.0 if g <= 0 else float(RAMP[min(g - 1, len(RAMP) - 1)])


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def reflect(x):
    x = np.mod(x, 2.0)
    return 1.0 - np.abs(x - 1.0)


run = np.genfromtxt(args.csv, delimiter=",", names=True)
run_gen = run["generation"].astype(int)
LX = (run["domain_extent"] - 10) / 120
LY = (run["cutoff"] - 2) / 6
ks = np.load(args.npz)
thumbs = []
for i in ks["idx"]:
    tr = ks[f"sim{int(i)}"][:, 0].astype(np.float32)
    for t in range(4, tr.shape[0], 3):
        f = np.asarray(Image.fromarray(tr[t, :96, :96], mode="F").resize((40, 40), Image.Resampling.BILINEAR))
        thumbs.append((f - f.mean()) / (f.std() + 1e-6))


def dense_subset(g, share_):
    idx = np.flatnonzero(run_gen == min(g, G_MAX))
    nb = int(round(share_ * len(idx)))
    if nb == 0:
        return np.empty((0, 2))
    xy = np.c_[LX[idx], LY[idx]]
    dens = np.exp(-((xy[:, None] - xy[None]) ** 2).sum(-1) / (2 * 0.08 ** 2)).sum(1)
    return xy[np.argsort(-dens)[:nb]]


speed = rng.uniform(0.75, 1.25, N_CLIENTS) / T_SIM
prog = rng.uniform(0.0, 0.3, N_CLIENTS)
src = np.zeros(N_CLIENTS, int)
queue = [0] * STAGED
finished = []
n_launched, ff_n0, since, n_res = N_CLIENTS, None, 0, 0
pending = []
resamplings = []
cl, n_disp = [], np.zeros(n_frames, int)
last_new = np.full(N_CLIENTS, -9.0)
for k, t in enumerate(times):
    prog += speed * rate[k] / FPS
    if is_ff[k] or t >= T_FF:
        ff_n0 = n_launched if ff_n0 is None else ff_n0
        u = min((t - T_NORMAL) / (T_FF - T_NORMAL), 1.0)
        n_disp[k] = int(round(ff_n0 + (N_BUDGET - ff_n0) * (1 - (1 - u) ** 2)))
    for p in [p for p in pending if p[0] <= t]:
        pending.remove(p)
        queue = list(p[1])
    for i in np.flatnonzero(prog >= 1.0):
        prog[i] -= 1.0
        finished.append(t)
        src[i] = queue.pop(0) if queue else 0
        last_new[i] = t
        n_launched += 1
        since += 1
        if since >= N_CLIENTS and t < T_FF:
            since = 0
            n_res += 1
            g = n_res if not is_ff[k] else max(int(n_disp[k]) // N_CLIENTS, n_res)
            fast = bool(is_ff[k])
            s_dur, lead = (0.3, 0.5) if fast else (1.8, 1.0)
            sh = share(n_res)
            srcs = list((rng.random(STAGED) < sh).astype(int))
            resamplings.append(dict(t0=t, t1=t + s_dur, t2=t + s_dur + lead, g=g, share=sh, fast=fast,
                                    pts=dense_subset(g, sh), srcs=srcs))
            pending.append((t + s_dur + lead, srcs))
    if not (is_ff[k] or t >= T_FF):
        n_disp[k] = n_launched
    cl.append((prog.copy(), src.copy(), last_new.copy(), list(queue), [f for f in finished if t - f < 6]))
n_res_at = np.array([sum(r_["t0"] <= t for r_ in resamplings) for t in times])
gen_disp = np.where(is_ff | (times >= T_FF), np.maximum(n_res_at, (n_disp - 1) // N_CLIENTS), n_res_at)
gen_disp = np.minimum(gen_disp, G_MAX)
for j, r_ in enumerate(resamplings):
    r_["t_end"] = resamplings[j + 1]["t1"] if j + 1 < len(resamplings) else np.inf

SO, RS, DS, OG = (14, 44, 174, 330), (284, 44, 364, 330), (426, 44, 726, 330), (646, 44, 986, 330)
CW, CH, CG, N_SHOW = 54, 16, 4, 10
cell_xy = [(64, 88 + i * (CH + CG)) for i in range(N_SHOW)]
CHIP_W, CHIP_H, CHIP_DY, N_CHIPS = 36, 8, 12, 17
FIN_X, WAIT_X = 18, 128
SW, SH_, SG, S_COLS, S_ROWS = 26, 15, 5, 1, 11
slot_xy = [((RS[0] + RS[2] - SW) / 2, 76 + s * (SH_ + SG)) for s in range(S_ROWS)]
N_SLOTS = len(slot_xy)
SUR_X, SUR_N, SUR_Y = [540, 572, 604, 636], [3, 5, 5, 3], 176
IN_X, OUT_X, TH = 452, 672, 40
BN, BX0, BY0, BW, BH = 10, 664, 92, 10, 22
KP = (852, 90, 970, 122)
DD_X, DD_N, DD_Y = [772, 796, 820, 844], [3, 5, 5, 2], 196
PG = (870, 150, 976, 256)
UB = (690, 246, 716, 272)
MIX = (815, 300)


class Path:
    def __init__(self, pts):
        self.p = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)

    def at(self, u):
        u = float(np.clip(u, 0, 1))
        return np.array([np.interp(u, self.s, self.p[:, 0]), np.interp(u, self.s, self.p[:, 1])])


def bezier(p0, p1, p2, n=20):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * np.array(p0) + 2 * (1 - s) * s * np.array(p1) + s ** 2 * np.array(p2)


def pg_xy(xy):
    xy = np.atleast_2d(xy)
    return np.c_[PG[0] + 4 + xy[:, 0] * (PG[2] - PG[0] - 8), PG[3] - 4 - xy[:, 1] * (PG[3] - PG[1] - 8)]


def net_nodes(xs, ns, yc, dy):
    return [[(x, yc + (j - (n - 1) / 2) * dy) for j in range(n)] for x, n in zip(xs, ns)]


SUR_NODES = net_nodes(SUR_X, SUR_N, SUR_Y, 16)
DD_NODES = net_nodes(DD_X, DD_N, DD_Y, 12)

from matplotlib.transforms import Affine2D  # noqa: E402

H, Y0 = 404, 44
PLACE = {"OG": (14, 1.08, False), "SO": (397, 1.0, True), "RS": (573, 1.0, False), "DS": (669, 1.0, False)}
RECT = {"OG": OG, "SO": SO, "RS": RS, "DS": DS}


def M(block, x, y=None):
    xy = np.asarray(x, float) if y is None else np.array([x, y], float)
    X, s, mirror = PLACE[block]
    x0 = RECT[block][2] if mirror else RECT[block][0]
    out = np.array(xy, float)
    out[..., 0] = X + (-(xy[..., 0] - x0) if mirror else (xy[..., 0] - x0)) * s
    out[..., 1] = Y0 + (xy[..., 1] - SO[1]) * s
    return out


def block_affine(block):
    X, s, mirror = PLACE[block]
    x0 = RECT[block][2] if mirror else RECT[block][0]
    return Affine2D().translate(-x0, -SO[1]).scale(-s if mirror else s, s).translate(X, Y0)


Y_TOP, Y_BOT = 33, 372
P_TRAIN = Path(M("OG", [(700, 126), (700, DD_NODES[0][0][1]), (DD_X[0] - 5, DD_NODES[0][0][1])]))
P_EPS = Path(M("OG", [(KP[0] + 20, KP[3] + 14), (DD_X[1], DD_NODES[1][0][1] - 8)]))
SIG0 = M("DS", OUT_X + TH / 2, SUR_Y - TH / 2 - 6)
B_IN = M("OG", BX0 - 3, BY0 + BH / 2)
P_SIG = Path([SIG0, (SIG0[0], Y_TOP), (B_IN[0] - 9, Y_TOP), (B_IN[0] - 9, B_IN[1]), (B_IN[0], B_IN[1])])
P_BATCH = Path([M("RS", RS[2] - 4, SUR_Y), M("DS", IN_X - 3, SUR_Y)])
MIX_C = M("OG", *MIX)
WAIT_IN = M("SO", WAIT_X + CHIP_W / 2, SO[3])
P_RET = [(MIX_C[0], MIX_C[1] + 11), (MIX_C[0], Y_BOT), (WAIT_IN[0], Y_BOT), (WAIT_IN[0], WAIT_IN[1] + 1)]

slot_col = np.full(N_SLOTS, -1)
slot_seen = np.zeros(N_SLOTS, bool)
slot_res = np.zeros(N_SLOTS, bool)
slot_evict = np.full(N_SLOTS, -9.0)
slot_read = np.full(N_SLOTS, -9.0)
inflight, arrivals_batch, arrivals_bar = [], [], []
state_packets, batch_packets, bar_packets, minibatches = [], [], [], []
sur_w = rng.normal(0, 0.6, sum(a * b for a, b in zip(SUR_N[:-1], SUR_N[1:])))
dd_w = rng.normal(0, 0.6, sum(a * b for a, b in zip(DD_N[:-1], DD_N[1:])))
sur_hist, dd_hist, sur_up, dd_up, dd_pending = [], [], [], [], []
hist, hist_flash, hist_states = [], np.full(BN, -9.0), []
res_states, thumb_hist = [], []
t_next_write = t_next_batch = t_next_train = 0.0
thumb_i = 0
for k, t in enumerate(times):
    r = rate[k]
    if r > 0 and t >= t_next_write:
        t_next_write = t + 1.0 / (7.0 * min(r, 2.0)) * rng.uniform(0.8, 1.2)
        empty = np.flatnonzero((slot_col < 0) & ~slot_res)
        seen = np.flatnonzero((slot_col >= 0) & slot_seen & ~slot_res)
        s = empty[0] if len(empty) else (rng.choice(seen) if len(seen) else None)
        if s is not None:
            i = int(rng.integers(N_SHOW))
            col = int(cl[k][1][i])
            slot_res[s] = True
            src_xy = M("SO", cell_xy[i][0], cell_xy[i][1] + CH / 2)
            dst = M("RS", slot_xy[s][0] + SW / 2, slot_xy[s][1] + SH_ / 2)
            ctrl = ((src_xy[0] + dst[0]) / 2, src_xy[1])
            state_packets.append((t, t + 0.6, Path(bezier(src_xy, ctrl, dst)), col))
            inflight.append((t + 0.6, s, col))
    for a in [a for a in inflight if a[0] <= t]:
        inflight.remove(a)
        _, s, col = a
        if slot_col[s] >= 0:
            slot_evict[s] = t
        slot_col[s], slot_seen[s], slot_res[s] = col, False, False
    filled = np.flatnonzero(slot_col >= 0)
    if r > 0 and len(filled) >= WATERMARK * N_SLOTS and t >= t_next_batch:
        t_next_batch = t + 1.0 / (3.5 * min(r, 2.0))
        pick = rng.choice(filled, min(4, len(filled)), replace=False)
        slot_read[pick] = t
        slot_seen[pick] = True
        batch_packets.append((t, t + 0.35, [int(slot_col[s]) for s in pick]))
        arrivals_batch.append(t + 0.35)
    for a in [a for a in arrivals_batch if a <= t]:
        arrivals_batch.remove(a)
        sur_w += rng.normal(0, 0.35, sur_w.shape)
        np.clip(sur_w, -1.2, 1.2, out=sur_w)
        sur_up.append(t)
        thumb_i = int(rng.integers(1, len(thumbs)))
        eps = float(np.clip(rng.gamma(2.0, 0.9) - 0.8, 0.05, 5.0))
        bar_packets.append((t + 0.15, t + 0.85, eps))
        arrivals_bar.append((t + 0.85, eps))
    for a in [a for a in arrivals_bar if a[0] <= t]:
        arrivals_bar.remove(a)
        hist.append(a[1])
        hist[:] = hist[-BN:]
        hist_flash[:] = np.roll(hist_flash, -1)
        hist_flash[-1] = t
    sampling = any(r_["t0"] <= t < r_["t1"] for r_ in resamplings)
    if r > 0 and not sampling and len(hist) >= 3 and t >= t_next_train:
        t_next_train = t + 0.7 / min(max(r, 1.0), 2.0)
        sel = rng.choice(len(hist), min(4, len(hist)), replace=False) + (BN - len(hist))
        hist_flash[sel] = np.maximum(hist_flash[sel], t - 0.01)
        minibatches.append((t, [BX0 + j * (BW + 2) + BW / 2 for j in sel], rng.normal(0, 1, (4, 2))))
        dd_pending.append(t + 0.6)
    for a in [a for a in dd_pending if a <= t]:
        dd_pending.remove(a)
        dd_w += rng.normal(0, 0.35, dd_w.shape)
        np.clip(dd_w, -1.2, 1.2, out=dd_w)
        dd_up.append(t)
    res_states.append((slot_col.copy(), slot_seen.copy(), slot_evict.copy(), slot_read.copy()))
    sur_hist.append(sur_w.copy())
    dd_hist.append(dd_w.copy())
    thumb_hist.append(thumb_i)
    hist_states.append((list(hist), hist_flash.copy()))

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
TF = {b: block_affine(b) + ax.transData for b in PLACE}
TXT = {"OG": 1.04, "SO": 1.0, "RS": 1.0, "DS": 1.0}


def text(x, y, s, size=10, color=FG, b=None, **kw):
    kw.setdefault("va", "center")
    if b is not None:
        kw["transform"] = TF[b]
        size *= TXT[b]
    return ax.text(x, y, s, fontsize=size, color=color, **kw)


def rbox(rect, fc=PANEL, ec=BORDER, r=6, lw=1.0, ls="-", z=0, b=None):
    x0, y0, x1, y1 = rect
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, ls=ls, zorder=z, transform=TF[b] if b else ax.transData)
    ax.add_patch(p)
    return p


def line(pts, color=FAINT, lw=1.1, head=False, z=1, ls="-", b=None):
    pts = np.asarray(pts, float)
    if b is not None:
        pts = M(b, pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="round")
    if head:
        d = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        n = np.array([-d[1], d[0]])
        ax.add_patch(plt.Polygon([pts[-1], pts[-1] - 6 * d + 3 * n, pts[-1] - 6 * d - 3 * n], color=color,
                                 zorder=z))


def draw_net(nodes, b):
    arts = []
    for la, lb in zip(nodes[:-1], nodes[1:]):
        for a in la:
            for c in lb:
                arts.append(ax.plot([a[0], c[0]], [a[1], c[1]], lw=0.8, color=FAINT, zorder=2, transform=TF[b])[0])
    circ = [[ax.add_patch(plt.Circle((x, y), 3.2, fc=PANEL, ec=DIM, lw=0.9, zorder=3, transform=TF[b]))
             for (x, y) in layer] for layer in nodes]
    return arts, circ


def thumb_stack(x0, yc, b):
    for j in (2, 1):
        rbox((x0 + 3 * j, yc - TH / 2 - 3 * j, x0 + TH + 3 * j, yc + TH / 2 - 3 * j), fc=INNER, r=2, z=2, b=b)
    rbox((x0, yc - TH / 2, x0 + TH, yc + TH / 2), fc=INNER, r=2, z=2, b=b)
    return ax.imshow(np.zeros((40, 40)), extent=(x0, x0 + TH, yc + TH / 2, yc - TH / 2), cmap="viridis",
                     vmin=-2.2, vmax=2.2, zorder=3, interpolation="nearest", transform=TF[b])


text(14, 16, "OGAS in the Melissa online loop", size=13, weight="bold", color=HL)
hdr = text(986, 16, "", size=10.5, ha="right")
ff_tag = text(730, 16, "fast-forward ▸▸", size=10, color=DIM, ha="right", alpha=0, family="DejaVu Sans")

rbox(OG, ls=(0, (4, 3)), ec="#388bfd", lw=1.3, b="OG")
text(656, 57, "OGAS", size=12, weight="bold", b="OG")
hd_train = text(724, 76, "Training", size=9, color=DIM, ha="center", style="italic", b="OG")
hd_samp = text(911, 76, "Resampling", size=9, color=DIM, ha="center", style="italic", b="OG")
hist_art = []
for j in range(BN):
    x = BX0 + j * (BW + 2)
    ax.add_patch(Rectangle((x, BY0), BW, BH, fc=INNER, ec="none", zorder=1, transform=TF["OG"]))
    hist_art.append(ax.add_patch(Rectangle((x, BY0 + BH), BW, 0, fc=SIG, ec="none", zorder=2, transform=TF["OG"])))
text(BX0 + BN * (BW + 2) + 4, BY0 + BH / 2, r"$\mathcal{B}$", size=10, color=DIM, b="OG")
text(BX0, BY0 + BH + 10, "evicts oldest", size=8, color=DIM, b="OG")
line(P_TRAIN.p, head=True)
text(706, 152, r"$(\lambda,\tilde\varepsilon)\sim\mathcal{U}_\mathcal{B}$", size=9, color=DIM, b="OG")
NB_K = 10
k_bars = []
for j in range(NB_K):
    h = (KP[3] - KP[1]) * (j + 0.5) / NB_K
    x = KP[0] + j * (KP[2] - KP[0]) / NB_K
    k_bars.append(ax.add_patch(Rectangle((x + 1, KP[3] - h), (KP[2] - KP[0]) / NB_K - 2, h, fc="#2d2140", ec="none",
                                         transform=TF["OG"])))
line([(KP[0], KP[3] + 1), (KP[2], KP[3] + 1)], color=DIM, lw=0.8, b="OG")
text(KP[2] + 3, KP[3] + 1, r"$\tilde\varepsilon$", size=9, color=DIM, b="OG")
text(KP[0] - 4, KP[1] + 6, r"$k$", size=9, color=DIM, ha="right", b="OG")
line(P_EPS.p, head=True)
dd_edges, dd_circ = draw_net(DD_NODES, "OG")
dd_layer_of = np.concatenate([[l_] * (a * b) for l_, (a, b) in enumerate(zip(DD_N[:-1], DD_N[1:]))])
text((DD_X[0] + DD_X[-1]) / 2, DD_Y + 44, r"DDPM $p_\phi(\lambda\mid\tilde\varepsilon)$", size=9, color=DIM,
     ha="center", b="OG")
line([(DD_X[-1] + 6, DD_Y), (PG[0] - 2, DD_Y)], head=True, b="OG")
rbox(PG, fc=INNER, r=3, b="OG")
text((PG[0] + PG[2]) / 2, PG[1] - 9, r"$p_g^{gen}(\lambda)$", size=9.5, color=DIM, ha="center", b="OG")
cloud = ax.scatter([], [], s=1.3, c="#6e7681", alpha=0.45, edgecolors="none", zorder=2)
pool_sc = ax.scatter([], [], edgecolors="none", zorder=5)
lead_sc = ax.scatter([], [], s=30, c=GEN, edgecolors="white", linewidths=1.0, zorder=7)
rbox(UB, fc=INNER, r=3, b="OG")
for a in range(3):
    for b_ in range(3):
        ax.add_patch(plt.Circle((UB[0] + 6 + 7 * a, UB[1] + 6 + 7 * b_), 1.5, color=UNI, zorder=3, transform=TF["OG"]))
text(UB[0] - 4, (UB[1] + UB[3]) / 2, r"$\mathcal{U}(\lambda)$", size=9.5, color=DIM, ha="right", b="OG")
ax.add_patch(plt.Circle(MIX, 9, fc=INNER, ec=DIM, lw=1.0, zorder=3, transform=TF["OG"]))
text(MIX[0], MIX[1] + 0.5, "+", size=11, color=FG, ha="center", zorder=4, b="OG")
line([(UB[2] + 2, (UB[1] + UB[3]) / 2), (MIX[0] - 9, MIX[1] - 3)], head=True, b="OG")
line([((PG[0] + PG[2]) / 2, PG[3] + 2), ((PG[0] + PG[2]) / 2, MIX[1]), (MIX[0] + 10, MIX[1])], head=True, b="OG")
text(760, 276, r"$\alpha$", size=10, color=UNI, ha="center", b="OG")
text(900, 290, r"$1-\alpha$", size=9.5, color=GEN, ha="center", b="OG")
text(MIX[0] - 14, MIX[1] + 14, r"$p_g(\lambda)$", size=9.5, color=FG, ha="right", b="OG")
ax.add_patch(Rectangle((872, 314), 104, 5, fc=INNER, ec="none", transform=TF["OG"]))
ramp_u = ax.add_patch(Rectangle((872, 314), 0, 5, fc=UNI, ec="none", transform=TF["OG"]))
ramp_g = ax.add_patch(Rectangle((872, 314), 0, 5, fc=GEN, ec="none", transform=TF["OG"]))
line(P_RET, head=True)
line(P_SIG.p, head=True)
text((SIG0[0] + B_IN[0]) / 2, Y_TOP - 9, r"$(\lambda,\tilde\varepsilon,t)$", size=9.5, ha="center", color=DIM)
text((MIX_C[0] + WAIT_IN[0]) / 2, Y_BOT - 9, r"$\lambda$", size=10, ha="center", color=GEN)

rbox(SO, b="SO")
so_left = M("SO", SO[2], SO[1])
text(so_left[0] + 9, so_left[1] + 12, r"Solvers $S(\lambda_i)$", size=10.5, weight="bold")
for x, lab, col in ((FIN_X + CHIP_W / 2, "Finished", FIN), (64 + CW / 2, "Running", RUN), (WAIT_X + CHIP_W / 2, "Waiting", DIM)):
    text(x, 76, lab, size=8.5, color=col, ha="center", style="italic", b="SO")
line([(123, 70), (123, 296)], color=RUN, lw=0.9, ls=(0, (3, 3)), b="SO")
line([(166, 312), (18, 312)], color=DIM, lw=0.9, head=True, b="SO")
text((SO[0] + SO[2]) / 2, 322, "time", size=8, color=DIM, ha="center", b="SO")
cell_fill, cell_out = [], []
for (x, y) in cell_xy:
    f = Rectangle((x + CW, y), 0, CH, fc=RUN, ec="none", alpha=0.6, zorder=2, transform=TF["SO"])
    o = FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0,rounding_size=2", fc="none", ec=UNI, lw=1.0, zorder=3,
                       transform=TF["SO"])
    ax.add_patch(f)
    ax.add_patch(o)
    cell_fill.append(f)
    cell_out.append(o)
fin_chips = [ax.add_patch(FancyBboxPatch((FIN_X, 88 + j * CHIP_DY), CHIP_W, CHIP_H, transform=TF["SO"],
                                         boxstyle="round,pad=0,rounding_size=2", fc=FIN, ec="none", alpha=0))
             for j in range(N_CHIPS)]
wait_chips = [ax.add_patch(FancyBboxPatch((WAIT_X, 88 + j * CHIP_DY), CHIP_W, CHIP_H, transform=TF["SO"],
                                          boxstyle="round,pad=0,rounding_size=2", fc=WAIT, ec="none", alpha=0))
              for j in range(N_CHIPS)]
wait_tags = [ax.add_patch(Rectangle((WAIT_X + 2, 88 + j * CHIP_DY + 2), 5, CHIP_H - 4, fc=UNI, ec="none", alpha=0,
                                    transform=TF["SO"])) for j in range(N_CHIPS)]
rbox(RS, b="RS")
text((RS[0] + RS[2]) / 2, 57, r"Reservoir $\mathcal{R}$", size=9.5, weight="bold", ha="center", b="RS")
slot_art = [ax.add_patch(FancyBboxPatch((x, y), SW, SH_, boxstyle="round,pad=0,rounding_size=2", fc=INNER,
                                        ec=BORDER, lw=0.8, zorder=2, transform=TF["RS"])) for (x, y) in slot_xy]
text((RS[0] + RS[2]) / 2, 311, "evicts\non write", size=8.5, color=DIM, ha="center", linespacing=1.1, b="RS")
a0_, a1_ = M("SO", SO[0] + 2, 186), M("RS", RS[0] + 2, 186)
line([a0_, a1_], head=True)
text((a0_[0] + a1_[0]) / 2, a0_[1] - 13, r"$X_t^{\lambda}$", size=10, ha="center", color=DIM)
rbox(DS, b="DS")
text(436, 57, "Deep Surrogate", size=10.5, weight="bold", b="DS")
in_img = thumb_stack(IN_X, SUR_Y, "DS")
out_img = thumb_stack(OUT_X, SUR_Y, "DS")
sur_edges, _ = draw_net(SUR_NODES, "DS")
line(P_BATCH.p, head=True)
text(IN_X + TH / 2, SUR_Y + 30, r"$X^{\lambda}_{[t-r]}$", size=9.5, ha="center", color=DIM, b="DS")
text((DS[0] + DS[2]) / 2, 296, "1.8 batches / simulation", size=8.5, color=DIM, ha="center", b="DS")

LEG = [("s", GEN, r"$\lambda\sim p_g^{gen}$"), ("s", UNI, r"$\lambda\sim\mathcal{U}$"), ("s", RUN, "running"),
       ("s", FIN, "finished"), ("s", WAIT, "waiting"), ("|", SIG, r"signal $\tilde\varepsilon$")]
lx = 430
for mk, col, lab in LEG:
    ax.scatter([lx + 4], [H - 14], marker=mk, s=26 if mk != "|" else 60, c=col, linewidths=2.5)
    text(lx + 12, H - 13.5, lab, size=8.5, color=DIM)
    lx += 90

pk_sq = ax.scatter([], [], marker="s", edgecolors="none", zorder=6)
pk_o = ax.scatter([], [], marker="o", edgecolors=BG, linewidths=0.6, zorder=6)
bar_pool = [ax.add_patch(Rectangle((0, 0), 0, 0, fc=SIG, ec="none", zorder=6)) for _ in range(8)]
COLS = (UNI, GEN)


def mix(c1, c2, a):
    c1, c2 = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple(c1 + (c2 - c1) * float(np.clip(a, 0, 1)))


def color_edges(arts, w, fl, accent, pulse=None, layer_of=None):
    for e, ln in enumerate(arts):
        a = min(abs(w[e]) / 1.2, 1.0)
        c = mix(mix(BORDER, "#58a6ff" if w[e] < 0 else "#d2a8ff", 0.25 + 0.6 * a), accent, 0.75 * fl)
        if pulse is not None:
            c = mix(c, GEN, max(0.0, 1 - abs(pulse * 3 - layer_of[e] - 0.5)))
        ln.set_color(c)
        ln.set_linewidth(0.4 + 1.1 * a)


def last_before(ts, t):
    return max([u for u in ts if u <= t], default=-9.0)


def render(k):
    t = times[k]
    pr, sr, new_t, q, fin = cl[k]
    for i in range(N_SHOW):
        x, y = cell_xy[i]
        w = round(CW * pr[i])
        cell_fill[i].set_bounds(x + CW - w, y, w, CH)
        fl = max(0.0, 1.0 - (t - new_t[i]) / 0.35)
        cell_out[i].set_edgecolor(mix(COLS[sr[i]], "#ffffff", fl))
        cell_out[i].set_linewidth(1.0 + 1.2 * fl)
    fin = sorted(fin, reverse=True)
    for j, ch in enumerate(fin_chips):
        ch.set_alpha(0 if j >= len(fin) else 0.85 * (1 - j / N_CHIPS) * min(1, (t - fin[j]) / 0.15 + 0.3))
    for j, (ch, tag) in enumerate(zip(wait_chips, wait_tags)):
        on = j < len(q)
        ch.set_alpha(0.8 if on else 0)
        tag.set_alpha(1 if on else 0)
        if on:
            tag.set_facecolor(COLS[q[j]])
    col_s, seen_s, ev_s, rd_s = res_states[k]
    for s, p in enumerate(slot_art):
        ev = max(0.0, 1.0 - (t - ev_s[s]) / 0.35)
        rd = max(0.0, 1.0 - (t - rd_s[s]) / 0.3)
        p.set_facecolor(INNER if col_s[s] < 0 else mix(INNER, COLS[col_s[s]], 0.3 if seen_s[s] else 0.85))
        p.set_edgecolor(FIN if ev > 0 else mix(BORDER, "#ffffff", rd))
        p.set_linewidth(0.8 + 1.2 * max(ev, rd))
    up = last_before(sur_up, t)
    color_edges(sur_edges, sur_hist[k], max(0.0, 1.0 - (t - up) / 0.25), TEAL)
    ti = thumb_hist[k]
    g_ = np.random.default_rng(ti).normal(0, 1, (40, 40))
    g_ = (g_ + np.roll(g_, 1, 0) + np.roll(g_, 1, 1) + np.roll(g_, -1, 0)) / 2
    in_img.set_data(thumbs[ti - 1] if ti else np.zeros((40, 40)))
    out_img.set_data(thumbs[ti] + 0.35 * g_ if ti else np.zeros((40, 40)))
    for im in (in_img, out_img):
        im.set_visible(up >= 0)
    hvals, hfl = hist_states[k]
    for j, b in enumerate(hist_art):
        jj = j - (BN - len(hvals))
        hh = 0 if jj < 0 else BH * min(hvals[jj], 5) / 5
        b.set_bounds(BX0 + j * (BW + 2), BY0 + BH - hh, BW, hh)
        b.set_facecolor(mix(SIG, "#ffffff", 0.7 * max(0.0, 1.0 - (t - hfl[j]) / 0.3)))
    o, o_c, o_s = [], [], []
    for (t0, xs, nz) in minibatches:
        u = (t - t0) / 0.6
        if 0 <= u < 1:
            for j, x in enumerate(xs):
                path = Path([tuple(M("OG", x, BY0 + BH))] + P_TRAIN.p.tolist())
                o.append(path.at(ease(u)) + 5 * u * nz[j])
                o_c.append(matplotlib.colors.to_rgba(SIG, 1 - 0.6 * u))
                o_s.append(14)
    training = any(0 <= t - m[0] < 0.9 for m in minibatches)
    samp, pulse, pxy, psz, pcc, lead = None, None, [], [], [], None
    for r_ in resamplings:
        if r_["t0"] <= t < r_["t_end"]:
            samp = r_
    dd_fl = max(0.0, 1.0 - (t - last_before(dd_up, t)) / 0.25)
    kb = np.zeros(NB_K)
    if samp is not None and t < samp["t1"] and not samp["fast"]:
        u = (t - samp["t0"]) / (samp["t1"] - samp["t0"])
        rn = np.random.default_rng(samp["g"]).normal(0, 1, samp["pts"].shape)
        q_ = np.clip((u - 0.2) / 0.8, 0, 1) * N_REV
        step = int(min(q_, N_REV - 1e-9))
        pulse = (q_ - step) if q_ > 0 else None
        cur = step + ease((q_ - step) / 0.35) if q_ > 0 else 0.0
        pos = reflect(samp["pts"] + 0.35 * (1 - cur / N_REV) ** 1.3 * rn)
        pxy += list(pos)
        psz += [9] * len(pos)
        pcc += [mix("#8b949e", GEN, cur / N_REV)] * len(pos)
        lead = pos[:1] if len(pos) else None
        rr = np.random.default_rng(samp["g"] + 7)
        kb[np.minimum((NB_K * np.sqrt(rr.random(12))).astype(int), NB_K - 1)] = 1.0 * (u < 0.35)
        if u < 0.35:
            o.append(P_EPS.at(ease(u / 0.35)))
            o_c.append(GEN)
            o_s.append(22)
    elif samp is not None:
        pxy += list(samp["pts"])
        psz += [9] * len(samp["pts"])
        pcc += [matplotlib.colors.to_rgba(GEN, 0.9 if t < samp["t2"] else 0.55)] * len(samp["pts"])
        if samp["fast"] and t < samp["t1"]:
            pulse = 0.5
    for j, b in enumerate(k_bars):
        b.set_facecolor(mix("#2d2140", GEN, kb[j]))
    color_edges(dd_edges, dd_hist[k], dd_fl, SIG, pulse, dd_layer_of)
    for li, layer in enumerate(dd_circ):
        hot = 0.0 if pulse is None else max(0.0, 1 - abs(pulse * 3 - li) / 0.8)
        for c_ in layer:
            c_.set_edgecolor(mix(DIM, GEN, hot))
    samp_on = samp is not None and t < samp["t1"]
    hd_train.set_color(SIG if training and not samp_on else DIM)
    hd_samp.set_color(GEN if samp_on else DIM)
    n = int(n_disp[k])
    cloud.set_offsets(M("OG", pg_xy(np.c_[LX[:n], LY[:n]])))
    pool_sc.set_offsets(M("OG", pg_xy(np.array(pxy))) if pxy else np.empty((0, 2)))
    pool_sc.set_sizes(psz if psz else [0])
    pool_sc.set_facecolors(pcc if pcc else [(0, 0, 0, 0)])
    lead_sc.set_offsets(M("OG", pg_xy(lead)) if lead is not None else np.empty((0, 2)))
    for r_ in resamplings:
        if r_["t1"] <= t < r_["t2"]:
            u = (t - r_["t1"]) / (r_["t2"] - r_["t1"])
            nb = int(round(10 * r_["share"]))
            for j in range(10):
                gen_ = j < nb
                a0 = M("OG", pg_xy(r_["pts"][j % max(len(r_["pts"]), 1)])[0]) if gen_ and len(r_["pts"]) else \
                    M("OG", UB[2], (UB[1] + UB[3]) / 2)
                uj = np.clip(u * 1.6 - 0.06 * j, 0, 1)
                if uj <= 0 or uj >= 1:
                    continue
                path = Path([tuple(a0), tuple(MIX_C)] + P_RET[1:])
                o.append(path.at(ease(uj)))
                o_c.append(COLS[int(gen_)])
                o_s.append(22)
    sq, sq_c, sq_s = [], [], []
    for (t0, t1, path, col) in state_packets:
        if t0 <= t < t1:
            sq.append(path.at(ease((t - t0) / (t1 - t0))))
            sq_c.append(COLS[col])
            sq_s.append(18)
    for (t0, t1, cols) in batch_packets:
        if t0 <= t < t1:
            c = P_BATCH.at(ease((t - t0) / (t1 - t0)))
            for j, col in enumerate(cols):
                sq.append(c + np.array([(j % 2) * 6 - 3, (j // 2) * 6 - 6]))
                sq_c.append(COLS[col])
                sq_s.append(16)
    for sc, xy_, c_, s_ in ((pk_sq, sq, sq_c, sq_s), (pk_o, o, o_c, o_s)):
        sc.set_offsets(np.round(np.array(xy_) * 2) / 2 if xy_ else np.empty((0, 2)))
        sc.set_facecolors(c_ if c_ else [(0, 0, 0, 0)])
        sc.set_sizes(s_ if s_ else [0])
    live = [(t0, t1, e) for (t0, t1, e) in bar_packets if t0 <= t < t1]
    for j, b in enumerate(bar_pool):
        if j < len(live):
            t0, t1, e = live[j]
            x, y = P_SIG.at(ease((t - t0) / (t1 - t0)))
            hh = 4 + 16 * e / 5
            b.set_bounds(round(x) - 2.5, round(y) - hh / 2, 5, hh)
        else:
            b.set_bounds(0, 0, 0, 0)
    g = int(gen_disp[k])
    hdr.set_text(f"generation {g:3d}   ·   {n:6,d} / {N_BUDGET:,} simulations")
    ff_tag.set_alpha(1.0 if is_ff[k] else 0.0)
    sh = share(int(n_res_at[k]))
    ramp_u.set_width(104 * (1 - sh))
    ramp_g.set_x(872 + 104 * (1 - sh))
    ramp_g.set_width(104 * sh)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return Image.open(buf).convert("RGB")


rgb = [render(k) for k in range(n_frames)]
last, first = np.asarray(rgb[-1], float), np.asarray(rgb[0], float)
n_fade = int(round(FADE_S * FPS))
for a in np.arange(1, n_fade + 1) / (n_fade + 1):
    rgb.append(Image.fromarray(np.round((1 - a) * last + a * first).astype(np.uint8)))
durations = [int(round(1000 / FPS))] * len(rgb)
if args.stills:
    os.makedirs(args.stills, exist_ok=True)
    for tt in (3.0, 6.0, 8.0, 10.5, 12.0, 16.0, 19.5):
        rgb[int(tt * FPS)].save(os.path.join(args.stills, f"loop_{tt:04.1f}.png"))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
kept = save_gif(args.out, rgb, durations, colors=254, reference=range(0, len(rgb), len(rgb) // 8))
n_saved, err = check_gif(args.out, [rgb[k] for k in kept])
print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB, {n_saved} frames, {rgb[0].size}, "
      f"{len(rgb) / FPS:.1f} s, loop=0, mean abs decode error {err:.2f}")
