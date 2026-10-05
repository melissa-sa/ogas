import argparse
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from gifsave import check_gif, save_gif  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--stills")
args = ap.parse_args()

BG, PANEL, INNER, BORDER, FAINT = "#0d1117", "#161b22", "#10151d", "#30363d", "#484f58"
FG, DIM, HL = "#c9d1d9", "#8b949e", "#e6edf3"
UNI, GEN, BUF, TEAL = "#3fb950", "#f0883e", "#bc8cff", "#39c5cf"
W, H, FPS = 1000, 400, 12
T_FILL, T_DISC, T_REW, T_AFTER, T_END = 5.0, 8.6, 11.0, 12.4, 14.4
FADE_S = 0.6
N_GEN, PER_GEN = 8, 70
RAMP = np.linspace(0.0, 0.7, 4)
BW_KDE = 0.07
rng = np.random.default_rng(5)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "text.color": FG,
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
                     "mathtext.it": "Helvetica:italic", "mathtext.bf": "Helvetica:bold"})


def share(g):
    return 0.0 if g <= 1 else float(RAMP[min(g - 1, 3)])


def crescent(n):
    out = []
    while len(out) < n:
        p = rng.random(2)
        if np.sum((p - (0.42, 0.5)) ** 2) < 0.3 ** 2 and np.sum((p - (0.56, 0.56)) ** 2) > 0.24 ** 2:
            out.append(p)
    return np.array(out)


pts, src, gen_of = [], [], []
for g in range(1, N_GEN + 1):
    nb = int(round(share(g) * PER_GEN))
    pts += list(rng.random((PER_GEN - nb, 2))) + list(crescent(nb))
    src += [0] * (PER_GEN - nb) + [1] * nb
    gen_of += [g] * PER_GEN
pts, src, gen_of = np.array(pts), np.array(src), np.array(gen_of)


def kde(xy, at, w=None):
    w = np.ones(len(xy)) if w is None else w
    d2 = ((at[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    return (np.exp(-d2 / (2 * BW_KDE ** 2)) * w).sum(1) / w.sum() / (2 * np.pi * BW_KDE ** 2)


w_pts = np.clip(1.0 / kde(pts, pts), 0.1, 10)
G = 36
gx = (np.arange(G) + 0.5) / G
grid = np.stack(np.meshgrid(gx, gx), -1).reshape(-1, 2)
w_map = np.log10(np.clip(1.0 / kde(pts, grid), 0.1, 10)).reshape(G, G)
dens_raw = kde(pts, grid).reshape(G, G)
dens_w = kde(pts, grid, w_pts).reshape(G, G)
vmax = np.percentile(dens_raw, 99)

n_frames = int(round(T_END * FPS))
times = np.arange(n_frames) / FPS

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def text(x, y, s, size=10, color=FG, **kw):
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, color=color, **kw)


def rbox(rect, fc=PANEL, ec=BORDER, r=6, lw=1.0, z=0):
    x0, y0, x1, y1 = rect
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def line(pts_, color=FAINT, lw=1.1, head=True):
    pts_ = np.asarray(pts_, float)
    ax.plot(pts_[:, 0], pts_[:, 1], color=color, lw=lw, zorder=1)
    if head:
        d = (pts_[-1] - pts_[-2]) / np.linalg.norm(pts_[-1] - pts_[-2])
        n = np.array([-d[1], d[0]])
        ax.add_patch(plt.Polygon([pts_[-1], pts_[-1] - 6 * d + 3 * n, pts_[-1] - 6 * d - 3 * n], color=color))


def to_px(xy, box):
    x0, y0, x1, y1 = box
    xy = np.atleast_2d(xy)
    return np.c_[x0 + 3 + xy[:, 0] * (x1 - x0 - 6), y1 - 3 - xy[:, 1] * (y1 - y0 - 6)]


def plot_box(box, title, sub=None):
    rbox(box, fc=INNER, r=3)
    text((box[0] + box[2]) / 2, box[1] - 10, title, size=9.5, ha="center")


def net(xs, ns, yc, dy):
    nodes = [[(x, yc + (j - (n - 1) / 2) * dy) for j in range(n)] for x, n in zip(xs, ns)]
    edges = []
    for la, lb in zip(nodes[:-1], nodes[1:]):
        for a in la:
            for b in lb:
                edges.append(ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.8, color=FAINT, zorder=2)[0])
    for layer in nodes:
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), 3.4, fc=PANEL, ec=DIM, lw=0.9, zorder=3))
    return nodes, edges


text(14, 20, "Density-ratio debiasing of the generator", size=13, weight="bold", color=HL)
text(360, 21, "concept, synthetic 2D data", size=9, color=DIM)
hdr = text(986, 20, "", size=10.5, ha="right")

rbox((14, 44, 432, 222))
text(24, 56, r"DDPM buffer $\mathcal{B}$", size=10.5, weight="bold")
P_U, P_G, P_B = (24, 84, 134, 194), (166, 84, 276, 194), (308, 84, 418, 194)
plot_box(P_U, r"$\alpha\,\mathcal{U}_\Lambda$")
plot_box(P_G, r"$(1-\alpha)\,p_g^{gen}(\lambda)$")
plot_box(P_B, r"$p_\mathcal{B}$")
text(150, 139, "+", size=16, color=DIM, ha="center")
text(292, 139, "=", size=16, color=DIM, ha="center")
sc_u = ax.scatter([], [], s=7, c=UNI, edgecolors="none", zorder=4)
sc_g = ax.scatter([], [], s=7, c=GEN, edgecolors="none", zorder=4)
sc_b = ax.scatter([], [], s=7, edgecolors="none", zorder=4)
ramp_u = ax.add_patch(plt.Rectangle((24, 206), 0, 5, fc=UNI, ec="none"))
ramp_g = ax.add_patch(plt.Rectangle((24, 206), 0, 5, fc=GEN, ec="none"))
ramp_t = text(418, 208.5, "", size=8.5, color=DIM, ha="right")

rbox((446, 44, 700, 222))
text(456, 56, r"Discriminator $D_\psi(\lambda)$", size=10.5, weight="bold")
D_NODES, D_EDGES = net([500, 530, 560, 590], [2, 5, 5, 1], 128, 14)
rbox((452, 160, 476, 184), fc=INNER, r=3)
for a in range(3):
    for b in range(3):
        ax.add_patch(plt.Circle((457 + 7 * a, 165 + 7 * b), 1.5, color=UNI, zorder=3))
text(464, 192, r"$\mathcal{U}_\Lambda$", size=9, color=DIM, ha="center")
text(D_NODES[0][0][0] - 8, D_NODES[0][0][1] - 9, r"$y{=}0$", size=8, color=BUF, ha="right")
text(D_NODES[0][1][0] - 8, D_NODES[0][1][1] + 11, r"$y{=}1$", size=8, color=UNI, ha="right")
text(598, 128, r"$D_\psi(\lambda)$", size=9.5, color=DIM)
text(573, 198, r"$w(\lambda) = \dfrac{D_\psi(\lambda)}{1 - D_\psi(\lambda)}$", size=10.5, ha="center")
line([(P_B[2] + 2, 121), (D_NODES[0][0][0] - 5, D_NODES[0][0][1])])
line([(476, 170), (D_NODES[0][1][0] - 5, D_NODES[0][1][1] + 2)])
line([(645, 128), (726, 128)])
disc_dots = ax.scatter([], [], s=16, edgecolors="none", zorder=5)

rbox((714, 44, 986, 222))
text(724, 56, r"weights $w(\lambda)$", size=10.5, weight="bold")
P_W = (736, 84, 846, 194)
plot_box(P_W, "")
w_im = ax.imshow(w_map, extent=(P_W[0] + 1, P_W[2] - 1, P_W[3] - 1, P_W[1] + 1), origin="upper", cmap="cividis",
                 vmin=-1, vmax=1, interpolation="bilinear", zorder=2, alpha=0)
w_pts_sc = ax.scatter([], [], s=2, c="#ffffff", alpha=0.35, edgecolors="none", zorder=3)
cb_x = 862
for j, v in enumerate(np.linspace(1, -1, 40)):
    ax.add_patch(plt.Rectangle((cb_x, 84 + j * 110 / 40), 8, 110 / 40 + 0.3,
                               fc=matplotlib.colormaps["cividis"]((v + 1) / 2), ec="none"))
text(cb_x + 13, 88, "10", size=8, color=DIM)
text(cb_x + 13, 101, "sparse: up-weight", size=8, color=DIM)
text(cb_x + 13, 139, "1", size=8, color=DIM)
text(cb_x + 13, 177, "dense: down-weight", size=8, color=DIM)
text(cb_x + 13, 190, "0.1", size=8, color=DIM)

rbox((14, 236, 432, 392))
text(24, 248, "Reweighted buffer", size=10.5, weight="bold")
P_R = (24, 270, 134, 380)
plot_box(P_R, "")
sc_r = ax.scatter([], [], edgecolors="none", zorder=4)
M_NODES, M_EDGES = net([200, 230, 260, 290], [3, 5, 5, 2], 318, 12)
text(245, 358, r"DDPM $p_\phi(\lambda\mid\tilde\varepsilon)$", size=9.5, color=DIM, ha="center")
text(360, 300, "modified\nobjective", size=9, color=DIM, ha="center", linespacing=1.3)
text(223, 378, r"$\mathbb{E}_{(\lambda,\tilde\varepsilon)\sim p_\mathcal{B}}"
     r"\left[-\,w(\lambda)\,\log p_\phi(\lambda\mid\tilde\varepsilon)\right]$", size=10, ha="center")
line([(P_R[2] + 2, 318), (M_NODES[0][1][0] - 6, 318)])
feed_dots = ax.scatter([], [], edgecolors="none", zorder=5)
rbox((446, 236, 986, 392))
text(456, 248, "What the DDPM learns", size=10.5, weight="bold")
P_A, P_C = (520, 278, 630, 388), (760, 278, 870, 388)
for box, lab in ((P_A, "without $w$"), (P_C, "with $w$")):
    plot_box(box, lab)
im_a = ax.imshow(dens_raw, extent=(P_A[0] + 1, P_A[2] - 1, P_A[3] - 1, P_A[1] + 1), origin="upper",
                 cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear", zorder=2, alpha=0)
im_c = ax.imshow(dens_w, extent=(P_C[0] + 1, P_C[2] - 1, P_C[3] - 1, P_C[1] + 1), origin="upper",
                 cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear", zorder=2, alpha=0)
lab_a = text(P_A[2] + 8, 333, "over-sampled\nregion dominates", size=8.5, color=DIM, linespacing=1.3,
             alpha=0)
lab_c = text(P_C[2] + 8, 333, "≈ uniform", size=8.5, color=DIM, alpha=0, family="DejaVu Sans")
line([(432, 318), (446, 318)])


def mix(c1, c2, a):
    c1, c2 = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple(c1 + (c2 - c1) * float(np.clip(a, 0, 1)))


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


rng_e = np.random.default_rng(9)
d_w = [rng_e.normal(0, 0.6, len(D_EDGES)) for _ in range(200)]
m_w = [rng_e.normal(0, 0.6, len(M_EDGES)) for _ in range(200)]
size_w = 2 + 30 * ((np.log10(w_pts) + 1) / 2) ** 1.5
x_in_b = np.array(D_NODES[0][0])
x_in_u = np.array(D_NODES[0][1])


def color_edges(edges, w, fl, accent):
    for e, ln in enumerate(edges):
        a = min(abs(w[e]) / 1.2, 1.0)
        ln.set_color(mix(mix(BORDER, "#58a6ff" if w[e] < 0 else "#d2a8ff", 0.25 + 0.6 * a), accent, 0.75 * fl))
        ln.set_linewidth(0.4 + 1.1 * a)


def render(t):
    g = int(np.clip(t / T_FILL * N_GEN + 0.999, 0, N_GEN)) if t > 0 else 0
    m = gen_of <= g
    fresh = m & (gen_of == g)
    for sc, sel in ((sc_u, m & (src == 0)), (sc_g, m & (src == 1))):
        box = P_U if sc is sc_u else P_G
        sc.set_offsets(to_px(pts[sel], box) if sel.any() else np.empty((0, 2)))
    sc_b.set_offsets(to_px(pts[m], P_B) if m.any() else np.empty((0, 2)))
    sc_b.set_facecolors([UNI if s == 0 else GEN for s in src[m]] if m.any() else [(0, 0, 0, 0)])
    sc_b.set_sizes(np.where(fresh[m], 16, 7) if m.any() else [0])
    sh = share(g) if g else 0.0
    ramp_u.set_width(394 * (1 - sh))
    ramp_g.set_x(24 + 394 * (1 - sh))
    ramp_g.set_width(394 * sh)
    ramp_t.set_text("")
    hdr.set_text(f"generation {g}   ·   " + rf"$\alpha$ = {1 - sh:.2f},  $1-\alpha$ = {sh:.2f}")
    u_d = (t - T_FILL) / (T_DISC - T_FILL)
    dxy, dcc = [], []
    fl_d = 0.0
    if 0 <= u_d < 1:
        k = int((t - T_FILL) * 5)
        ph = (t - T_FILL) * 5 - k
        rr = np.random.default_rng(100 + k)
        a0 = to_px(pts[rr.integers(len(pts))], P_B)[0]
        b0 = np.array((464, 172))
        dxy += [a0 + (x_in_b - a0) * ease(ph), b0 + (x_in_u - b0) * ease(ph)]
        dcc += [BUF, UNI]
        fl_d = max(0.0, 1 - abs(ph - 1.0) / 0.25) + max(0.0, 1 - ph / 0.25)
        w_idx = k
    else:
        w_idx = 0 if t < T_FILL else int((T_DISC - T_FILL) * 5)
    disc_dots.set_offsets(np.array(dxy) if dxy else np.empty((0, 2)))
    disc_dots.set_facecolors(dcc if dcc else [(0, 0, 0, 0)])
    color_edges(D_EDGES, d_w[w_idx], min(fl_d, 1), TEAL)
    w_im.set_alpha(0.95 * ease(u_d))
    w_pts_sc.set_offsets(to_px(pts, P_W) if u_d > 0 else np.empty((0, 2)))
    u_r = (t - T_DISC) / (T_REW - T_DISC)
    if u_r >= 0:
        sc_r.set_offsets(to_px(pts, P_R))
        sc_r.set_sizes(7 + (size_w - 7) * ease(u_r / 0.5))
        sc_r.set_facecolors([mix(UNI if s == 0 else GEN, "#ffffff", 0.1) for s in src])
    else:
        sc_r.set_offsets(np.empty((0, 2)))
    fxy, fsz = [], []
    fl_m = 0.0
    if u_r >= 0.45 and t < T_AFTER + 0.8:
        k = int((t - T_DISC) * 4)
        ph = (t - T_DISC) * 4 - k
        rr = np.random.default_rng(300 + k)
        for j in rr.integers(len(pts), size=3):
            a0 = to_px(pts[j], P_R)[0]
            end = np.array(M_NODES[0][1])
            fxy.append(a0 + (end - a0) * ease(ph))
            fsz.append(size_w[j] ** 1.6)
        fl_m = max(0.0, 1 - abs(ph - 1.0) / 0.25) + max(0.0, 1 - ph / 0.25)
        m_idx = k
    else:
        m_idx = 0 if t < T_DISC else int((T_AFTER + 0.8 - T_DISC) * 4)
    feed_dots.set_offsets(np.array(fxy) if fxy else np.empty((0, 2)))
    feed_dots.set_sizes(fsz if fsz else [0])
    feed_dots.set_facecolors([BUF] * len(fxy) if fxy else [(0, 0, 0, 0)])
    color_edges(M_EDGES, m_w[min(m_idx, 199)], min(fl_m, 1), BUF)
    u_a = ease((t - T_REW) / 0.9)
    u_c = ease((t - T_REW - 0.6) / 0.9)
    for art, a in ((im_a, u_a), (lab_a, u_a), (im_c, u_c), (lab_c, u_c)):
        art.set_alpha(a)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return Image.open(buf).convert("RGB")


rgb = [render(t) for t in times]
last, first = np.asarray(rgb[-1], float), np.asarray(rgb[0], float)
n_fade = int(round(FADE_S * FPS))
for a in np.arange(1, n_fade + 1) / (n_fade + 1):
    rgb.append(Image.fromarray(np.round((1 - a) * last + a * first).astype(np.uint8)))
durations = [int(round(1000 / FPS))] * len(rgb)
if args.stills:
    os.makedirs(args.stills, exist_ok=True)
    for tt in (2.5, 5.0, 7.0, 9.6, 13.5):
        rgb[int(tt * FPS)].save(os.path.join(args.stills, f"debias_{tt:04.1f}.png"))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
kept = save_gif(args.out, rgb, durations, colors=254, reference=range(0, len(rgb), len(rgb) // 8))
n_saved, err = check_gif(args.out, [rgb[k] for k in kept])
print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB, {n_saved} frames, {rgb[0].size}, "
      f"{len(rgb) / FPS:.1f} s, loop=0, mean abs decode error {err:.2f}")
