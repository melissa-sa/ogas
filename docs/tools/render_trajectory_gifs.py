import os
import sys

import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

BG = np.array([13, 17, 23])
FG = np.array([201, 209, 217])
DIM = np.array([139, 148, 158])
LEVELS = 32
SIZE = 192
FPS = 12
HOLD_LAST, HOLD_FIRST = 700, 350
FADE = 5
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except OSError:
    try:
        FONT = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        FONT = ImageFont.load_default()
PDES = {
    "gs": ("Gray-Scott", 1, "magma", False, "L = {:.1f}"),
    "ks": ("Kuramoto-Sivashinsky", 0, "viridis", True, "L = {:.0f}"),
    "ns": ("Kolmogorov flow", 0, "RdBu_r", True, "ν = {:.4f}"),
}
CMAPS = ["magma", "viridis", "RdBu_r"]

pal = [BG]
for col in (FG, DIM):
    pal += [BG + (col - BG) * a for a in (0.25, 0.5, 0.75, 1.0)]
CMAP_OFFSET = {}
for name in CMAPS:
    CMAP_OFFSET[name] = len(pal)
    lut = matplotlib.colormaps[name](np.linspace(0, 1, LEVELS))[:, :3] * 255
    pal += list(lut)
PALETTE = np.clip(np.round(np.array(pal)), 0, 255).astype(np.uint8)
assert len(PALETTE) <= 256
PAL_FLAT = PALETTE.flatten().tolist() + [0] * (768 - PALETTE.size)


def to_levels(frames, symmetric):
    f = np.stack([np.asarray(Image.fromarray(fr.astype(np.float32), mode="F").resize((SIZE, SIZE), Image.Resampling.LANCZOS))
                  for fr in frames])
    if symmetric:
        hi = np.percentile(np.abs(f), 99.5)
        lo = -hi
    else:
        lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.5)
    x = np.clip((f - lo) / (hi - lo), 0, 1)
    return np.round(x * (LEVELS - 1)).astype(np.uint8)


def draw_text(canvas, xy, text, shade_base):
    mask = Image.new("L", (canvas.shape[1], canvas.shape[0]), 0)
    ImageDraw.Draw(mask).text(xy, text, fill=255, font=FONT)
    m = np.asarray(mask)
    lvl = np.digitize(m, [32, 96, 160, 224])
    sel = lvl > 0
    canvas[sel] = shade_base + lvl[sel] - 1


def frame(panels, levels_of, t_label, width, height, top, pad, gap):
    c = np.zeros((height, width), np.uint8)
    for i, (label, lev, name) in enumerate(panels):
        x0 = pad + i * (lev.shape[2] + gap)
        c[top:top + lev.shape[1], x0:x0 + lev.shape[2]] = CMAP_OFFSET[name] + levels_of(lev)
        draw_text(c, (x0, 6), label, 1)
    draw_text(c, (width - pad - 52, 6), f"t = {t_label:2d}", 5)
    img = Image.fromarray(c, mode="P")
    img.putpalette(PAL_FLAT)
    return img


def render(panels, fname):
    n = len(panels)
    t_len, h, w = panels[0][1].shape
    pad, gap, top = 14, 12, 26
    geom = (2 * pad + n * w + (n - 1) * gap, top + h + pad, top, pad, gap)
    frames = [frame(panels, lambda lev: lev[t], t, *geom) for t in range(t_len)]
    durations = [int(1000 / FPS)] * t_len
    durations[0], durations[-1] = HOLD_FIRST, HOLD_LAST
    for a in np.arange(1, FADE + 1) / (FADE + 1):
        mix = lambda lev, a=a: np.round((1 - a) * lev[-1] + a * lev[0]).astype(np.uint8)
        frames.append(frame(panels, mix, t_len - 1 if a < 0.5 else 0, *geom))
        durations.append(int(1000 / FPS))
    path = os.path.join(OUT, fname)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=1)
    print(f"{path}: {os.path.getsize(path) / 1e6:.2f} MB, {len(frames)} frames, {geom[0]}x{geom[1]}")


data = {k: np.load(os.path.join(DATA, f"{k}.npz")) for k in PDES}

combined = []
for k, (title, ch, cmap, sym, fmt) in PDES.items():
    z = data[k]
    i = int(z["idx"][1])
    combined.append((title, to_levels(z[f"sim{i}"][:, ch], sym), cmap))
render(combined, "pde_trajectories.gif")

for k, (title, ch, cmap, sym, fmt) in PDES.items():
    z = data[k]
    panels = []
    for i, p in zip(z["idx"], z["phys"]):
        panels.append((fmt.format(float(p)), to_levels(z[f"sim{int(i)}"][:, ch], sym), cmap))
    render(panels, f"{k}_trajectories.gif")
