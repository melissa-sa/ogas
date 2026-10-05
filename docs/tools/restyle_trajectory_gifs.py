"""Convert the dark trajectory GIFs (old style of render_trajectory_gifs.py) to the paper style, in place.

    python restyle_trajectory_gifs.py ../assets/gs_trajectories.gif:"L = 1.5,L = 5.2,L = 9.1" ...

Each argument is GIF:LABELS. The three 192 px field panels are cropped from every frame, shrunk and laid out
on a white ground with serif labels; frame durations are kept. Use render_trajectory_gifs.py instead when the
simulation data are at hand.
"""
import sys

import numpy as np
import matplotlib
import matplotlib.font_manager
from PIL import Image, ImageDraw, ImageFont


X0S, Y0, SRC = (14, 218, 422), 26, 192          # panel geometry of the old GIFs
SIZE, PAD, GAP, TOP = 160, 2, 8, 24
FONT = ImageFont.truetype(matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(
    family=["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"])), 16)

for arg in sys.argv[1:]:
    path, labels = arg.split(":", 1)
    labels = labels.split(",")
    gif = Image.open(path)
    assert gif.size == (628, 232), f"{path}: not an old-style trajectory GIF"
    # keep the 32-level colormap palette of the old GIF; its dark ground and grey text levels become white and black
    pal = np.array(gif.getpalette()[:768], dtype=float).reshape(-1, 3)
    for k, a in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        pal[k] = 255 * (1 - a)
    ref = Image.new("P", (1, 1))
    ref.putpalette(np.round(pal).astype(np.uint8).flatten().tolist())
    w, h = 2 * PAD + 3 * SIZE + 2 * GAP, TOP + SIZE + PAD
    frames, durations, canvas = [], [], None
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info.get("duration", 83))
        fr = gif.convert("RGBA")
        canvas = fr if canvas is None else Image.alpha_composite(canvas, fr)
        out = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(out)
        for j, x0 in enumerate(X0S):
            panel = canvas.crop((x0, Y0, x0 + SRC, Y0 + SRC)).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
            x = PAD + j * (SIZE + GAP)
            out.paste(panel, (x, TOP))
            draw.text((x, 2), labels[j], fill="black", font=FONT)
        frames.append(out.quantize(palette=ref, dither=Image.Dither.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False,
                   disposal=1)
    print(f"{path}: {len(frames)} frames, {w}x{h}")
