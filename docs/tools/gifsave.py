import numpy as np
from PIL import Image

TRANSPARENT = 255


def save_gif(path, rgb_frames, durations, colors=128, reference=None, fixed=()):
    reference = range(0, len(rgb_frames), max(1, len(rgb_frames) // 6)) if reference is None else reference
    fixed = [tuple(int(v) for v in c) for c in fixed]
    ref = Image.fromarray(np.concatenate([np.asarray(rgb_frames[i]) for i in reference], 0))
    ref = ref.quantize(colors=min(colors, TRANSPARENT) - len(fixed), method=Image.Quantize.MEDIANCUT)
    rgb_pal = ref.getpalette()[:3 * (min(colors, TRANSPARENT) - len(fixed))] + [v for c in fixed for v in c]
    pal = rgb_pal + [255, 0, 255] * (256 - len(rgb_pal) // 3)
    ref = Image.new("P", (1, 1))
    ref.putpalette(pal)
    frames, kept, dur, prev = [], [], [], None
    for k, rgb in enumerate(rgb_frames):
        idx = np.asarray(rgb.quantize(palette=ref, dither=Image.Dither.NONE)).copy()
        if prev is not None and np.array_equal(idx, prev):
            dur[-1] += durations[k]
            continue
        out = idx.copy()
        if prev is not None:
            out[idx == prev] = TRANSPARENT
        prev = idx
        im = Image.fromarray(out, mode="P")
        im.putpalette(pal)
        frames.append(im)
        kept.append(k)
        dur.append(durations[k])
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                   transparency=TRANSPARENT, disposal=1, optimize=False)
    return kept


def check_gif(path, rgb_frames=None):
    im = Image.open(path)
    assert im.info.get("loop") == 0, f"{path}: no NETSCAPE loop=0 block"
    if rgb_frames is not None:
        worst = 0.0
        for i in range(im.n_frames):
            im.seek(i)
            got = np.asarray(im.convert("RGB"), dtype=float)
            worst = max(worst, np.abs(got - np.asarray(rgb_frames[i], dtype=float)).mean())
        return im.n_frames, worst
    return im.n_frames, None
