from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("APEBENCH_ROOT", str(REPO))

VAL_ROOT = os.path.join(os.environ.get("LEONARDO_PROJECT_ROOT", "."), "experiments/conditional_models_exp/val_data")

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
}


def load_config(path) -> dict:
    import rapidjson

    with open(path) as f:
        return rapidjson.load(f, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)


def run_config_path(run_dir) -> Path:
    run_dir = Path(run_dir)
    found = sorted(run_dir.glob("config_online_*.json"))
    found += [p for p in sorted(run_dir.glob("*.json")) if p not in found and '"study_options"' in p.read_text()]
    if not found:
        raise FileNotFoundError(f"no run config in {run_dir}")
    return found[0]


def offline_config_path(val_root: str, pde: str) -> str:
    return glob.glob(f"{val_root}/{pde}/config_offline*.json")[0]


def load_sampled_parameters(path, width: int, rows: str | None = None):
    raw = np.fromfile(path, dtype=np.float32)
    if raw[:2].view(np.uint8)[:6].tobytes() == b"\x93NUMPY":
        raw = np.load(path).astype(np.float32).ravel()
    lo, hi = (int(x) for x in (rows or f"0:{raw.size // width}").split(":"))
    return np.ascontiguousarray(raw.reshape(-1, width)[lo:hi]), (lo, hi)


def blob_indices(names, suffix: str) -> list:
    return [i for i, name in enumerate(names) if "blobs" in name and name.endswith(suffix)]
