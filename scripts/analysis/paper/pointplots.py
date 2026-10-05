#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper_common import (ARCH_NAMES, BASELINE, METRIC_LABEL, METRIC_PREFIX, PAPER_STATS, PDE_NAMES,  # noqa: E402
                          SCIENCE_STYLE, TABLE_STRATEGIES, load_per_seed, ordered, ratio_to_uniform)

STYLE = {**SCIENCE_STYLE,
         "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 18,
         "axes.labelsize": 21, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
         "legend.fontsize": 19, "legend.title_fontsize": 19, "figure.titlesize": 22, "axes.linewidth": 0.8,
         "grid.linewidth": 0.5, "lines.linewidth": 1.5, "lines.markersize": 6, "savefig.dpi": 300,
         "savefig.bbox": "tight", "savefig.pad_inches": 0.05}
MARKERS = ["o", "X", "s"]


def pointplot(df, cols: list, keys: list, mode: str, out: Path):
    if mode == "ratio":
        df = ratio_to_uniform(df, cols)
        df = df[df.strategy != BASELINE]
    pdes = ordered(df.pde, PDE_NAMES)
    archs = ordered(df.architecture, ARCH_NAMES)
    strategies = [s for s in TABLE_STRATEGIES if s != BASELINE and s in set(df.strategy)]
    if mode == "raw" and BASELINE in set(df.strategy):
        strategies.append(BASELINE)
    xpos = [2 * i for i in range(len(strategies))]
    colors = sns.color_palette("colorblind")[:len(MARKERS)]

    fig, axes = plt.subplots(len(pdes), len(cols), figsize=(5 * len(cols), 3 * len(pdes)), squeeze=False,
                             sharey="row" if mode == "ratio" else False, sharex=True, constrained_layout=True)
    for i, pde in enumerate(pdes):
        for j, (col, key) in enumerate(zip(cols, keys)):
            ax = axes[i, j]
            values_all = []
            for k, arch in enumerate(archs):
                marker, color = MARKERS[k % len(MARKERS)], colors[k % len(colors)]
                dx = (k - (len(archs) - 1) / 2) * (1.0 / len(archs))
                d = df[(df.pde == pde) & (df.architecture == arch)]
                for x, strat in zip(xpos, strategies):
                    v = d.loc[d.strategy == strat, col].dropna().to_numpy()
                    if len(v) == 0:
                        continue
                    values_all.extend(v)
                    ax.scatter([x + dx], [v.mean()], marker=marker, color=color, s=80, edgecolors="black",
                               linewidths=0.5, zorder=3, label=ARCH_NAMES.get(arch, arch) if x == 0 else "")
                    xs = x + dx + np.arange(1, len(v) + 1) * 0.01 - 0.01
                    ax.scatter(xs, v, marker=marker, color=color, s=80, alpha=0.5, edgecolors="white", linewidths=0.25)
                    if len(v) > 1:
                        ax.plot([x + dx] * 2, [v.min(), v.max()], color=color, alpha=0.4, linewidth=1.5, zorder=1)
                if mode == "raw":
                    uni = d.loc[d.strategy == BASELINE, col].dropna().to_numpy()
                    if len(uni):
                        ax.axhline(y=uni.mean(), color=color, linestyle="--", linewidth=2.5, alpha=0.6, zorder=0.5)
            if mode == "ratio":
                ax.axhline(y=1, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=0)
                if values_all:
                    lo, hi = min(values_all), max(values_all)
                    span = (hi - lo) or max(abs(hi) * 0.1, 0.1)
                    lo, hi = max(0, lo - 0.1 * span), hi + 0.1 * span
                    ax.set_ylim(lo, hi) if lo <= 1 <= hi else ax.set_ylim(min(lo, 0.95), max(hi, 1.05))
            elif values_all:
                ax.set_yscale("linear")
            if i == 0:
                ax.set_title(f"{METRIC_LABEL}-{key}")
            if j == 0:
                ax.set_ylabel(PDE_NAMES.get(pde, pde))
            ax.set_xticks(xpos)
            if i == len(pdes) - 1:
                ax.set_xticklabels([TABLE_STRATEGIES[s] for s in strategies], rotation=90, ha="right")
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3, which="both")
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                title = Line2D([], [], marker="", ls="")
                fig.legend([title] + list(by_label.values()), ["Architecture"] + list(by_label),
                           loc="outside upper right", ncol=4, bbox_to_anchor=(0., 1.02, 1., .025))
    fig.align_ylabels()
    fig.supxlabel("Strategy", fontsize=plt.rcParams["axes.labelsize"], y=0.03)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
        fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-seed CSV (legacy layout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", nargs="+", choices=["raw", "ratio"], default=["raw", "ratio"])
    ap.add_argument("--stats", default=PAPER_STATS, help=f"comma-separated statistics (default {PAPER_STATS})")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(STYLE)
    keys = [k.strip() for k in a.stats.split(",")]
    cols = [f"{METRIC_PREFIX}_{k}" for k in keys]
    df = load_per_seed(a.input)
    for mode in a.mode:
        pointplot(df, cols, keys, mode, out / f"strategy_pointplot_{mode}_custom_metrics.pdf")


if __name__ == "__main__":
    main()
