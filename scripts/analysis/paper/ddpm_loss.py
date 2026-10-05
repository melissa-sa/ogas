#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from paper_common import ARCH_NAMES, PDE_NAMES, SCIENCE_STYLE  # noqa: E402

STRATEGY = "ddpm_conf_ratio_proportional_normalized"
COLUMNS = ["_step", "DDPM/loss", "pde", "model", "seed", "strategy"]
STYLE = {**SCIENCE_STYLE, "font.family": "serif", "axes.labelsize": 14, "axes.titlesize": 16, "legend.fontsize": 12,
         "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 2, "figure.titlesize": 18,
         "savefig.dpi": 300}


def extract(export: str, out: str) -> None:
    n = 0
    with open(export, encoding="utf-8", errors="replace") as fin, open(out, "w", encoding="utf-8", newline="") as fout:
        reader, writer = csv.reader(fin), csv.writer(fout)
        header = [h.strip() for h in next(reader)]
        idx = [header.index(c) for c in COLUMNS]
        i_strat = header.index("strategy")
        writer.writerow(COLUMNS)
        for row in reader:
            if len(row) > i_strat and row[i_strat].strip() == STRATEGY:
                writer.writerow([row[i] if i < len(row) else "" for i in idx])
                n += 1
    print(f"wrote {out}: {n} rows")


def plot(history: str, out_dir: str, smooth: int) -> None:
    df = pd.read_csv(history)
    df["DDPM/loss"] = pd.to_numeric(df["DDPM/loss"], errors="coerce")
    df = df.dropna(subset=["DDPM/loss"]).sort_values(["pde", "model", "seed", "_step"])
    if smooth:
        df["DDPM/loss"] = df.groupby(["pde", "model", "seed"])["DDPM/loss"].transform(
            lambda x: x.rolling(window=smooth, min_periods=1).mean())
    df["Model"] = df.model.map(ARCH_NAMES).fillna(df.model)
    for pde, d in df.groupby("pde", sort=False):
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=d, x="_step", y="DDPM/loss", hue="Model", style="Model", estimator="mean",
                          errorbar="sd", palette="colorblind", linewidth=2, dashes=False)
        ax.set_yscale("log")
        plt.title(PDE_NAMES.get(pde, pde))
        plt.xlabel("Training Steps")
        plt.ylabel("DDPM Loss (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(title="Model")
        name = pde.replace("_low_res", "").replace("_2d", "") + (f"_smooth{smooth}" if smooth else "")
        path = os.path.join(out_dir, f"ddpm_loss_{name}.pdf")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="OGAS-L DDPM-loss history CSV (written when --wandb-export)")
    ap.add_argument("--wandb-export", help="full W&B export to extract --history from")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smooth", type=int, default=50, help="rolling-mean window in logged steps (0 = raw)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.wandb_export:
        extract(a.wandb_export, a.history)
    plt.rcParams.update(STYLE)
    plot(a.history, a.out_dir, a.smooth)


if __name__ == "__main__":
    main()
