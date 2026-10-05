from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PLOT_STYLE  # noqa: E402,F401  (re-exported for the figure scripts)
from report.make_seed_csv_from_fast import read_legacy_csv  # noqa: E402

PDE_NAMES = {
    "navier_stokes_kolmogorov_flow_2d_low_res": "Navier-Stokes",
    "kuramoto_sivashinsky_2d_low_res": "Kuramoto-Sivashinsky",
    "gray_scott_beta_low_res": "Gray-Scott",
}
ARCH_NAMES = {"unet_cond": "UNet", "fno_cond": "FNO", "scot_t256": "scOT"}

BASELINE = "no_resampling"
TABLE_STRATEGIES = {
    "ddpm_conf_ratio_proportional_normalized_sm": "OGAS-L",
    "ddpm_conf_ratio_proportional_normalized": "OGAS-L2",
    "ddpm_conf_ratio_proportional_uncertainty": "OGAS-U",
    "sbal": "SBAL",
    "sbal_topk": "Top-K",
    "no_resampling_sobol": "Sobol",
    BASELINE: "Uniform",
}
METRIC_PREFIX = "rmse_normalized"
STAT_NAMES = {"mean": "Mean", "std": "Std", "max": "Max", "min": "Min", "p10": "P10", "p25": "P25",
              "p50": "P50", "p75": "P75", "p90": "P90", "p95": "P95", "p99": "P99"}
PAPER_STATS = "mean,std,max,p99,p95,p75,p50"
METRIC_LABEL = "RMSE-normalized"

SCIENCE_STYLE = {
    "figure.figsize": [3.5, 2.5], "figure.dpi": 600,
    "xtick.direction": "in", "xtick.major.size": 3, "xtick.major.width": 0.5, "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5, "xtick.minor.visible": False, "xtick.top": True,
    "ytick.direction": "in", "ytick.major.size": 3, "ytick.major.width": 0.5, "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5, "ytick.minor.visible": True, "ytick.right": True,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
    "legend.borderaxespad": 0.8, "legend.borderpad": 0.2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"], "font.family": "serif", "font.size": 10,
    "axes.formatter.use_mathtext": True, "mathtext.fontset": "cm", "text.usetex": False,
}


def ordered(present, order) -> list:
    present = set(present)
    return [x for x in order if x in present] + sorted(present - set(order))


def load_per_seed(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(["object", "string"]).columns:
        df[c] = df[c].str.strip()
    df = df[df.strategy.isin(TABLE_STRATEGIES)]
    return df.groupby(["pde", "architecture", "strategy", "seed"]).mean(numeric_only=True).reset_index()


def ratio_to_uniform(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    nan = float("nan")
    base = df[df.strategy == BASELINE].groupby(["pde", "architecture"])[cols].mean().replace(0, nan)
    key = pd.MultiIndex.from_frame(df[["pde", "architecture"]])
    out[cols] = base.reindex(key).to_numpy() / df[cols].replace(0, nan).to_numpy()
    return out


def legacy_pairs(path) -> pd.DataFrame:
    return read_legacy_csv(path)[["pde", "architecture", "strategy", "seed", "checkpoint_kind"]].drop_duplicates()


def read_long(path, legacy) -> pd.DataFrame:
    long = pd.read_csv(path, low_memory=False).rename(columns={"method": "strategy"})
    long["seed"] = long.seed.astype(str)
    return long.merge(legacy_pairs(legacy), on=["pde", "architecture", "strategy", "seed", "checkpoint_kind"])


def per_seed_variants(long: pd.DataFrame, cols) -> pd.DataFrame:
    one = long[long.phase == "one_step"].copy()
    one["variant"] = "onestep_" + one.normalization + "_" + one.aggregation
    one["horizon"] = 0
    roll = long[long.phase == "rollout"].copy()
    roll["variant"] = "rollout_" + roll.normalization
    roll["horizon"] = roll.rollout_step.astype(int)
    v = pd.concat([one, roll], ignore_index=True)
    return v.groupby(["pde", "architecture", "strategy", "seed", "variant", "horizon"])[cols].mean().reset_index()
