#!/usr/bin/env python3
import argparse

import pandas as pd


def read_legacy_csv(path: str) -> pd.DataFrame:
    legacy = pd.read_csv(path)
    legacy.columns = legacy.columns.str.strip()
    for c in legacy.select_dtypes(["object", "string"]).columns:
        legacy[c] = legacy[c].str.strip()
    legacy["checkpoint_kind"] = legacy["model_best_path"].str.extract(r"checkpoints/(.*)\.pt$")[0]
    legacy["seed"] = legacy["seed"].astype(str)
    return legacy


STATS = ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--normalization", default="target")
    ap.add_argument("--aggregation", default="transition")
    a = ap.parse_args()

    legacy = read_legacy_csv(a.legacy)

    long = pd.read_csv(a.long, low_memory=False)
    long["seed"] = long["seed"].astype(str)
    sel = long[(long.phase == "one_step") & (long.normalization == a.normalization)
               & (long.aggregation == a.aggregation)].rename(columns={"method": "strategy"})

    keys = ["pde", "architecture", "strategy", "seed", "checkpoint_kind"]
    merged = legacy[keys + ["group", "experiment_dir", "is_ensemble", "target_step"]].merge(sel, on=keys, how="left")
    missing = merged["rmse_normalized_mean"].isna().sum()
    out = pd.DataFrame({
        "group": merged["group"], "pde": merged["pde"], "architecture": merged["architecture"],
        "strategy": merged["strategy"], "seed": merged["seed"],
        "experiment_dir": merged["experiment_dir"], "model_best_path": merged["checkpoint_path"],
        "is_ensemble": merged["is_ensemble"], "batch_idx": merged["checkpoint_step"],
        "target_step": merged["target_step"], "model_checkpoint_kind": merged["checkpoint_kind"],
        "avg_val_loss": merged["validation_batch_loss"], "rmse": merged["rmse_mean"],
        "rmse_normalized": merged["rmse_normalized_mean"],
    })
    for s in STATS:
        out[f"rmse_normalized_{s}"] = merged[f"rmse_normalized_{s}"]
        out[f"mse_{s}"] = merged[f"mse_{s}"]
    for b in ("0", "0.5", "0.8", "0.9", "0.95", "0.99"):
        out[f"rmse_normalized_cvar_{b}"] = merged[f"rmse_normalized_cvar_{b}"]
    out["normalization"] = a.normalization
    out["aggregation"] = a.aggregation
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, {missing} legacy rows without a fast evaluation")


if __name__ == "__main__":
    main()
