#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

QUANTILES = (10, 25, 50, 75, 90, 95, 99)
CVAR_LEVELS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)
HORIZONS = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def cvar(values, beta):
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    if beta == 0.0:
        return float(v.mean())
    mass = 1.0 / v.size
    tail = 1.0 - beta
    k = int(np.floor(tail / mass + 1e-12))
    total = v[:k].sum() * mass
    rest = tail - k * mass
    if rest > 1e-12 and k < v.size:
        total += rest * v[k]
    return float(total / tail)


def stats(values):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    risk = np.where(finite, v, np.inf)
    out = {"n": int(v.size), "failure_rate": float(1 - finite.mean())}
    if not finite.any():
        return out
    f = v[finite]
    out.update(mean=float(f.mean()), std=float(f.std()), min=float(f.min()), max=float(f.max()))
    for q in QUANTILES:
        out[f"p{q}"] = float(np.percentile(f, q))
    for b in CVAR_LEVELS:
        out[f"cvar_{b:g}"] = cvar(risk, b) if np.isfinite(risk).all() else float("inf")
    return out


def rows_for(meta, onestep, rollout, energy):
    e_t = np.maximum(energy[:, 1:], 1e-12)
    e_i = np.maximum(energy[:, :-1], 1e-12)
    base = {
        "pde": meta["pde"], "architecture": meta["arch"], "method": meta["method"],
        "seed": meta["seed"], "checkpoint_kind": os.path.splitext(os.path.basename(meta["ckpt"]))[0],
        "checkpoint_step": meta["checkpoint_step"], "checkpoint_path": meta["ckpt"],
        "val_dir": meta["val_dir"], "n_trajectories": onestep.shape[0],
        "validation_batch_loss": float(onestep.mean()),
    }
    out = []
    rmse1 = np.sqrt(onestep)
    for norm, den in (("target", e_t), ("input", e_i)):
        nrm = rmse1 / np.sqrt(den)
        for agg in ("transition", "trajectory"):
            red = (lambda a: a.reshape(-1)) if agg == "transition" else (lambda a: a.mean(axis=1))
            row = dict(base, phase="one_step", rollout_step=0, normalization=norm, aggregation=agg)
            for name, arr in (("mse", onestep), ("rmse", rmse1), ("rmse_normalized", nrm)):
                for k, val in stats(red(arr)).items():
                    row[f"{name}_{k}"] = val
            out.append(row)
    for h in HORIZONS:
        if h > rollout.shape[1]:
            continue
        mse_h = rollout[:, h - 1]
        rmse_h = np.sqrt(mse_h)
        for norm, den in (("target", energy[:, h]), ("ic", energy[:, 0])):
            row = dict(base, phase="rollout", rollout_step=h, normalization=norm, aggregation="endpoint")
            for name, arr in (("mse", mse_h), ("rmse", rmse_h),
                              ("rmse_normalized", rmse_h / np.sqrt(np.maximum(den, 1e-12)))):
                for k, val in stats(arr).items():
                    row[f"{name}_{k}"] = val
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    energies = {}
    for p in glob.glob(f"{a.output}/energy_*.npz"):
        if ".tmp." in p:
            continue
        z = np.load(p)
        energies[os.path.basename(p)[7:-4]] = (str(z["val_dir"]), z["energy"])
    rows, mismatched = [], []
    for path in sorted(glob.glob(f"{a.output}/raw/*.npz")):
        if ".tmp." in path:
            continue
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        val_dir, energy = energies.get(meta["pde"], (None, None))
        if val_dir != meta["val_dir"]:
            mismatched.append(path)
            continue
        rows.extend(rows_for(meta, z["onestep_mse"], z["rollout_mse"], energy))
    if mismatched:
        print(f"WARNING: {len(mismatched)} raw files skipped (val_dir differs from energy file)")
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.output}/metrics_long.csv", index=False)
    print("rows", len(df), "checkpoints", df.checkpoint_path.nunique())


if __name__ == "__main__":
    main()
