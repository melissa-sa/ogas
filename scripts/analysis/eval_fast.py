#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import socket
import time
from pathlib import Path

from common import load_config, run_config_path

import numpy as np
import torch

from apebench_online.scenarios.physics_ic_utils import physics_specs_from_registry
from apebench_online.scenarios.scenarios_utils import MelissaSpecificScenario
from apebench_online.validation.external_validation.torch_support import init_torch_model

CLAIM_STALE_S = 45 * 60


def tag_of(exp_dir: Path) -> str:
    digest = hashlib.sha1(str(exp_dir).encode()).hexdigest()[:10]
    return f"{digest}__" + "__".join(exp_dir.parts[-4:])


def list_tasks(roots):
    tasks = []
    for root in roots:
        for exp in sorted(p for p in Path(root).rglob("seed_*") if (p / "checkpoints").is_dir()):
            try:
                cfg = run_config_path(exp)
            except FileNotFoundError:
                continue
            pde, arch, method = exp.parts[-4:-1]
            for ckpt in sorted((exp / "checkpoints").glob("model_*.pt")):
                tasks.append({"pde": pde, "arch": arch, "method": method, "seed": exp.name[5:],
                              "exp": str(exp), "config": str(cfg), "ckpt": str(ckpt),
                              "out": f"{tag_of(exp)}__{ckpt.stem}.npz"})
    return tasks


def claim(out_dir: Path, name: str) -> bool:
    lock = out_dir / "claims" / name
    try:
        os.mkdir(lock)
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime < CLAIM_STALE_S:
                return False
            os.utime(lock)
        except FileNotFoundError:
            return False
    (lock / "owner").write_text(f"{socket.gethostname()} {os.getpid()} {time.ctime()}\n")
    return True


def load_dataset(val_dir: str, device):
    files = sorted(
        (f for f in glob.glob(f"{val_dir}/*.npy") if os.path.basename(f).startswith("sim")),
        key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit())),
    )
    first = np.load(files[0], mmap_mode="r")
    data = torch.empty((len(files),) + first.shape, dtype=torch.float32, device=device)
    for i, f in enumerate(files):
        data[i] = torch.from_numpy(np.load(f).astype(np.float32, copy=False))
    params = np.load(glob.glob(f"{val_dir}/*param*.npy")[0])
    return data, params, files


def build_model(config: dict, device, prediction_mode: str = "mean"):
    dl = dict(config["dl_config"])
    sc = dict(config["study_options"]["scenario_config"])
    sample_physics = bool(sc.get("sample_physics", True))
    specs = physics_specs_from_registry(str(sc.get("scenario_name", ""))) if sample_physics else []
    sc["physics_param_specs"] = specs
    scenario = MelissaSpecificScenario(**sc)
    dl["torch_device"] = str(device)
    tm = dict(dl["torch_model"])
    tm["prediction_mode"] = prediction_mode
    dl["torch_model"] = tm
    model, _, grid = init_torch_model(dl, scenario, len(specs))
    pmin = pmax = None
    if dl.get("normalize_physics", False) and specs:
        pmin = torch.tensor([s["min"] for s in specs], dtype=torch.float32, device=device)
        pmax = torch.tensor([s["max"] for s in specs], dtype=torch.float32, device=device)
    return model, grid, len(specs), pmin, pmax, sample_physics


def physics_for(params, traj_idx, physics_dim, sample_physics, pmin, pmax, device):
    if physics_dim == 0:
        return None
    raw = params[traj_idx, -physics_dim:] if sample_physics else np.zeros((len(traj_idx), physics_dim), np.float32)
    phys = torch.as_tensor(raw, dtype=torch.float32, device=device)
    if pmin is not None:
        phys = (phys - pmin) / (pmax - pmin).clamp_min(1e-8)
    return phys


@torch.inference_mode()
def evaluate(model, grid, data, phys_all, rollout_steps, batch):
    n, t = data.shape[:2]
    dims = tuple(range(1, data.ndim - 1))
    onestep = torch.empty((n, t - 1), dtype=torch.float32, device=data.device)
    pairs = torch.arange(n * (t - 1), device=data.device)
    for start in range(0, pairs.numel(), batch):
        p = pairs[start:start + batch]
        tr, ts = p // (t - 1), p % (t - 1)
        x, y = data[tr, ts], data[tr, ts + 1]
        phys = None if phys_all is None else phys_all[tr]
        pred = model(x, grid.expand(x.shape[0], *grid.shape[1:]), phys)
        onestep[tr, ts] = torch.mean((pred - y) ** 2, dim=dims)
    rollout = torch.empty((n, rollout_steps), dtype=torch.float32, device=data.device)
    for start in range(0, n, batch):
        tr = torch.arange(start, min(n, start + batch), device=data.device)
        cur = data[tr, 0]
        phys = None if phys_all is None else phys_all[tr]
        g = grid.expand(cur.shape[0], *grid.shape[1:])
        for s in range(1, rollout_steps + 1):
            cur = model(cur, g, phys)
            rollout[tr, s - 1] = torch.mean((cur - data[tr, s]) ** 2, dim=dims)
    return onestep.double().cpu().numpy(), rollout.double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--val", action="append", required=True, help="pde=trajectories_dir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rollout-steps", type=int, default=50)
    ap.add_argument("--deadline-min", type=float, default=1e9)
    ap.add_argument("--only", default=None, help="substring filter on checkpoint path")
    ap.add_argument("--prediction-mode", choices=["mean", "member0"], default="mean",
                    help="ensemble mean, or the first member alone (compare ensemble sizes fairly)")
    a = ap.parse_args()

    t_start = time.time()
    out = Path(a.output)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "claims").mkdir(exist_ok=True)
    val = dict(item.split("=", 1) for item in a.val)
    device = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(device)
    tasks = [x for x in list_tasks(a.root) if x["pde"] in val and (a.only is None or a.only in x["ckpt"])]
    pdes = sorted({x["pde"] for x in tasks})
    pdes = pdes[a.gpu % len(pdes):] + pdes[: a.gpu % len(pdes)]
    log = lambda msg: print(f"[gpu{a.gpu} {time.time() - t_start:7.0f}s] {msg}", flush=True)

    for pde in pdes:
        todo = [x for x in tasks if x["pde"] == pde and not (out / "raw" / x["out"]).exists()]
        if not todo:
            continue
        if time.time() - t_start > a.deadline_min * 60:
            break
        t0 = time.time()
        data, params, files = load_dataset(val[pde], device)
        log(f"{pde}: loaded {tuple(data.shape)} in {time.time() - t0:.0f}s, {len(todo)} checkpoints left")
        energy_path = out / f"energy_{pde}.npz"
        if not energy_path.exists():
            dims = tuple(range(2, data.ndim))
            energy = torch.mean(data ** 2, dim=dims).double().cpu().numpy()
            tmp = energy_path.with_name(f"{energy_path.stem}.{socket.gethostname()}.{os.getpid()}.tmp.npz")
            np.savez(tmp, energy=energy, params=params, val_dir=val[pde], n_files=len(files))
            os.replace(tmp, energy_path)
        current_cfg, model = None, None
        for x in todo:
            if time.time() - t_start > a.deadline_min * 60:
                log("deadline reached, stopping")
                break
            target = out / "raw" / x["out"]
            if target.exists() or not claim(out, x["out"]):
                continue
            t1 = time.time()
            if x["config"] != current_cfg:
                config = load_config(x["config"])
                model, grid, pdim, pmin, pmax, sphys = build_model(config, device, a.prediction_mode)
                phys_all = physics_for(params, np.arange(data.shape[0]), pdim, sphys, pmin, pmax, device)
                current_cfg = x["config"]
            state = torch.load(x["ckpt"], map_location=device, weights_only=False)
            model.load_state_dict(state.get("model_state", {}))
            model.eval()
            onestep, rollout = evaluate(model, grid, data, phys_all, a.rollout_steps, a.batch)
            meta = {k: x[k] for k in ("pde", "arch", "method", "seed", "exp", "config", "ckpt")}
            meta.update(checkpoint_step=int(state.get("batch_idx", -1)), val_dir=val[pde],
                        n_models=int(getattr(model, "n_models", 1)), prediction_mode=a.prediction_mode,
                        seconds=time.time() - t1,
                        batch=a.batch, host=socket.gethostname())
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, onestep_mse=onestep, rollout_mse=rollout, meta=json.dumps(meta))
            os.replace(tmp, target)
            log(f"{x['out']} step={meta['checkpoint_step']} mse1={onestep.mean():.4e} "
                f"roll50={rollout[:, -1].mean():.4e} in {meta['seconds']:.0f}s")
        del data
        torch.cuda.empty_cache()
    log("worker finished")


if __name__ == "__main__":
    main()
