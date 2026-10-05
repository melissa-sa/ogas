#!/usr/bin/env python3
"""Per-generation statistics and plots of the parameters sampled by an active-sampling run.

A generation is --gen-size consecutive parameter vectors (sim id // gen size). Writes
generation_parameter_stats.csv and, on request, mean/IQR evolution (--plot), histogram overlays
(--plot-hist), histogram time series (--plot-hist-ts), ridgelines (--plot-ridge) and scatter plots
of parameter pairs coloured by generation (--plot-2d). Besides parameter names or indices,
--params accepts mean_blob_amplitude / mean_blob_center_x / mean_blob_center_y.

Usage (see scripts/leo/job_ic_param_evolution_root.sh):
  parameter_generation_evolution.py --exp-dir <run>/seed_1 --output DIR --params domain_extent cutoff \
      [--plot] [--plot-hist] [--plot-hist-ts [--combine]] [--plot-ridge] [--plot-2d --pairs 'a,b;c,d']
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional styling
    sns = None

from common import blob_indices, load_config  # noqa: E402

from apebench_online.scenarios.physics_ic_utils import layout_from_scenario_config  # noqa: E402

DISPLAY_NAMES = {
    "diffusivity": "viscosity",
    "domain_extent": "Domain Size",
}
PDE_ABBREVIATIONS = {
    "navier_stokes_kolmogorov_flow_2d_low_res": "NS",
    "kuramoto_sivashinsky_2d_low_res": "KS",
    "gray_scott_beta_low_res": "GS",
}
ARCH_ABBREVIATIONS = {
    "fno_cond": "FNO",
    "unet_cond": "UNet",
    "scot_t256": "SCOT",
}
BLOB_MEANS = (
    ({"mean_blob_amplitude", "mean_amplitude"}, "amplitude"),
    ({"mean_blob_center_x", "mean_center_x"}, "center_x"),
    ({"mean_blob_center_y", "mean_center_y"}, "center_y"),
)


def _display_name(name: str) -> str:
    mapped = DISPLAY_NAMES.get(name, name)
    if mapped == name:
        mapped = mapped.replace("_", " ")
    return mapped.title()


class Layout:
    def __init__(self, config: Dict[str, Any], no_scale_physics: bool):
        layout = layout_from_scenario_config(config.get("study_options", {}).get("scenario_config", {}))
        self.ic_count = int(layout["ic_param_count"])
        self.physics_specs = list(layout["physics_specs"])
        self.names = list(layout["ic_param_names"])
        self.physics: Dict[str, Tuple[int, Dict[str, Any], int]] = {}
        for spec in self.physics_specs:
            dim = int(spec.get("dim", 1))
            base = spec.get("name", "physics")
            for i in range(dim):
                name = base if dim == 1 else f"{base}_{i}"
                self.physics[name] = self.physics[f"physics/{name}"] = (len(self.names), spec, i)
                self.names.append(name)
        self.no_scale_physics = no_scale_physics

    def values(self, chunk: np.ndarray, name: str) -> np.ndarray:
        for aliases, suffix in BLOB_MEANS:
            if name in aliases:
                indices = blob_indices(self.names, suffix)
                if not indices:
                    raise KeyError(f"No blob {suffix} parameters found")
                return np.mean(chunk[:, indices], axis=1)
        if name.isdigit():
            idx = int(name)
        elif name in self.names:
            idx = self.names.index(name)
        elif name in self.physics:
            idx = self.physics[name][0]
        else:
            raise KeyError(f"Unknown parameter '{name}'")
        values = chunk[:, idx].astype(np.float64)
        if not self.no_scale_physics and idx >= self.ic_count:
            for pidx, spec, comp in self.physics.values():
                if pidx == idx:
                    return _scale_physics_values(values, spec, comp)
        return values


def _scale_physics_values(raw: np.ndarray, spec: Dict[str, Any], comp: int) -> np.ndarray:
    if spec.get("sampling_scale", "linear") != "log":
        return raw
    min_val, max_val = spec.get("min"), spec.get("max")
    if isinstance(min_val, (list, tuple, np.ndarray)):
        min_val = min_val[comp]
    if isinstance(max_val, (list, tuple, np.ndarray)):
        max_val = max_val[comp]
    if min_val is None or max_val is None or max_val <= min_val:
        return raw
    frac = np.clip((raw - float(min_val)) / float(max_val - min_val), 0.0, 1.0)
    log_min, log_max = np.log(float(min_val)), np.log(float(max_val))
    return np.exp(log_min + frac * (log_max - log_min))


class Figures:
    def __init__(self, output_dir: Path, pdf_only: bool, pde_name: Optional[str], arch_name: Optional[str]):
        self.dir = output_dir
        self.pdf_only = pdf_only
        titles = [PDE_ABBREVIATIONS.get(pde_name, _display_name(pde_name))] if pde_name else []
        files = [PDE_ABBREVIATIONS.get(pde_name, pde_name)] if pde_name else []
        if arch_name:
            titles.append(ARCH_ABBREVIATIONS.get(arch_name, _display_name(arch_name)))
            files.append(ARCH_ABBREVIATIONS.get(arch_name, arch_name))
        self.title = " — ".join(titles) + " — " if titles else ""
        self.prefix = "__".join(files) + "__" if files else ""

    def save(self, fig, name: str, pdf_with_png: bool = True) -> None:
        if not self.pdf_only:
            fig.savefig(self.dir / f"{self.prefix}{name}.png", dpi=300)
        if pdf_with_png or self.pdf_only:
            fig.savefig(self.dir / f"{self.prefix}{name}.pdf")
        plt.close(fig)


def _load_parameters_array(params_path: Path, config: Dict[str, Any]) -> np.ndarray:
    try:
        params = np.load(params_path, mmap_mode="r", allow_pickle=True)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        return params
    except Exception:
        file_size = params_path.stat().st_size
        total_values = file_size // 4  # float32
        nb_params_config = len(config.get("study_options", {}).get("l_bounds", []))
        nb_params = None
        for test_nb in (nb_params_config, nb_params_config + 1, nb_params_config + 2):
            if test_nb > 0 and total_values % test_nb == 0:
                nb_params = test_nb
                break
        if nb_params is None:
            raise SystemExit(f"Could not determine parameter count from file size {file_size}")
        n_sims = total_values // nb_params
        return np.memmap(params_path, dtype=np.float32, mode="r").reshape((n_sims, nb_params))


def _auto_find_study_config(start_dir: Path) -> Optional[Path]:
    for directory in (start_dir, start_dir.parent, start_dir.parent.parent):
        for pattern in ("config_online_*.json", "config_offline_*.json"):
            candidates = sorted(directory.glob(pattern))
            if candidates:
                return candidates[0]
    return None


def _selected_generations(n_samples: int, gen_size: int, every: int) -> List[int]:
    n_gen = int(math.ceil(n_samples / gen_size))
    gens = list(range(0, n_gen, max(int(every), 1)))
    if (n_gen - 1) not in gens:
        gens.append(n_gen - 1)
    return gens


def _series(params: np.ndarray, layout: Layout, name: str, gen_size: int, gens: List[int]):
    series = []
    for g in gens:
        start, end = g * gen_size, min((g + 1) * gen_size, params.shape[0])
        if start >= end:
            continue
        values = layout.values(params[start:end], name)
        if values.size:
            series.append((g, values))
    return series


def _hist_bins(series, bins: int):
    all_vals = np.concatenate([v for _, v in series])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if vmin == vmax:
        vmax = vmin + 1e-6
    edges = np.linspace(vmin, vmax, bins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _stats_rows(params: np.ndarray, layout: Layout, names: List[str], gen_size: int, skip_missing: bool):
    rows = []
    for g in range(int(math.ceil(params.shape[0] / gen_size))):
        start, end = g * gen_size, min((g + 1) * gen_size, params.shape[0])
        for name in names:
            try:
                values = layout.values(params[start:end], name)
            except KeyError as exc:
                if skip_missing:
                    print(f"[WARN] {exc}; skipping {name}")
                    continue
                raise
            row = {"generation": g, "start": start, "end": end, "param": name,
                   "mean": float(np.nanmean(values)), "std": float(np.nanstd(values)),
                   "min": float(np.nanmin(values)), "max": float(np.nanmax(values))}
            for q in (10, 25, 50, 75, 90):
                row[f"p{q}"] = float(np.nanpercentile(values, q))
            row["count"] = int(end - start)
            rows.append(row)
    return rows


def _plot_param_evolution(rows: List[Dict[str, Any]], figs: Figures) -> None:
    by_param: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_param.setdefault(row["param"], []).append(row)
    for param, entries in by_param.items():
        entries.sort(key=lambda r: r["generation"])
        gens = [r["generation"] for r in entries]
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(gens, [r["mean"] for r in entries], label="mean")
        ax.fill_between(gens, [r["p25"] for r in entries], [r["p75"] for r in entries], alpha=0.2, label="IQR")
        ax.set_xlabel("generation")
        ax.set_ylabel(_display_name(param))
        ax.set_title(f"{figs.title}{_display_name(param)} evolution", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        figs.save(fig, f"evolution_{param}")


def _plot_param_histograms(params, layout, names, gen_size, every, bins, figs: Figures) -> None:
    gens = _selected_generations(params.shape[0], gen_size, every)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(gens), vmax=max(gens))
    for name in names:
        try:
            series = _series(params, layout, name, gen_size, gens)
        except KeyError:
            continue
        if not series:
            continue
        edges, centers = _hist_bins(series, bins)
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        for g, vals in series:
            hist, _ = np.histogram(vals, bins=edges, density=True)
            ax.plot(centers, hist, color=cmap(norm(g)), linewidth=1.2, alpha=0.9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("generation", fontsize=8)
        ax.set_title(f"{figs.title}{_display_name(name)} distribution over generations", fontweight="bold")
        ax.set_xlabel(_display_name(name))
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs.save(fig, f"hist_{name}")


def _plot_hist_timeseries(params, layout, names, gen_size, every, bins, figs: Figures, combine: bool) -> None:
    if not names:
        return
    gens = _selected_generations(params.shape[0], gen_size, every)
    fig = None
    if combine:
        fig, axes = plt.subplots(len(names), 1, figsize=(7.2, max(2.2, 2.0 * len(names))), sharex=False)
        axes = [axes] if len(names) == 1 else axes
    for idx, name in enumerate(names):
        try:
            series = _series(params, layout, name, gen_size, gens)
        except KeyError as exc:
            print(f"[WARN] {exc}; skipping {name}")
            continue
        if not series:
            continue
        edges, centers = _hist_bins(series, bins)
        hist_mat = np.asarray([np.histogram(v, bins=edges, density=True)[0] for _, v in series], dtype=np.float64)
        gen_arr = np.asarray([g for g, _ in series], dtype=np.int64)

        if combine:
            ax = axes[idx]
        else:
            fig, ax = plt.subplots(figsize=(7.2, 2.6))
        step = max(int(every), 1)
        if gen_arr.size >= 2:
            step = int(np.median(np.diff(gen_arr)))
        y_edges = np.concatenate(([gen_arr[0] - 0.5 * step], gen_arr + 0.5 * step))
        x_edges = np.concatenate(([centers[0] - (centers[1] - centers[0]) / 2], (centers[:-1] + centers[1:]) / 2,
                                  [centers[-1] + (centers[-1] - centers[-2]) / 2]))
        mesh = ax.pcolormesh(x_edges, y_edges, hist_mat, shading="auto", cmap="viridis")
        ax.set_ylabel("generation")
        ax.set_title(f"{figs.title}{_display_name(name)} distribution over generations", fontweight="bold")
        ax.grid(False)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("density", fontsize=8)
        if not combine:
            ax.set_xlabel(_display_name(name))
            fig.tight_layout()
            figs.save(fig, f"hist_ts_{name}", pdf_with_png=False)

    if combine and fig is not None:
        axes[-1].set_xlabel("value")
        if figs.title:
            fig.suptitle(f"{figs.title}Parameter distributions", fontweight="bold", y=0.98)
        fig.tight_layout()
        figs.save(fig, "hist_ts_combined")


def _plot_ridgeline(params, layout, names, gen_size, every, bins, figs: Figures) -> None:
    gens = _selected_generations(params.shape[0], gen_size, every)
    cmap = plt.get_cmap("Oranges")
    norm = plt.Normalize(vmin=min(gens), vmax=max(gens))
    step = 1.0
    if len(gens) > 1:
        step = float(np.median(np.diff(sorted(gens))))
        if step <= 0:
            step = 1.0
    for name in names:
        try:
            series = _series(params, layout, name, gen_size, gens)
        except KeyError:
            continue
        if not series:
            continue
        edges, centers = _hist_bins(series, bins)
        hists = [np.histogram(vals, bins=edges, density=True)[0] for _, vals in series]
        max_hist = float(np.nanmax(hists)) if hists else 1.0
        scale = (0.8 * step) / max_hist if max_hist > 0 else 1.0

        fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.18 * len(series) + 2.0)))
        for (g, _), hist in zip(series, hists):
            y = g + hist * scale
            ax.fill_between(centers, g, y, color=cmap(norm(g)), alpha=0.9, linewidth=0.6)
            ax.plot(centers, y, color="white", linewidth=0.6, alpha=0.9)
        ax.set_title(f"{figs.title}{_display_name(name)} — distribution per generation", fontweight="bold")
        ax.set_xlabel(_display_name(name))
        ax.set_ylabel("generation")
        ax.set_yticks(gens[:: max(1, len(gens) // 8)])
        ax.grid(False)
        fig.tight_layout()
        figs.save(fig, f"ridge_{name}")


def _parse_pairs(raw: Optional[str]) -> List[Tuple[str, str]]:
    pairs = []
    for chunk in (raw or "").split(";"):
        if "," not in chunk:
            continue
        a, b = [p.strip() for p in chunk.strip().split(",", 1)]
        if a and b:
            pairs.append((a, b))
    return pairs


def _plot_2d_pairs(params, layout, names, gen_size, figs: Figures, max_points: int, pairs) -> None:
    n_samples = params.shape[0]
    gens = np.arange(n_samples, dtype=np.int64) // max(int(gen_size), 1)
    rng = np.random.RandomState(0)
    if max_points > 0 and n_samples > max_points:
        idx = rng.choice(n_samples, size=max_points, replace=False)
    else:
        idx = np.arange(n_samples)
    for name_x, name_y in pairs or list(itertools.combinations(names, 2)):
        try:
            x = layout.values(params, name_x)
            y = layout.values(params, name_y)
        except KeyError as exc:
            print(f"[WARN] {exc}; skipping pair {name_x},{name_y}")
            continue
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        sc = ax.scatter(x[idx], y[idx], c=gens[idx], s=8, alpha=0.35, cmap="viridis", edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("generation", fontsize=8)
        ax.set_xlabel(_display_name(name_x))
        ax.set_ylabel(_display_name(name_y))
        ax.set_title(f"{figs.title}{_display_name(name_x)} vs {_display_name(name_y)}", fontweight="bold")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        figs.save(fig, f"pair_{name_x}__{name_y}".replace("/", "_"), pdf_with_png=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter distribution evolution by generation")
    parser.add_argument("--exp-dir", type=str, help="Experiment directory")
    parser.add_argument("--parameters", type=str, help="Path to parameters .npy file")
    parser.add_argument("--study-config", type=str, help="Path to config_online/offline JSON")
    parser.add_argument("--gen-size", type=int, default=56, help="Number of samples per generation")
    parser.add_argument("--params", nargs="*", help="Parameter names or indices (use 'all' for all)")
    parser.add_argument("--no-scale-physics", action="store_true", help="Disable log scaling for physics")
    parser.add_argument("--plot", action="store_true", help="Save mean/IQR evolution plots")
    parser.add_argument("--plot-hist", action="store_true", help="Save histogram evolution plots")
    parser.add_argument("--plot-hist-ts", action="store_true", help="Save histogram time-series plots")
    parser.add_argument("--plot-ridge", action="store_true", help="Save ridgeline (distribution) plots")
    parser.add_argument("--plot-2d", action="store_true", help="Save 2D parameter pair plots")
    parser.add_argument("--hist-every", type=int, default=10, help="Plot every N generations")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bins")
    parser.add_argument("--combine", action="store_true", help="Combine params in one time-series plot")
    parser.add_argument("--pairs", type=str, help="Semicolon-separated pairs like 'diffusivity,cutoff;domain_extent,cutoff'")
    parser.add_argument("--max-2d-points", type=int, default=20000, help="Max points for 2D scatter")
    parser.add_argument("--pdf-only", action="store_true", help="Only write PDF outputs (no PNG)")
    parser.add_argument("--pde-name", type=str, default=None, help="PDE name for titles/filenames")
    parser.add_argument("--arch-name", type=str, default=None, help="Architecture name for titles/filenames")
    parser.add_argument("--skip-missing", action="store_true", help="Skip params that are missing")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figs = Figures(output_dir, args.pdf_only, args.pde_name, args.arch_name)

    exp_dir = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else None
    params_path = Path(args.parameters).expanduser().resolve() if args.parameters else None
    if params_path is None and exp_dir is not None:
        for candidate in (exp_dir / "checkpoints" / "sampled_parameters.npy",
                          exp_dir / "trajectories" / "input_parameters.npy"):
            if candidate.exists():
                params_path = candidate
                break
    if params_path is None:
        raise SystemExit("Provide --parameters or --exp-dir with sampled_parameters.npy")

    study_config_path = Path(args.study_config).expanduser().resolve() if args.study_config else None
    if study_config_path is None:
        study_config_path = _auto_find_study_config(params_path.parent)
        if study_config_path is None:
            raise SystemExit("Could not auto-detect study config; pass --study-config")

    config = load_config(study_config_path)
    params = _load_parameters_array(params_path, config)
    layout = Layout(config, args.no_scale_physics)

    if not args.params:
        if layout.physics_specs:
            args.params = [spec.get("name", "physics") for spec in layout.physics_specs]
        else:
            args.params = layout.names[:1]
    if not args.params:
        raise SystemExit("No parameters to analyze")

    gen_size = max(int(args.gen_size), 1)
    rows = _stats_rows(params, layout, args.params, gen_size, args.skip_missing)
    out_csv = output_dir / "generation_parameter_stats.csv"
    if rows:
        headers = list(rows[0].keys())
        with out_csv.open("w") as f:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    print(f"Saved: {out_csv}")

    if sns is not None and (args.plot or args.plot_hist or args.plot_hist_ts or args.plot_ridge or args.plot_2d):
        sns.set_theme(style="whitegrid", context="paper")
    if args.plot:
        _plot_param_evolution(rows, figs)
        print(f"Evolution plots saved to {output_dir}")
    if args.plot_hist:
        _plot_param_histograms(params, layout, args.params, gen_size, args.hist_every, args.hist_bins, figs)
        print(f"Histogram plots saved to {output_dir}")
    if args.plot_hist_ts:
        _plot_hist_timeseries(params, layout, args.params, gen_size, args.hist_every, args.hist_bins, figs,
                              args.combine)
        print(f"Histogram time-series plots saved to {output_dir}")
    if args.plot_ridge:
        _plot_ridgeline(params, layout, args.params, gen_size, args.hist_every, args.hist_bins, figs)
        print(f"Ridgeline plots saved to {output_dir}")
    if args.plot_2d:
        _plot_2d_pairs(params, layout, args.params, gen_size, figs, args.max_2d_points, _parse_pairs(args.pairs))
        print(f"2D pair plots saved to {output_dir}")


if __name__ == "__main__":
    main()
