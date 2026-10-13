"""Unified helpers for physics and initial-condition parameter layouts.

This module is the single source of truth for:
- which IC family is used for each supported scenario,
- how the flat Melissa parameter vector is laid out,
- how that vector is decoded back into structured IC/physics configs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from .pde_registry import get_pde_config

_IC_FOURIER = "fourier"
_IC_GAUSSIAN_BLOBS = "gaussian_blobs"
_IC_SINE_SUP = "sine_sup"
_ALLOWED_IC_FAMILIES = (_IC_FOURIER, _IC_GAUSSIAN_BLOBS, _IC_SINE_SUP)

_FOURIER = {
    "cutoff": (2.0, 8.0),
    "taper": (1.0, 10.0),
    "amplitude": (-1.0, 1.0),
    "phase": (0.0, 2.0 * math.pi),
    "std_one": False,
    "max_one": True,
}

_GAUSSIAN = {
    "num_blobs": 8,
    "fields": (
        ("amplitude", 0.0, 1.0),
        ("sigma_x", 0.07071, 0.1),
        ("sigma_y", 0.07071, 0.1),
        ("cov_xy", 0.0, 0.1),
        ("center_x", 0.2, 0.8),
        ("center_y", 0.2, 0.8),
    ),
    "cov_xy_mode": "corr",
    "one_complement": False,
    "invert_channels": ["c0"],
}

_SINE_SUP = {
    "num_waves": 2,
    "amplitude": (-1.0, 1.0),
    "phase": (0.0, 2.0 * math.pi),
    "std_one": False,
    "max_one": False,
}

_LAYOUT_REQUIRED_KEYS = (
    "scenario_name",
    "ic_family",
    "sample_physics",
    "num_spatial_dims",
    "num_channels",
    "ic_param_names",
    "ic_l_bounds",
    "ic_u_bounds",
    "physics_specs",
)


def _as_float_list(value: Any, dim: int) -> List[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != dim:
            raise ValueError(f"Expected {dim} values, got {len(value)}.")
        return [float(v) for v in value]
    return [float(value)] * dim


def _canonical_base_name(name: str) -> str:
    out = str(name).strip()
    changed = True
    while changed:
        changed = False
        for suffix in ("_low_res", "_2d", "_1d"):
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def _scenario_family(base_name: str, config: Dict[str, Any] | None = None) -> str | None:
    base = _canonical_base_name(base_name)
    if base.startswith("gray_scott_"):
        return "gs"
    if base == "kuramoto_sivashinsky":
        return "ks"
    if base.startswith("navier_stokes_"):
        return "ns"
    if config is not None:
        if int(config.get("dim", 2)) == 1 and _IC_SINE_SUP in (
            config.get("initial_conditions") or {}
        ):
            return "sine_1d"
    if base in {"burgers", "diff_kdv", "diff_ks", "diff_ks_cons"}:
        return "sine_1d"
    return None


def _default_ic_family(family: str) -> str:
    if family == "gs":
        return _IC_GAUSSIAN_BLOBS
    if family == "sine_1d":
        return _IC_SINE_SUP
    return _IC_FOURIER


def resolve_supported_scenario(name: str) -> Dict[str, Any]:
    entry = get_pde_config(name)
    canonical = entry["name"]
    config = dict(entry["config"])

    base = _canonical_base_name(config.get("_base_pde_name", canonical))
    family = _scenario_family(base, config)
    if family is None:
        raise ValueError(
            "physics_ic_utils supports only NS/GS/KS and registered sine_sup 1D scenarios. "
            f"Received '{name}' (base '{base}')."
        )

    return {
        "requested_name": name,
        "canonical_name": canonical,
        "base_name": base,
        "family": family,
        "num_spatial_dims": int(config.get("dim", 2)),
        "num_channels": int(len(config.get("fields", ["Density"]))),
        "config": config,
    }


def _registry_constants(scenario_name: str) -> Dict[str, Any]:
    scenario = resolve_supported_scenario(scenario_name)
    return dict(get_pde_config(scenario["canonical_name"])["config"].get("physical_constant") or {})


def physics_specs_from_registry(scenario_name: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for base_name, meta in _registry_constants(scenario_name).items():
        if meta.get("min") is None or meta.get("max") is None:
            continue
        dim = int(meta.get("dim", 1))
        mins = _as_float_list(meta["min"], dim)
        maxs = _as_float_list(meta["max"], dim)
        scale = str(meta.get("sampling_scale", "linear"))
        for i in range(dim):
            specs.append(
                {
                    "name": (base_name if dim == 1 else f"{base_name}_{i}"),
                    "base": base_name,
                    "index": i,
                    "base_dim": dim,
                    "dim": 1,
                    "min": float(mins[i]),
                    "max": float(maxs[i]),
                    "sampling_scale": scale,
                }
            )
    return specs


def physics_defaults_from_registry(scenario_name: str) -> Dict[str, float | List[float]]:
    defaults: Dict[str, float | List[float]] = {}
    for base_name, meta in _registry_constants(scenario_name).items():
        dim = int(meta.get("dim", 1))
        if meta.get("value") is not None:
            values = _as_float_list(meta["value"], dim)
        elif meta.get("min") is not None and meta.get("max") is not None:
            mins = _as_float_list(meta["min"], dim)
            maxs = _as_float_list(meta["max"], dim)
            values = [0.5 * (lo + hi) for lo, hi in zip(mins, maxs)]
        else:
            continue
        defaults[base_name] = values[0] if dim == 1 else values
    return defaults


def derive_physics_param_specs(scenario_name: str) -> List[Dict[str, Any]]:
    """Backward-compatible alias returning registry-derived physics specs."""
    try:
        return physics_specs_from_registry(scenario_name)
    except Exception:
        return []


def ensure_physics_param_specs(
    scenario_config: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Populate ``physics_param_specs`` from the scenario registry.

    This helper is kept for compatibility with analysis scripts that still expect
    a ``(scenario_config, specs)`` pair.
    """
    scenario_config = dict(scenario_config)
    sample_physics = bool(scenario_config.get("sample_physics", False))
    specs = (
        derive_physics_param_specs(str(scenario_config.get("scenario_name", "")))
        if sample_physics
        else []
    )
    scenario_config["physics_param_specs"] = specs
    return scenario_config, specs


def _fourier_mode_count(num_spatial_dims: int) -> int:
    cutoff = int(round(_FOURIER["cutoff"][1]))
    return (cutoff + 1) * pow(2 * cutoff + 1, num_spatial_dims - 1)


def _sine_sup_options(scenario_config: Dict[str, Any] | None) -> Dict[str, Any]:
    source = ((scenario_config or {}).get("initial_conditions") or {}).get(_IC_SINE_SUP) or {}
    return {
        "num_waves": int(source.get("num_waves", _SINE_SUP["num_waves"])),
        "std_one": bool(source.get("std_one", _SINE_SUP["std_one"])),
        "max_one": bool(source.get("max_one", _SINE_SUP["max_one"])),
    }


def _ic_options_for_family(
    ic_family: str,
    scenario_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if ic_family == _IC_SINE_SUP:
        return _sine_sup_options(scenario_config)
    return {}


def _build_ic_layout(
    ic_family: str,
    dims: int,
    channels: int,
    scenario_config: Dict[str, Any] | None = None,
):
    if ic_family == _IC_FOURIER:
        mode_count = _fourier_mode_count(dims)
        names = ["cutoff", "taper_power"]
        lowers = [_FOURIER["cutoff"][0], _FOURIER["taper"][0]]
        uppers = [_FOURIER["cutoff"][1], _FOURIER["taper"][1]]
        for ch in range(channels):
            for i in range(mode_count):
                names.append(f"c{ch}/amplitudes/{i}")
                lowers.append(_FOURIER["amplitude"][0])
                uppers.append(_FOURIER["amplitude"][1])
                names.append(f"c{ch}/phases/{i}")
                lowers.append(_FOURIER["phase"][0])
                uppers.append(_FOURIER["phase"][1])
        return names, lowers, uppers

    if ic_family == _IC_GAUSSIAN_BLOBS:
        if dims != 2:
            raise ValueError("gaussian_blobs supports only 2D scenarios.")
        names: List[str] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for ch in range(channels):
            for i in range(int(_GAUSSIAN["num_blobs"])):
                for field_name, lo, hi in _GAUSSIAN["fields"]:
                    names.append(f"c{ch}/blobs/{i}/{field_name}")
                    lowers.append(float(lo))
                    uppers.append(float(hi))
        return names, lowers, uppers

    if ic_family == _IC_SINE_SUP:
        if dims != 1:
            raise ValueError("sine_sup supports only 1D scenarios.")
        if channels != 1:
            raise ValueError("sine_sup currently supports single-channel scenarios.")
        options = _sine_sup_options(scenario_config)
        names: List[str] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for wave_idx in range(int(options["num_waves"])):
            names.append(f"amplitudes/{wave_idx}")
            lowers.append(_SINE_SUP["amplitude"][0])
            uppers.append(_SINE_SUP["amplitude"][1])
            names.append(f"phases/{wave_idx}")
            lowers.append(_SINE_SUP["phase"][0])
            uppers.append(_SINE_SUP["phase"][1])
        return names, lowers, uppers

    raise ValueError(f"Unsupported ic_family '{ic_family}'.")


def _with_counts(layout: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(layout)
    out["ic_param_count"] = len(out["ic_param_names"])
    out["physics_param_count"] = len(out["physics_specs"]) if out["sample_physics"] else 0
    out["num_parameters"] = out["ic_param_count"] + out["physics_param_count"]
    return out


def full_bounds_from_layout(layout: Dict[str, Any]) -> tuple[List[float], List[float]]:
    lay = layout_from_dict(layout)
    lower = list(lay["ic_l_bounds"])
    upper = list(lay["ic_u_bounds"])
    if lay["sample_physics"]:
        lower.extend(float(spec["min"]) for spec in lay["physics_specs"])
        upper.extend(float(spec["max"]) for spec in lay["physics_specs"])
    return lower, upper


def build_param_layout(
    scenario_name: str,
    ic_family: str | None = None,
    sample_physics: bool = True,
) -> Dict[str, Any]:
    scenario = resolve_supported_scenario(scenario_name)
    chosen_ic_family = str(ic_family or _default_ic_family(scenario["family"])).strip()
    if chosen_ic_family not in _ALLOWED_IC_FAMILIES:
        allowed = ", ".join(_ALLOWED_IC_FAMILIES)
        raise ValueError(f"Unsupported ic_family '{chosen_ic_family}'. Allowed: {allowed}.")

    ic_names, ic_l, ic_u = _build_ic_layout(
        chosen_ic_family,
        int(scenario["num_spatial_dims"]),
        int(scenario["num_channels"]),
        scenario.get("config"),
    )
    specs = physics_specs_from_registry(str(scenario["canonical_name"])) if sample_physics else []

    return _with_counts(
        {
            "scenario_name": str(scenario["canonical_name"]),
            "ic_family": chosen_ic_family,
            "sample_physics": bool(sample_physics),
            "num_spatial_dims": int(scenario["num_spatial_dims"]),
            "num_channels": int(scenario["num_channels"]),
            "ic_param_names": list(ic_names),
            "ic_l_bounds": list(ic_l),
            "ic_u_bounds": list(ic_u),
            "ic_options": _ic_options_for_family(chosen_ic_family, scenario.get("config")),
            "physics_specs": list(specs),
        }
    )


def layout_from_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in _LAYOUT_REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(
            "physics/IC layout is missing required keys: " + ", ".join(sorted(missing))
        )

    return _with_counts(
        {
            "scenario_name": str(raw["scenario_name"]),
            "ic_family": str(raw["ic_family"]),
            "sample_physics": bool(raw["sample_physics"]),
            "num_spatial_dims": int(raw["num_spatial_dims"]),
            "num_channels": int(raw["num_channels"]),
            "ic_param_names": list(raw["ic_param_names"]),
            "ic_l_bounds": [float(v) for v in raw["ic_l_bounds"]],
            "ic_u_bounds": [float(v) for v in raw["ic_u_bounds"]],
            "ic_options": dict(raw.get("ic_options") or {}),
            "physics_specs": list(raw["physics_specs"]),
        }
    )


def layout_from_scenario_config(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    scenario_name = scenario_config.get("scenario_name")
    if scenario_name is None:
        raise ValueError("scenario_config.scenario_name is required to derive the IC/physics layout.")

    ic_family = scenario_config.get("ic_family")
    sample_physics = bool(scenario_config.get("sample_physics", True))
    return build_param_layout(
        scenario_name=str(scenario_name),
        ic_family=(str(ic_family) if ic_family is not None else None),
        sample_physics=sample_physics,
    )


def enrich_study_options_from_scenario_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    study_options = config_dict.setdefault("study_options", {})
    scenario_config = study_options.get("scenario_config") or {}
    layout = layout_from_scenario_config(scenario_config)
    lower, upper = full_bounds_from_layout(layout)
    n_ic = int(layout["ic_param_count"])
    given_l, given_u = study_options.get("l_bounds"), study_options.get("u_bounds")
    if given_l is not None and given_u is not None and len(given_l) == len(given_u) in (n_ic, len(lower)):
        lower[:n_ic], upper[:n_ic] = list(given_l[:n_ic]), list(given_u[:n_ic])
    study_options["nb_parameters"] = int(layout["num_parameters"])
    study_options["l_bounds"] = list(lower)
    study_options["u_bounds"] = list(upper)
    return config_dict


def decode_ic_parameters(ic_params: Sequence[float], layout: Dict[str, Any]) -> Dict[str, Any]:
    lay = layout_from_dict(layout)
    values = np.asarray(ic_params, dtype=float).reshape(-1)
    if values.size != lay["ic_param_count"]:
        raise ValueError(f"Expected {lay['ic_param_count']} IC params, got {values.size}.")

    if lay["ic_family"] == _IC_FOURIER:
        channels: Dict[str, Dict[str, List[float]]] = {}
        for ch in range(lay["num_channels"]):
            channels[f"c{ch}"] = {"amplitudes": [], "phases": []}

        for name, val in zip(lay["ic_param_names"][2:], values[2:]):
            if not isinstance(name, str) or not name.startswith("c"):
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            ch = parts[0]
            field = parts[1]
            if ch not in channels:
                continue
            if field == "amplitudes":
                channels[ch]["amplitudes"].append(float(val))
            elif field == "phases":
                channels[ch]["phases"].append(float(val))
        return {
            "type": _IC_FOURIER,
            "cutoff": float(values[0]),
            "taper_power": float(values[1]),
            "channels": channels,
            "std_one": bool(_FOURIER["std_one"]),
            "max_one": bool(_FOURIER["max_one"]),
        }

    if lay["ic_family"] == _IC_GAUSSIAN_BLOBS:
        idx = 0
        channels: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        for ch in range(lay["num_channels"]):
            blobs: List[Dict[str, float]] = []
            for _ in range(int(_GAUSSIAN["num_blobs"])):
                amplitude = float(values[idx])
                sigma_x = float(values[idx + 1])
                sigma_y = float(values[idx + 2])
                cov_xy_raw = float(values[idx + 3])
                center_x = float(values[idx + 4])
                center_y = float(values[idx + 5])
                idx += 6
                corr = float(np.clip(cov_xy_raw, -0.999, 0.999))
                blobs.append(
                    {
                        "amplitude": amplitude,
                        "sigma_x": sigma_x,
                        "sigma_y": sigma_y,
                        "cov_xy": corr * sigma_x * sigma_y,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )
            channels[f"c{ch}"] = {"blobs": blobs}
        return {
            "type": _IC_GAUSSIAN_BLOBS,
            "num_blobs": int(_GAUSSIAN["num_blobs"]),
            "channels": channels,
            "cov_xy_mode": str(_GAUSSIAN["cov_xy_mode"]),
            "one_complement": bool(_GAUSSIAN["one_complement"]),
            "invert_channels": list(_GAUSSIAN["invert_channels"]),
        }

    if lay["ic_family"] == _IC_SINE_SUP:
        amplitudes: List[float] = []
        phases: List[float] = []
        for name, val in zip(lay["ic_param_names"], values):
            if str(name).startswith("amplitudes/"):
                amplitudes.append(float(val))
            elif str(name).startswith("phases/"):
                phases.append(float(val))
        options = _sine_sup_options({"initial_conditions": {_IC_SINE_SUP: lay["ic_options"]}})
        return {
            "type": _IC_SINE_SUP,
            "amplitudes": amplitudes,
            "phases": phases,
            "std_one": bool(options["std_one"]),
            "max_one": bool(options["max_one"]),
        }

    raise ValueError(f"Unsupported ic_family '{lay['ic_family']}'.")


def decode_physics_parameters(
    physics_params: Sequence[float],
    layout: Dict[str, Any],
) -> Dict[str, Any]:
    lay = layout_from_dict(layout)
    if not lay["sample_physics"]:
        return {}

    values = np.asarray(physics_params, dtype=float).reshape(-1)
    if values.size != lay["physics_param_count"]:
        raise ValueError(
            f"Expected {lay['physics_param_count']} physics params, got {values.size}."
        )

    out: Dict[str, Any] = {}
    for spec, raw in zip(lay["physics_specs"], values):
        val = float(raw)
        if str(spec.get("sampling_scale", "linear")) == "log":
            lo = float(spec["min"])
            hi = float(spec["max"])
            if hi <= lo:
                raise ValueError(f"Invalid log bounds: min={lo}, max={hi}.")
            frac = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
            val = float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
        out[str(spec["name"])] = val
    return out


def decode_parameters(
    vector: Sequence[float],
    layout: Dict[str, Any],
) -> Dict[str, Any]:
    lay = layout_from_dict(layout)
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if vec.size != lay["num_parameters"]:
        raise ValueError(f"Expected {lay['num_parameters']} parameters, got {vec.size}.")

    ic_count = lay["ic_param_count"]
    return {
        "sampled_ic_config": decode_ic_parameters(vec[:ic_count], lay),
        "sampled_physics_config": decode_physics_parameters(vec[ic_count:], lay),
    }


def build_minimal_ic_config(layout: Dict[str, Any]) -> Dict[str, Any]:
    lay = layout_from_dict(layout)
    return {
        "type": str(lay["ic_family"]),
    }


__all__ = [
    "resolve_supported_scenario",
    "derive_physics_param_specs",
    "ensure_physics_param_specs",
    "build_param_layout",
    "full_bounds_from_layout",
    "enrich_study_options_from_scenario_config",
    "layout_from_dict",
    "layout_from_scenario_config",
    "decode_parameters",
    "decode_ic_parameters",
    "decode_physics_parameters",
    "physics_specs_from_registry",
    "physics_defaults_from_registry",
    "build_minimal_ic_config",
]
