"""Helpers for looking up registered PDE configs and building steppers.

This version keeps the refactoring that unifies 1D and 2D registries but
restores backwards compatibility with the ``main`` branch:

* names without a dimensional suffix still resolve to the original 2D
  scenarios (e.g. ``burgers``) so existing CSVs continue to work;
* explicit dimensional aliases (``*_1d`` / ``*_2d``) are supported for the
  new scenarios; and
* ``*_low_res`` aliases work for both 1D and 2D variants.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

import exponax as ex

_CONFIG_FILENAME_1D = "1d_config.yaml"
_CONFIG_FILENAME_2D = "2d_config.yaml"

# Allow lightweight variants of every registered PDE by appending one of these
# suffixes to the scenario name (e.g., ``burgers_low_res``). These variants
# reuse the same physical settings but force a smaller grid.
_LOW_RES_SUFFIXES = ("_low_res", "_lowres", "_lr")
_LOW_RES_RESOLUTION = 256

_STEPPERS = {
    "advection": ex.stepper.Advection,
    "diffusion": ex.stepper.Diffusion,
    "advection_diffusion": ex.stepper.AdvectionDiffusion,
    "dispersion": ex.stepper.Dispersion,
    "hyper_diffusion": ex.stepper.HyperDiffusion,
    "burgers": ex.stepper.Burgers,
    "korteweg_de_vries": ex.stepper.KortewegDeVries,
    "kuramoto_sivashinsky": ex.stepper.KuramotoSivashinsky,
    "fisher_kpp": ex.stepper.reaction.FisherKPP,
    "gray_scott_alpha": ex.stepper.reaction.GrayScott,
    "gray_scott_beta": ex.stepper.reaction.GrayScott,
    "gray_scott_gamma": ex.stepper.reaction.GrayScott,
    "gray_scott_delta": ex.stepper.reaction.GrayScott,
    "gray_scott_epsilon": ex.stepper.reaction.GrayScott,
    "gray_scott_theta": ex.stepper.reaction.GrayScott,
    "gray_scott_iota": ex.stepper.reaction.GrayScott,
    "gray_scott_kappa": ex.stepper.reaction.GrayScott,
    "swift_hohenberg": ex.stepper.reaction.SwiftHohenberg,
    "navier_stokes_decaying_turbulence": ex.stepper.NavierStokesVorticity,
    "navier_stokes_kolmogorov_flow": ex.stepper.KolmogorovFlowVorticity,
}


def _build_diff_kdv_stepper(
    dim: int,
    resolution: int,
    kwargs: Dict[str, Any],
) -> Any:
    dispersion_gamma = float(kwargs.pop("dispersion_gamma", -14.0))
    hyp_diffusion_gamma = float(kwargs.pop("hyp_diffusion_gamma", -9.0))
    convection_sc_delta = float(kwargs.pop("convection_sc_delta", -2.0))
    gammas = (0.0, 0.0, 0.0, dispersion_gamma, hyp_diffusion_gamma)
    deltas = (0.0, convection_sc_delta, 0.0)
    return ex.stepper.generic.DifficultyNonlinearStepper(
        num_spatial_dims=dim,
        num_points=resolution,
        linear_difficulties=gammas,
        nonlinear_difficulties=deltas,
        **kwargs,
    )


def _build_diff_ks_stepper(
    dim: int,
    resolution: int,
    kwargs: Dict[str, Any],
) -> Any:
    diffusion_gamma = float(kwargs.pop("diffusion_gamma", -1.2))
    hyp_diffusion_gamma = float(kwargs.pop("hyp_diffusion_gamma", -15.0))
    gradient_norm_delta = float(kwargs.pop("gradient_norm_delta", -6.0))
    gammas = (0.0, 0.0, diffusion_gamma, 0.0, hyp_diffusion_gamma)
    deltas = (0.0, 0.0, gradient_norm_delta)
    return ex.stepper.generic.DifficultyNonlinearStepper(
        num_spatial_dims=dim,
        num_points=resolution,
        linear_difficulties=gammas,
        nonlinear_difficulties=deltas,
        **kwargs,
    )


def _build_diff_ks_cons_stepper(
    dim: int,
    resolution: int,
    kwargs: Dict[str, Any],
) -> Any:
    diffusion_gamma = float(kwargs.pop("diffusion_gamma", -2.0))
    hyp_diffusion_gamma = float(kwargs.pop("hyp_diffusion_gamma", -18.0))
    convection_delta = float(kwargs.pop("convection_delta", -1.0))
    gammas = (0.0, 0.0, diffusion_gamma, 0.0, hyp_diffusion_gamma)
    return ex.stepper.generic.DifficultyConvectionStepper(
        num_spatial_dims=dim,
        num_points=resolution,
        linear_difficulties=gammas,
        convection_difficulty=-convection_delta,
        conservative=True,
        **kwargs,
    )


_CUSTOM_STEPPERS = {
    "diff_kdv": _build_diff_kdv_stepper,
    "diff_ks": _build_diff_ks_stepper,
    "diff_ks_cons": _build_diff_ks_cons_stepper,
}


def get_pde_config(
    pde_name: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Return the raw config block for a PDE."""

    canonical, config = _lookup_pde(pde_name, config_path)
    return {"name": canonical, "config": config}


def build_stepper(
    pde_name: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
    **physical_coefficients: Any,
) -> ex.BaseStepper:
    """Instantiate the PDE’s Exponax stepper using config defaults and overrides."""

    canonical, entry = _lookup_pde(pde_name, config_path)
    stepper_key = entry.get("_base_pde_name", canonical)

    dim = entry.get("dim", 2)
    resolution = entry["resolution"]
    dt = entry["dt"] / max(1, entry.get("substeps", 1))

    coeffs = dict(physical_coefficients)
    constants = _resolve_constants(entry, coeffs)
    coeffs = {**dict(entry.get("stepper_kwargs") or {}), **constants, **coeffs}
    domain_extent = (
        coeffs.pop("domain_extent", None)
        or entry.get("domain_extent")
        or constants.get("domain_extent")
    )
    if domain_extent is None:
        raise ValueError(f"domain_extent missing for '{canonical}'")

    kwargs = dict(coeffs)
    kwargs.pop("domain_extent", None)
    if stepper_key in _CUSTOM_STEPPERS:
        kwargs.pop("dt", None)
        kwargs.pop("substeps", None)
        return _CUSTOM_STEPPERS[stepper_key](dim, resolution, kwargs)

    ctor = _STEPPERS[stepper_key]
    return ctor(dim, domain_extent, resolution, dt, **kwargs)


def _lookup_pde(
    pde_name: str,
    config_path: Optional[Union[str, Path]],
) -> Tuple[str, Dict[str, Any]]:
    registry = _load_config(config_path)
    entries = registry.get("pde_list", {})
    requested = pde_name.strip()

    if requested in entries:
        entry = dict(entries[requested])
        entry.setdefault("_base_pde_name", entries[requested].get("_base_pde_name", requested))
        return requested, entry

    base = _find_low_res_base(requested, entries)
    if base is not None:
        entry = dict(entries[base])
        entry["resolution"] = _LOW_RES_RESOLUTION
        entry["_base_pde_name"] = entry.get("_base_pde_name", base)
        entry["_resolution_variant"] = "low"
        return requested, entry

    raise ValueError(f"PDE '{pde_name}' not found")


def _find_low_res_base(requested: str, entries: Dict[str, Any]) -> Optional[str]:
    """Return base PDE name if the requested name matches a low-res alias."""

    for suffix in _LOW_RES_SUFFIXES:
        if requested.endswith(suffix):
            base = requested[: -len(suffix)]
            if base in entries:
                return base
    return None


def _resolve_constants(entry: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    constants = {}
    for key, meta in (entry.get("physical_constant") or {}).items():
        if key in overrides:
            constants[key] = overrides.pop(key)
        elif "value" in meta:
            constants[key] = meta["value"]
        else:
            raise ValueError(f"Missing value for physical constant '{key}'")
    return constants


def _load_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load and merge 1D/2D registries with backward-compatible aliases."""

    path_1d = Path(__file__).with_name(_CONFIG_FILENAME_1D)
    path_2d = Path(__file__).with_name(_CONFIG_FILENAME_2D)

    with open(path_1d, "r", encoding="utf-8") as handle:
        config_1d = yaml.safe_load(handle) or {}
    with open(path_2d, "r", encoding="utf-8") as handle:
        config_2d = yaml.safe_load(handle) or {}

    registry: Dict[str, Any] = {"pde_list": {}}
    entries = registry["pde_list"]

    # 2D entries are the default for legacy names (main branch behaviour)
    for key, val in (config_2d.get("pde_list") or {}).items():
        base_entry = dict(val)
        # store base name so stepper resolution works for aliases
        base_entry.setdefault("_base_pde_name", key)
        entries[key] = dict(base_entry)

        alias_entry = dict(val)
        alias_entry["_base_pde_name"] = key
        entries[f"{key}_2d"] = alias_entry

    # 1D entries get explicit suffix; only add the bare name when free to avoid
    # clobbering 2D defaults.
    for key, val in (config_1d.get("pde_list") or {}).items():
        entries[f"{key}_1d"] = dict(val)
        if key not in entries:
            entries[key] = dict(val)

    return registry


__all__ = ["get_pde_config", "build_stepper"]
