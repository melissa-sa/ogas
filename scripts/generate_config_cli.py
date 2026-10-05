"""CLI tool for generating Melissa configuration files."""

import argparse
import json
import yaml
import pandas as pd
import os
import sys
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.scenario import ScenarioConfig
from configs.dl import DLConfig
from configs.melissa import MelissaConfig
from configs.active_sampling import ActiveSamplingConfig
from configs.utils import deep_update


def _json_default(obj):
    """Convert numpy scalar/array types to JSON-serializable Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _compute_wandb_group(model_name: str, sampling_name: str, has_physics: bool) -> str:
    """Build a deterministic W&B group name from model/sampling/physics flags."""
    model_tag = str(model_name).replace(" ", "_")
    sampling_tag = str(sampling_name).replace(" ", "_")
    phy_suffix = "_phy" if has_physics else ""
    return f"{model_tag}{phy_suffix}-{sampling_tag}"


def build_offline_json(
    default_tpl, scenario, dl, melissa, output_dir, use_detail_val=False
):
    """Build offline (validation) configuration JSON.

    Args:
        default_tpl: Default template configuration
        scenario: ScenarioConfig instance
        dl: DLConfig instance
        melissa: MelissaConfig instance
        output_dir: Output directory for validation data

    Returns:
        Complete offline configuration dictionary
    """
    conf = copy.deepcopy(default_tpl["offline"])

    # Build scenario config for offline
    scenario_config = scenario.config_dict.copy()
    scenario_config["network_config"] = "MLP;1;1;relu"

    updates = {
        "output_dir": output_dir,
        "study_options": {
            "scenario_config": scenario_config,
            "parameter_sweep_size": melissa.total_nb_simulations_offline,
            "nb_time_steps": dl.valid_nb_time_steps,
            "nb_parameters": scenario.num_parameters,
            "l_bounds": scenario.l_bounds,
            "u_bounds": scenario.u_bounds,
            "seed": 12345678,
            "use_detail_val": bool(use_detail_val),
        },
        "launcher_config": {
            "http_port": int(np.random.randint(8000, 9000)),
        },
    }

    # on an accelerator, only one batch can run efficiently
    if "solver_trajectory_batch_size" in scenario_config:
        assert scenario_config["solver_trajectory_batch_size"] > 1
        updates["launcher_config"]["job_limit"] = 2

    return deep_update(conf, updates)


def build_online_json(
    default_tpl, scenario, dl, melissa, active, seed, output_dir, val_dir
):
    """Build online (training) configuration JSON.

    Args:
        default_tpl: Default template configuration
        scenario: ScenarioConfig instance
        dl: DLConfig instance
        melissa: MelissaConfig instance
        active: ActiveSamplingConfig instance
        seed: Random seed
        output_dir: Output directory for training
        val_dir: Validation data directory

    Returns:
        Complete online configuration dictionary
    """
    conf = copy.deepcopy(default_tpl["online"])

    # Determine W&B group
    model_name = (dl.network_config or "").split(";")[0]
    has_physics = bool(scenario.config_dict.get("sample_physics"))
    wandb_group = _compute_wandb_group(model_name, active.regime, has_physics)

    selected_val_dir = val_dir
    if dl.use_detail_val and val_dir:
        scenario_root = os.path.dirname(os.path.normpath(val_dir))
        selected_val_dir = os.path.join(scenario_root, "detail_val", "trajectories")

    # Build scenario config for online
    scenario_config = scenario.config_dict.copy()
    scenario_config["network_config"] = dl.network_config
    scenario_config["optim_config"] = dl.optim_config

    updates = {
        "output_dir": output_dir,
        "study_options": {
            "scenario_config": scenario_config,
            "parameter_sweep_size": melissa.total_nb_simulations_online,
            "nb_time_steps": dl.nb_time_steps,
            "nb_parameters": scenario.num_parameters,
            "l_bounds": scenario.l_bounds,
            "u_bounds": scenario.u_bounds,
            "seed": seed,
            "zmq_hwm": melissa.zmq_hwm,
            "use_detail_val": dl.use_detail_val,
        },
        "active_sampling_config": active.config_dict,
        "dl_config": {
            "validation_directory": selected_val_dir,
            "wandb_project": f"apebench-{scenario.scenario_name}",
            "valid_rollout": dl.valid_rollout,
            "valid_batch_size": dl.valid_batch_size,
            "valid_nb_time_steps": dl.valid_nb_time_steps,
            "valid_num_samples": dl.valid_num_samples,
            "nb_batches_update": dl.nb_batches_update,
            "batch_size": dl.batch_size,
            "per_server_watermark": melissa.per_server_watermark,
            "buffer_size": melissa.buffer_size,
            "wandb_group": wandb_group,
            "use_ema_for_validation": dl.use_ema_for_validation,
            "ema_decay": dl.ema_decay,
            "ema_start_step": dl.ema_start_step,
            "ema_update_every": dl.ema_update_every,
            "use_detail_val": dl.use_detail_val,
            "normalize_physics": dl.normalize_physics,
            "train_bf_16": dl.train_bf_16,
            "train_random_t_plus_1_or_2": dl.train_random_t_plus_1_or_2,
            "use_ada_grad": dl.use_ada_grad,
            "external_validation": True,
        },
        "launcher_config": {
            "timer_delay": melissa.timer_delay,
            "http_port": int(np.random.randint(8000, 9000)),
        },
    }

    if dl.train_random_t_plus_1_or_2:
        # Need the extra t+2 target for the random horizon training mode.
        updates["study_options"]["node_names"] = [
            "preposition",
            "position",
            "position_plus1",
        ]
    if dl.rate_limit_speed_batch_per_sim is not None:
        updates["dl_config"]["rate_limit_speed_batch_per_sim"] = (
            dl.rate_limit_speed_batch_per_sim
        )

    # Set job_limit based on scenario (burgers_2d uses 112, others use 56)
    if "burgers_2d" in scenario.scenario_name.lower():
        updates["launcher_config"]["job_limit"] = 112
    else:
        updates["launcher_config"]["job_limit"] = 56

    # Compute T_max for Cosine scheduler
    sweep_size = melissa.total_nb_simulations_online
    speed_per_sim = updates["dl_config"].get(
        "rate_limit_speed_batch_per_sim",
        (default_tpl.get("online", {}).get("dl_config", {}) or {}).get(
            "rate_limit_speed_batch_per_sim"
        ),
    )

    if speed_per_sim is not None:
        try:
            t_max_guess = int(round(float(speed_per_sim) * sweep_size))
        except Exception:
            t_max_guess = int(sweep_size * 2)
    else:
        t_max_guess = int(sweep_size * 2)

    t_max_guess = max(1, t_max_guess)
    updates["dl_config"].setdefault("num_training_steps", t_max_guess)
    updates["study_options"]["soft_limit_batch_budget"] = t_max_guess

    # Add Torch-specific configuration (conditional models)
    torch_payload = {}
    if dl.torch_model is not None:
        torch_payload["torch_model"] = dl.torch_model
    if dl.torch_optimizer is not None:
        torch_payload["torch_optimizer"] = dl.torch_optimizer
    if dl.torch_scheduler is not None:
        sched_cfg = copy.deepcopy(dl.torch_scheduler)
        if (
            isinstance(sched_cfg, dict)
            and sched_cfg.get("type", "").lower() == "cosine"
        ):
            sched_cfg.setdefault("t_max", t_max_guess)
        torch_payload["torch_scheduler"] = sched_cfg
    if dl.torch_device is not None:
        torch_payload["torch_device"] = dl.torch_device
    if dl.torch_max_grad_norm is not None:
        torch_payload["torch_max_grad_norm"] = dl.torch_max_grad_norm

    if torch_payload:
        updates["dl_config"].update(torch_payload)
        updates["server_filename"] = (
            "$APEBENCH_ROOT/apebench_online/core/apebench_torch_server.py"
        )
        updates["server_class"] = "APEBenchTorchServer"
    else:
        updates["server_filename"] = (
            "$APEBENCH_ROOT/apebench_online/core/apebench_server.py"
        )
        updates["server_class"] = "APEBenchServer"

    return deep_update(conf, updates)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Melissa configuration files for APEBench scenarios"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        required=True,
        help="Configuration mode: offline (validation) or online (training)",
    )
    parser.add_argument("--scenario", required=True, help="Scenario name from CSV")
    parser.add_argument(
        "--csv", required=True, help="Path to CSV file with scenario specifications"
    )
    parser.add_argument(
        "--template", required=True, help="Path to default configuration template JSON"
    )
    parser.add_argument(
        "--yaml", required=True, help="Path to global YAML configuration"
    )
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for data")

    # Online-only arguments
    parser.add_argument(
        "--regime", default="uniform", help="Active sampling regime (for online mode)"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed (for online mode)"
    )
    parser.add_argument(
        "--validation_dir",
        default=None,
        help="Validation data directory (for online mode)",
    )
    parser.add_argument(
        "--torch_variant",
        default=None,
        help="Torch model variant (for conditional models)",
    )
    parser.add_argument("--use-detail-val", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override the generated W&B project for an online config.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Override the generated W&B group for an online config.",
    )

    args = parser.parse_args()

    # Load configuration files
    with open(args.yaml, "r") as f:
        global_yaml = yaml.safe_load(f)
    with open(args.template, "r") as f:
        default_template = json.load(f)

    df = pd.read_csv(args.csv)
    row = df[df["scenario_name"] == args.scenario].iloc[0].copy()

    # Deep copy defaults
    dl_defaults = copy.deepcopy(global_yaml["dl_defaults"])
    if args.use_detail_val:
        dl_defaults["use_detail_val"] = True
    melissa_defaults = copy.deepcopy(global_yaml["melissa_defaults"])
    torch_variants = global_yaml.get("torch_variants", {})

    # Apply torch variant if specified
    variant_key = args.torch_variant
    if variant_key:
        if variant_key not in torch_variants:
            raise ValueError(
                f"Unknown torch variant '{variant_key}'. "
                f"Available: {', '.join(torch_variants.keys())}"
            )
        variant_cfg = torch_variants.get(variant_key) or {}
        if not isinstance(variant_cfg, dict):
            raise ValueError(
                f"Torch variant '{variant_key}' must be a mapping, found "
                f"{type(variant_cfg).__name__}"
            )

        # Apply variant config to appropriate places
        dl_default_keys = set(dl_defaults.keys())
        dl_variant_fields = {
            "torch_model",
            "torch_optimizer",
            "torch_scheduler",
            "torch_device",
            "torch_max_grad_norm",
        }
        scenario_variant_fields = {"sample_physics"}

        for key, value in variant_cfg.items():
            if key in dl_default_keys:
                dl_defaults[key] = value
            elif key in dl_variant_fields:
                row[key] = value
            elif key in scenario_variant_fields:
                row[key] = value

        # Ensure torch fields are preserved
        for field in dl_variant_fields:
            if field in variant_cfg and field not in row:
                row[field] = variant_cfg[field]

    # Enable ensemble for online regimes unless explicitly marked single-model.
    # For loss-based regimes: use mean loss across ensemble members
    # For uncertainty-based regimes (sbal, uncertainty): use ensemble variance
    if args.mode == "online":
        single_model_regime = "sm" in args.regime.split("_")

        def _set_torch_ensemble(cfg, use_ensemble: bool) -> None:
            if not isinstance(cfg, dict):
                return
            cfg["use_ensemble"] = use_ensemble
            if use_ensemble:
                cfg.setdefault("n_models", 2)
            else:
                cfg["n_models"] = 1

        if "torch_model" in row:
            _set_torch_ensemble(row["torch_model"], not single_model_regime)
        elif "torch_model" in dl_defaults:
            _set_torch_ensemble(dl_defaults.get("torch_model"), not single_model_regime)

    # Initialize configuration objects
    scenario_cfg = ScenarioConfig(row)
    dl_cfg = DLConfig(dl_defaults, row)

    # Adjust buffer size for DL breeder and SBAL regimes
    ogas_breeder_regimes = set(global_yaml.get("regimes", {}).get("ogas_breeder", []))
    sbal_regimes = set(global_yaml.get("regimes", {}).get("sbal", []))
    reduced_buffer_regimes = ogas_breeder_regimes | sbal_regimes
    default_buffer_pct = melissa_defaults.get("buffer_size_pct", 0.05)
    breeder_buffer_pct = melissa_defaults.pop(
        "buffer_size_pct_ogas_breeder", default_buffer_pct * 0.1
    )
    melissa_defaults["buffer_size_pct"] = (
        breeder_buffer_pct
        if args.regime in reduced_buffer_regimes
        else default_buffer_pct
    )

    melissa_cfg = MelissaConfig(melissa_defaults, row, dl_cfg)

    # Build configuration based on mode
    if args.mode == "offline":
        final_config = build_offline_json(
            default_template,
            scenario_cfg,
            dl_cfg,
            melissa_cfg,
            args.output_dir,
            use_detail_val=args.use_detail_val,
        )
    else:
        # Online mode
        if args.regime in ogas_breeder_regimes:
            melissa_cfg.buffer_size = int(melissa_cfg.buffer_size)

        active_cfg = ActiveSamplingConfig(
            args.regime, dl_cfg, melissa_cfg, scenario_cfg
        )
        final_config = build_online_json(
            default_template,
            scenario_cfg,
            dl_cfg,
            melissa_cfg,
            active_cfg,
            args.seed,
            args.output_dir,
            args.validation_dir,
        )
        if args.wandb_project:
            final_config["dl_config"]["wandb_project"] = args.wandb_project
        if args.wandb_group:
            final_config["dl_config"]["wandb_group"] = args.wandb_group

    if isinstance(global_yaml.get("monitoring_config"), dict):
        final_config["monitoring_config"] = deep_update(
            final_config.get("monitoring_config", {}),
            copy.deepcopy(global_yaml["monitoring_config"]),
        )

    # Inject CONFIG_FILE environment variable
    cfg_fname = os.path.abspath(args.output_file)
    client_conf = final_config.setdefault("client_config", {})
    preproc = client_conf.get("preprocessing_commands", []) or []

    # Ensure JAX env is present
    if "export JAX_PLATFORMS=cpu" not in preproc:
        preproc.insert(0, "export JAX_PLATFORMS=cpu")

    # Remove old CONFIG_FILE exports and add new one
    preproc = [cmd for cmd in preproc if not cmd.startswith("export CONFIG_FILE=")]
    preproc.append(f"export CONFIG_FILE={cfg_fname}")
    client_conf["preprocessing_commands"] = preproc
    final_config["client_config"] = client_conf

    launcher_conf = final_config.setdefault("launcher_config", {})
    for key in ("scheduler_server_command", "scheduler_client_command"):
        if key in launcher_conf and isinstance(launcher_conf[key], str):
            launcher_conf[key] = os.path.expandvars(launcher_conf[key])

    # Write output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(final_config, f, indent=4, default=_json_default)

    print(f"✓ Generated {args.mode} config: {args.output_file}")


if __name__ == "__main__":
    main()
