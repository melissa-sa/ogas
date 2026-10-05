#!/usr/bin/env python
import os
import yaml
import sys
import argparse
import warnings
import logging
import gc
from typing import Any, Dict, Optional

# Suppress numpy subnormal warnings (common on some CPU architectures)
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")

# Suppress JAX CUDA plugin discovery errors when running on CPU-only
# This must be set BEFORE importing jax
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
logging.getLogger("jax_plugins").setLevel(logging.CRITICAL)

if "jax" in sys.modules and os.getenv("APEBENCH_ALLOW_PREIMPORTED_JAX") != "1":
    raise AssertionError(
        "Make sure JAX is not imported before setting threading flags."
    )
# make only a single device visible
SIM_ID = int(
    os.environ.get("MELISSA_SIMU_ID") or os.environ.get("MELISSA_SIM_ID") or "0"
)
CUDA_DEVICE_IDS = os.getenv("CUDA_VISIBLE_DEVICES", "")
if CUDA_DEVICE_IDS:
    CUDA_DEVICE_IDS = CUDA_DEVICE_IDS.split(",")
    DEVICE_COUNT = len(CUDA_DEVICE_IDS)
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_IDS[SIM_ID % DEVICE_COUNT]
    os.environ["JAX_PLATFORMS"] = "cuda"
else:
    # otherwise use multi-threaded CPU based on allocated cpus per task
    scpt = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    os.environ["OMP_NUM_THREADS"] = str(scpt)
    os.environ["MKL_NUM_THREADS"] = str(scpt)
    os.environ["OPENBLAS_NUM_THREADS"] = str(scpt)
    os.environ["NUMEXPR_NUM_THREADS"] = str(scpt)
    force_single_device = "--xla_force_host_platform_device_count=1"
    xla_flags = os.getenv("XLA_FLAGS", "")
    if xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} {force_single_device}"
    else:
        os.environ["XLA_FLAGS"] = force_single_device
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import exponax as ex
import rapidjson
from mpi4py import MPI
import numpy as np
from melissa.client.api import (
    melissa_fetch_next_parameters,
    melissa_finalize,
    melissa_init,
    melissa_send_float32,
)
from melissa.launcher.schema import CONFIG_PARSE_MODE
from apebench_online.scenarios.scenarios_utils import MelissaSpecificScenario
from apebench_online.core.constants import (
    FIELD_PREV_POSITION,
    FIELD_POSITION,
    FIELD_POSITION_PLUS1,
    VALIDATION_DIR,
)

from apebench_online.scenarios import get_pde_config
from apebench_online.scenarios.physics_ic_utils import (
    decode_parameters,
    layout_from_scenario_config,
)

CONFIG_PATH = os.environ.get("CONFIG_FILE")
if CONFIG_PATH:
    with open(CONFIG_PATH) as json_file:
        CONFIG_DICT = rapidjson.load(json_file, parse_mode=CONFIG_PARSE_MODE)
else:
    CONFIG_DICT = {}


STUDY_OPTIONS = CONFIG_DICT.get("study_options", {})
SCENARIO_CONFIG = STUDY_OPTIONS.get("scenario_config", {})


def _resolve_ts(config):
    ts_local = config.get("ts")
    if ts_local is None:
        scenario_name = config.get("scenario_name")
        if scenario_name:
            ts_local = get_pde_config(scenario_name)["config"].get("ts")
            config["ts"] = ts_local
    if ts_local is None:
        raise ValueError("scenario_config missing 'ts' and registry lookup failed")
    return ts_local


TS_DEFAULT = _resolve_ts(SCENARIO_CONFIG) if SCENARIO_CONFIG else None


def _resolve_ts_val(config):
    ts_v = config.get("ts_val")
    if ts_v is None:
        scenario_name = config.get("scenario_name")
        if scenario_name:
            ts_v = get_pde_config(scenario_name)["config"].get("ts_val")
            if ts_v is not None:
                config["ts_val"] = ts_v
    return ts_v


TS_VAL = _resolve_ts_val(SCENARIO_CONFIG) if SCENARIO_CONFIG else None


class Solver:
    def __init__(
        self,
        config_dict: Dict[str, Any],
        study_options: Dict[str, Any],
        scenario_config: Dict[str, Any],
    ):
        self.config_dict = config_dict
        self.study_options = study_options
        self.scenario_config = scenario_config
        self.timings: Dict[str, float] = {}
        self.use_detail_val = bool(self.study_options.get("use_detail_val", False))

    def _wrap_substeps(self, stepper: ex.BaseStepper, substeps: int):
        substeps = int(substeps or 1)
        if substeps > 1:
            return ex.RepeatedStepper(stepper, substeps)
        return stepper

    def _abort_on_nan(self, arr, context_msg: str = ""):
        if jnp.isnan(arr).any():
            if context_msg:
                print(context_msg, file=sys.stderr)
            print("Aborting the solver.", file=sys.stderr)
            os._exit(1)

    def _run_warmup(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        u,
        warmup_steps: int,
        label: str = "warm-up",
        batched: bool = False,
    ):
        warmup_steps = int(warmup_steps or 0)
        if warmup_steps <= 0:
            return u

        if warmup_steps == 1 and not batched:
            u = stepper(u)
            self._abort_on_nan(u, f"NaN values encountered during {label} at t=0.")
            return u

        rollout_stepper = ex.rollout(stepper, warmup_steps - 1, include_init=True)

        if batched:
            trajectory = jax.vmap(rollout_stepper)(u)  # type: ignore
            self._abort_on_nan(trajectory, f"NaN values encountered during {label}.")
            return trajectory[:, -1]  # multiple ics
        else:
            trajectory = rollout_stepper(u)  # type: ignore
            self._abort_on_nan(trajectory, f"NaN values encountered during {label}.")
            return trajectory[-1]

    def _send_field(self, field: str, arr: np.ndarray):
        payload = np.ascontiguousarray(np.asarray(arr).reshape(-1), dtype=np.float32)
        melissa_send_float32(field, payload)

    def _initialize_online_nodes(self, send_t_plus_2: bool):
        comm = MPI.COMM_SELF
        melissa_init(FIELD_PREV_POSITION, comm)
        melissa_init(FIELD_POSITION, comm)
        if send_t_plus_2:
            melissa_init(FIELD_POSITION_PLUS1, comm)

    def run_online(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        u,
        flattened_mesh_size: int,
        nb_steps: int,
        warmup_steps: int = 0,
    ):
        send_t_plus_2 = bool(
            (self.config_dict.get("dl_config") or {}).get(
                "train_random_t_plus_1_or_2", False
            )
        )
        self._initialize_online_nodes(send_t_plus_2)
        self._run_online_trajectory(
            stepper,
            u,
            flattened_mesh_size,
            nb_steps,
            warmup_steps,
            send_t_plus_2,
        )
        melissa_finalize()

    def _run_online_trajectory(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        u,
        flattened_mesh_size: int,
        nb_steps: int,
        warmup_steps: int,
        send_t_plus_2: bool,
    ):
        del flattened_mesh_size

        self._abort_on_nan(u, "NaN values encountered in IC.")
        u = self._run_warmup(stepper, u, warmup_steps)

        device_type = jax.devices()[0].platform

        if device_type == "cpu":
            if send_t_plus_2 and nb_steps > 0:
                u0 = u
                u1 = stepper(u0)
                self._abort_on_nan(u1, "NaN values encountered at t=1 (t+1 warmup).")
                u2 = stepper(u1)
                self._abort_on_nan(u2, "NaN values encountered at t=2 (t+2 warmup).")
                for t in range(nb_steps):
                    self._send_field(FIELD_PREV_POSITION, u0)
                    self._send_field(FIELD_POSITION, u1)
                    self._send_field(FIELD_POSITION_PLUS1, u2)
                    if t == nb_steps - 1:
                        break
                    u0, u1 = u1, u2
                    u2 = stepper(u1)
                    self._abort_on_nan(u2, f"NaN values encountered at t={t + 3}.")
            else:
                for t in range(nb_steps):
                    self._send_field(FIELD_PREV_POSITION, u)
                    u = stepper(u)
                    self._abort_on_nan(
                        u,
                        f"NaN values encountered at t={t}.",
                    )
                    self._send_field(FIELD_POSITION, u)
        else:
            rollout_steps = nb_steps + 1 if send_t_plus_2 else nb_steps
            rollout_stepper = ex.rollout(stepper, rollout_steps, include_init=True)
            trajectory = rollout_stepper(u)  # type: ignore
            self._abort_on_nan(
                trajectory,
                "NaN values encountered in the trajectory (device rollout).",
            )
            host_traj = jax.device_get(trajectory)
            for t in range(nb_steps):
                prev = host_traj[t].ravel()
                pos = host_traj[t + 1].ravel()
                self._send_field(FIELD_PREV_POSITION, prev)
                self._send_field(FIELD_POSITION, pos)
                if send_t_plus_2:
                    pos_p1 = host_traj[t + 2].ravel()
                    self._send_field(FIELD_POSITION_PLUS1, pos_p1)

    def _run_trajectory(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        ic,
        nb_steps: int,
        batched: bool = False,
    ):
        """Run trajectory computation with or without batching."""
        rollout_stepper = ex.rollout(stepper, nb_steps - 1, include_init=True)

        if batched:
            trajectory = jax.vmap(rollout_stepper)(ic)  # type: ignore
            self._abort_on_nan(
                trajectory.flatten(), "NaN values encountered in the trajectories."
            )
        else:
            trajectory = rollout_stepper(ic)  # type: ignore
            self._abort_on_nan(trajectory, "NaN values encountered in the trajectory.")

        return trajectory

    def _save_trajectory(self, trajectory: jnp.ndarray, sim_id: int):
        """Save a single trajectory to disk."""
        path = f"{VALIDATION_DIR}/sim{sim_id}.npy"
        if os.path.exists(path):
            warnings.warn(
                f"Overwriting existing trajectory file: {path}",
                RuntimeWarning,
                stacklevel=2,
            )

        jnp.save(path, trajectory)

    def run_offline(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        ic,
        nb_steps: int,
        warmup_steps: int = 0,
    ):
        ic = self._run_warmup(stepper, ic, warmup_steps, batched=False)

        trajectory = self._run_trajectory(stepper, ic, nb_steps, batched=False)
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        self._save_trajectory(trajectory, SIM_ID)

    def run_offline_batched(
        self,
        stepper: ex.BaseStepper | ex.RepeatedStepper,
        multiple_ics,
        nb_steps: int,
        warmup_steps: int = 0,
    ):
        multiple_ics = self._run_warmup(
            stepper, multiple_ics, warmup_steps, label="warmup-batched", batched=True
        )

        # Trajectory computation
        multiple_trajectories = self._run_trajectory(
            stepper, multiple_ics, nb_steps, batched=True
        )

        # Save trajectories
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        for offset in range(multiple_trajectories.shape[0]):
            self._save_trajectory(multiple_trajectories[offset], SIM_ID + offset)

    def _produce_scenario_per_config(
        self, ic_config_path: str
    ) -> MelissaSpecificScenario:
        with open(ic_config_path, "r") as f:
            sampled_config = yaml.safe_load(f)

        sampled_ic_config = sampled_config["sampled_ic_config"]
        sampled_physics_config = sampled_config.get("sampled_physics_config", {})

        # Assemble scenario kwargs
        scenario_kwargs = dict(self.scenario_config)
        scenario_kwargs["ts"] = TS_DEFAULT
        if TS_VAL is not None:
            scenario_kwargs["ts_val"] = TS_VAL

        scenario = MelissaSpecificScenario(
            sampled_ic_config=sampled_ic_config,
            sampled_physics_config=sampled_physics_config,
            **scenario_kwargs,
        )
        return scenario

    def _sampled_config_from_parameters(self, parameters) -> Dict[str, Any]:
        layout = layout_from_scenario_config(self.scenario_config)
        return decode_parameters(parameters, layout)

    def _produce_scenario_from_parameters(self, parameters) -> MelissaSpecificScenario:
        sampled_config = self._sampled_config_from_parameters(parameters)
        scenario_kwargs = dict(self.scenario_config)
        scenario_kwargs["ts"] = TS_DEFAULT
        if TS_VAL is not None:
            scenario_kwargs["ts_val"] = TS_VAL
        return MelissaSpecificScenario(**sampled_config, **scenario_kwargs)

    def _write_sampled_config(self, sim_id: int, parameters) -> None:
        sampled_config = self._sampled_config_from_parameters(parameters)
        os.makedirs("sampled_configs", exist_ok=True)
        path = f"sampled_configs/sim_{sim_id}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(sampled_config, f, sort_keys=False)

    def run_persistent(self):
        send_t_plus_2 = bool(
            (self.config_dict.get("dl_config") or {}).get(
                "train_random_t_plus_1_or_2", False
            )
        )
        self._initialize_online_nodes(send_t_plus_2)
        clear_every, n_runs = max(1, int(self.study_options.get("jax_clear_caches_every", 1))), 0
        try:
            while True:
                fetched = melissa_fetch_next_parameters()
                if fetched.is_terminated:
                    break

                parameters = np.asarray(fetched.parameters, dtype=np.float64)
                self._write_sampled_config(fetched.sim_id, parameters)
                scenario = self._produce_scenario_from_parameters(parameters)
                try:
                    self._run_scenario_online(scenario, send_t_plus_2)
                finally:
                    # Persistent clients otherwise retain JAX/XLA dispatch and
                    # executable caches for every sampled scenario.  A long
                    # sweep can then exhaust the host cgroup during LLVM
                    del scenario
                    n_runs += 1
                    if n_runs % clear_every == 0:
                        jax.clear_caches()
                        gc.collect()
        finally:
            melissa_finalize()

    def _run_scenario_online(
        self, scenario: MelissaSpecificScenario, send_t_plus_2: bool
    ):
        stepper = self._wrap_substeps(
            scenario.get_stepper(), getattr(scenario, "substeps")
        )
        input_fn_config = self.scenario_config.get("input_fn_config", {})
        ic = scenario.get_ic_mesh(**input_fn_config)
        warmup_steps = self._resolve_warmup_steps(
            getattr(scenario, "warmup_steps"),
            offline=False,
            warmup_override=None,
        )
        self.scenario = scenario
        self._run_online_trajectory(
            stepper,
            ic,
            int(np.prod(scenario.get_shape())),
            int(TS_DEFAULT),
            warmup_steps,
            send_t_plus_2,
        )

    def run_solver(
        self, ic_config_path: str, offline: bool, warmup_override: Optional[int] = None
    ):
        with open(ic_config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        batched_scenarios = None
        scenario = self._produce_scenario_per_config(ic_config_path)
        if "sampled_batch_paths" in yaml_config:
            batched_scenarios = [scenario] + [
                self._produce_scenario_per_config(ic_config_path)
                for ic_config_path in yaml_config["sampled_batch_paths"]
            ]

        # decide number of steps
        nb_steps = (
            int(TS_VAL + 1) if offline and TS_VAL is not None else int(TS_DEFAULT)
        )

        substeps = getattr(scenario, "substeps")
        default_warmup = getattr(scenario, "warmup_steps")
        warmup_steps = self._resolve_warmup_steps(
            default_warmup, offline, warmup_override
        )

        stepper = scenario.get_stepper()
        stepper = self._wrap_substeps(stepper, substeps)
        data_shape = scenario.get_shape()

        self.scenario = scenario
        mesh_shape_for_send = data_shape

        flattened_mesh_size = int(np.prod(mesh_shape_for_send))
        input_fn_config = self.scenario_config.get("input_fn_config", {})

        if batched_scenarios is not None:
            ic = None
            multiple_ics = jnp.stack(
                [sc.get_ic_mesh(**input_fn_config) for sc in batched_scenarios], axis=0
            )
        else:
            ic = scenario.get_ic_mesh(**input_fn_config)
            multiple_ics = None

        if offline:
            if batched_scenarios is not None:
                self.run_offline_batched(
                    stepper,
                    multiple_ics,
                    nb_steps,
                    scenario.warmup_steps,
                )
            else:
                self.run_offline(stepper, ic, nb_steps, scenario.warmup_steps)
        else:
            self.run_online(
                stepper,
                ic,
                flattened_mesh_size,
                nb_steps,
                warmup_steps,
            )

    def _resolve_warmup_steps(
        self, default_warmup: int, offline: bool, warmup_override: Optional[int]
    ) -> int:
        if warmup_override is not None:
            return int(warmup_override)
        if not self.use_detail_val:
            return int(default_warmup)
        if offline:
            warmup = self._detail_val_offline_warmup(int(default_warmup))
            return warmup
        return 0

    def _detail_val_offline_warmup(self, default_warmup: int) -> int:
        total = int(self.study_options.get("parameter_sweep_size", 0) or 0)
        if total <= 0:
            return int(default_warmup)
        split = total // 2
        if SIM_ID < split:
            return 0
        max_warmup = int(default_warmup)
        if max_warmup <= 0:
            return 0
        rng_seed = int(self.study_options.get("seed", 0) or 0)
        rng = np.random.default_rng(rng_seed + SIM_ID)
        return int(rng.integers(0, max_warmup + 1))


def get_default_parser():
    parser = argparse.ArgumentParser(
        description="APEBench PDE solver with YAML-based configuration"
    )
    parser.add_argument(
        "--ic-config-path",
        type=str,
        required=False,
        help="Path to YAML file containing ic_config and physics_config",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run in offline mode (store trajectories instead of sending to server)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        help="Override the number of warmup steps for this run",
    )

    return parser


def main():
    if not CONFIG_DICT:
        raise RuntimeError("CONFIG_FILE must point to the Melissa JSON config.")
    parser = get_default_parser()
    args = parser.parse_args()

    solver = Solver(CONFIG_DICT, STUDY_OPTIONS, SCENARIO_CONFIG)

    if not args.offline and bool(STUDY_OPTIONS.get("persistent_client_mode", False)):
        solver.run_persistent()
        return

    if not args.ic_config_path:
        parser.error("--ic-config-path is required outside persistent client mode")

    solver.run_solver(
        args.ic_config_path,
        args.offline,
        args.warmup_steps,
    )


if __name__ == "__main__":
    main()
