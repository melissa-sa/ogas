import logging

from typing import List

import numpy as np
from typing_extensions import override

from melissa.server.base_server import BaseServer
from melissa.server.offline_server import OfflineServer
from melissa.server.deep_learning import active_sampling

from apebench_online.sampler.ic_sampler import get_sampler_class_type
from apebench_online.scenarios.physics_ic_utils import (
    enrich_study_options_from_scenario_config,
    full_bounds_from_layout,
    layout_from_scenario_config,
)
from apebench_online.scenarios.scenarios_utils import MelissaSpecificScenario

logger = logging.getLogger("melissa")


class BaseAPEBenchServer:
    """Shared server logic used by offline and PyTorch training servers.

    This module is intentionally torch-first and does not provide a JAX training
    backend anymore.
    """

    def __init__(self, config_dict: dict, is_valid: bool = False):
        enrich_study_options_from_scenario_config(config_dict)
        self.output_dir = config_dict["output_dir"]
        self.study_options = config_dict["study_options"]
        self.scenario_config = self.study_options["scenario_config"]

        l_bounds, u_bounds = self._parse_bounds(self.study_options)

        if not is_valid:
            self.__update_scenario_config(config_dict, self.scenario_config)

        self.scenario = MelissaSpecificScenario(**self.scenario_config)
        self._sync_scenario_time_settings(is_valid)

        self.mesh_shape = self.scenario.get_model_shape()
        self.mesh_axes = tuple(range(1, self.scenario.num_spatial_dims + 2))

        self.ac_config = config_dict.get("active_sampling_config", {})
        self.breed_params = self.ac_config.get("breed_params", {})
        self.breeder_type_to_use = self.ac_config.get("breeder_backend", "default")
        self.value_to_register = self.ac_config.get("value_to_register", "delta_loss")

        self.loss_type = str(self.ac_config.get("loss_type", "rmse")).lower()
        if self.loss_type not in {"mse", "rmse", "nrmse"}:
            raise ValueError(
                "active_sampling_config.loss_type must be one of: mse, rmse, nrmse"
            )

        logger.info(
            f"[ActiveSampling] breeder_type={self.breeder_type_to_use}, "
            f"value_to_register={self.value_to_register}, "
            f"loss_type={self.loss_type}"
        )

        # To solve multiple ICs in batched mode.
        self.traj_batch_size = self.scenario_config.get(
            "solver_trajectory_batch_size", -1
        )
        self.traj_batch_size = (
            self.traj_batch_size if is_valid else -1
        )  # only supported for offline mode

        self._configure_physics_and_sampler(config_dict, is_valid, l_bounds, u_bounds)

    # ------------------------------------------------------------------ #
    # Helpers shared by offline and torch servers
    # ------------------------------------------------------------------ #
    def _extract_field_array(self, msg, field_name):
        return msg.payload[field_name]

    def get_breed_ratio(self, sim_id_list):
        is_bred_list = []
        for sim_id in sim_id_list:
            is_bred_list.append(
                self._parameter_sampler.current_metadata_list[sim_id].is_bred
            )
        return np.mean(is_bred_list)

    def _active_sampling_loss_values(self, loss_data, normalized_data):
        if self.loss_type == "mse":
            return loss_data, "MSE"
        if self.loss_type == "rmse":
            return np.sqrt(np.maximum(loss_data, 0.0)), "RMSE"
        if normalized_data is None:
            raise ValueError("loss_type='nrmse' requires normalized_loss_per_sample.")
        return normalized_data, "nRMSE"

    def _register_active_sampling_metrics(
        self,
        loss_per_sample,
        sim_ids,
        time_steps,
        batch_idx,
        normalized_loss_per_sample=None,
        uncertainty_per_sample=None,
    ):
        loss_data = np.asarray(loss_per_sample)
        normalized_data = (
            np.asarray(normalized_loss_per_sample)
            if normalized_loss_per_sample is not None
            else None
        )
        uncertainty_data = (
            np.asarray(uncertainty_per_sample)
            if uncertainty_per_sample is not None
            else None
        )

        value_key = str(self.value_to_register or "delta_loss").strip().lower()
        if value_key == "delta_loss":
            data_for_delta, label = self._active_sampling_loss_values(
                loss_data, normalized_data
            )
            if batch_idx % 500 == 0:
                logger.info(
                    f"[Breed] Using {label} for delta: mean={np.mean(data_for_delta):.4e}, "
                    f"MSE mean={np.mean(loss_data):.4e}"
                )
            values_to_record = active_sampling.calculate_delta_loss(data_for_delta)
        elif value_key == "loss":
            values_to_record, label = self._active_sampling_loss_values(
                loss_data, normalized_data
            )
            if batch_idx % 500 == 0:
                logger.info(
                    f"[Breed] Using {label}: mean={np.mean(values_to_record):.4e}, "
                    f"MSE mean={np.mean(loss_data):.4e}"
                )
        elif value_key == "uncertainty":
            # Use ensemble uncertainty for active sampling.
            if uncertainty_data is not None:
                values_to_record = uncertainty_data
                if batch_idx % 500 == 0:
                    logger.info(
                        f"[Breed] Using UNCERTAINTY: mean={np.mean(values_to_record):.4e}, "
                        f"L2 mean={np.mean(loss_data):.4e}"
                    )
            else:
                # Fallback to loss if uncertainty not available.
                logger.warning(
                    "[Breed] value_to_register='uncertainty' but no uncertainty data available. "
                    "Falling back to L2 loss. Ensure use_ensemble=True in torch_model config."
                )
                values_to_record = loss_data
        elif value_key == "ecrps":
            if uncertainty_data is None:
                raise ValueError(
                    "value_to_register='ecrps' requires an ensemble acquisition signal"
                )
            values_to_record = uncertainty_data
        else:
            logger.warning(
                f"Unknown value_to_register: {self.value_to_register}. "
                "Defaulting to delta_loss."
            )
            values_to_record = active_sampling.calculate_delta_loss(loss_data)

        if (
            hasattr(self, "breeder_type_to_use")
            and self.breeder_type_to_use == "ogas_breeder"
        ):
            active_sampling.record_increments(
                sim_ids, time_steps, values_to_record, batch_idx
            )
        else:
            active_sampling.record_increments(sim_ids, time_steps, values_to_record)

    def _parse_bounds(self, study_options) -> tuple[list, list]:
        if "l_bounds" not in study_options or "u_bounds" not in study_options:
            enrich_study_options_from_scenario_config({"study_options": study_options})
        return list(study_options["l_bounds"]), list(study_options["u_bounds"])

    def _sync_scenario_time_settings(self, is_valid: bool):
        self.scenario_config["ts"] = getattr(self.scenario, "ts", None)
        if getattr(self.scenario, "ts_val", None) is not None:
            self.scenario_config["ts_val"] = self.scenario.ts_val

        if is_valid and getattr(self.scenario, "ts_val", None) is not None:
            self.study_options["nb_time_steps"] = int(self.scenario.ts_val)
        else:
            self.study_options["nb_time_steps"] = int(self.scenario.ts)

    def _log_parameter_layout(self, layout: dict) -> None:
        if getattr(self, "rank", 0) != 0:
            return

        logger.info(
            "[PARAM_LAYOUT] ic_family=%s ic_param_count=%d physics_param_count=%d sample_physics=%s",
            layout["ic_family"],
            int(layout["ic_param_count"]),
            int(layout["physics_param_count"]),
            bool(layout["sample_physics"]),
        )
        logger.info("[PARAM_LAYOUT] ic_placeholders=%s", list(layout["ic_param_names"]))
        logger.info("[PARAM_LAYOUT] ic_l_bounds=%s", list(layout["ic_l_bounds"]))
        logger.info("[PARAM_LAYOUT] ic_u_bounds=%s", list(layout["ic_u_bounds"]))

        if layout["sample_physics"]:
            physics_names = [str(spec["name"]) for spec in layout["physics_specs"]]
            physics_l_bounds = [float(spec["min"]) for spec in layout["physics_specs"]]
            physics_u_bounds = [float(spec["max"]) for spec in layout["physics_specs"]]
            logger.info("[PARAM_LAYOUT] physics_placeholders=%s", physics_names)
            logger.info("[PARAM_LAYOUT] physics_l_bounds=%s", physics_l_bounds)
            logger.info("[PARAM_LAYOUT] physics_u_bounds=%s", physics_u_bounds)
        else:
            logger.info("[PARAM_LAYOUT] physics_placeholders=[]")
            logger.info("[PARAM_LAYOUT] physics_l_bounds=[]")
            logger.info("[PARAM_LAYOUT] physics_u_bounds=[]")

    def _configure_physics_and_sampler(
        self, config_dict, is_valid, ic_l_bounds, ic_u_bounds
    ):
        """Encapsulates sampler setup for IC + optional physics conditioning."""
        layout = layout_from_scenario_config(self.scenario_config)
        self._log_parameter_layout(layout)

        # Determine sampler type.
        self.sampler_t = get_sampler_class_type(
            ic_type=self.scenario_config["ic_config"]["type"],
            use_classic=is_valid,
            breeder_type=self.breeder_type_to_use if not is_valid else "default",
        )

        self.sample_physics = bool(layout["sample_physics"])
        self.physics_param_specs = list(
            layout["physics_specs"] if self.sample_physics else []
        )
        ic_param_count = int(layout["ic_param_count"])
        expected_l_bounds, expected_u_bounds = full_bounds_from_layout(layout)
        full_l_bounds = list(ic_l_bounds)
        full_u_bounds = list(ic_u_bounds)

        if len(full_l_bounds) != len(expected_l_bounds) or len(full_u_bounds) != len(
            expected_u_bounds
        ):
            raise ValueError(
                "Bounds length mismatch with derived IC/physics layout: "
                f"got {len(full_l_bounds)}/{len(full_u_bounds)}, "
                f"expected {len(expected_l_bounds)}/{len(expected_u_bounds)}."
            )

        if not self.sample_physics:
            self.fixed_physics_params = []
            for _, values in self.scenario.get_physical_args().items():
                if isinstance(values, float):
                    self.fixed_physics_params.append(values)
                # list stores values per dimension of physical_constant
                elif isinstance(values, np.ndarray):
                    self.fixed_physics_params.extend(values.tolist())

        self.study_options["l_bounds"] = full_l_bounds
        self.study_options["u_bounds"] = full_u_bounds

        # Ensure the base server attributes stay in sync with the updated bounds.
        total_params = len(full_l_bounds)
        self.study_options["nb_parameters"] = total_params
        self._nb_parameters = total_params
        self.ic_param_count = ic_param_count

        sampler_kwargs = {
            "ic_config": self.scenario_config["ic_config"],
            "scenario_config": self.scenario_config,
            "is_valid": is_valid,
            "seed": self.study_options["seed"],
            "l_bounds": full_l_bounds,
            "u_bounds": full_u_bounds,
            "dtype": np.float32,
            "solver_trajectory_batch_size": self.traj_batch_size,
            "persistent_client_mode": self.persistent_client_mode,
            **self.breed_params,
        }

        if not is_valid and self.breeder_type_to_use == "ogas_breeder":
            sampler_kwargs["nb_sims"] = self.nb_simulations

        self.set_parameter_sampler(sampler_t=self.sampler_t, **sampler_kwargs)  # type: ignore

    def __update_scenario_config(self, config_dict, scenario_config):
        if "optim_config" not in scenario_config:
            scenario_config["optim_config"] = "adam;warmup_cosine;1e-5;1e-3;0.05"
        optim_config = scenario_config["optim_config"].split(";")
        optim_name = optim_config[0]
        study_options = config_dict["study_options"]
        expected_batches_min = int(
            study_options["parameter_sweep_size"]
            * study_options["nb_time_steps"]
            / config_dict["dl_config"]["batch_size"]
        )
        speed_batch_per_sim = config_dict["dl_config"].get(
            "rate_limit_speed_batch_per_sim"
        )
        if speed_batch_per_sim is not None and float(speed_batch_per_sim) > 0:
            num_training_steps_total = int(
                study_options["parameter_sweep_size"] * float(speed_batch_per_sim)
            )
        else:
            num_training_steps_total = expected_batches_min
        num_training_steps = int(num_training_steps_total * 2)

        scheduler_name = optim_config[1]
        if scheduler_name == "warmup_cosine":
            scheduler_start = float(optim_config[2])
            scheduler_peak = float(optim_config[3])

            scheduler_warmup = float(optim_config[4])
            if scheduler_warmup <= 1.0:
                scheduler_warmup = int(scheduler_warmup * num_training_steps_total)
            else:
                scheduler_warmup = int(scheduler_warmup)

            if scheduler_warmup >= num_training_steps:
                scheduler_warmup = num_training_steps - 1
            if scheduler_warmup <= 0:
                scheduler_warmup = 1

            scenario_config["optim_config"] = (
                f"{optim_name};{num_training_steps};{scheduler_name};"
                f"{scheduler_start};{scheduler_peak};{scheduler_warmup}"
            )
        elif scheduler_name == "constant":
            scheduler_start = float(optim_config[2])
            scenario_config["optim_config"] = (
                f"{optim_name};{num_training_steps};{scheduler_name};"
                f"{scheduler_start}"
            )
        else:
            logger.warning(
                f"Scheduler {scheduler_name} not recognized. "
                "Using default warmup_cosine scheduler."
            )
            scenario_config["optim_config"] = (
                f"{optim_name};{num_training_steps};warmup_cosine;"
                f"1e-5;1e-3;{int(0.05 * expected_batches_min)}"
            )


class APEBenchOfflineServer(BaseAPEBenchServer, OfflineServer):
    def __init__(self, config_dict):
        enrich_study_options_from_scenario_config(config_dict)
        OfflineServer.__init__(self, config_dict)
        BaseAPEBenchServer.__init__(self, config_dict, is_valid=True)

    @override
    def _launch_simulations(self, sim_ids: List[int]) -> None:  # type: ignore
        """Submits strided groups when solver runs multiple trajectories."""
        sim_ids = list(range(0, self.nb_simulations))
        if self.traj_batch_size > 1:
            if self._job_limit > 1:
                raise RuntimeError(
                    "Preferably run a single solver at a time as the possibility of sharing an accelerator increases."
                    "Set `job_limit` to 2 and increase the `solver_trajectory_batch_size` to launch as many trajectories "
                    "as possible on the current accelerator."
                )
            sim_ids = sim_ids[:: self.traj_batch_size]
        self._offline_expected_jobs = len(sim_ids)
        return BaseServer._launch_simulations(self, sim_ids)

    @property
    def all_simulations_finished(self) -> bool:
        expected = getattr(self, "_offline_expected_jobs", self.nb_simulations)
        return self.nb_finished_simulations == expected


class APEBenchServer:
    """Removed JAX training backend.

    Use `APEBenchTorchServer` from
    `apebench_online/core/apebench_torch_server.py`.
    """

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "APEBenchServer (JAX) has been removed. "
            "Please switch config server_class to 'APEBenchTorchServer' "
            "and server_filename to '$APEBENCH_ROOT/apebench_online/core/apebench_torch_server.py'."
        )


__all__ = ["BaseAPEBenchServer", "APEBenchOfflineServer", "APEBenchServer"]
