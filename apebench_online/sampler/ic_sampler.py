import os
import yaml
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional
from typing_extensions import override

from melissa.server.parameters import (  # type: ignore
    BaseExperiment,
    HaltonSamplerMixIn,
    ParameterSamplerType,
    QMCSamplerMixIn,
    RandomUniformSamplerMixIn,
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder,
    OGASBreeder,
)
from melissa.server.deep_learning.active_sampling.breeder.base import QMC_SAMPLERS  # type: ignore
from apebench_online.core.constants import VALIDATION_DIR, VALIDATION_INPUT_PARAM_FILE
from apebench_online.scenarios.physics_ic_utils import (
    decode_ic_parameters,
    decode_parameters,
    decode_physics_parameters,
    layout_from_scenario_config,
    layout_from_dict,
)


def _extract_base_sampler_kwargs(source_kwargs):
    keys = ("scenario_config", "solver_trajectory_batch_size", "persistent_client_mode")
    extracted = {}
    for key in keys:
        if key in source_kwargs:
            extracted[key] = source_kwargs.pop(key)
    return extracted


class BaseCustomSamplerMixIn(RandomUniformSamplerMixIn, HaltonSamplerMixIn):
    nb_params: int
    seed: int
    parameters: np.ndarray
    _non_breed_sampler_t: ParameterSamplerType  # Inherited from ExperimentBreeder

    def __init__(
        self,
        ic_config: Dict[str, Any],
        is_valid: bool = False,
        scenario_config: Optional[Dict[str, Any]] = None,
        solver_trajectory_batch_size: int = -1,
        persistent_client_mode: bool = False,
    ):
        if not isinstance(ic_config, dict):
            raise ValueError(f"ic_config must be a dict, got {type(ic_config)}")
        if isinstance(scenario_config, dict):
            self.layout = layout_from_scenario_config(scenario_config)
        elif isinstance(ic_config.get("layout"), dict):
            self.layout = layout_from_dict(ic_config["layout"])
        else:
            raise ValueError(
                "Missing scenario metadata needed to derive the IC/physics layout."
            )
        self.ic_type = self.layout["ic_family"]
        self.is_valid = is_valid
        self.unsampled_constants = None
        self.solver_trajectory_batch_size = solver_trajectory_batch_size
        self.persistent_client_mode = persistent_client_mode

        if self.is_valid:
            HaltonSamplerMixIn.__init__(self, self.nb_params, self.seed)
            os.makedirs(VALIDATION_DIR, exist_ok=True)

    def base_sample(self, nb_samples: int) -> np.ndarray:
        """Generates samples using the appropriate sampler.

        For validation (is_valid=True): always uses Halton sampling.
        For training (is_valid=False): uses the sampler specified by _non_breed_sampler_t.

        This override fixes the MRO issue where RandomUniformSamplerMixIn.base_sample
        would always be called regardless of which sampler was configured.

        Note: Uses getattr for is_valid because base_sample may be called during
        BaseExperiment.__init__ before BaseCustomSamplerMixIn.__init__ sets is_valid.
        """
        # Use getattr because this may be called before __init__ completes
        is_valid = getattr(self, "is_valid", False)

        if is_valid:
            # Validation always uses Halton sampling
            return HaltonSamplerMixIn.base_sample(self, nb_samples)
        else:
            # Training uses the sampler configured via non_breed_sampling_strategy
            if getattr(self, "_non_breed_sampler_t", None) in QMC_SAMPLERS:
                return QMCSamplerMixIn.base_sample(self, nb_samples)
            return RandomUniformSamplerMixIn.base_sample(self, nb_samples)

    def _fill_ic_config(self, ic_params: np.ndarray) -> Dict[str, Any]:
        if len(ic_params) != int(self.layout["ic_param_count"]):
            raise ValueError(
                f"Expected {self.layout['ic_param_count']} IC params, got {len(ic_params)}"
            )
        return decode_ic_parameters(ic_params, self.layout)

    def _split_parameters(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ic_count = int(self.layout["ic_param_count"])
        physics_count = (
            int(self.layout["physics_param_count"])
            if self.layout["sample_physics"]
            else 0
        )
        total = len(parameters)

        if ic_count + physics_count != total:
            raise ValueError(
                f"Parameter count mismatch: expected {ic_count + physics_count}, got {total}"
            )

        ic_params = parameters[:ic_count]
        physics_params = parameters[ic_count : ic_count + physics_count]
        return ic_params, physics_params

    def _fill_physics_config_as_specs(
        self, physics_params: np.ndarray
    ) -> Dict[str, float]:
        if not self.layout["sample_physics"]:
            return {}
        return decode_physics_parameters(physics_params, self.layout)

    @override
    def process_drawn(self, parameters: np.ndarray) -> Dict[str, Any]:  # type: ignore
        if self.is_valid and not os.path.exists(VALIDATION_INPUT_PARAM_FILE):
            jnp.save(VALIDATION_INPUT_PARAM_FILE, self.parameters)
        return decode_parameters(parameters, self.layout)

    @override
    def draw(self, sim_id: int) -> List[str] | np.ndarray:  # type: ignore
        main_config = self.process_drawn(self.parameters[sim_id])

        os.makedirs("sampled_configs", exist_ok=True)
        main_config_path = f"sampled_configs/sim_{sim_id}.yaml"

        # with batched mode, on top of its own sampled_configs,
        # the solver will take extra configs to run
        if (
            self.solver_trajectory_batch_size > 1
            and sim_id % self.solver_trajectory_batch_size == 0
        ):
            assert self.is_valid, "Batched rollouts are not supported in online mode."
            batch_paths = []
            for offset in range(1, self.solver_trajectory_batch_size):
                if (sim_id + offset) >= self.nb_sims:
                    break
                batch_config = self.process_drawn(self.parameters[sim_id + offset])
                batch_config_path = f"sampled_configs/sim_{sim_id + offset}.yaml"
                with open(batch_config_path, "w") as f:
                    yaml.dump(batch_config, f)
                batch_paths.append(batch_config_path)

            main_config["sampled_batch_paths"] = batch_paths

        with open(main_config_path, "w") as f:
            yaml.dump(main_config, f)

        if self.persistent_client_mode:
            return np.asarray(self.parameters[sim_id], dtype=self.dtype)
        return [f"--ic-config-path={main_config_path}"]


class ICSamplerClassic(BaseCustomSamplerMixIn, BaseExperiment):
    def __init__(self, ic_config: Dict[str, Any], is_valid: bool = False, **kwargs):
        base_kwargs = _extract_base_sampler_kwargs(kwargs)
        # Set is_valid BEFORE BaseExperiment.__init__ because it calls base_sample()
        self.is_valid = is_valid
        # Initialize the appropriate sampler before BaseExperiment calls base_sample
        if is_valid:
            HaltonSamplerMixIn.__init__(
                self, kwargs.get("nb_params", 1), kwargs.get("seed")
            )
        else:
            RandomUniformSamplerMixIn.__init__(self)
        BaseExperiment.__init__(self, **kwargs)
        BaseCustomSamplerMixIn.__init__(
            self, ic_config=ic_config, is_valid=is_valid, **base_kwargs
        )


class ICSamplerBreed(BaseCustomSamplerMixIn, DefaultBreeder):
    def __init__(self, ic_config: Dict[str, Any], is_valid: bool = False, **kwargs):
        base_kwargs = _extract_base_sampler_kwargs(kwargs)
        DefaultBreeder.__init__(self, **kwargs)
        BaseCustomSamplerMixIn.__init__(
            self, ic_config=ic_config, is_valid=is_valid, **base_kwargs
        )


class ICSamplerOGAS(BaseCustomSamplerMixIn, OGASBreeder):
    def __init__(self, ic_config: Dict[str, Any], is_valid: bool = False, **kwargs):
        base_kwargs = _extract_base_sampler_kwargs(kwargs)
        OGASBreeder.__init__(self, **kwargs)
        BaseCustomSamplerMixIn.__init__(
            self, ic_config=ic_config, is_valid=is_valid, **base_kwargs
        )


def get_sampler_class_type(
    ic_type: str, use_classic: bool, breeder_type: str = "default"
):
    del ic_type
    if use_classic:
        return ICSamplerClassic
    if breeder_type == "ogas_breeder":
        return ICSamplerOGAS
    if breeder_type == "sbal":
        # Import here to avoid circular dependency
        from apebench_online.sbal.ic_sampler_sbal import ICSamplerSBAL

        return ICSamplerSBAL
    if breeder_type == "default":
        return ICSamplerBreed
    raise ValueError(f"Unknown breeder_type: {breeder_type}")
