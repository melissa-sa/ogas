from __future__ import annotations

import jax
import apebench
import optax
import numpy as np
import apebench_online.sampler.ic_generation as icgen
from apebench_online.scenarios import get_pde_config, build_stepper
from apebench_online.scenarios.physics_ic_utils import (
    physics_defaults_from_registry,
    physics_specs_from_registry,
)
import exponax as ex

class CustomApebenchScenario(apebench.BaseScenario):
    """Wrapper that populates :mod:`apebench` scenarios from the 2D registry."""

    def __init__(
        self,
        scenario_name: str,
        network_config=None,
        optim_config=None,
        *args,
        **kwargs,
    ):
        entry = get_pde_config(scenario_name)
        canonical = entry["name"]
        config = dict(entry["config"])

        config_num_spatial_dims = int(config.get("dim", 2))
        config_num_points = int(config.get("resolution", 160))
        config_num_channels = int(len(config.get("fields", ["Density"])))
        config_num_warmup_steps = int(config.get("warmup_steps", 0))
        config_ts_val = int(config.get("ts_val", 0))

        num_spatial_dims = int(kwargs.pop("num_spatial_dims", config_num_spatial_dims))
        num_points = int(kwargs.pop("num_points", config_num_points))
        num_channels = int(kwargs.pop("num_channels", config_num_channels))
        num_warmup_steps = int(kwargs.pop("num_warmup_steps", config_num_warmup_steps))
        ts_val = int(kwargs.pop("ts_val", config_ts_val))

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            num_channels=num_channels,
            num_warmup_steps=num_warmup_steps,
            *args,
            **kwargs,
        )

        default_optim = optim_config or config.get(
            "optim_config", "adam;warmup_cosine;1e-5;1e-3;0.05"
        )
        object.__setattr__(self, "scenario_name", canonical)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "network_config", network_config)
        object.__setattr__(self, "optim_config", default_optim)
        object.__setattr__(self, "ts_val", ts_val)

    def get_ref_stepper(self, **physical_coefficient):
        return build_stepper(self.scenario_name, **physical_coefficient)

    def get_coarse_stepper(self, *args, **kwargs) -> ex.BaseStepper:
        return self.get_ref_stepper(*args, **kwargs)

    def get_scenario_name(self) -> str:
        return self.scenario_name


class MelissaSpecificScenario:
    """Scenario helper that exposes Melissa-friendly metadata."""

    def __init__(
        self,
        scenario_name,
        sampled_ic_config=None,
        sampled_physics_config=None,
        network_config="MLP;64;3;relu",
        optim_config="adam;warmup_cosine;0.001;0.0001;5000",
        **scenario_kwargs,
    ):
        overrides = {
            key: scenario_kwargs[key]
            for key in ("num_points", "num_spatial_dims", "num_channels", "num_warmup_steps")
            if key in scenario_kwargs
        }

        self.scenario = CustomApebenchScenario(
            scenario_name=scenario_name,
            network_config=network_config,
            optim_config=optim_config,
            **overrides,
        )

        self.scenario_name = self.scenario.get_scenario_name()
        self.network_config = network_config
        self.sampled_ic_config = sampled_ic_config or {}

        self.domain_extent = self.scenario.config.get("domain_extent")

        if (
            sampled_physics_config is not None
            and sampled_physics_config.get("domain_extent") is not None
        ):
            self.domain_extent = float(sampled_physics_config.get("domain_extent"))
        self.dt = self.scenario.config.get("dt")
        self.substeps = int(
            scenario_kwargs.get("substeps", self.scenario.config.get("substeps", 1))
        )
        self.warmup_steps = int(
            scenario_kwargs.get("warmup_steps", self.scenario.config.get("warmup_steps", 0))
        )
        self.ts = scenario_kwargs.get("ts", self.scenario.config.get("ts"))
        # Optional different horizon for validation only
        self.ts_val = scenario_kwargs.get("ts_val", self.scenario.config.get("ts_val", None))
        self.train_temporal_horizon = None
        self.num_channels = self.scenario.num_channels
        self.num_spatial_dims = self.scenario.num_spatial_dims
        self.num_points = self.scenario.num_points

        self._physics_specs = physics_specs_from_registry(self.scenario_name)
        self._physical_defaults = physics_defaults_from_registry(self.scenario_name)
        self._sampled_physics_config = (
            sampled_physics_config if isinstance(sampled_physics_config, dict) else {}
        )

    def get_shape(self):
        return (self.num_channels,) + (self.num_points,) * self.num_spatial_dims

    def get_model_shape(self):
        return self.get_shape()

    def get_model_resolution(self):
        return self.num_points

    def get_network(self):
        return self.scenario.get_network(
            network_config=self.network_config, key=jax.random.PRNGKey(0)
        )

    def get_physical_args(self):
        if not self._physics_specs:
            return {}

        grouped: dict[str, list[float | None]] = {}

        for base, value in self._physical_defaults.items():
            if isinstance(value, (list, tuple)):
                grouped[str(base)] = [float(v) for v in value]
            else:
                grouped[str(base)] = [float(value)]

        sampled = self._sampled_physics_config
        for spec in self._physics_specs:
            base = str(spec.get("base", spec["name"]))
            name = str(spec["name"])
            idx = int(spec.get("index", 0))
            base_dim = int(spec.get("base_dim", 1))
            grouped.setdefault(base, [None] * base_dim)

            resolved = None
            if name in sampled:
                raw = sampled[name]
                if isinstance(raw, (list, tuple)):
                    resolved = float(raw[idx])
                else:
                    resolved = float(raw)
            elif base in sampled:
                raw = sampled[base]
                if isinstance(raw, (list, tuple)):
                    resolved = float(raw[idx])
                else:
                    resolved = float(raw)
            elif grouped[base][idx] is not None:
                resolved = float(grouped[base][idx])  # type: ignore[arg-type]
            else:
                resolved = 0.5 * (float(spec["min"]) + float(spec["max"]))
            grouped[base][idx] = resolved

        args = {}
        for base, values in grouped.items():
            dense = [
                (float(v) if v is not None else 0.0)
                for v in values
            ]
            if len(dense) == 1:
                args[base] = dense[0]
            else:
                args[base] = np.asarray(dense, dtype=float)
        return args

    def get_optimizer(self, with_lr_scheduler=False, faux=False):
        if faux:
            return apebench.components.optimizer_dict["adam"](
                "adam;10_000;warmup_cosine;0.0;1e-3;2_000"
            )(optax.constant_schedule(1e-3))
        optim_args = self.scenario.optim_config.split(";")
        optimizer_name = optim_args[0]
        num_training_steps = int(optim_args[1])
        scheduler_args = optim_args[2:]
        scheduler_name = scheduler_args[0]

        lr_scheduler = apebench.components.lr_scheduler_dict[scheduler_name.lower()](
            ";".join(scheduler_args), num_training_steps
        )
        optimizer = apebench.components.optimizer_dict[optimizer_name.lower()](
            self.scenario.optim_config
        )(lr_scheduler)
        if with_lr_scheduler:
            return optimizer, lr_scheduler
        else:
            return optimizer

    def get_stepper(self):
        physical_args = self.get_physical_args()
        return self.scenario.get_ref_stepper(**physical_args)

    def get_ic_mesh(self, **input_fn_config):
        if self.sampled_ic_config is None:
            raise ValueError("sampled_ic_config must be initialized before calling get_ic_mesh")

        ic_maker = icgen.get_ic_maker(
            sampled_ic_config=self.sampled_ic_config,
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            num_channels=self.num_channels,
        )
        assert ic_maker.num_spatial_dims == self.num_spatial_dims, (
            f"This is due to having IC with {ic_maker.num_spatial_dims}D "
            f"while the scenario is set with {self.num_spatial_dims}D "
            "Ensure proper flow of arguments between ICMaker and Scenario classes."
        )

        ic = ic_maker(**input_fn_config)
        expected_shape = self.get_shape()
        if tuple(ic.shape) != expected_shape:
            raise ValueError(
                f"Initial condition shape {tuple(ic.shape)} does not match expected "
                f"{expected_shape} for scenario '{self.scenario_name}'."
            )
        return ic

    def __repr__(self):
        return str(self.__dict__)
