"""Scenario configuration for PDE scenarios."""

from dataclasses import dataclass
import pandas as pd

from apebench_online.scenarios import get_pde_config
from apebench_online.scenarios.physics_ic_utils import (
    build_minimal_ic_config,
    build_param_layout,
    full_bounds_from_layout,
)

from .utils import coerce_optional, parse_bool


@dataclass
class ScenarioConfig:
    """Configuration for a PDE scenario with IC and physics parameters."""

    def __init__(self, row: pd.Series):
        """Initialize scenario configuration from CSV row.

        Args:
            row: Pandas Series containing scenario specification
        """
        self.scenario_name: str = str(row["scenario_name"])  # type: ignore

        # Get PDE configuration from registry
        entry = get_pde_config(self.scenario_name)
        pde_conf = entry.get("config", {})
        self.scenario_name = entry.get("name", self.scenario_name)

        self.num_points = int(pde_conf.get("resolution", 0) or 0)
        self.num_spatial_dims = int(pde_conf.get("dim", 2))
        self.num_channels = len(pde_conf.get("fields", ["Density"]))
        stbs = pde_conf.get("solver_trajectory_batch_size", -1)
        self.solver_trajectory_batch_size = stbs if stbs > 1 else -1

        sample_physics_raw = parse_bool(row.get("sample_physics"))
        sample_physics = bool(sample_physics_raw) if sample_physics_raw is not None else True
        ic_family_raw = coerce_optional(row.get("ic_family"))
        ic_family = str(ic_family_raw).strip() if ic_family_raw is not None else None

        layout = build_param_layout(
            scenario_name=self.scenario_name,
            ic_family=ic_family,
            sample_physics=sample_physics,
        )
        self.ic_config = build_minimal_ic_config(layout)

        self.l_bounds, self.u_bounds = full_bounds_from_layout(layout)
        self.num_parameters = int(layout["num_parameters"])

        # Build configuration dictionary for JSON output
        self.config_dict = {
            "ic_config": self.ic_config,
            "scenario_name": self.scenario_name,
            "ic_family": str(layout["ic_family"]),
            "sample_physics": sample_physics,
            "ts": pde_conf.get("ts"),
        }
        if stbs > 1:
            self.config_dict["solver_trajectory_batch_size"] = stbs

        # Network and optimizer overrides
        if not pd.isna(row.get("network_config")):
            self.config_dict["network_config"] = row.get("network_config")
        if not pd.isna(row.get("optim_config")):
            self.config_dict["optim_config"] = row.get("optim_config")
