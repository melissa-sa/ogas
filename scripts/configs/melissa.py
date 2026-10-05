from dataclasses import dataclass
import pandas as pd
import copy
from .utils import coerce_optional
from .dl import DLConfig


@dataclass
class MelissaConfig:
    # Scenarios requiring more clients (high-res 2D grids)
    HIGH_RES_2D_SCENARIOS = {"burgers_2d"}
    
    def __init__(self, defaults: dict, row: pd.Series, dl: DLConfig):
        conf = copy.deepcopy(defaults)

        # 1. Logic: Increase timeout for long horizons (Keep this, it's useful)
        th = coerce_optional(row.get("temporal_horizon"))
        if th is not None and int(float(th)) >= 200:
            conf["timeout_minutes"] = 120
            # Optional: Adjust total simulations for long runs if desired
            # conf["total_nb_simulations_training"] = 4000

        row_timeout = coerce_optional(row.get("timeout_minutes"))
        if row_timeout:
            conf["timeout_minutes"] = int(float(row_timeout))

        self.timeout_minutes = conf["timeout_minutes"]
        
        # Determine nb_clients based on scenario
        scenario_name = row.get("scenario_name", "")
        if scenario_name in self.HIGH_RES_2D_SCENARIOS:
            self.nb_clients = conf.get("nb_clients_2d_high_res", 112)
        else:
            self.nb_clients = conf["nb_clients"]
        
        self.timer_delay = conf.get("timer_delay", 5)
        self.zmq_hwm = 337

        # 2. Simple Buffer Logic
        # No byte calculation, no caps. Just simple math.
        buffer_size_pct = conf.get("buffer_size_pct", 0.05)
        total_sims = conf["total_nb_simulations_training"]

        # Calculate buffer size based on percentage
        buffer_num_sim = round(total_sims * buffer_size_pct)

        self.total_nb_simulations_online = total_sims
        self.total_nb_simulations_offline = conf["total_nb_simulations_validation"]

        # Actual buffer size in time-steps (minus ZMQ overhead)
        self.buffer_size = (buffer_num_sim * dl.nb_time_steps) - self.zmq_hwm
        self.per_server_watermark = (
            conf.get("watermark_num_sim", self.nb_clients) * dl.nb_time_steps
        )

        # Pass through DL settings
        self.valid_num_samples = dl.valid_num_samples
