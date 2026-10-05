from .dl import DLConfig
from .melissa import MelissaConfig
from .scenario import ScenarioConfig


class ActiveSamplingConfig:
    def __init__(
        self,
        regime: str,
        dl: DLConfig,
        melissa: MelissaConfig,
        scenario: ScenarioConfig,
    ):
        self.regime = regime
        self.dl = dl
        self.melissa = melissa
        self.scenario = scenario

        self.buffer_samples = round(melissa.buffer_size // dl.nb_time_steps)

        if regime.startswith("ddpm"):
            self.config_dict = self._get_OGASBreeder_config(regime, self.buffer_samples)
        elif regime.startswith("sbal"):
            self.config_dict = self._get_sbal_config(regime, self.buffer_samples)
        else:
            self.config_dict = self._get_standard_config(regime)

    def _get_standard_config(self, regime):
        # Default parameters (random)
        # sigma is now a ratio of (u_bound - l_bound) for each dimension
        # The actual sigma array is computed at runtime in DefaultBreeder._sigma_init

        tokens = regime.split("_")
        loss_tokens = {"mse", "rmse", "nrmse"}
        explicit_loss_type = next((token for token in tokens if token in loss_tokens), None)
        loss_type = explicit_loss_type or "rmse"
        base_tokens = [
            token
            for token in tokens
            if token not in loss_tokens and token != "sm"
        ]
        base_regime = "_".join(base_tokens) if base_tokens else regime
        sigma = 0.1  # 10% of parameter range
        start = 0.0
        end = 0.7
        breakpoint_ = 3
        sliding_window_size = self.buffer_samples
        fitness_min_steps = round(0.25 * self.dl.nb_time_steps)
        min_finished = round((self.melissa.per_server_watermark // self.dl.nb_time_steps) * 2)
        updates = self.buffer_samples * 2
        strategy = "random"
        if base_regime == "precise":
            sigma = 0.05
            start = 0.5
            end = 0.75
            breakpoint_ = 5
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.9 * self.dl.nb_time_steps)
            min_finished = self.buffer_samples
            updates = self.buffer_samples * 5
        elif base_regime == "broad":
            sigma = 0.1
            start = 0.0
            end = 0.7
            breakpoint_ = 3
            sliding_window_size = 2 * self.buffer_samples
            fitness_min_steps = round(0.25 * self.dl.nb_time_steps)
            min_finished = round((self.melissa.per_server_watermark // self.dl.nb_time_steps) * 2)
            updates = self.buffer_samples * 2
        elif base_regime == "mixed":
            sigma = 0.1
            start = 0.0
            end = 0.7
            breakpoint_ = 3
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.33 * self.dl.nb_time_steps)
            min_finished = round((self.melissa.per_server_watermark // self.dl.nb_time_steps) * 2)
            updates = self.buffer_samples * 3
        elif base_regime == "soft":
            sigma = 0.1
            start = 0.75
            end = 0.5
            breakpoint_ = 5
            sliding_window_size = self.buffer_samples * 3
            fitness_min_steps = round(0.5 * self.dl.nb_time_steps)
            min_finished = round(0.9 * self.buffer_samples)
            updates = self.buffer_samples * 1
        elif base_regime == "uniform":
            # same as mixed but r-value = 0
            sigma = 0.1
            start = 0.0
            end = 0.0
            breakpoint_ = 3
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.33 * self.dl.nb_time_steps)
            min_finished = round((self.melissa.per_server_watermark // self.dl.nb_time_steps) * 2)
            updates = self.buffer_samples * 3
        elif base_regime == "no_resampling":
            sigma = 0.0  # Not used but keep as scalar
            start = 0.0
            end = 0.0
            breakpoint_ = 1
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.9 * self.dl.nb_time_steps)
            min_finished = self.buffer_samples
            updates = -1
        elif base_regime == "no_resampling_lhs":
            sigma = 0.0
            start = 0.0
            end = 0.0
            breakpoint_ = 1
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.9 * self.dl.nb_time_steps)
            min_finished = self.buffer_samples
            updates = -1
            strategy = "LHS"
        elif base_regime == "no_resampling_halton":
            sigma = 0.0
            start = 0.0
            end = 0.0
            breakpoint_ = 1
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.9 * self.dl.nb_time_steps)
            min_finished = self.buffer_samples
            updates = -1
            strategy = "HALTON"
        elif base_regime == "no_resampling_sobol":
            # Sobol sequences - good for online sampling (extensible low-discrepancy)
            sigma = 0.0
            start = 0.0
            end = 0.0
            breakpoint_ = 1
            sliding_window_size = self.buffer_samples
            fitness_min_steps = round(0.9 * self.dl.nb_time_steps)
            min_finished = self.buffer_samples
            updates = -1
            strategy = "SOBOL"
        else:
            raise ValueError("Custom regime requires kwargs to be set")

        # sigma is now a scalar ratio - the actual array is computed at runtime
        # based on (u_bounds - l_bounds) in DefaultBreeder._sigma_init

        # Build Dictionary
        config = {
            "nn_updates": updates,
            "min_nb_finished_simulations": min_finished,
            "delta_loss_min_nb_time_steps": fitness_min_steps,
            "non_resampling_threshold": 1,
            "breed_params": {
                "non_breed_sampling_strategy": strategy,
                "sigma": sigma,  # scalar ratio, not array
                "start": start,
                "end": end,
                "breakpoint": breakpoint_,
                "sliding_window_size": sliding_window_size,
                "use_true_mixing": True,
            },
        }
        config["loss_type"] = loss_type

        return config

    def _get_OGASBreeder_config(self, regime: str, nb_buffer_trajectories: int):

        base = {
            "nn_updates": 100,
            "breeder_backend": "ogas_breeder",
            "value_to_register": "loss",
            "loss_type": "rmse",
            "non_resampling_threshold": 10,
            "per_batch_synchronization": True,
            "min_nb_finished_simulations": 0,
            "delta_loss_min_nb_time_steps": 1,
            "min_nb_finished_simulations_period": int(nb_buffer_trajectories * 0.3),
            "breed_params": {
                "data": {
                    "ema_loss_alpha": 0.3,
                    "normalize_losses": True,
                    "min_max_queue_length": 20,
                    "clip_loss_norm_to_3_sigma": False,
                    "max_loss_value": 5,
                    "batch_size": 128,
                    "buffer_size": 6000,
                },
                "model": {
                    "name": "ddpm",
                    "lr": 2e-4,
                    "timesteps": 100,
                    "guidance_scale": 5.0,
                    "model_dim": 1024,
                    "n_train_steps_per_epoch": 1,
                },
                "breeder": {
                    "breakpoint": 4,
                    "activate_breed": True,
                    "start_breed_ratio": 0.0,
                    "max_epoch_per_epoch": 50,
                    "max_mlp_steps_per_collect": 4,
                    "end_breed_ratio": 0.7,
                    "loss_sampling_strategy": "proportional",
                    "learn_sampling_ratio": True,
                    "resample_oob": True,
                    "loss_sampling_alpha": 1,
                    "loss_sampling_quantile": 0.9,
                },
            },
        }

        tokens = regime.split("_")
        for loss_type in ("mse", "rmse", "nrmse"):
            if loss_type in tokens:
                base["loss_type"] = loss_type
                break
        if "normalized" in tokens:
            base["loss_type"] = "nrmse"

        for token in tokens:
            if not token.startswith("maxloss"):
                continue
            raw_value = token[len("maxloss") :]
            if raw_value:
                base["breed_params"]["data"]["max_loss_value"] = float(raw_value)

        if "no_correction" in regime:
            base["breed_params"]["breeder"].update(
                {
                    "learn_sampling_ratio": False,
                    "train_only_with_non_breed_samples": False,
                }
            )
        elif "random_only" in regime:  # Fixed name to match your commented out original
            base["breed_params"]["breeder"].update(
                {
                    "learn_sampling_ratio": False,
                    "train_only_with_non_breed_samples": True,
                }
            )
        elif "ratio_weight" in regime:
            base["breed_params"]["breeder"].update(
                {
                    "learn_sampling_ratio": True,
                    "train_only_with_non_breed_samples": False,
                }
            )
        elif "proportional" in regime:
            base["breed_params"]["breeder"].update(
                {
                    "learn_sampling_ratio": True,
                    "train_only_with_non_breed_samples": False,
                    "loss_sampling_strategy": "proportional",
                    "loss_sampling_alpha": 1,
                }
            )
            if "hard" in regime:
                base["breed_params"]["breeder"]["end_breed_ratio"] = 0.9
        
        # Enable ensemble uncertainty for active sampling
        if "uncertainty" in regime:
            base["value_to_register"] = "uncertainty"
        if "ecrps" in regime:
            base["value_to_register"] = "ecrps"
        
        return base

    def _get_sbal_config(self, regime: str, nb_buffer_trajectories: int):
        """Get configuration for SBAL (Simulation-Based Active Learning).
        
        SBAL uses ensemble model uncertainty to select high-uncertainty 
        parameter points from a generated candidate pool.
        
        Regime variants:
        - sbal: Basic SBAL with power selection (AL4PDE default)
        - sbal_topk: Top-k selection with random exploration
        - sbal_proportional: Proportional sampling weighted by uncertainty
        
        Hyperparameters aligned with AL4PDE for fair comparison.
        Resampling frequency: 10 times across the whole study.
        """
        
        # Calculate resampling period to have ~10 resamplings total
        total_sims = self.melissa.total_nb_simulations_online
        resampling_period = max(1, total_sims // 10)
        
        base = {
            # nn_updates=1 means check every batch, but actual resampling is controlled
            # by min_nb_finished_simulations_period (has_min_period_budget condition)
            "nn_updates": 1,
            "breeder_backend": "sbal",
            "value_to_register": "uncertainty",
            "non_resampling_threshold": 10,
            "per_batch_synchronization": True,
            "min_nb_finished_simulations": 100,  # Initial warm-up: wait for 100 sims before first AL (like AL4PDE)
            "delta_loss_min_nb_time_steps": 1,
            "min_nb_finished_simulations_period": resampling_period,  # ~10 resamplings total
            "breed_params": {
                "log_extra": False,
                "non_breed_sampling_strategy": "random",
                "sbal_config": {
                    "pool_size": 5000,  # Reduced for efficiency (AL4PDE uses 100k on smaller problems)
                    "rollout_steps_rel": 0.5,  # 50% of trajectory for faster uncertainty computation
                    "selection_strategy": "power",  # AL4PDE default
                    "power_beta": 1.0,
                    "uncertainty_aggregation": "mean",
                    # Breeding schedule: 100% SBAL selection (no random)
                    "start_breed_ratio": 1.0,
                    "end_breed_ratio": 1.0,
                    "breakpoint": 1,
                },
            },
        }

        # Parse regime variants
        if "topk" in regime or "top_k" in regime:
            base["breed_params"]["sbal_config"]["selection_strategy"] = "top_k"
            # Mix top-k with random candidates from the pool
            base["breed_params"]["sbal_config"]["top_k_ratio"] = 0.7
            # Use SBAL for all parameters (randomness comes from top_k_ratio)
            base["breed_params"]["sbal_config"]["start_breed_ratio"] = 1.0
            base["breed_params"]["sbal_config"]["end_breed_ratio"] = 1.0
        elif "proportional" in regime:
            base["breed_params"]["sbal_config"]["selection_strategy"] = "proportional"
        
        return base
