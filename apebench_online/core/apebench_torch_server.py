import os
import logging
import copy
import contextlib
import math
from typing_extensions import override
import numpy as np

import torch
import torch.distributed as dist

from melissa.server.deep_learning.active_sampling.torch_server import (
    ExperimentalTorchActiveSamplingServer,
)
from melissa.server.deep_learning.frameworks import FrameworkType

import apebench_online.core.apebench_server as jax_server
from apebench_online.models import build_torch_model
from apebench_online.scenarios.physics_ic_utils import (
    enrich_study_options_from_scenario_config,
)
from apebench_online.sbal.sbal_torch_server import SBALModeMixin

logger = logging.getLogger("melissa")


class APEBenchTorchServer(
    SBALModeMixin, jax_server.BaseAPEBenchServer, ExperimentalTorchActiveSamplingServer
):
    """Torch variant of the APEBench server.

    Mirrors the behaviour of the JAX server. With R > 1 training ranks, the ensemble members are
    spread over the ranks (one GPU each, possibly on several nodes; see models/ensemble.py).
    """

    def __init__(self, config_dict):
        enrich_study_options_from_scenario_config(config_dict)
        ExperimentalTorchActiveSamplingServer.__init__(self, config_dict)
        jax_server.BaseAPEBenchServer.__init__(self, config_dict)

        self.cur_batch_idx = 0
        self.monitoring_config = config_dict.get("monitoring_config", dict())

        self.torch_scheduler = None
        self.checkpoint_model_freq = self.monitoring_config.get(
            "checkpoint_model_each_step", 1
        )
        self.checkpoint_model_max_num_files = self.monitoring_config.get(
            "checkpoint_model_max_num_files", None
        )
        if self.checkpoint_model_max_num_files is not None:
            self.chkpt_id = -1

        self.torch_model_cfg = self.dl_config.get("torch_model")
        if self.torch_model_cfg is None:
            raise ValueError(
                "APEBenchTorchServer requires dl_config.torch_model. "
                "Use APEBenchServer for JAX models."
            )

        self.physics_dim = len(self.physics_param_specs or [])
        self.enable_physics_normalization = bool(
            self.dl_config.get("normalize_physics", False)
        )
        self.use_ema_for_validation = self.dl_config.get(
            "use_ema_for_validation", False
        )
        self.ema_decay = float(self.dl_config.get("ema_decay", 0.999))
        self.ema_start_step = int(self.dl_config.get("ema_start_step", 0))
        self.ema_update_every = int(self.dl_config.get("ema_update_every", 1))
        self.ema_model = None
        self.train_bf16 = bool(self.dl_config.get("train_bf_16", False))
        self._autocast_device_type = None
        self.train_random_t_plus_1_or_2 = bool(
            self.dl_config.get("train_random_t_plus_1_or_2", False)
        )

        self._grid_tensor = None
        self._physics_min = None
        self._physics_max = None
        self._physics_log_done = False
        self.max_grad_norm = self.dl_config.get("torch_max_grad_norm")

        # Adaptive gradient norm (al4pde style)
        self.use_adaptive_grad_norm = bool(self.dl_config.get("use_ada_grad", False))
        self._ada_grad_norm = None  # Will be initialized on first batch
        self._ada_grad_multiplier = float(
            self.dl_config.get("ada_grad_multiplier", 5.0)
        )
        self._ada_grad_ema_alpha = float(self.dl_config.get("ada_grad_ema_alpha", 0.05))
        # Warmup steps for adaptive grad norm (uses same warmup as scheduler by default)
        sched_cfg = self.dl_config.get("torch_scheduler", {})
        self._ada_grad_warmup_steps = int(
            self.dl_config.get(
                "ada_grad_warmup_steps", sched_cfg.get("warmup_steps", 0) or 0
            )
        )

        if self.no_fault_tolerance and self.checkpoint_model_freq < 100:
            logger.warning(
                (
                    f"Fault tolerance is not active, but model checkpointing is "
                    f"requested every {self.checkpoint_model_freq} batch."
                    "Frequency is changed to 100."
                )
            )
            self.checkpoint_model_freq = 100

        self.external_validator_stop_fname = config_dict.get(
            "external_validator_stop_fname", "STOP_VALIDATION"
        )
        if os.path.exists(self.external_validator_stop_fname):
            logger.warning(
                "Removing stale external validator stop file: %s",
                self.external_validator_stop_fname,
            )
            os.remove(self.external_validator_stop_fname)

        self._framework_t = FrameworkType.TORCH
        self.criterion = torch.nn.MSELoss()

        # SBAL mode detection and state
        self._init_sbal_state(config_dict)
        self._split_data_budget_over_ranks()

    def _split_data_budget_over_ranks(self):
        self.ensemble_batch_size = self.batch_size
        n_ranks = self.comm_size
        if n_ranks == 1 or self.world_rank == self.breed_rank:
            return
        if self._sbal_enabled() or self.use_ema_for_validation:
            raise NotImplementedError("SBAL and EMA validation require a single training rank.")
        for key in ("batch_size", "buffer_size", "per_server_watermark"):
            value = math.ceil(getattr(self, key) / n_ranks)
            setattr(self, key, value)
            self.dl_config[key] = value
        if self._rate_limit_enabled:
            self.dl_config["rate_limit_samples_per_batch"] = self._rate_threshold_samples_per_batch / n_ranks
            self._init_rate_limiter(self.dl_config)

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    @override
    def prepare_training_attributes(self):
        self.torch_device = torch.device(getattr(self, "device", "cpu"))
        self._grid_tensor = (
            self._build_grid_tensor().to(torch.float32).to(self.torch_device)
        )
        self._init_physics_scaler()

        if self.train_bf16:
            dev_type = "cuda" if str(self.torch_device).startswith("cuda") else "cpu"
            if dev_type == "cuda":
                bf16_supported = True
                support_fn = getattr(torch.cuda, "is_bf16_supported", None)
                if callable(support_fn):
                    bf16_supported = support_fn()
                if bf16_supported:
                    self._autocast_device_type = "cuda"
                    logger.info("train_bf_16 enabled: using CUDA bfloat16 autocast.")
                else:
                    logger.warning(
                        "train_bf_16 requested but CUDA device does not support bfloat16; falling back to fp32."
                    )
                    self.train_bf16 = False
            else:
                logger.warning(
                    "train_bf_16 requested but only CUDA devices are supported; falling back to fp32."
                )
                self.train_bf16 = False

        if self.torch_device.type == "cuda":
            torch.cuda.set_device(self.torch_device)
        group = self.dist_group if dist.is_initialized() and self.comm_size > 1 else None
        seed = self.config_dict["study_options"].get("seed", 0)
        model = build_torch_model(self.torch_model_cfg, self.scenario, self.physics_dim, group=group, seed=seed)
        model.acquisition_signal = str(self.value_to_register).lower()
        model = model.to(self.torch_device)

        # Initialize per-model optimizers and schedulers inside EnsembleModel
        optimizer = self._build_torch_optimizer(model)
        self._init_ema_model(model)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"TRAINING:00000:\nModel parameters count: {param_count}"
            f"\nEnsemble: {model.n_models} model(s), {len(model.models)} on this rank; local batch "
            f"{self.batch_size} (global {self.ensemble_batch_size}), buffer {self.buffer_size}, watermark "
            f"{self.per_server_watermark}, new samples/batch {self._rate_threshold_samples_per_batch:.2f}"
            f"\nOptimizer config: {self.dl_config.get('torch_optimizer', {})}"
            f"\nAPEBench scenario: {self.scenario.scenario}"
        )

        # Initialize SBAL sampler with model reference if using SBAL breeding
        self._init_sbal_sampler_reference(model)

        return model, optimizer

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @override
    def training_step(self, batch, batch_idx):
        self._torch_training_step(batch, batch_idx)
        if batch_idx % self.checkpoint_model_freq == 0:
            self._handle_periodic_checkpoint(batch_idx)

        # SBAL mode: trigger resampling on training rank (no breed_rank needed)
        if self._sbal_enabled():
            self._sbal_trigger_resampling_if_needed(batch_idx)

    def _torch_training_step(self, batch, batch_idx):
        self.cur_batch_idx = batch_idx
        if len(batch) == 4:
            u_prev, u_next, sim_ids_list, time_step_list = batch
            u_next2 = None
        elif len(batch) == 5:
            u_prev, u_next, u_next2, sim_ids_list, time_step_list = batch
        else:
            raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")

        if self.train_random_t_plus_1_or_2 and u_next2 is None:
            raise ValueError(
                "train_random_t_plus_1_or_2 is enabled but no t+2 target was received. "
                "Ensure configs set study_options.node_names to include 'position_plus1' "
                "and clients send it."
            )

        u_prev_t = self._convert_batch_tensor(u_prev)
        u_next_t = self._convert_batch_tensor(u_next)
        u_next2_t = self._convert_batch_tensor(u_next2) if u_next2 is not None else None
        sim_ids = self._extract_sim_ids(sim_ids_list)
        time_steps = self._extract_sim_ids(time_step_list)
        physics = self._get_physics_tensor(sim_ids)

        autocast_cm = (
            torch.autocast(device_type=self._autocast_device_type, dtype=torch.bfloat16)
            if self.train_bf16 and self._autocast_device_type is not None
            else contextlib.nullcontext()
        )

        with autocast_cm:
            preds, uncertainty_tensor, loss = self.model.training_step(
                state=u_prev_t,
                grid=self._grid_tensor,
                target=u_next_t,
                criterion=self.criterion,
                physics=physics if self.physics_dim > 0 else None,
                step=batch_idx,
                target2=u_next2_t if self.train_random_t_plus_1_or_2 else None,
                batch_size=self.ensemble_batch_size,
            )
        kept = preds.shape[0]
        targets_t = (u_next2_t if self.train_random_t_plus_1_or_2 else u_next_t)[:kept]
        sim_ids, time_steps = sim_ids[:kept], time_steps[:kept]
        self._update_ema()

        # Compute metrics
        with torch.no_grad():
            preds_fp32 = preds.detach().to(torch.float32)
            targets_fp32 = targets_t.detach().to(torch.float32)
            reduce_axes = tuple(range(1, targets_fp32.ndim))
            per_sample = torch.mean((preds_fp32 - targets_fp32) ** 2, dim=reduce_axes)

            rmse = torch.sqrt(per_sample)
            target_rms = torch.sqrt(torch.mean(targets_fp32**2, dim=reduce_axes))
            target_rms = torch.clamp(target_rms, min=1e-6)
            normalized_loss = rmse / target_rms

            uncertainty_np = None
            if uncertainty_tensor is not None:
                uncertainty_np = uncertainty_tensor.detach().cpu().numpy()
                if batch_idx % 500 == 0:
                    signal_name = (
                        "ecrps" if str(self.value_to_register).lower() == "ecrps" else "uncertainty"
                    )
                    logger.info(
                        f"[Ensemble] {signal_name} signal: mean={uncertainty_np.mean():.4e}, "
                        f"std={uncertainty_np.std():.4e}"
                    )

        loss_per_sample = per_sample.detach().cpu().numpy()
        normalized_loss_np = normalized_loss.detach().cpu().numpy()

        loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
        self.metric_logger.log_scalar("Loss/train", loss_value, batch_idx)
        self.metric_logger.log_scalar(
            "Learning_rate", self.get_learning_rate(), batch_idx
        )
        self.metric_logger.log_scalar(
            "Breed_ratio/batch", self.get_breed_ratio(sim_ids), batch_idx
        )
        if batch_idx % 50 == 0:
            logger.info(f"TRAINING:{batch_idx:05d}: Batch loss: {loss_value:.2e}")

        self._register_active_sampling_metrics(
            loss_per_sample,
            sim_ids,
            time_steps,
            batch_idx,
            normalized_loss_per_sample=normalized_loss_np,
            uncertainty_per_sample=uncertainty_np,
        )

    # ------------------------------------------------------------------ #
    # Optimizer / Scheduler
    # ------------------------------------------------------------------ #
    def _build_torch_optimizer(self, model):
        """Build optimizer(s) for the model.

        For EnsembleModel: initializes per-model optimizers inside the ensemble
        For regular models: returns a single optimizer
        """
        cfg = self.dl_config.get("torch_optimizer", {"type": "adam", "lr": 1e-3})
        opt_type = cfg.get("type", "adam").lower()
        betas = cfg.get("betas")
        if betas is not None:
            betas = tuple(float(x) for x in betas)
        eps = cfg.get("eps")

        # Prepare optimizer kwargs
        optimizer_kwargs = {
            "lr": cfg.get("lr", 1e-3),
            "weight_decay": cfg.get(
                "weight_decay", 0.0 if opt_type == "adam" else 0.01
            ),
            "betas": betas or (0.9, 0.999),
            "eps": float(eps) if eps is not None else 1e-8,
        }

        # Choose optimizer class
        if opt_type == "adam":
            optimizer_class = torch.optim.Adam
        elif opt_type == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer type '{opt_type}'")

        # Build scheduler builder function
        scheduler_builder = self._make_scheduler_builder()

        optimizers = model.init_optimizers(
            optimizer_class,
            optimizer_kwargs,
            scheduler_builder,
            adaptive=self.use_adaptive_grad_norm,
            max_norm=self.max_grad_norm,
            multiplier=self._ada_grad_multiplier,
            alpha=self._ada_grad_ema_alpha,
            warmup_steps=self._ada_grad_warmup_steps,
        )
        # Return first optimizer for compatibility with parent class
        return optimizers[0]

    def _make_scheduler_builder(self):
        """Create a scheduler builder function from config."""
        cfg = self.dl_config.get("torch_scheduler")
        if not cfg:
            return None

        sched_type = cfg.get("type", "cosine").lower()

        if sched_type == "none":
            return None

        if sched_type == "cosine":
            t_max = int(
                cfg.get("t_max", self.dl_config.get("num_training_steps", 1000))
            )
            eta_min = float(cfg.get("eta_min", 0.0))
            warmup_steps = int(cfg.get("warmup_steps", 0) or 0)

            def build_cosine_scheduler(opt):
                if warmup_steps <= 0:
                    return torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=t_max, eta_min=eta_min
                    )

                base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]

                def make_lambda(base_lr: float):
                    if base_lr <= 0:
                        return lambda step: 1.0
                    eta_min_factor = eta_min / base_lr
                    eta_min_factor = max(0.0, min(1.0, eta_min_factor))
                    denom = max(1, t_max - warmup_steps)

                    def lr_lambda(step: int):
                        if step < warmup_steps:
                            frac = float(step + 1) / float(max(1, warmup_steps))
                            return eta_min_factor + (1.0 - eta_min_factor) * frac
                        if step >= t_max:
                            return eta_min_factor
                        progress = float(step - warmup_steps) / float(denom)
                        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                        return eta_min_factor + (1.0 - eta_min_factor) * cosine

                    return lr_lambda

                lrs = [make_lambda(lr) for lr in base_lrs]
                return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lrs)

            return build_cosine_scheduler

        if sched_type == "step":
            step_size = cfg.get("step_size", 1000)
            gamma = cfg.get("gamma", 0.5)

            def build_step_scheduler(opt):
                return torch.optim.lr_scheduler.StepLR(
                    opt, step_size=step_size, gamma=gamma
                )

            return build_step_scheduler

        raise ValueError(f"Unsupported scheduler type '{sched_type}'")

    # ------------------------------------------------------------------ #
    # EMA utilities
    # ------------------------------------------------------------------ #
    def _init_ema_model(self, model):
        if not self.use_ema_for_validation:
            self.ema_model = None
            return
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema_for_validation or self.ema_model is None:
            return
        if self.cur_batch_idx < self.ema_start_step:
            return
        if (
            self.ema_update_every > 1
            and (self.cur_batch_idx % self.ema_update_every) != 0
        ):
            return
        decay = self.ema_decay
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                # Handle multi-device ensemble: move model param to EMA device if needed
                p_data = p.detach()
                if ema_p.device != p_data.device:
                    p_data = p_data.to(ema_p.device)
                ema_p.mul_(decay).add_(p_data, alpha=1 - decay)
            for ema_buf, buf in zip(self.ema_model.buffers(), self.model.buffers()):
                if ema_buf.device != buf.device:
                    ema_buf.copy_(buf.to(ema_buf.device))
                else:
                    ema_buf.copy_(buf)

    # ------------------------------------------------------------------ #
    # Batch utilities
    # ------------------------------------------------------------------ #
    def _convert_batch_tensor(self, value, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(self.torch_device, dtype=dtype)

    def _extract_sim_ids(self, sim_ids):
        if isinstance(sim_ids, torch.Tensor):
            return [int(x) for x in sim_ids.detach().cpu().tolist()]
        if isinstance(sim_ids, np.ndarray):
            return [int(x) for x in sim_ids.tolist()]
        return [int(x) for x in sim_ids]

    def _get_physics_tensor(self, sim_ids):
        if self.physics_dim == 0:
            return torch.zeros(len(sim_ids), 0, device=self.torch_device)

        if self.enable_physics_normalization:
            self._init_physics_scaler()

        if self.sample_physics:
            params = []
            for sim_id in self._extract_sim_ids(sim_ids):
                meta: Payload = self._parameter_sampler.current_metadata_list[sim_id]
                if hasattr(meta, "parameters"):
                    vec = meta.parameters
                else:
                    vec = self._parameter_sampler.parameters[sim_id]
                params.append(vec[-self.physics_dim :])
        else:
            fixed = self.fixed_physics_params or [0.0] * self.physics_dim
            params = [fixed for _ in sim_ids]

        # Convert list of numpy arrays to single numpy array first to avoid slow path
        params_array = np.array(params, dtype=np.float32)
        physics_raw = torch.from_numpy(params_array).to(device=self.torch_device)

        if (
            self.enable_physics_normalization
            and self._physics_min is not None
            and self._physics_max is not None
        ):
            denom = (self._physics_max - self._physics_min).clamp_min(1e-8)
            physics = (physics_raw - self._physics_min) / denom
        else:
            physics = physics_raw

        return physics

    def _init_physics_scaler(self):
        if not self.enable_physics_normalization or self.physics_dim == 0:
            self._physics_min = None
            self._physics_max = None
            return
        specs = self.physics_param_specs or []
        if len(specs) < self.physics_dim:
            self._physics_min = None
            self._physics_max = None
            return
        mins = [spec.get("min", 0.0) for spec in specs[-self.physics_dim :]]
        maxs = [spec.get("max", 1.0) for spec in specs[-self.physics_dim :]]
        self._physics_min = torch.tensor(
            mins, device=self.torch_device, dtype=torch.float32
        )
        self._physics_max = torch.tensor(
            maxs, device=self.torch_device, dtype=torch.float32
        )

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _handle_periodic_checkpoint(self, batch_idx: int) -> None:
        if self.checkpoint_model_max_num_files is not None:
            self.chkpt_id = (self.chkpt_id + 1) % self.checkpoint_model_max_num_files
            suffix = f"cur{self.chkpt_id}"
        else:
            suffix = f"{(batch_idx * self.comm_size)}"
        logger.info(f"TRAINING:{batch_idx:05d}: Saving model.")
        self.checkpoint_model(batch_idx, suffix)

    @override
    def checkpoint(self, batch_idx, path="checkpoints"):
        pass

    @override
    def on_train_end(self):
        if self.monitoring_config.get("checkpoint_model_last", False):
            logger.info("Saving last model.")
            self.checkpoint_model(batch_idx=self.cur_batch_idx, suffix="last")

    def checkpoint_model(self, batch_idx, suffix=None):
        model_state = self.model.full_state_dict(optimizers=suffix == "last")
        if self.rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_model_path = "checkpoints/model.pt" if suffix is None else f"checkpoints/model_{suffix}.pt"
            state = {"model_state": model_state, "batch_idx": batch_idx}

            if self.use_ema_for_validation and self.ema_model is not None:
                state["ema_state"] = self.ema_model.state_dict()
            torch.save(state, checkpoint_model_path)
            logger.info(
                f"TRAINING:{batch_idx:05d}: Saved torch model checkpoint to {checkpoint_model_path}"
            )
            self.ext_val_send_checkpoint_signal(
                {
                    "checkpoint_path": os.path.abspath(checkpoint_model_path),
                    "batch_idx": batch_idx,
                }
            )

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _build_grid_tensor(self):
        dims = getattr(self.scenario, "num_spatial_dims", 2)
        num_points = getattr(self.scenario, "num_points", 256)
        if num_points <= 0:
            raise ValueError(
                "Scenario must expose num_points to build the grid tensor."
            )

        grid_resolution = num_points
        axes = [torch.linspace(0.0, 1.0, grid_resolution) for _ in range(dims)]
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        mesh = mesh * 2.0 - 1.0
        return mesh.unsqueeze(0)

    @override
    def process_simulation_data(self, msg, config_dict):
        u_prev = self._extract_field_array(msg, "preposition").reshape(*self.mesh_shape)
        u_next = self._extract_field_array(msg, "position").reshape(*self.mesh_shape)
        if self.train_random_t_plus_1_or_2:
            u_next2 = self._extract_field_array(msg, "position_plus1").reshape(
                *self.mesh_shape
            )
            return u_prev, u_next, u_next2, msg.simulation_id, msg.time_step
        return u_prev, u_next, msg.simulation_id, msg.time_step

    def get_learning_rate(self):
        return self.model.get_learning_rate()
