import argparse
import logging
import os
import time
from typing import Callable

import equinox as eqx
import exponax as ex  # noqa: F401  # kept for backwards compatibility rollouts
import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing_extensions import override

import apebench_online.core.dl_utils as dl_utils
from apebench_online.models import (
    build_torch_model,
)  # noqa: F401  # re-export for legacy callers
from apebench_online.scenarios.scenarios_utils import MelissaSpecificScenario
from apebench_online.scenarios.physics_ic_utils import physics_specs_from_registry
from melissa.utility.external_validation import BaseExternalValidator, get_default_kwargs

from .logger_setup import TBCompatLogger, create_validation_logger
from .metrics import MetricsTracker, compute_pointwise_metrics
from .rollout import run_jax_rollout, run_torch_rollout
from .torch_support import init_torch_model, validation_physics_tensor
import matplotlib.pyplot as plt

logger = logging.getLogger("melissa")


class APEBenchExternalValidator(BaseExternalValidator):
    """
    Cleaner, functionally equivalent rewrite of the original external validator.
    The public surface (class name, CLI, return values) is unchanged.
    """

    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        checkpoint_dir: str | None = None,
        tblog_dir: str | None = None,
        poll_interval: float = 10.0,
        stop_file: str = "",
        max_ckpt_age: float = 0.0,
        max_iterations: int = 1,
    ):
        if config_path is not None:
            super().__init__(config_path=config_path, poll_interval=poll_interval)
            config_dict = self.config_dict
            checkpoint_dir = f"{self.study_dir}/checkpoints"
            tblog_dir = tblog_dir or f"{self.study_dir}/tensorboard/external_validation"
            stop_file = ""
            max_ckpt_age = 0.0
            max_iterations = 1
        elif config_dict is None:
            raise ValueError("Either config_path or config_dict must be provided.")
        else:
            self._config_dict = config_dict
            self._study_dir = config_dict.get("output_dir", "")
            self.running = False

        print("Initializing APEBenchExternalValidator...")
        self.checkpoint_dir = checkpoint_dir or f"{config_dict['output_dir']}/checkpoints"
        self.poll_interval = poll_interval
        self.stop_file = stop_file
        self.max_ckpt_age = max_ckpt_age
        self.max_iterations = max_iterations
        
        # Specific state for this validator
        self.cur_model_id = -1
        self.last_validation = False
        self.last_hash = None
        self.iterations = 0

        study_options = config_dict["study_options"]
        self.monitoring_config = config_dict.get("monitoring_config", dict())

        # Validation data
        self.nb_time_steps = study_options["nb_time_steps"]
        self.dl_config = config_dict["dl_config"]
        self.torch_model_cfg = self.dl_config.get("torch_model")
        self.using_torch_model = self.torch_model_cfg is not None

        self.torch_model = None
        self.torch_device = None
        self._grid_tensor = None
        self.physics_dim = 0
        self.sample_physics = True
        self.fixed_physics_params = None
        self.mesh_axes = None
        self.scenario = None
        self.physics_param_specs: list[dict] | None = []
        self.enable_physics_normalization = bool(self.dl_config.get("normalize_physics", False))
        self._physics_min = None
        self._physics_max = None
        self._physics_log_done = False

        self.valid_rollout = self.dl_config.get("valid_rollout", -1)
        self.valid_batch_size = self.dl_config.get("valid_batch_size", 32)
        self.valid_nb_time_steps = self.dl_config.get("valid_nb_time_steps", self.nb_time_steps)
        self.valid_num_samples = self.dl_config.get("valid_num_samples", 1)
        self.valid_rollout_each_nth = self.dl_config.get("valid_rollout_each_nth", 1)
        self.one_small_batch = self.dl_config.get("valid_rollout_fast", False)
        self.rollout_memory_efficient = self.dl_config.get("valid_rollout_memory_efficient", False)
        self.best_val_loss = np.inf
        self.use_ema_for_validation = self.dl_config.get("use_ema_for_validation", False)

        v_dict = dl_utils.load_validation_dataset(
            validation_dir=self.dl_config.get("validation_directory", None),
            validation_file=self.dl_config.get("validation_file", None),
            nb_time_steps=self.valid_nb_time_steps,
            batch_size=self.valid_batch_size,
            rollout_size=self.valid_rollout,
            num_samples=self.valid_num_samples,
        )
        self.valid_dataset = v_dict["dataset"]
        self.valid_parameters = v_dict["input_parameters"]
        self.valid_dataloader = v_dict["dataloader"]
        self.valid_dataloader_rollout = v_dict["dataloader_rollout"]

        study_detail_flag = bool(study_options.get("use_detail_val", False))
        dl_detail_flag = bool(self.dl_config.get("use_detail_val", False))
        self.use_detail_val = study_detail_flag or dl_detail_flag
        self.detail_groups: tuple[tuple[str, int, int], ...] = tuple()
        self._detail_group_names: list[str] = []
        self._detail_total_samples: int = 0
        self._detail_nb_time_steps: int = self.valid_nb_time_steps
        self._last_detail_means: dict[str, dict[str | None, float]] = {}

        if self.use_detail_val:
            self._init_detail_groups()

        self.metrics = MetricsTracker()

        self.metric_logger: TBCompatLogger = create_validation_logger(config_dict, tblog_dir)

        self.__set_pytree_from_scenario(study_options["scenario_config"])
        if not self.using_torch_model:
            self.__use_cuda_device_0()

    @override
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        print(f"Loading checkpoint from {checkpoint_path}")
        if self.using_torch_model:
            if self.torch_model is None:
                raise RuntimeError("Torch model not initialized before loading checkpoint.")
            state = torch.load(checkpoint_path, map_location=self.torch_device)
            ema_state = state.get("ema_state")
            model_state = (
                ema_state
                if self.use_ema_for_validation and ema_state is not None
                else state.get("model_state", {})
            )
            if isinstance(self.torch_model, torch.nn.parallel.DistributedDataParallel):
                load_target = self.torch_model.module
            else:
                load_target = self.torch_model

            load_target.load_state_dict(model_state)
            load_target.eval()
            out = {
                "model": load_target,
                "model_state": model_state,
                "batch_idx": state.get("batch_idx", 0),
            }
            if ema_state is not None:
                out["ema_state"] = ema_state

            return out
        checkpoint_dict = eqx.tree_deserialise_leaves(checkpoint_path, self.pytree_struct)
        return checkpoint_dict

    @override
    def validation_entrypoint(self, metadata_or_ckpt: dict | None) -> dict | None:
        metadata_or_ckpt = metadata_or_ckpt or {}
        if "checkpoint_path" in metadata_or_ckpt and "model" not in metadata_or_ckpt:
            checkpoint_path = metadata_or_ckpt["checkpoint_path"]
            logger.info(
                f"[Validator] Checkpoint signal received: {checkpoint_path}, "
                f"timestamp: {time.ctime(os.path.getmtime(checkpoint_path))}"
            )
            try:
                ckpt = self.load_checkpoint(checkpoint_path)
            except Exception as exc:
                logger.error(f"[Validator] Error loading checkpoint: {exc}")
                return None
            if "batch_idx" in metadata_or_ckpt:
                ckpt["batch_idx"] = metadata_or_ckpt["batch_idx"]
            if self.cur_model_id == ckpt["batch_idx"]:
                logger.info("[Validator] Skipping validation for the same model id.")
                return None
            return self._validate_checkpoint(ckpt)

        return self._validate_checkpoint(metadata_or_ckpt)

    def _validate_checkpoint(self, ckpt: dict) -> dict | None:
        self.cur_model_id = ckpt["batch_idx"]
        out = dict()

        start_overall = time.time()

        self.on_validation_start()
        batch_durations = []
        for v_batch_idx, v_batch in enumerate(self.valid_dataloader):
            start_batch = time.time()
            self.validation_step(v_batch_idx, v_batch, ckpt)
            end_batch = time.time()
            batch_durations.append(end_batch - start_batch)
        self.on_validation_end(ckpt)

        start_rollout = time.time()
        if self.iterations % self.valid_rollout_each_nth == 0 or self.last_validation:
            out = self.validation_rollout(
                ckpt,
                one_small_batch=self.one_small_batch,
                memory_efficient=self.rollout_memory_efficient,
            )

        end_overall = time.time()

        logger.info("=====================================================================")
        peak_gpu_line = ""
        if self.using_torch_model and self.torch_device and self.torch_device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated(self.torch_device)
            limit = torch.cuda.get_device_properties(self.torch_device).total_memory
            peak_gpu_line = f"\n\tPeak GPU allocation     : {alloc / 1024 / 1024 / 1024:.2f}/{limit / 1024 / 1024 / 1024:.2f} GB"
        elif not self.using_torch_model:
            stats = jax.devices()[0].memory_stats()
            peak_gpu_line = (
                f"\n\tPeak GPU allocation     : {stats['largest_alloc_size'] / 1024 / 1024 / 1024:.2f}"
                f"/{stats['bytes_limit'] / 1024 / 1024 / 1024:.2f} GB"
            )
        logger.info(
            f"\n\tValidation for model {self.cur_model_id} completed."
            f"\n\tValidation {self.iterations} Phase Durations"
            f"\n\tNumber of batches       : {len(batch_durations)}"
            f"\n\tAverage batch duration  : {np.mean(batch_durations):.4f} s"
            f"\n\tBatch loop duration     : {np.sum(batch_durations):.4f} s"
            f"\n\tRollout duration        : {end_overall - start_rollout:.4f} s"
            f"\n\tTotal validation time   : {end_overall - start_overall:.4f} s"
            f"{peak_gpu_line}"
        )
        logger.info("=====================================================================")

        out["avg_val_loss"] = self.metrics.mean_batch_loss
        if (
            getattr(self.metrics, "rmse_per_sample", None) is not None
            and self.metrics.rmse_per_sample.size
        ):
            out["rmse"] = float(np.nanmean(self.metrics.rmse_per_sample))
        if (
            getattr(self.metrics, "rmse_normalized_per_sample", None) is not None
            and self.metrics.rmse_normalized_per_sample.size
        ):
            out["rmse_normalized"] = float(np.nanmean(self.metrics.rmse_normalized_per_sample))
        if (
            getattr(self.metrics, "nrmse_per_sample", None) is not None
            and self.metrics.nrmse_per_sample.size
        ):
            out["nrmse_normalized"] = float(np.nanmean(self.metrics.nrmse_per_sample))

        if self.use_detail_val and self._last_detail_means:
            loss_means = self._last_detail_means.get("Loss_valid_stats", {})
            for name in getattr(self, "_detail_group_names", []):
                if name in loss_means:
                    out[f"avg_val_loss_{name}"] = float(loss_means[name])

        return out

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def __use_cuda_device_0(self):
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        jax.config.update("jax_default_device", jax.devices("gpu")[0])

    def __set_pytree_from_scenario(self, scenario_config):
        scenario_config = dict(scenario_config)
        sample_physics = bool(scenario_config.get("sample_physics", True))
        physics_specs = (
            physics_specs_from_registry(str(scenario_config.get("scenario_name", "")))
            if sample_physics
            else []
        )
        scenario_config["physics_param_specs"] = physics_specs
        self.scenario = MelissaSpecificScenario(**scenario_config)
        self.mesh_axes = tuple(range(1, self.scenario.num_spatial_dims + 2))
        self.physics_dim = len(physics_specs)
        self.physics_param_specs = physics_specs
        self.sample_physics = sample_physics
        self.fixed_physics_params = scenario_config.get("fixed_physics_params")

        if self.using_torch_model:
            self._init_torch_model()
            self.pytree_struct = None
            self._init_physics_scaler()
            return

        model = self.scenario.get_network()
        self.pytree_struct = {
            "model": model,
            "optimizer": object(),
            "opt_state": object(),
            "batch_idx": 1,
        }

    def _init_torch_model(self):
        # Use model resolution for grid building
        model_resolution = self.scenario.get_model_resolution()
        if model_resolution != self.scenario.num_points:
            print(
                f"[VAL][TORCH] Using model resolution {model_resolution} instead of solver resolution {self.scenario.num_points}"
            )

        self.torch_model, self.torch_device, self._grid_tensor = init_torch_model(
            self.dl_config, self.scenario, self.physics_dim
        )

    def _init_physics_scaler(self):
        if not self.enable_physics_normalization or not self.using_torch_model:
            self._physics_min = None
            self._physics_max = None
            return
        if self.physics_dim == 0:
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
        device = self.torch_device or torch.device("cpu")
        self._physics_min = torch.tensor(mins, device=device, dtype=torch.float32)
        self._physics_max = torch.tensor(maxs, device=device, dtype=torch.float32)

    def _log_example_rollouts(self, model, num_traj: int = 3, steps=(0, 10, 20, 30, 40, 50)):
        """Log GT vs pred for fixed trajectories (0,250,500 if available) at specified timesteps."""
        try:

            model.eval()
            batches = list(self.valid_dataloader_rollout)
            if not batches:
                return
            ic_idx_batch, rollout_idx_batch = batches[0]
            if len(ic_idx_batch) == 0:
                return
            # rollout_idx_batch is shaped [time][batch]; transpose so we can index per trajectory
            rollout_idx_per_sample = (
                list(map(list, zip(*rollout_idx_batch))) if rollout_idx_batch else []
            )
            if not rollout_idx_per_sample:
                return
            desired = [0, 250, 500]
            chosen = [d for d in desired if d in ic_idx_batch]
            if len(chosen) < num_traj:
                for i in ic_idx_batch:
                    if i not in chosen:
                        chosen.append(i)
                    if len(chosen) == num_traj:
                        break
            positions = [ic_idx_batch.index(i) for i in chosen[:num_traj]]
            ic_idx = [ic_idx_batch[i] for i in positions]
            rollout_idx = [rollout_idx_per_sample[i][: self.valid_rollout] for i in positions]

            u_step = np.asarray(self.valid_dataset[ic_idx], dtype=np.float32)
            current_norm = torch.as_tensor(u_step, dtype=torch.float32, device=self.torch_device)
            is_1d_rollout = u_step.ndim == 3 and np.asarray(u_step[0, 0]).ndim == 1

            # run rollout up to max step
            preds_per_step = {}
            available_rollout_steps = min((len(items) for items in rollout_idx), default=0)
            max_step = min(int(self.valid_rollout), int(available_rollout_steps))
            if not is_1d_rollout:
                max_step = min(max(steps), max_step)
            grid = self._grid_tensor.expand(current_norm.shape[0], *self._grid_tensor.shape[1:])
            physics = self._get_validation_physics_tensor(ic_idx)
            # include step 0 snapshot
            preds_per_step[0] = current_norm.detach().cpu().numpy()
            for s in range(1, max_step + 1):
                with torch.no_grad():
                    current_norm = model(
                        current_norm, grid, physics if physics is not None else None
                    )
                if is_1d_rollout or s in steps:
                    # store the model output at each requested rollout step
                    preds_per_step[s] = current_norm.detach().cpu().numpy()

            if is_1d_rollout:
                for t_idx, traj in enumerate(range(len(ic_idx))):
                    gt_rows = []
                    pred_rows = []
                    for step in range(0, max_step + 1):
                        if step == 0:
                            gt = u_step[traj]
                        else:
                            gt = np.asarray(
                                self.valid_dataset[rollout_idx[traj][step - 1]],
                                dtype=np.float32,
                            )
                        pred = preds_per_step.get(step)
                        if pred is None:
                            continue
                        gt_rows.append(np.asarray(gt[0], dtype=np.float32))
                        pred_rows.append(np.asarray(pred[traj][0], dtype=np.float32))
                    if not gt_rows:
                        continue
                    gt_map = np.stack(gt_rows, axis=0)
                    pred_map = np.stack(pred_rows, axis=0)
                    diff_map = np.abs(gt_map - pred_map)
                    mse_mean = float(np.mean((gt_map - pred_map) ** 2))

                    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                    axes[0].imshow(gt_map, aspect="auto", cmap="viridis", origin="lower")
                    axes[0].set_title("GT")
                    axes[1].imshow(pred_map, aspect="auto", cmap="viridis", origin="lower")
                    axes[1].set_title(f"Pred | MSE mean: {mse_mean:.2e}")
                    axes[2].imshow(diff_map, aspect="auto", cmap="hot", origin="lower")
                    axes[2].set_title("Abs error")
                    for ax in axes:
                        ax.set_ylabel("rollout step")
                    axes[-1].set_xlabel("x")
                    fig.tight_layout()
                    self.metric_logger.add_figure(
                        f"Rollout_examples_1d/traj_{traj}",
                        fig,
                        self.cur_model_id,
                    )
                    plt.close(fig)
                return

            # build figure for each traj
            for t_idx, traj in enumerate(range(len(ic_idx))):
                fig, axes = plt.subplots(3, len(steps), figsize=(3 * len(steps), 9))
                mse_per_step = []  # collect MSE for each displayed step
                l2_per_step = []  # collect L2 norm of diff for each displayed step
                for j, step in enumerate(steps):
                    # guard if requested step exceeds rollout length
                    if step == 0:
                        gt = u_step[traj]
                    elif step - 1 < len(rollout_idx[traj]):
                        gt = np.asarray(
                            self.valid_dataset[rollout_idx[traj][step - 1]],
                            dtype=np.float32,
                        )
                    else:
                        continue
                    if step in preds_per_step:
                        pred = preds_per_step[step][traj]
                    else:
                        continue
                    gt_img = gt[0]
                    pred_img = pred[0]
                    # Compute MSE and L2 diff for this step
                    diff = gt - pred
                    step_mse = float(np.mean(diff ** 2))
                    step_l2 = float(np.sqrt(np.sum(diff ** 2)))
                    mse_per_step.append(step_mse)
                    l2_per_step.append(step_l2)
                    # Plot GT
                    axes[0, j].imshow(gt_img, cmap="viridis")
                    axes[0, j].axis("off")
                    axes[0, j].set_title(f"GT t+{step}")
                    # Plot Pred
                    axes[1, j].imshow(pred_img, cmap="viridis")
                    axes[1, j].axis("off")
                    axes[1, j].set_title(f"Pred t+{step}\nMSE: {step_mse:.2e}")
                    # Plot Diff (L2)
                    diff_img = np.abs(diff[0])
                    im_diff = axes[2, j].imshow(diff_img, cmap="hot")
                    axes[2, j].axis("off")
                    axes[2, j].set_title(f"Diff t+{step}\nL2: {step_l2:.2e}")
                # Compute trajectory summary stats
                if mse_per_step:
                    mean_mse = float(np.mean(mse_per_step))
                    max_mse = float(np.max(mse_per_step))
                    mean_l2 = float(np.mean(l2_per_step))
                    max_l2 = float(np.max(l2_per_step))
                    fig.suptitle(f"Trajectory {traj} | MSE (mean/max): {mean_mse:.2e}/{max_mse:.2e} | L2 (mean/max): {mean_l2:.2e}/{max_l2:.2e}")
                else:
                    fig.suptitle(f"Trajectory {traj}")
                self.metric_logger.add_figure(f"Rollout_examples/traj_{traj}", fig, self.cur_model_id)
                plt.close(fig)
        except Exception as exc:
            logger.warning(f"Failed to log example rollouts: {exc}")

    def _get_validation_physics_tensor(self, indices):
        physics = validation_physics_tensor(
            indices=indices,
            physics_dim=self.physics_dim,
            sample_physics=self.sample_physics,
            fixed_physics_params=self.fixed_physics_params,
            valid_dataset=self.valid_dataset,
            valid_parameters=self.valid_parameters,
            nb_time_steps=self.valid_nb_time_steps,
            torch_device=self.torch_device,
            physics_min=(self._physics_min if self.enable_physics_normalization else None),
            physics_max=(self._physics_max if self.enable_physics_normalization else None),
        )
        return physics

    def _init_detail_groups(self):
        dataset = self.valid_dataset
        if dataset is None:
            self.use_detail_val = False
            return
        total = int(
            getattr(dataset, "global_num_samples", None) or getattr(dataset, "num_samples", 0)
        )
        if total <= 1:
            self.use_detail_val = False
            return
        split = total // 2
        if split <= 0:
            self.use_detail_val = False
            return
        self.detail_groups = (
            ("easy", 0, split),
            ("hard", split, total),
        )
        self._detail_group_names = [name for name, _, _ in self.detail_groups]
        self._detail_total_samples = total
        self._detail_nb_time_steps = getattr(dataset, "nb_time_steps", self.valid_nb_time_steps)

    def _detail_group_specs(self) -> list[tuple[str, int, int]] | None:
        if not self.use_detail_val or not self.detail_groups:
            return None
        return [(name, start, end) for name, start, end in self.detail_groups]

    def _detail_rollout_config(self):
        specs = self._detail_group_specs()
        if not specs:
            return None
        return {
            "names": [name for name, _, _ in specs],
            "bounds": [(start, end) for _, start, end in specs],
        }

    def _sample_ids_from_indices(self, indices) -> np.ndarray:
        if not indices:
            return np.array([], dtype=np.int64)
        steps = getattr(self.valid_dataset, "nb_time_steps", None) or self.valid_nb_time_steps
        if steps <= 0:
            return np.zeros(len(indices), dtype=np.int64)
        flat = np.asarray(indices, dtype=np.int64)
        return flat // int(steps)

    # -------------------------------------------------------------------------
    # Validation loop
    # -------------------------------------------------------------------------
    def on_validation_start(self):
        self.metrics.reset()
        if self.enable_physics_normalization:
            self._physics_log_done = False
        self._last_detail_means = {}

    def validation_step(self, valid_batch_idx, valid_batch, ckpt):
        """Loss across all trajectories and their time steps t[i] -> t[i + 1]."""
        if self.using_torch_model:
            self._validation_step_torch(valid_batch_idx, valid_batch, ckpt)
            return
        print(f"Running validation step {valid_batch_idx}...")
        model = ckpt["model"]

        assert self.valid_dataset is not None
        u_step = jnp.asarray(self.valid_dataset[valid_batch[0]])
        u_next = jnp.asarray(self.valid_dataset[valid_batch[1]])
        prediction = jax.vmap(model)(u_step)
        (
            loss_per_sample,
            rmse_per_sample,
            rmse_normalized_per_sample,
            nrmse_per_sample,
        ) = compute_pointwise_metrics(prediction, u_next, self.mesh_axes, normalizer=u_step)
        den = jnp.mean(u_step**2, axis=self.mesh_axes)
        den = jnp.where(den > 0, den, 1.0)
        mse_normalized_per_sample = loss_per_sample / den
        batch_loss = jnp.mean(loss_per_sample)

        if not np.isfinite(batch_loss.item()):
            logger.info(f"NaN or Inf loss encountered at batch {valid_batch_idx}.")
            logger.info(f"LOSSES = {loss_per_sample}")

        sample_ids = self._sample_ids_from_indices(valid_batch[0])
        self.metrics.record(
            loss_per_sample,
            rmse_per_sample,
            rmse_normalized_per_sample,
            nrmse_per_sample,
            batch_loss,
            sample_ids,
            mse_normalized_per_sample=mse_normalized_per_sample,
        )

    def _validation_step_torch(self, valid_batch_idx, valid_batch, ckpt):
        t0 = time.time()
        u_prev_idx, u_next_idx = valid_batch
        if not u_prev_idx:
            return
        
        # Time data loading from dataset
        t1 = time.time()
        u_prev = self.valid_dataset[u_prev_idx]
        u_next = self.valid_dataset[u_next_idx]
        t_data_load = time.time() - t1
        
        # Time CPU -> GPU transfer
        t2 = time.time()
        u_prev_t = torch.as_tensor(u_prev, dtype=torch.float32, device=self.torch_device)
        u_next_t = torch.as_tensor(u_next, dtype=torch.float32, device=self.torch_device)
        grid = self._grid_tensor.expand(u_prev_t.shape[0], *self._grid_tensor.shape[1:])
        physics = self._get_validation_physics_tensor(u_prev_idx)
        t_h2d = time.time() - t2
        
        # Time model forward pass
        model = ckpt["model"]
        model.eval()
        t3 = time.time()
        with torch.no_grad():
            preds = model(u_prev_t, grid, physics if physics is not None else None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure forward is complete for accurate timing
        t_forward = time.time() - t3
        
        # Time metrics computation
        t4 = time.time()
        reduce_axes = tuple(self.mesh_axes)
        with torch.no_grad():
            diff = preds - u_next_t
            mse = torch.mean(diff**2, dim=reduce_axes)
            rmse = torch.sqrt(mse)
            den = torch.mean(u_prev_t**2, dim=reduce_axes)
            den = torch.clamp(den, min=1e-6)
            rmse_normalized = rmse / torch.sqrt(den)
            mse_normalized = mse / den
            nrmse = rmse_normalized
        t_metrics_gpu = time.time() - t4

        # Time GPU -> CPU transfer
        t5 = time.time()
        loss_per_sample = mse.detach().cpu().numpy()
        rmse_per_sample = rmse.detach().cpu().numpy()
        rmse_normalized_per_sample = rmse_normalized.detach().cpu().numpy()
        nrmse_per_sample = nrmse.detach().cpu().numpy()
        mse_normalized_per_sample = mse_normalized.detach().cpu().numpy()
        t_d2h = time.time() - t5
        
        batch_loss = float(np.mean(loss_per_sample))

        if not np.isfinite(batch_loss):
            logger.info(f"NaN or Inf loss encountered at batch {valid_batch_idx}.")
            logger.info(f"LOSSES = {loss_per_sample}")

        # Time metrics recording
        t6 = time.time()
        sample_ids = self._sample_ids_from_indices(u_prev_idx)
        self.metrics.record(
            loss_per_sample,
            rmse_per_sample,
            rmse_normalized_per_sample,
            nrmse_per_sample,
            batch_loss,
            sample_ids,
            mse_normalized_per_sample=mse_normalized_per_sample,
        )
        t_record = time.time() - t6
        
        t_total = time.time() - t0
        
        # Log timing every 50 batches
        if valid_batch_idx % 50 == 0:
            logger.info(
                f"[VAL TIMING] Batch {valid_batch_idx}: "
                f"data_load={t_data_load*1000:.1f}ms, "
                f"h2d={t_h2d*1000:.1f}ms, "
                f"forward={t_forward*1000:.1f}ms, "
                f"metrics_gpu={t_metrics_gpu*1000:.1f}ms, "
                f"d2h={t_d2h*1000:.1f}ms, "
                f"record={t_record*1000:.1f}ms, "
                f"TOTAL={t_total*1000:.1f}ms"
            )

    def on_validation_end(self, ckpt):
        self.metric_logger.add_scalar(
            "Loss_valid/mean", self.metrics.mean_batch_loss, self.cur_model_id
        )
        mean_registry = self.metrics.log_distributions(
            self.metric_logger,
            self.cur_model_id,
            group_specs=self._detail_group_specs(),
        )
        self._last_detail_means = mean_registry or {}
        self.__save_best_model(ckpt)

    # -------------------------------------------------------------------------
    # Rollout evaluation
    # -------------------------------------------------------------------------
    def validation_rollout(
        self,
        ckpt,
        memory_efficient=True,
        plot_rollout_loss=False,
        one_small_batch=False,
    ):
        group_config = self._detail_rollout_config()
        label_nb_steps = getattr(self.valid_dataset, "nb_time_steps", self.valid_nb_time_steps)
        if self.using_torch_model:
            rollout_metrics = run_torch_rollout(
                model=ckpt["model"],
                valid_dataset=self.valid_dataset,
                valid_dataloader_rollout=self.valid_dataloader_rollout,
                valid_rollout=self.valid_rollout,
                nb_time_steps=self.nb_time_steps,
                mesh_axes=self.mesh_axes,
                valid_batch_size=self.valid_batch_size,
                valid_num_samples=self.valid_num_samples,
                tb_logger=self.metric_logger,
                cur_model_id=self.cur_model_id,
                get_physics=self._get_validation_physics_tensor,
                grid_tensor=self._grid_tensor,
                one_small_batch=one_small_batch,
                memory_efficient=memory_efficient,
                plot_rollout_loss=plot_rollout_loss,
                group_config=group_config,
                label_nb_time_steps=label_nb_steps,
            )
            self._log_example_rollouts(ckpt["model"])
            return rollout_metrics

        return run_jax_rollout(
            model=ckpt["model"],
            valid_dataset=self.valid_dataset,
            valid_dataloader_rollout=self.valid_dataloader_rollout,
            valid_rollout=self.valid_rollout,
            nb_time_steps=self.nb_time_steps,
            mesh_axes=self.mesh_axes,
            valid_batch_size=self.valid_batch_size,
            valid_num_samples=self.valid_num_samples,
            tb_logger=self.metric_logger,
            cur_model_id=self.cur_model_id,
            one_small_batch=one_small_batch,
            memory_efficient=memory_efficient,
            plot_rollout_loss=plot_rollout_loss,
            group_config=group_config,
            label_nb_time_steps=label_nb_steps,
        )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def __save_best_model(self, ckpt):
        if not self.monitoring_config.get("checkpoint_model_best", False):
            return

        avg_val_loss = self.metrics.mean_batch_loss
        if avg_val_loss >= self.best_val_loss:
            return

        self.best_val_loss = avg_val_loss
        if self.using_torch_model:
            best_model_path = f"{self.checkpoint_dir}/model_best.pt"
            torch.save(
                {
                    "model_state": ckpt.get("model_state", ckpt["model"].state_dict()),
                    "batch_idx": ckpt.get("batch_idx", self.cur_model_id),
                },
                best_model_path,
            )
            return

        best_model_path = f"{self.checkpoint_dir}/model_best.eqx"
        eqx.tree_serialise_leaves(best_model_path, ckpt)

# -------------------------------------------------------------------------
# CLI helpers
# -------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Run external validation from Melissa signals.")
    parser.add_argument("--config", dest="config_path", type=str, required=True)
    parser.add_argument("--poll_interval", type=float, default=10.0)
    return parser


def get_dir_if_absolute(path: str) -> str | None:
    """Forcing absolute path."""
    if os.path.isabs(path):
        return os.path.dirname(path)
    return os.getcwd()


def main():
    APEBenchExternalValidator(**get_default_kwargs()).run()


if __name__ == "__main__":
    main()
