"""SBAL-specific helpers for the Torch server.

This module keeps SBAL logic outside of APEBenchTorchServer by providing a mixin
that mirrors the in-server behavior, without changing algorithms.
"""

import logging
import os
import time
from typing import Dict, Any

import numpy as np
import torch

from melissa.utility.rank_helper import rank_zero_only

logger = logging.getLogger("melissa")


class SBALModeMixin:
    """Mixin that adds SBAL-specific behavior to APEBenchTorchServer."""

    def _init_sbal_state(self, config_dict: Dict[str, Any]) -> None:
        self._is_sbal_mode = is_sbal_config(config_dict)
        self._sbal_last_submitted_id = -1
        self._sbal_resampling_in_progress = False
        self._sbal_last_finished_count = 0
        if self._is_sbal_mode:
            logger.info(
                "[SBAL] SBAL mode enabled - resampling will run on training rank"
            )

    def _sbal_enabled(self) -> bool:
        return bool(getattr(self, "_is_sbal_mode", False))

    def start(self) -> None:
        """Override start to handle SBAL mode specially."""
        if self._sbal_enabled() and self.world_rank == self.breed_rank:
            logger.info(
                f"[SBAL] Rank {self.world_rank} is breed_rank - idling (not needed for SBAL)"
            )
            self.setup_environment()
            self._sbal_breed_rank_idle_loop()
            return

        super().start()

    def _sbal_breed_rank_idle_loop(self) -> None:
        """Idle loop for breed_rank in SBAL mode."""
        logger.info("[SBAL] Breed rank entering idle loop")

        output_dir = self.config_dict.get("output_dir", ".")
        sentinel_file = os.path.join(output_dir, ".sbal_terminate_breed_rank")

        if os.path.exists(sentinel_file):
            try:
                os.remove(sentinel_file)
            except Exception:
                pass

        poll_interval = 2.0
        while True:
            time.sleep(poll_interval)

            if os.path.exists(sentinel_file):
                logger.info("[SBAL] Breed rank detected termination sentinel")
                break

            if getattr(self, "terminate_breeding", False):
                logger.info("[SBAL] Breed rank detected terminate_breeding flag")
                break

            if not getattr(self, "is_receiving", True):
                logger.info("[SBAL] Breed rank detected is_receiving=False")
                break

        logger.info("[SBAL] Breed rank exiting idle loop")
        self._server_finalize(exit_=0)

    def _sbal_trigger_resampling_if_needed(self, batch_idx: int) -> bool:
        if not self._sbal_enabled():
            return False

        if getattr(self, "_sbal_resampling_in_progress", False):
            return False

        nn_threshold = self.ac_config.get("nn_updates", 100)
        min_finished = self.ac_config.get("min_nb_finished_simulations", 0)
        min_period = self.ac_config.get("min_nb_finished_simulations_period", -1)

        current_finished = self.nb_finished_simulations
        finished_since_last = current_finished - self._sbal_last_finished_count

        has_period_budget = min_period < 0 or finished_since_last >= min_period

        should_trigger = (
            batch_idx > 0
            and current_finished >= min_finished
            and nn_threshold > 0
            and has_period_budget
            and (batch_idx + 1) % nn_threshold == 0
        )

        if should_trigger:
            self._sbal_last_finished_count = current_finished
            return self._do_sbal_resampling(batch_idx)

        return False

    @rank_zero_only
    def _do_sbal_resampling(self, batch_idx: int) -> bool:
        self._sbal_resampling_in_progress = True
        t_start = time.time()

        try:
            sampler = getattr(self, "_parameter_sampler", None)
            if sampler is None:
                logger.warning("[SBAL] No parameter sampler available")
                return False

            if self._sbal_last_submitted_id < 0:
                self._sbal_last_submitted_id = self.current_max_submitted_sim_id

            remaining = sampler.nb_sims - (self._sbal_last_submitted_id + 1)
            if remaining <= 0:
                logger.info("[SBAL] All simulations submitted")
                return False

            min_period = self.ac_config.get("min_nb_finished_simulations_period", 1000)
            margin = 1.05
            target_count = int(min_period * margin)
            max_count = min(target_count, remaining)

            logger.info(
                f"[SBAL] Resampling at batch {batch_idx}: "
                f"last_submitted={self._sbal_last_submitted_id}, remaining={remaining}, "
                f"generating up to {max_count} params"
            )

            sampler.next_parameters(max_breeding_count=max_count)

            first_id, last_id = sampler.concretize_resampled_parameters(
                self._sbal_last_submitted_id
            )

            if first_id >= 0:
                self._sbal_last_submitted_id = last_id
                n_generated = last_id - first_id + 1
                t_elapsed = time.time() - t_start

                logger.info(
                    f"[SBAL] Generated {n_generated} params (sims {first_id}-{last_id}) "
                    f"in {t_elapsed:.2f}s"
                )

                if self.metric_logger is not None:
                    self.metric_logger.log_scalar(
                        "SBAL/num_generated", n_generated, batch_idx
                    )
                    self.metric_logger.log_scalar(
                        "SBAL/time_seconds", t_elapsed, batch_idx
                    )
                    self.metric_logger.log_scalar(
                        "SBAL/breed_ratio",
                        getattr(sampler, "current_breed_ratio", 0),
                        batch_idx,
                    )

            return True

        except Exception as e:
            logger.error(f"[SBAL] Resampling failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self._sbal_resampling_in_progress = False

    def _on_batch_end(self, batch_idx: int) -> None:
        """Bypass breed_rank MPI sync in SBAL mode."""
        if self._sbal_enabled():
            from melissa.server.deep_learning.base_dl_server import DeepMelissaServer

            DeepMelissaServer._on_batch_end(self, batch_idx)
            return

        super()._on_batch_end(batch_idx)

    def _server_offline(self) -> None:
        super()._server_offline()
        if self._sbal_enabled() and self.rank == 0:
            output_dir = self.config_dict.get("output_dir", ".")
            sentinel_file = os.path.join(output_dir, ".sbal_terminate_breed_rank")
            try:
                open(sentinel_file, "w").close()
                logger.info(f"[SBAL] Created termination sentinel: {sentinel_file}")
            except Exception as e:
                logger.warning(f"[SBAL] Could not create sentinel file: {e}")

    def _init_sbal_sampler_reference(self, model) -> None:
        sampler = getattr(self, "_parameter_sampler", None)
        if sampler is None:
            return

        sampler_class_name = type(sampler).__name__
        if sampler_class_name != "ICSamplerSBAL":
            return

        from apebench_online.sbal.ic_sampler_sbal import ICSamplerSBAL

        if isinstance(sampler, ICSamplerSBAL):
            logger.info("Initializing SBAL sampler with model reference")

            def ic_generator(params):
                from apebench_online.sampler.ic_generation import get_ic_maker

                ic_params, _ = sampler._split_parameters(params)
                filled_ic_config = sampler._fill_ic_config(ic_params)
                ic_maker = get_ic_maker(
                    sampled_ic_config=filled_ic_config,
                    domain_extent=self.scenario.domain_extent,
                    num_points=self.scenario.num_points,
                    num_spatial_dims=self.scenario.num_spatial_dims,
                    num_channels=self.scenario.num_channels,
                )
                ic = ic_maker()
                return torch.from_numpy(np.array(ic)).float().to(self.torch_device)

            def physics_extractor(params):
                if self.physics_dim == 0:
                    return None
                _, physics_params = sampler._split_parameters(params)
                physics_tensor = torch.tensor(physics_params, dtype=torch.float32)
                physics_tensor = physics_tensor.to(self.torch_device)
                if (
                    hasattr(self, "enable_physics_normalization")
                    and self.enable_physics_normalization
                    and self._physics_min is not None
                ):
                    denom = (self._physics_max - self._physics_min).clamp_min(1e-8)
                    physics_tensor = (physics_tensor - self._physics_min) / denom
                return physics_tensor

            sampler.set_model_reference(
                model=model,
                grid_tensor=self._grid_tensor,
                ic_generator=ic_generator,
                physics_extractor=physics_extractor,
                tb_logger=getattr(self, "_tb_logger", None),
            )


def integrate_sbal_into_server(server_instance) -> None:
    """Backwards-compatible patching for SBAL methods."""
    import types

    mixin = SBALModeMixin()
    for name in dir(mixin):
        if name.startswith("_sbal") or name in {
            "start",
            "_on_batch_end",
            "_server_offline",
        }:
            method = getattr(mixin, name)
            if callable(method):
                bound = types.MethodType(method.__func__, server_instance)
                setattr(server_instance, name, bound)

    server_instance._init_sbal_state(getattr(server_instance, "config_dict", {}))
    logger.info("[SBAL] Server instance patched with SBAL methods")


def is_sbal_config(config_dict: Dict[str, Any]) -> bool:
    """Check if configuration is for SBAL mode.

    Args:
        config_dict: Server configuration dictionary

    Returns:
        True if this is an SBAL configuration
    """
    ac_config = config_dict.get("active_sampling_config", {})
    return ac_config.get("breeder_backend") == "sbal"
