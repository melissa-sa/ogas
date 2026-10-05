import logging
from typing import Callable

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import torch

from .metrics import compute_pointwise_metrics

logger = logging.getLogger("melissa")


def _empty_rollout_result():
    return {
        "rollout_loss": np.nan,
        "rollout_known_loss": None,
        "rollout_rmse": np.nan,
        "rollout_rmse_known": None,
        "rollout_rmse_normalized": np.nan,
        "rollout_rmse_normalized_known": None,
        "rollout_losses_mean": None,
        "rollout_trajectory_ids": np.array([], dtype=np.int64),
        "rollout_horizons": np.array([], dtype=np.int64),
        "rollout_horizon_errors": np.empty((0, 0), dtype=np.float64),
        "rollout_metric_steps": {
            name: np.empty((0, 0), dtype=np.float64)
            for name in (
                "mse", "mse_normalized", "rmse", "rmse_normalized"
            )
        },
        "group_rollout": {},
    }


def _summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p10": float(np.nanpercentile(arr, 10)),
    }


def _init_group_tracker(group_config, valid_rollout: int):
    if not group_config:
        return None
    names = list(group_config.get("names") or [])
    bounds = list(group_config.get("bounds") or [])
    if not names or not bounds:
        return None
    num_groups = len(names)
    return {
        "names": names,
        "bounds": bounds,
        "nrmse": np.zeros((num_groups, valid_rollout), dtype=np.float64),
        "rmse": np.zeros((num_groups, valid_rollout), dtype=np.float64),
        "rmse_norm": np.zeros((num_groups, valid_rollout), dtype=np.float64),
        "counts": np.zeros(num_groups, dtype=np.int64),
    }


def _group_labels_from_ic_idx(ic_idx, nb_time_steps: int, tracker):
    if tracker is None or nb_time_steps <= 0:
        return None
    sample_ids = np.asarray(ic_idx, dtype=np.int64) // nb_time_steps
    labels = np.full(sample_ids.shape, -1, dtype=np.int64)
    for gid, (lower, upper) in enumerate(tracker["bounds"]):
        mask = (sample_ids >= lower) & (sample_ids < upper)
        labels[mask] = gid
    return labels


def _update_group_counts(tracker, labels):
    if tracker is None or labels is None:
        return
    valid = labels >= 0
    if not np.any(valid):
        return
    counts = np.bincount(labels[valid], minlength=tracker["counts"].size)
    tracker["counts"][: len(counts)] += counts


def _log_group_rollout_results(
    tracker,
    tb_logger,
    cur_model_id,
    valid_rollout: int,
    nb_time_steps: int,
):
    if tracker is None:
        return {}
    results = {}
    for gid, name in enumerate(tracker["names"]):
        count = tracker["counts"][gid]
        if count == 0:
            continue
        denom_long = count * valid_rollout
        loss_long = tracker["nrmse"][gid].sum() / denom_long
        rmse_long = tracker["rmse"][gid].sum() / denom_long
        rmse_norm_long = tracker["rmse_norm"][gid].sum() / denom_long

        tb_logger.add_scalar(
            f"Loss_valid_{name}/rollout_longer_n{valid_rollout}_nRMSE",
            loss_long,
            cur_model_id,
        )
        tb_logger.add_scalar(
            f"Loss_valid_{name}/rollout_longer_rmse",
            rmse_long,
            cur_model_id,
        )
        tb_logger.add_scalar(
            f"Loss_valid_{name}/rollout_longer_rmse_normalized",
            rmse_norm_long,
            cur_model_id,
        )

        seen_stats = {"loss": None, "rmse": None, "rmse_norm": None}
        if valid_rollout > nb_time_steps:
            denom_seen = count * nb_time_steps
            seen_stats["loss"] = tracker["nrmse"][gid, :nb_time_steps].sum() / denom_seen
            seen_stats["rmse"] = tracker["rmse"][gid, :nb_time_steps].sum() / denom_seen
            seen_stats["rmse_norm"] = tracker["rmse_norm"][gid, :nb_time_steps].sum() / denom_seen
            tb_logger.add_scalar(
                f"Loss_valid_{name}/rollout_seen_n{nb_time_steps}_nRMSE",
                seen_stats["loss"],
                cur_model_id,
            )
            tb_logger.add_scalar(
                f"Loss_valid_{name}/rollout_seen_rmse",
                seen_stats["rmse"],
                cur_model_id,
            )
            tb_logger.add_scalar(
                f"Loss_valid_{name}/rollout_seen_rmse_normalized",
                seen_stats["rmse_norm"],
                cur_model_id,
            )

        results[name] = {
            "rollout_loss": loss_long,
            "rollout_rmse": rmse_long,
            "rollout_rmse_normalized": rmse_norm_long,
            "rollout_seen_loss": seen_stats["loss"],
            "rollout_seen_rmse": seen_stats["rmse"],
            "rollout_seen_rmse_normalized": seen_stats["rmse_norm"],
        }

    return results


def prepare_rollout_config(
    valid_dataset,
    valid_batch_size,
    valid_num_samples,
    one_small_batch,
    memory_efficient,
):
    """
    Decide how many validation trajectories to roll out and whether to use the
    memory‑efficient streaming path.
    """
    batch_size = None
    total_samples = getattr(valid_dataset, "num_samples", None)

    if one_small_batch:
        batch_size = min(valid_batch_size, total_samples or valid_num_samples or 500)
        total_samples = batch_size

    return batch_size, total_samples, memory_efficient


def run_jax_rollout(
    *,
    model,
    valid_dataset,
    valid_dataloader_rollout,
    valid_rollout: int,
    nb_time_steps: int,
    mesh_axes,
    valid_batch_size: int,
    valid_num_samples: int,
    tb_logger,
    cur_model_id,
    one_small_batch=False,
    memory_efficient=True,
    plot_rollout_loss=False,
    group_config=None,
    label_nb_time_steps=None,
):
    batch_size, total_samples, memory_efficient = prepare_rollout_config(
        valid_dataset,
        valid_batch_size,
        valid_num_samples,
        one_small_batch,
        memory_efficient,
    )

    if valid_rollout <= 0 or total_samples in (0, None):
        logger.warning(
            "Skipping rollout metrics logging because valid_rollout is not positive or total_samples is zero."
        )
        return _empty_rollout_result()

    rollout_nrmse_steps = np.zeros(valid_rollout, dtype=np.float64)
    rollout_rmse_steps = np.zeros_like(rollout_nrmse_steps)
    rollout_rmse_norm_steps = np.zeros_like(rollout_nrmse_steps)
    rollout_losses = np.zeros(valid_rollout, dtype=np.float64) if plot_rollout_loss else None
    group_tracker = _init_group_tracker(group_config, valid_rollout) if memory_efficient else None

    label_nb_time_steps = label_nb_time_steps or nb_time_steps

    for ic_idx, rollout_idx in valid_dataloader_rollout:
        active_ic_idx = ic_idx
        if one_small_batch:
            active_ic_idx = ic_idx[:batch_size]

        group_labels = None
        if group_tracker is not None:
            group_labels = _group_labels_from_ic_idx(
                active_ic_idx, label_nb_time_steps, group_tracker
            )
            _update_group_counts(group_tracker, group_labels)

        u_step = jnp.asarray(valid_dataset[active_ic_idx])

        if memory_efficient:
            metrics = jax_rollout_memory_efficient(
                model=model,
                u_step=u_step,
                rollout_idx=rollout_idx,
                batch_size=batch_size,
                one_small_batch=one_small_batch,
                valid_dataset=valid_dataset,
                rollout_losses=rollout_losses,
                plot_rollout_loss=plot_rollout_loss,
                group_assignments=group_labels,
                num_groups=len(group_tracker["names"]) if group_tracker else 0,
            )
        else:
            metrics = jax_rollout_full_trajectory(
                model=model,
                u_step=u_step,
                rollout_idx=rollout_idx,
                batch_size=batch_size,
                one_small_batch=one_small_batch,
                valid_dataset=valid_dataset,
                valid_rollout=valid_rollout,
            )

        rollout_nrmse_steps += np.asarray(metrics["nrmse_step_sum"])
        rollout_rmse_steps += np.asarray(metrics["rmse_step_sum"])
        rollout_rmse_norm_steps += np.asarray(metrics["rmse_normalized_step_sum"])
        if group_tracker and metrics.get("group_nrmse_step_sum") is not None:
            group_tracker["nrmse"] += np.asarray(metrics["group_nrmse_step_sum"])
            group_tracker["rmse"] += np.asarray(metrics["group_rmse_step_sum"])
            group_tracker["rmse_norm"] += np.asarray(metrics["group_rmse_normalized_step_sum"])

        if one_small_batch:
            break

    divisor_longer = total_samples * valid_rollout
    rollout_loss = rollout_nrmse_steps.sum() / divisor_longer
    rollout_rmse = rollout_rmse_steps.sum() / divisor_longer
    rollout_rmse_normalized = rollout_rmse_norm_steps.sum() / divisor_longer

    rollout_known_loss = None
    rollout_known_rmse = None
    rollout_known_rmse_norm = None
    if valid_rollout > nb_time_steps:
        divisor_seen = total_samples * nb_time_steps
        rollout_known_loss = rollout_nrmse_steps[:nb_time_steps].sum() / divisor_seen
        rollout_known_rmse = rollout_rmse_steps[:nb_time_steps].sum() / divisor_seen
        rollout_known_rmse_norm = rollout_rmse_norm_steps[:nb_time_steps].sum() / divisor_seen

    tb_logger.add_scalar(
        f"Loss_valid/rollout_longer_n{valid_rollout}_nRMSE", rollout_loss, cur_model_id
    )
    tb_logger.add_scalar("Loss_valid/rollout_longer_rmse", rollout_rmse, cur_model_id)
    tb_logger.add_scalar(
        "Loss_valid/rollout_longer_rmse_normalized",
        rollout_rmse_normalized,
        cur_model_id,
    )

    if rollout_known_loss is not None:
        tb_logger.add_scalar(
            f"Loss_valid/rollout_seen_n{nb_time_steps}_nRMSE",
            rollout_known_loss,
            cur_model_id,
        )
    if rollout_known_rmse is not None:
        tb_logger.add_scalar("Loss_valid/rollout_seen_rmse", rollout_known_rmse, cur_model_id)
    if rollout_known_rmse_norm is not None:
        tb_logger.add_scalar(
            "Loss_valid/rollout_seen_rmse_normalized",
            rollout_known_rmse_norm,
            cur_model_id,
        )

    group_results = _log_group_rollout_results(
        group_tracker, tb_logger, cur_model_id, valid_rollout, nb_time_steps
    )

    rollout_losses_mean = None
    if plot_rollout_loss and rollout_losses is not None and total_samples:
        rollout_losses_mean = rollout_losses / float(total_samples)

    stats = {}
    for prefix, values in (
        ("rollout_longer_nrmse", mean_nrmse_per_traj),
        ("rollout_longer_rmse", mean_rmse_per_traj),
        ("rollout_edges_nrmse", edge_nrmse_per_traj),
        ("rollout_edges_rmse", edge_rmse_per_traj),
    ):
        summary = _summary_stats(values)
        for key, val in summary.items():
            stats[f"{prefix}_{key}"] = val

    return {
        "rollout_loss": rollout_loss,
        "rollout_known_loss": rollout_known_loss,
        "rollout_rmse": rollout_rmse,
        "rollout_rmse_known": rollout_known_rmse,
        "rollout_rmse_normalized": rollout_rmse_normalized,
        "rollout_rmse_normalized_known": rollout_known_rmse_norm,
        "rollout_losses_mean": rollout_losses_mean,
        **stats,
        "group_rollout": group_results,
    }


def run_torch_rollout(
    *,
    model,
    valid_dataset,
    valid_dataloader_rollout,
    valid_rollout: int,
    nb_time_steps: int,
    mesh_axes,
    valid_batch_size: int,
    valid_num_samples: int,
    tb_logger,
    cur_model_id,
    get_physics: Callable,
    grid_tensor,
    one_small_batch=False,
    memory_efficient=True,
    plot_rollout_loss=False,
    group_config=None,
    label_nb_time_steps=None,
):
    batch_size, total_samples, _ = prepare_rollout_config(
        valid_dataset,
        valid_batch_size,
        valid_num_samples,
        one_small_batch,
        memory_efficient,
    )

    if valid_rollout <= 0 or total_samples in (0, None):
        logger.warning(
            "Skipping rollout metrics logging because valid_rollout is not positive or total_samples is zero."
        )
        return _empty_rollout_result()

    if not memory_efficient:
        logger.warning("Torch validator currently forces memory-efficient rollout evaluation.")

    rollout_nrmse_steps = np.zeros(valid_rollout, dtype=np.float64)
    rollout_rmse_steps = np.zeros_like(rollout_nrmse_steps)
    rollout_rmse_norm_steps = np.zeros_like(rollout_nrmse_steps)
    rollout_losses = np.zeros(valid_rollout, dtype=np.float64) if plot_rollout_loss else None
    group_tracker = _init_group_tracker(group_config, valid_rollout)
    per_traj_mean_nrmse = []
    per_traj_mean_rmse = []
    per_traj_edge_nrmse = []
    per_traj_edge_rmse = []
    per_traj_nrmse_steps = []
    trajectory_ids = []
    per_traj_metric_steps = {
        "mse": [],
        "mse_normalized": [],
        "rmse": [],
        "rmse_normalized": [],
    }

    label_nb_time_steps = label_nb_time_steps or nb_time_steps

    for ic_idx, rollout_idx in valid_dataloader_rollout:
        active_ic_idx = ic_idx
        if one_small_batch:
            active_ic_idx = ic_idx[:batch_size]
            rollout_idx = [idx[: len(active_ic_idx)] for idx in rollout_idx]

        group_labels = _group_labels_from_ic_idx(active_ic_idx, label_nb_time_steps, group_tracker)
        _update_group_counts(group_tracker, group_labels)

        u_step = np.asarray(valid_dataset[active_ic_idx], dtype=np.float32)
        metrics = torch_rollout_memory_efficient(
            model=model,
            u_step=u_step,
            ic_idx=active_ic_idx,
            rollout_idx=rollout_idx,
            rollout_losses=rollout_losses,
            plot_rollout_loss=plot_rollout_loss,
            grid_tensor=grid_tensor,
            get_physics=get_physics,
            mesh_axes=mesh_axes,
            valid_dataset=valid_dataset,
            group_assignments=group_labels,
            num_groups=len(group_tracker["names"]) if group_tracker else 0,
        )

        rollout_nrmse_steps += np.asarray(metrics["nrmse_step_sum"])
        rollout_rmse_steps += np.asarray(metrics["rmse_step_sum"])
        rollout_rmse_norm_steps += np.asarray(metrics["rmse_normalized_step_sum"])
        batch_mean_nrmse = metrics.get("per_sample_mean_nrmse")
        if batch_mean_nrmse is not None and np.size(batch_mean_nrmse):
            per_traj_mean_nrmse.append(np.asarray(batch_mean_nrmse))
        batch_nrmse_steps = metrics.get("per_sample_nrmse_steps")
        if batch_nrmse_steps is not None and np.size(batch_nrmse_steps):
            per_traj_nrmse_steps.append(np.asarray(batch_nrmse_steps))
        for metric_name in per_traj_metric_steps:
            values = metrics.get(f"per_sample_{metric_name}_steps")
            if values is not None and np.size(values):
                per_traj_metric_steps[metric_name].append(np.asarray(values))
        trajectory_ids.append(
            np.asarray(active_ic_idx, dtype=np.int64) // int(label_nb_time_steps)
        )
        batch_mean_rmse = metrics.get("per_sample_mean_rmse")
        if batch_mean_rmse is not None and np.size(batch_mean_rmse):
            per_traj_mean_rmse.append(np.asarray(batch_mean_rmse))
        batch_edge_nrmse = metrics.get("per_sample_edge_nrmse")
        if batch_edge_nrmse is not None and np.size(batch_edge_nrmse):
            per_traj_edge_nrmse.append(np.asarray(batch_edge_nrmse))
        batch_edge_rmse = metrics.get("per_sample_edge_rmse")
        if batch_edge_rmse is not None and np.size(batch_edge_rmse):
            per_traj_edge_rmse.append(np.asarray(batch_edge_rmse))
        if group_tracker and metrics.get("group_nrmse_step_sum") is not None:
            group_tracker["nrmse"] += np.asarray(metrics["group_nrmse_step_sum"])
            group_tracker["rmse"] += np.asarray(metrics["group_rmse_step_sum"])
            group_tracker["rmse_norm"] += np.asarray(metrics["group_rmse_normalized_step_sum"])

        if one_small_batch:
            break

    divisor_longer = total_samples * valid_rollout
    rollout_loss = rollout_nrmse_steps.sum() / divisor_longer
    rollout_rmse = rollout_rmse_steps.sum() / divisor_longer
    rollout_rmse_normalized = rollout_rmse_norm_steps.sum() / divisor_longer

    rollout_known_loss = None
    rollout_known_rmse = None
    rollout_known_rmse_norm = None
    if valid_rollout > nb_time_steps:
        divisor_seen = total_samples * nb_time_steps
        rollout_known_loss = rollout_nrmse_steps[:nb_time_steps].sum() / divisor_seen
        rollout_known_rmse = rollout_rmse_steps[:nb_time_steps].sum() / divisor_seen
        rollout_known_rmse_norm = rollout_rmse_norm_steps[:nb_time_steps].sum() / divisor_seen

    mean_nrmse_per_traj = (
        np.concatenate(per_traj_mean_nrmse) if per_traj_mean_nrmse else np.array([])
    )
    mean_rmse_per_traj = np.concatenate(per_traj_mean_rmse) if per_traj_mean_rmse else np.array([])
    edge_nrmse_per_traj = (
        np.concatenate(per_traj_edge_nrmse) if per_traj_edge_nrmse else np.array([])
    )
    edge_rmse_per_traj = np.concatenate(per_traj_edge_rmse) if per_traj_edge_rmse else np.array([])
    nrmse_by_step = (
        np.concatenate(per_traj_nrmse_steps, axis=0)
        if per_traj_nrmse_steps
        else np.empty((0, valid_rollout))
    )
    rollout_trajectory_ids = (
        np.concatenate(trajectory_ids)
        if trajectory_ids
        else np.array([], dtype=np.int64)
    )
    rollout_metric_steps = {
        metric_name: (
            np.concatenate(values, axis=0)
            if values
            else np.empty((0, valid_rollout), dtype=np.float64)
        )
        for metric_name, values in per_traj_metric_steps.items()
    }

    tb_logger.add_scalar(
        f"Loss_valid/rollout_longer_n{valid_rollout}_nRMSE", rollout_loss, cur_model_id
    )
    tb_logger.add_scalar("Loss_valid/rollout_longer_rmse", rollout_rmse, cur_model_id)
    tb_logger.add_scalar(
        "Loss_valid/rollout_longer_rmse_normalized",
        rollout_rmse_normalized,
        cur_model_id,
    )

    if rollout_known_loss is not None:
        tb_logger.add_scalar(
            f"Loss_valid/rollout_seen_n{nb_time_steps}_nRMSE",
            rollout_known_loss,
            cur_model_id,
        )
    if rollout_known_rmse is not None:
        tb_logger.add_scalar("Loss_valid/rollout_seen_rmse", rollout_known_rmse, cur_model_id)
    if rollout_known_rmse_norm is not None:
        tb_logger.add_scalar(
            "Loss_valid/rollout_seen_rmse_normalized",
            rollout_known_rmse_norm,
            cur_model_id,
        )

    def _log_per_traj_stats(tag_prefix: str, values: np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        tb_logger.add_scalar(f"{tag_prefix}_min", float(np.nanmin(arr)), cur_model_id)
        tb_logger.add_scalar(f"{tag_prefix}_max", float(np.nanmax(arr)), cur_model_id)
        tb_logger.add_scalar(f"{tag_prefix}_std", float(np.nanstd(arr)), cur_model_id)

    _log_per_traj_stats("Loss_valid/rollout_longer_nRMSE", mean_nrmse_per_traj)
    _log_per_traj_stats("Loss_valid/rollout_longer_rmse", mean_rmse_per_traj)
    _log_per_traj_stats("Loss_valid/rollout_edges_nRMSE", edge_nrmse_per_traj)
    _log_per_traj_stats("Loss_valid/rollout_edges_rmse", edge_rmse_per_traj)

    group_results = _log_group_rollout_results(
        group_tracker, tb_logger, cur_model_id, valid_rollout, nb_time_steps
    )

    rollout_losses_mean = None
    if plot_rollout_loss and rollout_losses is not None and total_samples:
        rollout_losses_mean = rollout_losses / float(total_samples)

    return {
        "rollout_loss": rollout_loss,
        "rollout_known_loss": rollout_known_loss,
        "rollout_rmse": rollout_rmse,
        "rollout_rmse_known": rollout_known_rmse,
        "rollout_rmse_normalized": rollout_rmse_normalized,
        "rollout_rmse_normalized_known": rollout_known_rmse_norm,
        "rollout_losses_mean": rollout_losses_mean,
        "rollout_trajectory_ids": rollout_trajectory_ids,
        "rollout_horizons": np.arange(1, nrmse_by_step.shape[1] + 1, dtype=np.int64),
        "rollout_horizon_errors": nrmse_by_step,
        "rollout_metric_steps": rollout_metric_steps,
        "group_rollout": group_results,
    }


def jax_rollout_memory_efficient(
    *,
    model,
    u_step,
    rollout_idx,
    batch_size,
    one_small_batch,
    valid_dataset,
    rollout_losses=None,
    plot_rollout_loss=False,
    group_assignments=None,
    num_groups=0,
):
    """Stream rollout evaluation so we never hold the full tensor in memory."""
    batched_model = jax.jit(jax.vmap(model), donate_argnums=0)
    reduce_axes = tuple(range(1, u_step.ndim))
    input_energy = jnp.mean(u_step**2, axis=reduce_axes)
    num_steps = len(rollout_idx)
    nrmse_step_sum = np.zeros(num_steps, dtype=np.float64)
    rmse_step_sum = np.zeros_like(nrmse_step_sum)
    rmse_norm_step_sum = np.zeros_like(nrmse_step_sum)

    active_batch = u_step.shape[0]
    group_step_nrmse = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_step_rmse = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_step_rmse_norm = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_indices = None
    if group_assignments is not None:
        labels = np.asarray(group_assignments, dtype=np.int64)
        group_indices = [np.where(labels == gid)[0] for gid in range(num_groups)]

    for step, roll_idx in enumerate(rollout_idx):
        current_idx = roll_idx
        if one_small_batch or batch_size is not None:
            current_idx = roll_idx[:active_batch]

        u_step = batched_model(u_step)
        u_next = jnp.asarray(valid_dataset[current_idx])

        diff = u_step - u_next
        rmse_batch = jnp.sqrt(jnp.mean(diff**2, axis=reduce_axes))

        den = jnp.where(input_energy > 0, input_energy, 1.0)
        rmse_norm_batch = rmse_batch / jnp.sqrt(den)
        nrmse_batch = rmse_norm_batch

        nrmse_step_sum[step] = float(jnp.sum(nrmse_batch))
        rmse_step_sum[step] = float(jnp.sum(rmse_batch))
        rmse_norm_step_sum[step] = float(jnp.sum(rmse_norm_batch))

        if plot_rollout_loss and rollout_losses is not None:
            rollout_losses[step] += float(jnp.sum(nrmse_batch))

        if group_indices is not None:
            nrmse_np = np.asarray(nrmse_batch)
            rmse_np = np.asarray(rmse_batch)
            rmse_norm_np = np.asarray(rmse_norm_batch)
            for gid, idxs in enumerate(group_indices):
                if idxs.size == 0:
                    continue
                group_step_nrmse[gid, step] = float(np.sum(nrmse_np[idxs]))
                group_step_rmse[gid, step] = float(np.sum(rmse_np[idxs]))
                group_step_rmse_norm[gid, step] = float(np.sum(rmse_norm_np[idxs]))

        del u_next

    return {
        "nrmse_step_sum": nrmse_step_sum,
        "rmse_step_sum": rmse_step_sum,
        "rmse_normalized_step_sum": rmse_norm_step_sum,
        "group_nrmse_step_sum": group_step_nrmse,
        "group_rmse_step_sum": group_step_rmse,
        "group_rmse_normalized_step_sum": group_step_rmse_norm,
    }


def jax_rollout_full_trajectory(
    *,
    model,
    u_step,
    rollout_idx,
    batch_size,
    one_small_batch,
    valid_dataset,
    valid_rollout,
):
    trj_idx = np.vstack(rollout_idx).T
    if one_small_batch:
        trj_idx = trj_idx[:batch_size]

    flat_trj_idx = trj_idx.reshape(-1)
    flat_ref_batch = jnp.asarray(valid_dataset[flat_trj_idx])
    trj_batch = flat_ref_batch.reshape(trj_idx.shape + flat_ref_batch.shape[1:])
    logger.info(f"Ref batch shape: {trj_batch.shape}, u_step shape: {u_step.shape}")

    rollout_pred = jax.vmap(ex.rollout(model, valid_rollout))(u_step)
    logger.info(f"Rollout shape: {rollout_pred.shape}")

    reduce_axes = tuple(range(2, rollout_pred.ndim))
    diff = rollout_pred - trj_batch
    rmse_batch = jnp.sqrt(jnp.mean(diff**2, axis=reduce_axes))
    input_energy = jnp.mean(u_step**2, axis=tuple(range(1, u_step.ndim)))
    input_energy = jnp.where(input_energy > 0, input_energy, 1.0)
    rmse_norm_batch = rmse_batch / jnp.sqrt(input_energy[:, None])
    nrmse_batch = rmse_norm_batch

    return {
        "nrmse_step_sum": jnp.sum(nrmse_batch, axis=0),
        "rmse_step_sum": jnp.sum(rmse_batch, axis=0),
        "rmse_normalized_step_sum": jnp.sum(rmse_norm_batch, axis=0),
    }


def _compute_edge_error(first_step_vals, last_step_vals):
    if first_step_vals is None or last_step_vals is None:
        return np.array([], dtype=np.float64)
    first = np.asarray(first_step_vals, dtype=np.float64)
    last = np.asarray(last_step_vals, dtype=np.float64)
    length = min(first.shape[0], last.shape[0])
    if length == 0:
        return np.array([], dtype=np.float64)
    return 0.5 * (first[:length] + last[:length])


def torch_rollout_memory_efficient(
    *,
    model,
    u_step,
    ic_idx,
    rollout_idx,
    rollout_losses,
    plot_rollout_loss,
    grid_tensor,
    get_physics: Callable,
    mesh_axes,
    valid_dataset,
    group_assignments=None,
    num_groups=0,
):
    current = torch.as_tensor(u_step, dtype=torch.float32, device=grid_tensor.device)
    active_batch = current.shape[0]
    physics = get_physics(ic_idx)
    if physics is not None and physics.shape[0] != active_batch:
        physics = physics[:active_batch]

    num_steps = len(rollout_idx)
    nrmse_step_sum = np.zeros(num_steps, dtype=np.float64)
    rmse_step_sum = np.zeros_like(nrmse_step_sum)
    rmse_norm_step_sum = np.zeros_like(nrmse_step_sum)
    per_sample_nrmse_sum = np.zeros(active_batch, dtype=np.float64)
    per_sample_rmse_sum = np.zeros(active_batch, dtype=np.float64)
    per_sample_mse_steps = np.zeros((active_batch, num_steps), dtype=np.float64)
    per_sample_mse_normalized_steps = np.zeros((active_batch, num_steps), dtype=np.float64)
    per_sample_rmse_steps = np.zeros((active_batch, num_steps), dtype=np.float64)
    per_sample_rmse_normalized_steps = np.zeros((active_batch, num_steps), dtype=np.float64)
    per_sample_nrmse_steps = np.zeros((active_batch, num_steps), dtype=np.float64)
    first_step_nrmse = None
    first_step_rmse = None
    last_step_nrmse = None
    last_step_rmse = None
    group_step_nrmse = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_step_rmse = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_step_rmse_norm = (
        np.zeros((num_groups, num_steps), dtype=np.float64)
        if group_assignments is not None
        else None
    )
    group_indices = None
    if group_assignments is not None:
        labels = np.asarray(group_assignments, dtype=np.int64)
        group_indices = [np.where(labels == gid)[0] for gid in range(num_groups)]

    for step, roll_idx in enumerate(rollout_idx):
        grid = grid_tensor.expand(current.shape[0], *grid_tensor.shape[1:])
        with torch.no_grad():
            preds = model(current, grid, physics if physics is not None else None)
        current = preds.detach()

        u_next = np.asarray(valid_dataset[roll_idx], dtype=np.float32)
        prediction = preds.detach().cpu().numpy()
        (mse_batch, rmse_batch, rmse_norm_batch, nrmse_batch) = compute_pointwise_metrics(
            jnp.asarray(prediction),
            jnp.asarray(u_next),
            mesh_axes,
            normalizer=jnp.asarray(u_step),
        )

        mse_np = np.asarray(mse_batch)
        nrmse_np = np.asarray(nrmse_batch)
        rmse_np = np.asarray(rmse_batch)
        rmse_norm_np = np.asarray(rmse_norm_batch)
        input_energy = np.maximum(
            np.mean(u_step**2, axis=mesh_axes), 1e-6
        )
        mse_normalized_np = mse_np / input_energy

        per_sample_nrmse_sum[: nrmse_np.shape[0]] += nrmse_np
        per_sample_rmse_sum[: rmse_np.shape[0]] += rmse_np
        per_sample_mse_steps[: mse_np.shape[0], step] = mse_np
        per_sample_mse_normalized_steps[: mse_normalized_np.shape[0], step] = mse_normalized_np
        per_sample_rmse_steps[: rmse_np.shape[0], step] = rmse_np
        per_sample_rmse_normalized_steps[: rmse_norm_np.shape[0], step] = rmse_norm_np
        per_sample_nrmse_steps[: nrmse_np.shape[0], step] = nrmse_np
        if first_step_nrmse is None:
            first_step_nrmse = nrmse_np.copy()
            first_step_rmse = rmse_np.copy()
        last_step_nrmse = nrmse_np.copy()
        last_step_rmse = rmse_np.copy()

        nrmse_step_sum[step] = float(np.sum(nrmse_np))
        rmse_step_sum[step] = float(np.sum(rmse_np))
        rmse_norm_step_sum[step] = float(np.sum(rmse_norm_np))

        if plot_rollout_loss and rollout_losses is not None:
            rollout_losses[step] += float(np.sum(nrmse_np))

        if group_indices is not None:
            for gid, idxs in enumerate(group_indices):
                if idxs.size == 0:
                    continue
                group_step_nrmse[gid, step] = float(np.sum(nrmse_np[idxs]))
                group_step_rmse[gid, step] = float(np.sum(rmse_np[idxs]))
                group_step_rmse_norm[gid, step] = float(np.sum(rmse_norm_np[idxs]))

    return {
        "nrmse_step_sum": nrmse_step_sum,
        "rmse_step_sum": rmse_step_sum,
        "rmse_normalized_step_sum": rmse_norm_step_sum,
        "group_nrmse_step_sum": group_step_nrmse,
        "group_rmse_step_sum": group_step_rmse,
        "group_rmse_normalized_step_sum": group_step_rmse_norm,
        "per_sample_mean_nrmse": per_sample_nrmse_sum / max(num_steps, 1),
        "per_sample_nrmse_steps": per_sample_nrmse_steps,
        "per_sample_mean_rmse": per_sample_rmse_sum / max(num_steps, 1),
        "per_sample_mse_steps": per_sample_mse_steps,
        "per_sample_mse_normalized_steps": per_sample_mse_normalized_steps,
        "per_sample_rmse_steps": per_sample_rmse_steps,
        "per_sample_rmse_normalized_steps": per_sample_rmse_normalized_steps,
        "per_sample_edge_nrmse": _compute_edge_error(first_step_nrmse, last_step_nrmse),
        "per_sample_edge_rmse": _compute_edge_error(first_step_rmse, last_step_rmse),
    }
