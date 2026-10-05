import numpy as np
import jax.numpy as jnp


def compute_pointwise_metrics(prediction, target, reduce_axes, normalizer=None):
    diff = prediction - target
    mse = jnp.mean(diff**2, axis=reduce_axes)
    rmse = jnp.sqrt(mse)

    reference = target if normalizer is None else normalizer
    den = jnp.mean(reference**2, axis=reduce_axes)
    den = jnp.where(den > 0, den, 1.0)
    rmse_normalized = rmse / jnp.sqrt(den)
    nrmse = rmse_normalized
    return mse, rmse, rmse_normalized, nrmse


class MetricsTracker:
    """Accumulates per-batch validation metrics and logs consolidated statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses_per_sample: list[np.ndarray] = []
        self.rmse_per_sample: list[np.ndarray] = []
        self.rmse_normalized_per_sample: list[np.ndarray] = []
        self.mse_normalized_per_sample: list[np.ndarray] = []
        self.nrmse_per_sample: list[np.ndarray] = []
        self.batch_val_losses: list[float] = []
        self.sample_ids: list[np.ndarray] = []

    def record(
        self,
        loss_per_sample,
        rmse_per_sample,
        rmse_normalized_per_sample,
        nrmse_per_sample,
        batch_loss,
        sample_ids,
        mse_normalized_per_sample=None,
    ):
        self.batch_val_losses.append(float(batch_loss))
        self.losses_per_sample.append(np.asarray(loss_per_sample).ravel())
        self.rmse_per_sample.append(np.asarray(rmse_per_sample).ravel())
        self.rmse_normalized_per_sample.append(np.asarray(rmse_normalized_per_sample).ravel())
        self.mse_normalized_per_sample.append(
            np.asarray(
                mse_normalized_per_sample
                if mse_normalized_per_sample is not None
                else np.asarray(rmse_normalized_per_sample) ** 2
            ).ravel()
        )
        self.nrmse_per_sample.append(np.asarray(nrmse_per_sample).ravel())
        self.sample_ids.append(np.asarray(sample_ids, dtype=np.int64).ravel())

    def _finalize_arrays(self):
        self.losses_per_sample = (
            np.concatenate(self.losses_per_sample, axis=0)
            if self.losses_per_sample
            else np.array([])
        )
        self.rmse_per_sample = (
            np.concatenate(self.rmse_per_sample, axis=0) if self.rmse_per_sample else np.array([])
        )
        self.rmse_normalized_per_sample = (
            np.concatenate(self.rmse_normalized_per_sample, axis=0)
            if self.rmse_normalized_per_sample
            else np.array([])
        )
        self.mse_normalized_per_sample = (
            np.concatenate(self.mse_normalized_per_sample, axis=0)
            if self.mse_normalized_per_sample
            else np.array([])
        )
        self.nrmse_per_sample = (
            np.concatenate(self.nrmse_per_sample, axis=0) if self.nrmse_per_sample else np.array([])
        )
        self.sample_ids = (
            np.concatenate(self.sample_ids, axis=0)
            if self.sample_ids
            else np.array([], dtype=np.int64)
        )

    def _build_group_entries(self, group_specs):
        entries: list[tuple[str, np.ndarray | None, str | None]] = [("", None, None)]
        if group_specs:
            for name, lower, upper in group_specs:
                if self.sample_ids.size == 0:
                    continue
                mask = (self.sample_ids >= lower) & (self.sample_ids < upper)
                if not mask.any():
                    continue
                entries.append((f"_{name}", mask, name))
        return entries

    def _log_scalar_means(
        self,
        tb_logger,
        cur_model_id,
        base_tag: str,
        values: np.ndarray,
        group_entries,
        include_overall: bool = True,
    ) -> dict[str | None, float]:
        payload = {}
        arr = np.asarray(values).ravel()
        if arr.size == 0:
            return payload
        for suffix, mask, group_name in group_entries:
            if mask is None and not include_overall:
                continue
            subset = arr if mask is None else arr[mask]
            if subset.size == 0:
                continue
            mean_val = float(np.nanmean(subset))
            tag = (
                base_tag
                if group_name is None
                else base_tag.replace("Loss_valid", f"Loss_valid_{group_name}", 1)
            )
            tb_logger.add_scalar(tag, mean_val, cur_model_id)
            payload[group_name] = mean_val
        return payload

    def log_distributions(self, tb_logger, cur_model_id, group_specs=None):
        """Push percentile and mean statistics for each metric to the logger."""
        self._finalize_arrays()
        group_entries = self._build_group_entries(group_specs)

        metrics = [
            ("Loss_valid_stats", self.losses_per_sample),
            ("Loss_valid_rmse_stats", self.rmse_per_sample),
            ("Loss_valid_rmse_normalized_stats", self.rmse_normalized_per_sample),
            ("Loss_valid_mse_normalized_stats", self.mse_normalized_per_sample),
        ]

        mean_registry: dict[str, dict[str | None, float]] = {}
        for prefix, values in metrics:
            if values is None:
                continue
            values = np.asarray(values)
            if values.size == 0:
                continue
            for suffix, mask, group_name in group_entries:
                subset = values if mask is None else values[mask]
                if subset.size == 0:
                    continue
                mean_tmp = np.nanmean(subset)
                std_tmp = np.nanstd(subset)
                stats = {
                    "max": np.nanmax(subset),
                    "min": np.nanmin(subset),
                    "std": std_tmp,
                    "mean_pstd": mean_tmp + std_tmp,
                    "mean_mean": mean_tmp,
                    "mean_mstd": max(max(mean_tmp - std_tmp, 0), max(mean_tmp - 0.5 * std_tmp, 0)),
                    "p90": np.nanpercentile(subset, 90),
                    "p99": np.nanpercentile(subset, 99),
                    "p95": np.nanpercentile(subset, 95),
                    "p75": np.nanpercentile(subset, 75),
                    "p50": np.nanpercentile(subset, 50),
                    "p25": np.nanpercentile(subset, 25),
                    "p10": np.nanpercentile(subset, 10),
                }
                if group_name is None:
                    tag_prefix = prefix
                else:
                    tag_prefix = prefix.replace("Loss_valid_", f"Loss_valid_{group_name}_", 1)
                for key, val in stats.items():
                    tb_logger.add_scalar(f"{tag_prefix}/{key}", val, cur_model_id)
                key = group_name if group_name is not None else "overall"
                mean_registry.setdefault(prefix, {})[key] = mean_tmp
                if prefix == "Loss_valid_stats" and group_name is not None:
                    tb_logger.add_scalar(f"Loss_valid_{group_name}/mean", mean_tmp, cur_model_id)

        self._log_scalar_means(
            tb_logger,
            cur_model_id,
            "Loss_valid/mean",
            self.losses_per_sample,
            group_entries,
            include_overall=False,
        )
        self._log_scalar_means(
            tb_logger,
            cur_model_id,
            "Loss_valid/mean_rmse",
            self.rmse_per_sample,
            group_entries,
        )
        self._log_scalar_means(
            tb_logger,
            cur_model_id,
            "Loss_valid/mean_rmse_normalized",
            self.rmse_normalized_per_sample,
            group_entries,
        )
        self._log_scalar_means(
            tb_logger,
            cur_model_id,
            "Loss_valid/mean_nrmse_normalized",
            self.nrmse_per_sample,
            group_entries,
        )
        self._log_scalar_means(
            tb_logger,
            cur_model_id,
            "Loss_valid/mean_mse_normalized",
            self.mse_normalized_per_sample,
            group_entries,
        )

        return mean_registry

    @property
    def mean_batch_loss(self):
        if not self.batch_val_losses:
            return np.inf
        return np.mean(self.batch_val_losses)
