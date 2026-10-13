import logging
import os

from melissa.server.deep_learning.metric_logger import make_metric_logger, LoggerType


class TBCompatLogger:
    """Thin wrapper so the rest of the code can talk to TB and W&B identically."""

    def __init__(self, inner):
        self._inner = inner

    def add_scalar(self, tag, scalar_value, step):
        return self._inner.log_scalar(tag, scalar_value, step)

    def add_figure(self, tag, figure, step, close=True):
        return self._inner.log_figure(tag, figure, step, close=close)

    def add_custom_scalars(self, layout):
        if hasattr(self._inner, "writer") and self._inner.writer is not None:
            try:
                return self._inner.writer.add_custom_scalars(layout)
            except Exception:
                # Gracefully ignore if backend does not expose the hook (e.g. W&B).
                pass
        return None


def create_validation_logger(
    config_dict: dict,
    tblog_dir: str,
    wandb_prefix: str = "validation",
    wandb_name: str | None = None,
    wandb_job_type: str | None = None,
    log_label: str = "external validator",
) -> TBCompatLogger:
    """Instantiate the TensorBoard or W&B logger based on dl_config flags."""
    del wandb_name, wandb_job_type
    dl_config = config_dict["dl_config"]
    use_wandb = dl_config.get("wandb", False)
    logger_type = LoggerType.WANDB if use_wandb else LoggerType.TENSORBOARD
    wandb_project = dl_config.get("wandb_project")
    wandb_group = dl_config.get("wandb_group")

    only_dir = config_dict.get("output_dir") or os.path.dirname(tblog_dir)
    logdir = only_dir if use_wandb else tblog_dir

    base_logger = make_metric_logger(
        framework_t=None,  # treated as DEFAULT by factory
        rank=-1,
        logdir=logdir,
        disable=False,
        debug=False,
        logger_type=logger_type,
        wandb_project=wandb_project,
        wandb_mode=dl_config.get("wandb_mode"),
        wandb_prefix=wandb_prefix,
        config=config_dict,
        wandb_group=wandb_group,
    )

    logging.info(f"Using {'W&B' if use_wandb else 'TensorBoard'} logger for {log_label}")
    return TBCompatLogger(base_logger)
