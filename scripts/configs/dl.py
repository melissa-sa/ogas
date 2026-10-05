from dataclasses import dataclass
import pandas as pd
import copy
import json
from .utils import coerce_optional, parse_bool


@dataclass
class DLConfig:
    def __init__(self, defaults: dict, row: pd.Series):
        conf = copy.deepcopy(defaults)
        # Optional Torch defaults (ignored unless explicitly provided)
        torch_model_default = conf.get("torch_model")
        torch_opt_default = conf.get("torch_optimizer")
        torch_sched_default = conf.get("torch_scheduler")
        torch_device_default = conf.get("torch_device")
        torch_grad_clip_default = conf.get("torch_max_grad_norm")
        rate_limit_default = conf.get("rate_limit_speed_batch_per_sim")

        # CSV Overrides
        ts = row.get("ts")
        if ts is not None and not pd.isna(ts):
            conf["nb_time_steps"] = int(ts)

        th = coerce_optional(row.get("temporal_horizon"))
        if th is not None:
            conf["temporal_horizon"] = int(float(th))

        detail_override = parse_bool(row.get("use_detail_val"))
        if detail_override is not None:
            conf["use_detail_val"] = detail_override

        physics_norm_override = parse_bool(row.get("normalize_physics"))
        if physics_norm_override is not None:
            conf["normalize_physics"] = physics_norm_override

        train_bf16_override = parse_bool(row.get("train_bf_16"))
        if train_bf16_override is not None:
            conf["train_bf_16"] = train_bf16_override

        horizon_override = parse_bool(row.get("train_random_t_plus_1_or_2"))
        if horizon_override is not None:
            conf["train_random_t_plus_1_or_2"] = horizon_override

        ada_grad_override = parse_bool(row.get("use_ada_grad"))
        if ada_grad_override is not None:
            conf["use_ada_grad"] = ada_grad_override

        # Computed properties
        self.batch_size = conf["batch_size"]
        self.nb_time_steps = conf["nb_time_steps"]
        self.valid_nb_time_steps = conf["temporal_horizon"] + 1
        self.valid_rollout = conf["temporal_horizon"]
        self.valid_batch_size = conf.get("valid_batch_size", 4 * self.batch_size)
        self.valid_num_samples = conf["valid_num_samples"]
        self.nb_batches_update = conf.get("nb_batches_update", 250)
        self.use_detail_val = bool(conf.get("use_detail_val", False))
        self.normalize_physics = bool(conf.get("normalize_physics", False))
        self.train_bf_16 = bool(conf.get("train_bf_16", False))
        self.train_random_t_plus_1_or_2 = bool(conf.get("train_random_t_plus_1_or_2", False))
        self.use_ada_grad = bool(conf.get("use_ada_grad", False))

        # Torch-specific overrides (from CSV if present)
        def _parse_json_maybe(val):
            val = coerce_optional(val)
            if val is None:
                return None
            if isinstance(val, (dict, list)):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    # allow simple tokens like "cuda" for device strings
                    return val
            return val

        self.torch_model = _parse_json_maybe(row.get("torch_model")) or torch_model_default
        self.torch_optimizer = _parse_json_maybe(row.get("torch_optimizer")) or torch_opt_default
        self.torch_scheduler = _parse_json_maybe(row.get("torch_scheduler")) or torch_sched_default
        self.torch_device = coerce_optional(row.get("torch_device")) or torch_device_default
        grad_clip = coerce_optional(row.get("torch_max_grad_norm"))
        self.torch_max_grad_norm = (
            float(grad_clip) if grad_clip is not None else torch_grad_clip_default
        )
        rate_limit = coerce_optional(row.get("rate_limit_speed_batch_per_sim"))
        self.rate_limit_speed_batch_per_sim = (
            float(rate_limit) if rate_limit is not None else rate_limit_default
        )

        # Strings for JSON
        activation = conf.get("activation", "relu")
        lr_start = conf["lr_start"]
        lr_peak = conf.get("lr_peak", lr_start)
        lr_interval = conf.get("lr_interval", 0.99)

        self.network_config = (
            f"{conf['model_name']};{conf['num_channels']};{conf['num_blocks']};{activation}"
        )
        self.optim_config = f"adam;warmup_cosine;{lr_start};{lr_peak};{lr_interval}"

        # EMA settings (pass-through to final config)
        self.use_ema_for_validation = conf.get("use_ema_for_validation", False)
        self.ema_decay = conf.get("ema_decay", 0.999)
        self.ema_start_step = conf.get("ema_start_step", 0)
        self.ema_update_every = conf.get("ema_update_every", 1)
