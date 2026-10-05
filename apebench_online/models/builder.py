from typing import Any, Dict

import torch

from apebench_online.models.torch_wrappers import (
    ConditionalFNO1D,
    ConditionalFNO2D,
    ConditionalScOT2D,
    ConditionalUNet1D,
    ConditionalUNet2D,
    ConditionalSineNet2D,
    optimize_for_training,
    resolve_padding_mode,
)
from apebench_online.models.ensemble import EnsembleModel, local_member_ids, member_seed
from apebench_online.vendor.scot import MODEL_MAP, ScOTConfig


def _build_single_model(model_cfg: Dict[str, Any], scenario, physics_dim: int):
    """Build a single model instance based on configuration."""
    model_type = model_cfg.get("type", "").lower()
    if not model_type:
        raise ValueError("`torch_model.type` must be provided in the configuration.")
    history = model_cfg.get("history", 1)
    spatial_dim = getattr(scenario, "num_spatial_dims", 2)
    padding_mode = model_cfg.get("padding_mode", resolve_padding_mode(scenario.domain_extent))
    kwargs = dict(
        num_scalar_channels=scenario.num_channels,
        cond_dim=physics_dim,
        history=history,
    )

    if model_type == "fno":
        predict_delta = model_cfg.get("predict_delta", False)
        width = model_cfg.get("width", 64)
        if spatial_dim == 1:
            model = ConditionalFNO1D(
                **kwargs,
                modes=model_cfg.get("modes", 16),
                width=width,
                predict_delta=predict_delta,
            )
        else:
            modes = model_cfg.get("modes", 20)
            model = ConditionalFNO2D(
                **kwargs,
                modes1=model_cfg.get("modes1", modes),
                modes2=model_cfg.get("modes2", modes),
                width=width,
                predict_delta=predict_delta,
            )
    elif model_type == "unet":
        predict_delta = model_cfg.get("predict_delta", False)
        delta_scale = model_cfg.get("delta_scale", 1.0)
        common_args = dict(
            hidden_channels=model_cfg.get("hidden_channels", 64),
            activation=model_cfg.get("activation", "gelu"),
            padding_mode=padding_mode,
            ch_mults=tuple(model_cfg.get("ch_mults", (1, 2, 2, 4))),
            is_attn=tuple(model_cfg.get("is_attention", (False, False, False, False))),
            mid_attn=model_cfg.get("mid_attention", False),
            n_blocks=model_cfg.get("n_blocks", 2),
            use_scale_shift_norm=model_cfg.get("use_scale_shift_norm", False),
            use1x1=model_cfg.get("use_pointwise", False),
            predict_delta=predict_delta,
            delta_scale=delta_scale,
        )
        if spatial_dim == 1:
            model = ConditionalUNet1D(
                **kwargs,
                **common_args,
            )
        else:
            model = ConditionalUNet2D(
                **kwargs,
                **common_args,
            )
    elif model_type == "sinenet":
        if spatial_dim == 1:
            raise ValueError("SineNet is only implemented for 2D scenarios.")
        predict_delta = model_cfg.get("predict_delta", False)
        delta_scale = model_cfg.get("delta_scale", 1.0)
        mult = model_cfg.get("mult", model_cfg.get("channel_multiplier", 2))
        model = ConditionalSineNet2D(
            **kwargs,
            hidden_channels=model_cfg.get("hidden_channels", 64),
            padding_mode=padding_mode,
            activation=model_cfg.get("activation", "gelu"),
            num_layers=model_cfg.get("num_layers", 4),
            num_waves=model_cfg.get("num_waves", 2),
            num_blocks=model_cfg.get("num_blocks", 1),
            mult=mult,
            residual=model_cfg.get("residual", True),
            disentangle=model_cfg.get("disentangle", True),
            down_pool=model_cfg.get("down_pool", True),
            avg_pool=model_cfg.get("avg_pool", True),
            up_interpolation=model_cfg.get("up_interpolation", True),
            interpolation_mode=model_cfg.get("interp_mode", "bicubic"),
            use_scale_shift_norm=model_cfg.get("use_scale_shift_norm", False),
            predict_delta=predict_delta,
            delta_scale=delta_scale,
        )
    elif model_type == "scot":
        if spatial_dim != 2:
            raise ValueError("ScOT is only implemented for 2D scenarios.")

        variant = model_cfg.get("variant") or model_cfg.get("size")
        preset = MODEL_MAP.get(str(variant).upper(), {}) if variant else {}

        def _cfg(name, default):
            return model_cfg.get(name, preset.get(name, default))

        image_size = model_cfg.get("image_size")
        if image_size is None:
            res_getter = getattr(scenario, "get_model_resolution", None)
            if callable(res_getter):
                image_size = int(res_getter())
            else:
                image_size = getattr(scenario, "num_points", None)
        if image_size is None:
            raise ValueError("Could not infer image_size for the ScOT model.")

        config = ScOTConfig(
            image_size=image_size,
            patch_size=_cfg("patch_size", 4),
            num_channels=scenario.num_channels * history,
            num_out_channels=scenario.num_channels,
            cond_dim=max(physics_dim, 1),
            embed_dim=_cfg("embed_dim", 48),
            depths=_cfg("depths", [2, 2, 6, 2]),
            num_heads=_cfg("num_heads", [3, 6, 12, 24]),
            skip_connections=_cfg("skip_connections", [2, 2, 2, 0]),
            window_size=_cfg("window_size", 16),
            mlp_ratio=_cfg("mlp_ratio", 4.0),
            qkv_bias=_cfg("qkv_bias", True),
            hidden_dropout_prob=_cfg("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=_cfg("attention_probs_dropout_prob", 0.0),
            drop_path_rate=_cfg("drop_path_rate", 0.0),
            use_absolute_embeddings=_cfg("use_absolute_embeddings", False),
            use_conditioning=physics_dim > 0,
            use_mask_token=_cfg("use_mask_token", False),
            residual_model=_cfg("residual_model", "convnext"),
            learn_residual=_cfg("learn_residual", False),
            output_hidden_states=model_cfg.get("output_hidden_states", False),
        )
        model = ConditionalScOT2D(
            num_scalar_channels=scenario.num_channels,
            cond_dim=physics_dim,
            history=history,
            config=config,
            predict_delta=model_cfg.get("predict_delta", False),
            delta_scale=model_cfg.get("delta_scale", 1.0),
        )
    else:
        raise ValueError(f"Unknown torch model type '{model_type}'")
    return model


def build_torch_model(model_cfg: Dict[str, Any], scenario, physics_dim: int, group=None, seed: int = 0):
    """Build the (always ensemble) PyTorch surrogate: `n_models` members (2 by default with `use_ensemble`).

    With a torch.distributed `group`, only this rank's members are instantiated (see EnsembleModel). Each
    member is initialised from its own seed, so the ensemble is the same whatever the rank layout.
    """
    n_models = int(model_cfg.get("n_models", 2 if model_cfg.get("use_ensemble", False) else 1))
    if model_cfg.get("use_ensemble", False):
        n_models = max(n_models, 2)
    member_ids = local_member_ids(n_models, group)
    members = []
    for i in member_ids:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(member_seed(seed, i))
            member = _build_single_model(model_cfg, scenario, physics_dim)
        members.append(optimize_for_training(member) if model_cfg.get("optimize", False) else member)
    return EnsembleModel(
        members,
        member_ids=member_ids,
        n_members=n_models,
        group=group,
        prediction_mode=model_cfg.get("prediction_mode", "mean"),
        acquisition_signal=model_cfg.get("acquisition_signal", "uncertainty"),
    )
