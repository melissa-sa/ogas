from typing import List, Dict

import math
import torch

from torch.optim.lr_scheduler import LambdaLR
from apebench_online.vendor.scot.modules import ConditionalLayerNorm, LayerNorm



def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    if forbidden_layer_names is None:
        forbidden_layer_names = []
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_layer_names)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k
        for k in model._parameters.keys()
        if not any(forbidden in k.lower() for forbidden in forbidden_layer_names)
    ]
    return result


def get_conditional_norm_params(model):
    params = []
    for name, module in model.named_modules():
        if isinstance(module, ConditionalLayerNorm):
            for param_name, _ in module.named_parameters():
                params.append(f"{name}.{param_name}")
    return params


def get_decay_parameter_names(model) -> List[str]:
    ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, LayerNorm, ConditionalLayerNorm]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def get_optimizer_grouped_parameters(model, config) -> List[Dict]:
    decay_parameters = get_decay_parameter_names(model)
    if config["lr_embedding_recovery"] is not None:
        if config["lr_time_embedding"] is not None:
            time_embedding_params = get_conditional_norm_params(model)
            params = {
                "standard": [],
                "no_weight_decay": [],
                "embeddings": [],
                "time_embedding": [],
            }
            for n, p in model.named_parameters():
                if ("embeddings" in n or "patch_recovery" in n) and p.requires_grad:
                    params["embeddings"].append(p)
                elif n in decay_parameters and p.requires_grad:
                    params["standard"].append(p)
                elif p.requires_grad:
                    if n in time_embedding_params:
                        params["time_embedding"].append(p)
                    else:
                        params["no_weight_decay"].append(p)
            optimizer_grouped_parameters = [
                {
                    "params": params["standard"],
                    "weight_decay": config["weight_decay"],
                },
                {
                    "params": params["no_weight_decay"],
                    "weight_decay": 0.0,
                },
                {
                    "params": params["embeddings"],
                    "lr": config["lr_embedding_recovery"],
                    "weight_decay": config["weight_decay"],
                },
                {
                    "params": params["time_embedding"],
                    "lr": config["lr_time_embedding"],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = {"standard": [], "no_weight_decay": [], "embeddings": []}
            for n, p in model.named_parameters():
                if ("embeddings" in n or "patch_recovery" in n) and p.requires_grad:
                    params["embeddings"].append(p)
                elif n in decay_parameters and p.requires_grad:
                    params["standard"].append(p)
                elif p.requires_grad:
                    params["no_weight_decay"].append(p)
            optimizer_grouped_parameters = [
                {
                    "params": params["standard"],
                    "weight_decay": config["weight_decay"],
                },
                {
                    "params": params["no_weight_decay"],
                    "weight_decay": 0.0,
                },
                {
                    "params": params["embeddings"],
                    "lr": config["lr_embedding_recovery"],
                    "weight_decay": config["weight_decay"],
                },
            ]
    elif config["lr_time_embedding"] is not None:
        time_embedding_params = get_conditional_norm_params(model)
        params = {"standard": [], "no_weight_decay": [], "time_embedding": []}
        for n, p in model.named_parameters():
            if n in decay_parameters and p.requires_grad:
                params["standard"].append(p)
            elif p.requires_grad:
                if n in time_embedding_params:
                    params["time_embedding"].append(p)
                else:
                    params["no_weight_decay"].append(p)
        optimizer_grouped_parameters = [
            {
                "params": params["standard"],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": params["no_weight_decay"],
                "weight_decay": 0.0,
            },
            {
                "params": params["time_embedding"],
                "lr": config["lr_time_embedding"],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

    return optimizer_grouped_parameters
