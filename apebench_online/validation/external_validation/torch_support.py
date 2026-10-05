import numpy as np
import torch

from apebench_online.models import build_torch_model


def select_device(dl_config: dict) -> torch.device:
    device_name = dl_config.get("torch_device")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def build_grid_tensor(scenario):
    """Create a normalized spatial grid tensor based on the scenario mesh.

    """
    if scenario is None:
        raise RuntimeError("Scenario must be initialized before building the grid tensor.")
    dims = getattr(scenario, "num_spatial_dims", 2)
    num_points = getattr(scenario, "num_points", 0)
    if num_points <= 0:
        raise ValueError("Scenario must expose num_points to build the grid tensor.")

    model_num_points = num_points

    axes = [torch.linspace(0.0, 1.0, model_num_points) for _ in range(dims)]
    mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    mesh = mesh * 2.0 - 1.0
    return mesh.unsqueeze(0).to(torch.float32)


def init_torch_model(dl_config: dict, scenario, physics_dim: int):
    device = select_device(dl_config)
    model_cfg = dl_config.get("torch_model")

    model = build_torch_model(dict(model_cfg), scenario, physics_dim)
    model.to(device)
    model.eval()
    grid_tensor = build_grid_tensor(scenario).to(device)
    return model, device, grid_tensor


def validation_physics_tensor(
    indices,
    physics_dim: int,
    sample_physics: bool,
    fixed_physics_params,
    valid_dataset,
    valid_parameters,
    nb_time_steps: int,
    torch_device,
    physics_min=None,
    physics_max=None,
):
    if physics_dim == 0:
        return None
    if indices is None or len(indices) == 0:
        return torch.zeros(0, physics_dim, device=torch_device)

    nb_steps = getattr(valid_dataset, "nb_time_steps", nb_time_steps)
    sample_ids = (np.asarray(indices, dtype=np.int64) // nb_steps).astype(int)
    if sample_physics and valid_parameters is not None:
        params = valid_parameters[sample_ids, -physics_dim:]
    elif sample_physics:
        params = np.zeros((len(sample_ids), physics_dim), dtype=np.float32)
    else:
        fixed = np.asarray(fixed_physics_params or [0.0] * physics_dim, dtype=np.float32)
        params = np.repeat(fixed[None, :], len(sample_ids), axis=0)
    physics = torch.as_tensor(params, dtype=torch.float32, device=torch_device)

    if physics_min is not None and physics_max is not None and physics.numel() > 0:
        if not isinstance(physics_min, torch.Tensor):
            physics_min = torch.as_tensor(physics_min, dtype=torch.float32, device=torch_device)
        if not isinstance(physics_max, torch.Tensor):
            physics_max = torch.as_tensor(physics_max, dtype=torch.float32, device=torch_device)
        denom = (physics_max - physics_min).clamp_min(1e-8)
        physics = (physics - physics_min) / denom

    return physics
