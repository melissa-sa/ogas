import copy
import math
from typing import Optional, Sequence

import torch
from torch import nn

from ..vendor.al4pde.modules.fno import FNO1d, FNO2d
from ..vendor.al4pde.modules.sinenet_cond_2d import SineNet
from ..vendor.al4pde.modules.unet_cond_1d import Unet1D
from ..vendor.al4pde.modules.unet_cond_2d import Unet2D
from ..vendor.scot import ScOT, ScOTConfig


class BaseConditionalModel(nn.Module):
    """Base helper that standardizes the forward interface."""

    def forward_step(
        self,
        state: torch.Tensor,
        grid: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        state: torch.Tensor,
        grid: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_step(state, grid, physics)


def _expand_grid(grid: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast a cached grid tensor to the current batch size."""
    return grid.expand((batch_size,) + tuple(grid.shape[1:]))


class ConditionalFNO2D(BaseConditionalModel):
    """Wrap the AL4PDE FNO implementation into a simple conditioned one-step model."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        history: int = 1,
        modes1: int = 20,
        modes2: int = 20,
        width: int = 64,
        predict_delta: bool = False,
    ):
        super().__init__()
        if history != 1:
            raise ValueError("Only history=1 is currently supported for the FNO wrapper.")
        self.history = history
        self.cond_dim = cond_dim
        self.num_scalar_channels = num_scalar_channels
        self.predict_delta = predict_delta
        input_channels = num_scalar_channels + cond_dim
        self.fno = FNO2d(
            in_channels=input_channels,
            out_channels=num_scalar_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            initial_step=history,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        grid: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # state: [B, C, H, W]
        if state.ndim != 4:
            raise ValueError(f"Expected state with shape [B, C, H, W], got {state.shape}")
        bsz, channels, height, width = state.shape
        x = state.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        if self.cond_dim > 0:
            if physics is None:
                raise ValueError("physics tensor must be provided when conditioning is enabled.")
            cond = physics.view(bsz, 1, 1, self.cond_dim).expand(-1, height, width, -1)
            x = torch.cat([x, cond], dim=-1)
        grid = _expand_grid(grid, bsz)
        pred = self.fno(x, grid).squeeze(-2)  # -> [B, H, W, total_C]
        pred = pred[..., : self.num_scalar_channels]
        pred = pred.permute(0, 3, 1, 2)
        if self.predict_delta:
            pred = pred + state
        return pred


class ConditionalFNO1D(BaseConditionalModel):
    """1D variant of the conditional FNO wrapper."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        history: int = 1,
        modes: int = 16,
        width: int = 64,
        predict_delta: bool = False,
    ):
        super().__init__()
        if history != 1:
            raise ValueError("Only history=1 is currently supported for the FNO wrapper.")
        self.history = history
        self.cond_dim = cond_dim
        self.num_scalar_channels = num_scalar_channels
        self.predict_delta = predict_delta
        input_channels = num_scalar_channels + cond_dim
        self.fno = FNO1d(
            in_channels=input_channels,
            out_channels=num_scalar_channels,
            modes=modes,
            width=width,
            initial_step=history,
            predict_delta=False,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        grid: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"Expected state with shape [B, C, X], got {state.shape}")
        bsz, channels, length = state.shape
        x = state.permute(0, 2, 1)  # -> [B, X, C]
        if self.cond_dim > 0:
            if physics is None:
                raise ValueError("physics tensor must be provided when conditioning is enabled.")
            cond = physics.view(bsz, 1, self.cond_dim).expand(-1, length, -1)
            x = torch.cat([x, cond], dim=-1)
        grid = _expand_grid(grid, bsz)
        pred = self.fno(x, grid).squeeze(-2)  # -> [B, X, total_C]
        pred = pred[..., : self.num_scalar_channels]
        pred = pred.permute(0, 2, 1)
        if self.predict_delta:
            pred = pred + state
        return pred


class ConditionalUNet2D(BaseConditionalModel):
    """Wrapper around the modern U-Net implementation with scalar conditioning."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        history: int = 1,
        hidden_channels: int = 64,
        activation: str = "gelu",
        padding_mode: str = "zeros",
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        is_attn: Sequence[bool] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use_scale_shift_norm: bool = False,
        use1x1: bool = False,
        predict_delta: bool = False,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        cond = f"scalar_{cond_dim}" if cond_dim > 0 else None
        self.history = history
        self.predict_delta = predict_delta
        self.delta_scale = float(delta_scale)
        self.unet = Unet2D(
            n_input_scalar_components=num_scalar_channels,
            n_input_vector_components=0,
            n_output_scalar_components=num_scalar_channels,
            n_output_vector_components=0,
            time_history=history,
            time_future=1,
            hidden_channels=hidden_channels,
            activation=activation,
            padding_mode=padding_mode,
            norm=True,
            ch_mults=tuple(ch_mults),
            is_attn=tuple(is_attn),
            mid_attn=mid_attn,
            n_blocks=n_blocks,
            param_conditioning=cond,
            use_scale_shift_norm=use_scale_shift_norm,
            use1x1=use1x1,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        _: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.unet.param_conditioning and physics is None:
            raise ValueError("physics tensor must be provided when conditioning is enabled.")
        bsz = state.shape[0]
        x = state.unsqueeze(1)  # [B, 1, C, H, W]
        pred = self.unet(x, time=None, z=physics)
        pred = pred.squeeze(1)
        if self.predict_delta:
            pred = self.delta_scale * pred + state
        return pred


class ConditionalUNet1D(BaseConditionalModel):
    """Wrapper for the 1D modern U-Net implementation."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        history: int = 1,
        hidden_channels: int = 64,
        activation: str = "gelu",
        padding_mode: str = "zeros",
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        is_attn: Sequence[bool] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use_scale_shift_norm: bool = False,
        use1x1: bool = False,
        predict_delta: bool = False,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        cond = f"scalar_{cond_dim}" if cond_dim > 0 else None
        self.history = history
        self.cond_dim = cond_dim
        self.predict_delta = predict_delta
        self.delta_scale = float(delta_scale)
        self.unet = Unet1D(
            padding_mode=padding_mode,
            n_input_scalar_components=num_scalar_channels,
            n_input_vector_components=0,
            n_output_scalar_components=num_scalar_channels,
            n_output_vector_components=0,
            time_history=history,
            time_future=1,
            hidden_channels=hidden_channels,
            activation=activation,
            norm=True,
            ch_mults=tuple(ch_mults),
            is_attn=tuple(is_attn),
            mid_attn=mid_attn,
            n_blocks=n_blocks,
            param_conditioning=cond,
            use_scale_shift_norm=use_scale_shift_norm,
            use1x1=use1x1,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        _: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = state.shape[0]
        if self.cond_dim <= 0:
            physics = None
        if self.cond_dim > 0 and physics is None:
            raise ValueError("physics tensor must be provided when conditioning is enabled.")
        x = state.unsqueeze(1)  # [B, 1, C, X]
        time_embed = None
        if physics is None:
            time_embed = torch.zeros(bsz, device=x.device, dtype=x.dtype)
        pred = self.unet(x, time=time_embed, z=physics)
        pred = pred.squeeze(1)
        if self.predict_delta:
            pred = self.delta_scale * pred + state
        return pred


class ConditionalSineNet2D(BaseConditionalModel):
    """Wrapper around the stacked U-Net (SineNet) architecture."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        history: int = 1,
        hidden_channels: int = 64,
        padding_mode: str = "zeros",
        activation: str = "gelu",
        num_layers: int = 4,
        num_waves: int = 2,
        num_blocks: int = 1,
        mult: int = 2,
        residual: bool = True,
        disentangle: bool = True,
        down_pool: bool = True,
        avg_pool: bool = True,
        up_interpolation: bool = True,
        interpolation_mode: str = "bicubic",
        use_scale_shift_norm: bool = False,
        predict_delta: bool = False,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        cond = f"scalar_{cond_dim}" if cond_dim > 0 else None
        self.history = history
        self.predict_delta = predict_delta
        self.delta_scale = float(delta_scale)
        self.sinenet = SineNet(
            n_input_scalar_components=num_scalar_channels,
            n_input_vector_components=0,
            n_output_scalar_components=num_scalar_channels,
            n_output_vector_components=0,
            time_history=history,
            time_future=1,
            hidden_channels=hidden_channels,
            padding_mode=padding_mode,
            activation=activation,
            num_layers=num_layers,
            num_waves=num_waves,
            num_blocks=num_blocks,
            norm=True,
            mult=mult,
            residual=residual,
            wave_residual=residual,
            disentangle=disentangle,
            down_pool=down_pool,
            avg_pool=avg_pool,
            up_interpolation=up_interpolation,
            interpolation_mode=interpolation_mode,
            param_conditioning=cond,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        _: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.sinenet.param_conditioning and physics is None:
            raise ValueError("physics tensor must be provided when conditioning is enabled.")
        x = state.unsqueeze(1)
        pred = self.sinenet(x, time=None, z=physics)
        pred = pred.squeeze(1)
        if self.predict_delta:
            pred = self.delta_scale * pred + state
        return pred


class ConditionalScOT2D(BaseConditionalModel):
    """Wrapper around the ScOT architecture with physics-only conditioning."""

    def __init__(
        self,
        num_scalar_channels: int,
        cond_dim: int,
        config: ScOTConfig,
        history: int = 1,
        predict_delta: bool = False,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        if history < 1:
            raise ValueError("history must be >= 1 for the ScOT wrapper.")

        self.history = history
        self.num_scalar_channels = num_scalar_channels
        self.total_in_channels = num_scalar_channels * history
        self.cond_dim = cond_dim
        self.predict_delta = predict_delta
        self.delta_scale = float(delta_scale)

        cfg = copy.deepcopy(config)
        cfg.num_channels = self.total_in_channels
        cfg.num_out_channels = num_scalar_channels
        cfg.use_conditioning = cond_dim > 0
        cfg.output_hidden_states = bool(getattr(cfg, "output_hidden_states", False))
        cfg.cond_dim = max(cond_dim, 1)
        self.scot_config = cfg
        self.scot = ScOT(cfg, use_mask_token=cfg.use_mask_token)

        self.cond_projector = None
        if cond_dim > 0:
            hidden = max(cond_dim, 16)
            self.cond_projector = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, cfg.cond_dim),
            )

    def forward_step(
        self,
        state: torch.Tensor,
        _: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state.ndim != 4:
            raise ValueError(f"Expected state with shape [B, C, H, W], got {state.shape}")
        bsz, channels, height, width = state.shape
        if channels != self.total_in_channels:
            raise ValueError(
                f"Expected {self.total_in_channels} input channels "
                f"(num_channels * history), got {channels}"
            )

        time_token = None
        if self.cond_projector is not None:
            if physics is None:
                raise ValueError("physics tensor must be provided when conditioning is enabled.")
            cond = physics
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            if cond.dim() > 2:
                cond = cond.view(cond.shape[0], -1)
            if cond.shape[0] != bsz:
                raise ValueError(
                    f"physics batch size {cond.shape[0]} does not match state batch {bsz}"
                )
            cond = cond.to(device=state.device, dtype=state.dtype)
            if cond.shape[1] != self.cond_dim:
                raise ValueError(f"Expected physics dim {self.cond_dim}, got {cond.shape[1]}")
            time_token = self.cond_projector(cond)

        outputs = self.scot(pixel_values=state, time=time_token)
        pred = outputs.output
        if self.predict_delta:
            baseline = state[:, -self.num_scalar_channels :, ...]
            pred = self.delta_scale * pred + baseline
        return pred


class _CatCircularConv2d(nn.Conv2d):
    def _conv_forward(self, x, weight, bias):
        ph, pw = self.padding
        if pw:
            x = torch.cat([x[..., -pw:], x, x[..., :pw]], dim=-1)
        if ph:
            x = torch.cat([x[..., -ph:, :], x, x[..., :ph, :]], dim=-2)
        return nn.functional.conv2d(x, weight, bias, self.stride, 0, self.dilation, self.groups)


def optimize_for_training(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if type(m) is nn.Conv2d and m.padding_mode == "circular":
            m.__class__ = _CatCircularConv2d
    model.compile()
    return model


def resolve_padding_mode(domain_extent) -> str:
    """Heuristic to keep circular padding for periodic domains."""
    if isinstance(domain_extent, (tuple, list)):
        periodic = all(math.isclose(domain_extent[0], d, rel_tol=1e-6) for d in domain_extent)
    else:
        periodic = True
    return "circular" if periodic else "zeros"
