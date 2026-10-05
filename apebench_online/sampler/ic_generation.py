import jax.numpy as jnp
from typing import Dict, Any, List, Union

from apebench_online.sampler.ic_types import (
    ensure_builtin_ic_types_loaded,
)
from apebench_online.sampler.ic_types.base import BaseICMaker
from apebench_online.sampler.ic_types.registry import get_ic_maker_class

# Backwards compatibility re-exports for existing imports
from apebench_online.sampler.ic_types.sine import (  # noqa: F401
    SineWave,
    SupSineWave,
)
from apebench_online.sampler.ic_types.fourier import (  # noqa: F401
    RandomTruncatedFourierSeries,
)
from apebench_online.sampler.ic_types.gaussian_blobs import (  # noqa: F401
    GaussianBlobsManual,
)


ensure_builtin_ic_types_loaded()


def _build_single_maker(
    channel_key: tuple[str] | str,
    sampled_ic_config: Dict[str, Any],
    domain_extent: float,
    num_points: int,
    num_spatial_dims: int,
    num_channels: int,
    extra_args: Dict[str, Any],
) -> BaseICMaker:
    """Build a single IC maker from a config dictionary.

    Args:
        channel_key: Key ("c0", "c1") to the channel config dictionary on which IC maker works on.
        sampled_ic_config: Configuration dictionary containing 'type' and IC-specific parameters
        domain_extent: Spatial domain extent
        num_points: Number of grid points
        num_spatial_dims: Number of spatial dimensions
        num_channels: Number of channels for this IC
        extra_args: Additional arguments to pass to the IC maker

    Returns:
        An instance of the appropriate IC maker class
    """
    ic_type = sampled_ic_config.get("type")
    if not ic_type:
        raise ValueError("IC config must contain a 'type' field")

    ic_cls = get_ic_maker_class(ic_type)
    return ic_cls(
        channel_key=channel_key,
        sampled_ic_config=sampled_ic_config,
        domain_extent=domain_extent,
        num_points=num_points,
        num_spatial_dims=num_spatial_dims,
        num_channels=num_channels,
        **extra_args,
    )


class MultiChannelIC(BaseICMaker):
    """Composite IC maker to handle multi-channel configurations.

    This class manages multiple IC makers, one per channel, and concatenates
    their outputs along the channel dimension.
    """

    def __init__(
        self,
        channel_key: tuple[str] | str,
        sampled_ic_config: Dict[str, Any],
        domain_extent: float,
        num_points: int,
        num_spatial_dims: int,
        makers: List[BaseICMaker],
    ):
        super().__init__(
            channel_key=channel_key,
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            num_channels=len(makers),
        )
        self.makers = makers

    def __call__(self, **extra_args) -> jnp.ndarray:
        """Generate initial conditions for all channels.

        Returns:
            Array with shape (num_channels, *spatial_dims)
        """
        ics = [maker(**extra_args) for maker in self.makers]
        return jnp.concatenate(ics, axis=0)


def _normalize_channel_indices(
    channels: Any,
    num_channels: int,
) -> list[int]:
    if channels is None:
        return []
    if isinstance(channels, (str, int)):
        channels = [channels]
    indices: list[int] = []
    for ch in channels:
        if isinstance(ch, str):
            ch = ch.strip()
            if ch.startswith("c") and ch[1:].isdigit():
                idx = int(ch[1:])
            else:
                raise ValueError(f"Invalid channel key '{ch}' in invert_channels")
        else:
            idx = int(ch)
        if idx < 0 or idx >= num_channels:
            raise ValueError(
                f"invert_channels index {idx} out of range for num_channels={num_channels}"
            )
        if idx not in indices:
            indices.append(idx)
    return indices


class PostProcessedIC(BaseICMaker):
    """Wraps an IC maker and applies simple post-processing to its output."""

    def __init__(self, inner: BaseICMaker, invert_channels: list[int]):
        super().__init__(
            channel_key=inner.channel_key,
            sampled_ic_config=inner.sampled_ic_config,
            domain_extent=inner.domain_extent,
            num_points=inner.num_points,
            num_spatial_dims=inner.num_spatial_dims,
            num_channels=inner.num_channels,
        )
        self.inner = inner
        self.invert_channels = invert_channels

    def __call__(self, **extra_args) -> jnp.ndarray:
        ic = self.inner(**extra_args)
        if not self.invert_channels:
            return ic
        ic = jnp.asarray(ic)
        for idx in self.invert_channels:
            ic = ic.at[idx].set(1.0 - ic[idx])
        return ic


def get_ic_maker(
    sampled_ic_config: Dict[str, Any],
    num_spatial_dims: int,
    domain_extent: float,
    num_points: int,
    num_channels: int = 1,
    **extra_args,
) -> BaseICMaker:
    """
    Create an IC maker from a configuration dictionary.

    Args:
        sampled_ic_config: Configuration dictionary
        num_spatial_dims: Number of spatial dimensions (1, 2, or 3)
        domain_extent: Spatial extent of the domain
        num_points: Number of grid points per dimension
        num_channels: Expected number of channels
        **extra_args: Additional arguments passed to IC makers

    Returns:
        An IC maker instance (either single or multi-channel)
    """

    if not isinstance(sampled_ic_config, dict):
        raise ValueError(
            f"sampled_ic_config must be a dict or string, got {type(sampled_ic_config)}"
        )

    # If the config explicitly provides per-channel sections, honor them.
    # Otherwise, honor the scenario-provided `num_channels` (e.g. `fourier_simple`
    # does not have a `channels:` block but still supports multi-channel output).
    expected_num_channels = int(num_channels)
    channels = sampled_ic_config.get("channels") or {}

    if not isinstance(channels, dict):
        raise ValueError(f"sampled_ic_config['channels'] must be a dict, got {type(channels)}")

    explicit_channel_count = len(channels)
    if explicit_channel_count > 0 and expected_num_channels != explicit_channel_count:
        raise ValueError(
            f"IC config provides {explicit_channel_count} channel(s) via 'channels', "
            f"but scenario expects num_channels={expected_num_channels}."
        )

    # Multi-channel config via explicit per-channel blocks
    if explicit_channel_count > 1:
        makers = []
        for channel_key in channels.keys():
            makers.append(
                _build_single_maker(
                    channel_key=channel_key,
                    sampled_ic_config=sampled_ic_config,
                    domain_extent=domain_extent,
                    num_points=num_points,
                    num_spatial_dims=num_spatial_dims,
                    num_channels=1,
                    extra_args=extra_args,
                )
            )
        maker: BaseICMaker = MultiChannelIC(
            channel_key=tuple(channels.keys()),
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            makers=makers,
        )
    else:
        # Single-maker config (may still generate multiple channels internally).
        maker = _build_single_maker(
            channel_key=next(iter(channels.keys()), "c0"),
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            num_channels=expected_num_channels,
            extra_args=extra_args,
        )

    invert_channels = sampled_ic_config.get("invert_channels")
    indices = _normalize_channel_indices(invert_channels, maker.num_channels)
    if indices:
        maker = PostProcessedIC(maker, indices)
    return maker


__all__ = [
    "BaseICMaker",
    "SineWave",
    "SupSineWave",
    "RandomTruncatedFourierSeries",
    "GaussianBlobsManual",
    "get_ic_maker",
    "MultiChannelIC",
]
