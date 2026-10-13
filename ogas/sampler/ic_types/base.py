import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

import jax.numpy as jnp

IC_DIR = "initial_conditions"


class BaseICMaker(ABC):
    """Base class for initial condition makers.

    All IC makers should inherit from this class and implement the __call__ method.
    The sampled_ic_config is now expected to be a dictionary containing the IC type
    and all necessary parameters.
    """

    def __init__(
        self,
        channel_key: tuple[str] | str,
        sampled_ic_config: Dict[str, Any],
        domain_extent: float,
        num_points: int,
        num_spatial_dims: int = 2,
        num_channels: int = 1,
    ):
        """Initialize the base IC maker."""
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points
        self.sampled_ic_config = sampled_ic_config
        self.num_channels = num_channels
        self.ic_maker = None

        if isinstance(channel_key, tuple) and len(channel_key) != num_channels:
            raise ValueError(
                "Must pass the channel keys to existing in sampled_ic_config channels section "
                "for extraction."
            )
        self.channel_key = channel_key[0] if len(channel_key) == 1 else channel_key

        # Validate that config is a dictionary
        if isinstance(sampled_ic_config, dict):
            self._validate_config(sampled_ic_config)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate the configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if "type" not in config:
            raise ValueError("IC config must contain 'type' field")

    def input_from_file(self) -> jnp.ndarray:
        """Load initial condition from a saved file.

        Returns:
            Array with shape (num_channels, *spatial_dims)
        """
        sim_id = os.environ.get("MELISSA_SIMU_ID") or os.environ.get("MELISSA_SIM_ID")
        if sim_id is None:
            raise KeyError(
                "Neither MELISSA_SIMU_ID nor MELISSA_SIM_ID is set; cannot load per-sim IC file."
            )
        arr = jnp.load(f"{IC_DIR}/sim{sim_id}.npy")
        return self._ensure_channel_dim(arr)

    @abstractmethod
    def __call__(self, **input_fn_args) -> jnp.ndarray:
        """Generate the initial condition.

        Args:
            **input_fn_args: Additional arguments for IC generation

        Returns:
            Array with shape (num_channels, *spatial_dims)
        """
        raise NotImplementedError

    def _ensure_channel_dim(self, ic: jnp.ndarray) -> jnp.ndarray:
        """Ensure returned array has the configured channel dimension.

        This method handles various input shapes and ensures the output always
        has the correct channel dimension as the first axis.

        Args:
            ic: Initial condition array

        Returns:
            Array with shape (num_channels, *spatial_dims)

        Raises:
            ValueError: If the array shape is incompatible with num_channels
        """
        ic = jnp.asarray(ic)

        # Add channel dimension if missing
        if ic.ndim == self.num_spatial_dims:
            ic = ic[jnp.newaxis, ...]

        # Check if channel dimension matches
        if ic.shape[0] == self.num_channels:
            return ic

        # Broadcast single channel to multiple channels if needed
        if ic.shape[0] == 1 and self.num_channels > 1:
            target_shape = (self.num_channels,) + ic.shape[1:]
            return jnp.broadcast_to(ic, target_shape)

        # Format config for error message
        config_str = self._format_config_for_error()

        raise ValueError(
            f"Initial condition shape {ic.shape} incompatible with expected "
            f"({self.num_channels}, ...) for config: {config_str}"
        )

    def _format_config_for_error(self) -> str:
        """Format the config for error messages.

        Returns:
            Human-readable string representation of the config
        """
        if isinstance(self.sampled_ic_config, dict):
            ic_type = self.sampled_ic_config.get("type", "unknown")
            return f"type={ic_type}"
        raise ValueError("sampled_ic_config must be a dictionary.")
