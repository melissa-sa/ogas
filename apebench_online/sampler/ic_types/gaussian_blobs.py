import exponax as ex
import jax.numpy as jnp
from typing import Dict, Any, List

from exponax.ic._gaussian_blob import GaussianBlob

from .base import BaseICMaker
from .registry import register_ic_maker


@register_ic_maker("gaussian_blobs")
class GaussianBlobsManual(BaseICMaker):
    """Manual Gaussian blob IC builder (2D) with full covariance per blob.

    Each blob is defined by:
        - amplitude: Height/intensity of the blob
        - sigma_x: Standard deviation in x direction
        - sigma_y: Standard deviation in y direction
        - cov_xy: Covariance between x and y
        - center_x: X coordinate of blob center
        - center_y: Y coordinate of blob center

    Config format:
        {
            "blobs": {
                "0": {
                    "amplitude": 1.0,
                    "sigma_x": 0.1,
                    "sigma_y": 0.15,
                    "cov_xy": 0.02,
                    "center_x": 0.5,
                    "center_y": 0.5
                },
                ...
            },
            "one_complement": False
        }

    Notes:
        - centers/sigmas/cov_xy are interpreted in a unit domain and scaled by domain_extent
        - blobs are combined as an amplitude-weighted average (sum / num_blobs)
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
        super().__init__(
            channel_key=channel_key,
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            num_channels=num_channels,
        )

        if self.num_spatial_dims != 2:
            raise ValueError("gaussian_blobs currently supports only 2D with full covariance.")

        if not isinstance(sampled_ic_config, dict):
            raise ValueError("sampled_ic_config must be a dictionary")

        # Extract blob configurations
        assert isinstance(self.channel_key, str)
        channel_config = sampled_ic_config.get("channels", {}).get(self.channel_key, {})
        blobs_config = channel_config.get("blobs", {})
        if not isinstance(blobs_config, (list, dict)):
            raise ValueError("'blobs' must be a list or dict of blob configurations")

        if len(blobs_config) == 0:
            raise ValueError("At least one blob must be specified")

        self.num_blobs = len(blobs_config)
        self.blobs = []

        # Parse each blob
        if isinstance(blobs_config, dict):
            blob_items = blobs_config.items()
        else:
            blob_items = enumerate(blobs_config)

        for b, blob_cfg in blob_items:
            if not isinstance(blob_cfg, dict):
                raise ValueError(f"Blob {b} configuration must be a dictionary")

            try:
                amplitude = float(blob_cfg.get("amplitude", 1.0))
                sigma_x = max(float(blob_cfg.get("sigma_x", 0.1)), 1e-8)
                sigma_y = max(float(blob_cfg.get("sigma_y", 0.1)), 1e-8)
                cov_xy = float(blob_cfg.get("cov_xy", 0.0))
                center_x = float(blob_cfg.get("center_x", 0.5))
                center_y = float(blob_cfg.get("center_y", 0.5))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid parameter in blob {b} configuration") from exc

            # Scale by domain extent to match exponax semantics
            sigma_x = sigma_x * self.domain_extent
            sigma_y = sigma_y * self.domain_extent
            cov_xy = cov_xy * (self.domain_extent**2)
            center_x = center_x * self.domain_extent
            center_y = center_y * self.domain_extent

            # Construct covariance matrix (full 2D)
            cov = jnp.array([[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]])

            # Check positive definiteness
            if jnp.linalg.det(cov) <= 0:
                raise ValueError(
                    f"Covariance matrix for blob {b} is not positive definite. "
                    f"Ensure sigma_x * sigma_y > |cov_xy|"
                )

            self.blobs.append(
                {
                    "amplitude": amplitude,
                    "cov": cov,
                    "centers": (center_x, center_y),
                }
            )

        # Optional flags
        self.one_complement = sampled_ic_config.get("one_complement", False)

    def _compose_field(self, grid: jnp.ndarray) -> jnp.ndarray:
        """Compose the field from all blobs.

        Args:
            grid: Spatial grid with shape (num_spatial_dims, *spatial_shape)

        Returns:
            Array with shape (num_channels, *spatial_shape)
        """
        field = jnp.zeros(grid.shape[1:], dtype=jnp.float32)

        # Sum all Gaussian blobs (with per-blob amplitude)
        for blob in self.blobs:
            gb = GaussianBlob(jnp.array(blob["centers"]), blob["cov"], one_complement=False)
            gaussian = gb(grid).squeeze(0)
            field = field + blob["amplitude"] * gaussian

        # Match exponax.GaussianBlobs: average over blobs (amplitude-weighted sum)
        if self.num_blobs > 0:
            field = field / float(self.num_blobs)

        # Handle multi-channel output
        channels = [field]

        # Add complement channel if requested
        if self.one_complement and self.num_channels >= 2:
            channels.append(1.0 - field)

        # Replicate field to fill remaining channels
        while len(channels) < self.num_channels:
            channels.append(field)

        # Stack channels
        data = jnp.stack(channels[: self.num_channels], axis=0)

        return data

    def __call__(self, **extra_args) -> jnp.ndarray:
        """Generate the Gaussian blobs initial condition.

        Args:
            **extra_args: Additional arguments passed to exponax.make_grid

        Returns:
            Array with shape (num_channels, *spatial_shape)
        """
        grid = ex.make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            **extra_args,
        )
        ic = self._compose_field(grid)
        return self._ensure_channel_dim(ic)
