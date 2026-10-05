import os
import hashlib
import json
import numpy as np

import exponax as ex
import jax
import jax.numpy as jnp
from typing import Dict, Any

from .base import BaseICMaker
from .registry import register_ic_maker
import jax.numpy as jnp
from jaxtyping import Array, Float

from exponax.ic._base_ic import BaseIC
from exponax._spectral import (
    low_pass_filter_mask,
    build_scaling_array,
    ifft,
    build_wavenumbers,
)


@register_ic_maker("fourier")
class RandomTruncatedFourierSeries(BaseICMaker):
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

        # Extract parameters from the dictionary
        if not isinstance(sampled_ic_config, dict):
            raise ValueError(
                f"sampled_ic_config must be a dictionary {type(sampled_ic_config)}"
                f"{sampled_ic_config}"
            )

        cutoff = int(sampled_ic_config.get("cutoff", 5))
        taper_power = float(sampled_ic_config.get("taper_power", 4.0))

        # Get amplitudes and phases
        assert isinstance(self.channel_key, str)
        channel_config = sampled_ic_config.get("channels", {}).get(self.channel_key, {})
        amplitudes = channel_config.get("amplitudes", [])
        phases = channel_config.get("phases", [])

        if not amplitudes or not phases:
            raise ValueError("fourier config must contain 'amplitudes' and 'phases' lists")

        # Validate coefficient counts
        cutoff_int = int(round(cutoff))
        required_pairs = (cutoff_int + 1) * pow((2 * cutoff_int + 1), num_spatial_dims - 1)

        if len(amplitudes) < required_pairs or len(phases) < required_pairs:
            raise ValueError(
                f"fourier needs {required_pairs} amplitude/phase pairs for cutoff={cutoff} "
                f"(dims={num_spatial_dims}), got {len(amplitudes)} amplitudes and {len(phases)} phases."
            )

        # Truncate if over-specified (allow extra coefficients for max cutoff case)
        amps = tuple(amplitudes[:required_pairs])
        phs = tuple(phases[:required_pairs])

        std_one = sampled_ic_config.get("std_one", False)
        max_one = sampled_ic_config.get("max_one", False)

        self.ic_maker = TruncatedFourierSeries(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            amplitudes=amps,
            phases=phs,
            cutoff=cutoff,
            taper_power=taper_power,
            std_one=std_one,
            max_one=max_one,
        )

    def __call__(self, **extra_args):
        del extra_args
        return self._ensure_channel_dim(self.ic_maker(self.num_points))


@register_ic_maker("fourier_simple")
class RandomTruncatedFourierSimple(BaseICMaker):
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

        # Extract parameters from the dictionary
        if not isinstance(sampled_ic_config, dict):
            raise ValueError(
                f"sampled_ic_config must be a dictionary {type(sampled_ic_config)}"
                f"{sampled_ic_config}"
            )
        self.cutoff = sampled_ic_config["cutoff"]
        self.taper_power = sampled_ic_config["taper_power"]
        self.std_one = sampled_ic_config.get("std_one", False)
        self.max_one = sampled_ic_config.get("max_one", False)
        self.seed_offset = int(sampled_ic_config.get("seed", 0))

        cutoff_int = max(0, int(round(self.cutoff)))
        self.num_modes = int((cutoff_int + 1) * pow((2 * cutoff_int + 1), num_spatial_dims - 1))

    def _derive_seed(self):
        # Melissa uses MELISSA_SIMU_ID (note the 'U'). Keep MELISSA_SIM_ID as a
        # backward-compatible alias for older scripts.
        sim_id = os.environ.get("MELISSA_SIMU_ID") or os.environ.get("MELISSA_SIM_ID")
        if sim_id is not None:
            try:
                return (self.seed_offset + int(sim_id)) % (2**32)
            except ValueError:
                pass
        try:
            payload = json.dumps(
                self.sampled_ic_config,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        except Exception:
            payload = repr(self.sampled_ic_config).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        stable_seed = int.from_bytes(digest, "little", signed=False)
        return (self.seed_offset + stable_seed) % (2**32)

    def __call__(self, **extra_args):
        del extra_args
        base_seed = self._derive_seed()
        samples = []
        for channel_idx in range(max(1, int(self.num_channels))):
            channel_seed = (base_seed + channel_idx * 7919) % (2**32)
            rng = np.random.default_rng(channel_seed)
            amps = rng.uniform(-1.0, 1.0, size=self.num_modes)
            phs = rng.uniform(0.0, 2 * np.pi, size=self.num_modes)
            ic_builder = TruncatedFourierSeries(
                num_spatial_dims=self.num_spatial_dims,
                domain_extent=self.domain_extent,
                amplitudes=tuple(float(val) for val in amps),
                phases=tuple(float(val) for val in phs),
                cutoff=self.cutoff,
                taper_power=self.taper_power,
                std_one=self.std_one,
                max_one=self.max_one,
            )
            samples.append(ic_builder(self.num_points))
        ic = jnp.concatenate(samples, axis=0)
        return self._ensure_channel_dim(ic)


class TruncatedFourierSeries(BaseIC):
    num_spatial_dims: int
    domain_extent: float
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    cutoff: int
    std_one: bool
    max_one: bool

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        amplitudes: tuple[float, ...],
        phases: tuple[float, ...],
        cutoff: int = 5,
        std_one: bool = False,
        max_one: bool = False,
        taper_power: float = 4.0,  # even number => sharper roll-off
    ):
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent

        if len(amplitudes) != len(phases):
            raise ValueError("Amplitudes and phases must have the same length.")
        self.amplitudes = jnp.array(amplitudes).ravel()
        self.phases = jnp.array(phases).ravel()
        self.cutoff = int(cutoff)
        self.std_one = std_one
        self.max_one = max_one
        object.__setattr__(self, "taper_power", float(taper_power))

    def __call__(self, num_points: int) -> Float[Array, "1 N"]:
        low_pass_filter = low_pass_filter_mask(
            self.num_spatial_dims, num_points, cutoff=self.cutoff, axis_separate=True
        )

        waves = self.amplitudes * jnp.exp(1j * self.phases)
        active_modes = int(low_pass_filter.sum())
        if waves.size < active_modes:
            raise ValueError(
                f"Expected {active_modes} Fourier coefficients for cutoff={self.cutoff} "
                f"with num_points={num_points} (dims={self.num_spatial_dims}), got {waves.size}. "
                "Provide at least one amplitude/phase pair for every mode retained by the "
                "low_pass_filter_mask (no extra padding needed)."
            )
        if waves.size > active_modes:
            waves = waves[:active_modes]  # allow over-specified coefficient lists

        fourier_noise = jnp.zeros(low_pass_filter.shape, dtype=waves.dtype)
        fourier_noise = fourier_noise.at[jnp.where(low_pass_filter)].set(waves)

        # Smoothly attenuate higher modes instead of a hard cutoff.
        kc = jnp.maximum(self.cutoff, 1e-12)
        wn = build_wavenumbers(self.num_spatial_dims, num_points)
        k_norm = jnp.linalg.norm(wn, axis=0, keepdims=True)
        soft_taper = jnp.exp(-((k_norm / kc) ** self.taper_power))
        fourier_noise = fourier_noise * soft_taper

        fourier_noise = fourier_noise * build_scaling_array(
            self.num_spatial_dims,
            num_points,
            mode="coef_extraction",
        )

        u = ifft(
            fourier_noise,
            num_spatial_dims=self.num_spatial_dims,
            num_points=num_points,
        )

        if self.std_one:
            u = u / jnp.std(u)

        if self.max_one:
            u /= jnp.max(jnp.abs(u))

        return u
