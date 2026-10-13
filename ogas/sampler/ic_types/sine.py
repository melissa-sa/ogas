from typing import Any, Dict, Sequence

import exponax as ex

from .base import BaseICMaker
from .registry import register_ic_maker


def _as_float_tuple(value: Any, *, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(v) for v in default)
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def _legacy_sine_config(payload: str) -> Dict[str, Any]:
    parts = str(payload).split(";")
    if len(parts) < 4:
        raise ValueError(f"Invalid legacy sine config: {payload!r}")
    return {
        "type": parts[0],
        "amplitudes": [float(v) for v in parts[1:-2:2]],
        "phases": [float(v) for v in parts[2:-2:2]],
        "std_one": parts[-2].lower() == "true",
        "max_one": parts[-1].lower() == "true",
    }


@register_ic_maker("sine")
class SineWave(BaseICMaker):
    def __init__(
        self,
        channel_key: tuple[str] | str,
        sampled_ic_config: Dict[str, Any],
        domain_extent: float,
        num_points: int,
        num_spatial_dims: int = 1,
        num_channels: int = 1,
    ):
        if isinstance(sampled_ic_config, str):
            sampled_ic_config = _legacy_sine_config(sampled_ic_config)
        super().__init__(
            channel_key=channel_key,
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            num_channels=num_channels,
        )
        if self.num_spatial_dims != 1:
            raise ValueError("sine IC supports only 1D scenarios.")

        amplitudes = _as_float_tuple(sampled_ic_config.get("amplitudes"), default=(1.0,))
        phases = _as_float_tuple(sampled_ic_config.get("phases"), default=(0.0,))
        if len(amplitudes) != 1 or len(phases) != 1:
            raise ValueError("sine expects exactly one amplitude and one phase.")
        wavenumbers = _as_float_tuple(sampled_ic_config.get("wavenumbers"), default=(1.0,))
        if len(wavenumbers) != 1:
            raise ValueError("sine expects exactly one wavenumber.")

        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=amplitudes,
            wavenumbers=wavenumbers,
            phases=phases,
            std_one=bool(sampled_ic_config.get("std_one", False)),
            max_one=bool(sampled_ic_config.get("max_one", False)),
        )
        self.normalise = None

    def __call__(self, **extra_args):
        grid = ex.make_grid(
            self.num_spatial_dims, self.domain_extent, self.num_points, **extra_args
        )
        ic_mesh = self.ic_maker(grid)
        if self.normalise is not None:
            ic_mesh = ic_mesh / self.normalise
        return self._ensure_channel_dim(ic_mesh)


@register_ic_maker("sine_sup")
class SupSineWave(SineWave):
    def __init__(
        self,
        channel_key: tuple[str] | str,
        sampled_ic_config: Dict[str, Any],
        domain_extent: float,
        num_points: int,
        num_spatial_dims: int = 1,
        num_channels: int = 1,
    ):
        if isinstance(sampled_ic_config, str):
            sampled_ic_config = _legacy_sine_config(sampled_ic_config)

        amplitudes = _as_float_tuple(sampled_ic_config.get("amplitudes"), default=(1.0, 1.0))
        phases = _as_float_tuple(sampled_ic_config.get("phases"), default=(0.0,) * len(amplitudes))
        if len(amplitudes) != len(phases):
            raise ValueError(
                f"sine_sup needs matching amplitudes/phases, got {len(amplitudes)} and {len(phases)}."
            )
        default_wavenumbers = tuple(float(i) for i in range(1, len(amplitudes) + 1))
        wavenumbers = _as_float_tuple(
            sampled_ic_config.get("wavenumbers"),
            default=default_wavenumbers,
        )
        if len(wavenumbers) != len(amplitudes):
            raise ValueError(
                f"sine_sup needs one wavenumber per amplitude, got {len(wavenumbers)} and "
                f"{len(amplitudes)}."
            )

        BaseICMaker.__init__(
            self,
            channel_key=channel_key,
            sampled_ic_config=sampled_ic_config,
            domain_extent=domain_extent,
            num_points=num_points,
            num_spatial_dims=num_spatial_dims,
            num_channels=num_channels,
        )
        if self.num_spatial_dims != 1:
            raise ValueError("sine_sup IC supports only 1D scenarios.")

        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=amplitudes,
            wavenumbers=wavenumbers,
            phases=phases,
            std_one=bool(sampled_ic_config.get("std_one", False)),
            max_one=False,
        )
        self.normalise = len(amplitudes) if bool(sampled_ic_config.get("max_one", False)) else None
