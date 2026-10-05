"""Utility functions for configuration handling."""

import pandas as pd


def coerce_optional(value):
    """Convert various null-like values to None."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        val = value.strip()
        return None if val == "" else val
    return value


def parse_bool(value):
    """Parse boolean value from CSV or config."""
    val = coerce_optional(value)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"1", "true", "yes", "y", "t"}


def deep_update(d, u):
    """Recursively update nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def calculate_data_size(num_points, nb_time_steps, num_spatial_dims=1, dtype_size=8):
    """Calculate data size for memory estimation.

    Args:
        num_points: Number of grid points per dimension
        nb_time_steps: Number of time steps
        num_spatial_dims: Number of spatial dimensions
        dtype_size: Size of data type in bytes (default: 8 for float64)

    Returns:
        Total data size in bytes
    """
    total_points = pow(num_points, num_spatial_dims)
    return total_points * nb_time_steps * dtype_size
