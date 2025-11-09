# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Provides geographic conversion helpers between meters and degrees latitude/longitude.
- Exposes utilities for physics kernels to express diffusion and advection consistently.
- Based on WGS84 ellipsoid approximations suitable for mesoscale ocean drift modeling.
"""

from __future__ import annotations

from typing import Union

import numpy as np

Number = Union[float, np.ndarray]


def meters_per_deg_lat(latitude: Number) -> Number:
    """Return meridional arc length in metres for one degree latitude at ``latitude``.

    Uses a third-order series in sine multiples consistent with Snyder (1987).
    The approximation is accurate to <1 m over valid latitude ranges.
    """
    lat_rad = np.deg2rad(latitude)
    sin_lat = np.sin(lat_rad)
    return (
        111_132.954
        - 559.822 * np.cos(2.0 * lat_rad)
        + 1.175 * np.cos(4.0 * lat_rad)
        - 0.0023 * np.cos(6.0 * lat_rad)
    )


def meters_per_deg_lon(latitude: Number) -> Number:
    """Return zonal arc length in metres for one degree longitude at ``latitude``."""
    lat_rad = np.deg2rad(latitude)
    return (
        111_319.459 * np.cos(lat_rad)
        - 162.239 * np.cos(3.0 * lat_rad)
        + 0.705 * np.cos(5.0 * lat_rad)
    )


def deg_lat_per_meter(latitude: Number) -> Number:
    """Return degrees latitude per metre displacement at ``latitude``."""
    return 1.0 / meters_per_deg_lat(latitude)


def deg_lon_per_meter(latitude: Number) -> Number:
    """Return degrees longitude per metre displacement at ``latitude``."""
    return 1.0 / meters_per_deg_lon(latitude)
