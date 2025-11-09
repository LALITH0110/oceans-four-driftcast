"""
File Summary:
- Generates synthetic 10 m wind fields with seasonal modulation.
- Exposes configuration dataclasses and API returning (u10, v10) arrays.
- Used by driftcast.particles.physics for windage calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WindFieldConfig:
    """Parameters controlling the analytic wind field generator."""

    trade_wind_max: float = 8.0  # m/s easterlies in subtropics
    westerly_max: float = 12.0  # m/s westerlies in mid-latitudes
    seasonal_amp: float = 0.3  # 30% amplitude
    seasonal_phase_days: float = 45.0
    transition_lat: float = 35.0


def seasonal_wind_field(
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: float | np.ndarray = 0.0,
    config: WindFieldConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return synthetic 10 m winds over the domain.

    The model blends trade easterlies and mid-latitude westerlies using
    smooth hyperbolic tangent transitions together with temporal sinusoids.

    Args:
        lon: Longitudes in degrees east (unused except to keep shape).
        lat: Latitudes in degrees north.
        time_days: Time since simulation start in days.
        config: Optional :class:`WindFieldConfig`.

    Returns:
        Tuple of ``(u10, v10)`` arrays matched to the broadcast shape.
    """
    config = config or WindFieldConfig()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_b, lat_b = np.broadcast_arrays(lon, lat)
    time_days = np.asarray(time_days, dtype=float)
    seasonal = 1.0 + config.seasonal_amp * np.sin(
        2.0 * math.pi * (time_days - config.seasonal_phase_days) / 365.0
    )
    seasonal = np.broadcast_to(seasonal, lon_b.shape)

    # Easterly trades (negative zonal wind) for low latitudes.
    trades = -config.trade_wind_max * np.exp(-(lat_b / config.transition_lat) ** 2)
    # Mid-latitude westerlies positive zonal.
    westerlies = config.westerly_max * np.exp(
        -((lat_b - 45.0) / config.transition_lat) ** 2
    )
    zonal = seasonal * (trades + westerlies)

    # Meridional component: weak cross-equatorial flow using sine.
    meridional = 2.0 * np.sin(np.deg2rad(lat_b)) * seasonal
    return zonal, meridional
