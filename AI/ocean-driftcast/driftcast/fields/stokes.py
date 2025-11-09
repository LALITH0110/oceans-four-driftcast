"""
File Summary:
- Builds a simple synthetic wave-induced Stokes drift velocity field.
- Provides configuration hooks to scale magnitude and cross-shore decay.
- Consumed by driftcast.particles.physics when Stokes coupling is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class StokesConfig:
    """Parameters for the analytic Stokes drift representation."""

    reference_speed: float = 0.12  # m/s typical open ocean Stokes drift
    decay_scale_km: float = 150.0
    directional_shift_deg: float = -15.0


def stokes_drift_velocity(
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: float | np.ndarray = 0.0,
    config: StokesConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return synthetic Stokes drift velocities.

    Args:
        lon: Longitudes in degrees east.
        lat: Latitudes in degrees north.
        time_days: Time since simulation start (currently unused but reserved).
        config: Optional configuration overrides.

    Returns:
        Tuple of ``(u, v)`` components in m/s.
    """
    config = config or StokesConfig()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_b, lat_b = np.broadcast_arrays(lon, lat)

    # Idealized magnitude strongest mid-basin, decays exponentially toward coasts.
    lon_norm = (lon_b + 60.0) / 50.0
    lat_norm = (lat_b - 40.0) / 20.0
    radius2 = lon_norm**2 + lat_norm**2
    magnitude = config.reference_speed * np.exp(-radius2 / 4.0)

    theta = np.deg2rad(config.directional_shift_deg + 10.0 * np.sin(np.deg2rad(lat_b)))
    u = magnitude * np.cos(theta)
    v = magnitude * np.sin(theta)
    return u, v
