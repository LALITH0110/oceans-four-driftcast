"""
File Summary:
- Implements an idealized two-gyre streamfunction for the North Atlantic surface.
- Provides dataclasses and helpers to compute ψ, u, v with seasonal modulation.
- Pair with driftcast.fields.winds and driftcast.sim.integrators for full runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from driftcast.fields.winds import WindFieldConfig, seasonal_wind_field

EARTH_RADIUS_KM = 6371.0
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s


@dataclass(frozen=True)
class GyreSpec:
    """Describe an individual gyre component in the analytic streamfunction.

    Args:
        center_lon: Longitude of the gyre core in degrees east.
        center_lat: Latitude of the gyre core in degrees north.
        amplitude: Peak streamfunction strength (m^2 s^-1).
        radius: E-folding radius in degrees for the Gaussian envelope.
    """

    center_lon: float
    center_lat: float
    amplitude: float
    radius: float


@dataclass(frozen=True)
class GyreFieldConfig:
    """Container for gyre pair parameters and seasonal modulation constants.

    Args:
        subtropical: Warm-core subtropical gyre specification.
        subpolar: Cool-core subpolar gyre specification.
        seasonal_cycle_days: Period of the seasonal modulation (default 365).
        seasonal_amp: Fractional amplitude applied when seasonal_enabled is True.
        base_time_days: Reference start of the modulation (default 0).
        seasonal_enabled: Toggle to apply the seasonal ramp.
        ekman_enabled: Enable Ekman surface drift approximation.
        ekman_alpha: Scaling applied to the Ekman solution (dimensionless).
        ekman_drag_coeff: Quadratic drag coefficient used for wind stress.
        ekman_air_density: Air density (kg/m^3) used in the stress calculation.
        ekman_water_density: Water density (kg/m^3) for Ekman velocity scaling.
    """

    subtropical: GyreSpec = GyreSpec(center_lon=-55.0, center_lat=28.0, amplitude=32.0, radius=9.0)
    subpolar: GyreSpec = GyreSpec(center_lon=-40.0, center_lat=54.0, amplitude=-20.0, radius=11.0)
    seasonal_cycle_days: float = 365.0
    seasonal_amp: float = 0.1
    base_time_days: float = 0.0
    seasonal_enabled: bool = False
    ekman_enabled: bool = False
    ekman_alpha: float = 0.03
    ekman_drag_coeff: float = 1.4e-3
    ekman_air_density: float = 1.225
    ekman_water_density: float = 1025.0


def _seasonal_factor(time_days: float | np.ndarray, config: GyreFieldConfig) -> float | np.ndarray:
    """Return the multiplicative seasonal modulation term."""
    if not config.seasonal_enabled or config.seasonal_amp == 0.0:
        return 1.0
    omega = 2.0 * math.pi / config.seasonal_cycle_days
    phase = omega * (time_days - config.base_time_days)
    return 1.0 + config.seasonal_amp * np.cos(phase)


def _angular_distance(lon: np.ndarray, lat: np.ndarray, spec: GyreSpec) -> np.ndarray:
    """Compute squared angular distance in radians^2 for Gaussian envelope."""
    lon_c = np.deg2rad(spec.center_lon)
    lat_c = np.deg2rad(spec.center_lat)
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    dlon = (lon_r - lon_c) * np.cos(0.5 * (lat_r + lat_c))
    dlat = lat_r - lat_c
    return dlon**2 + dlat**2


def _western_boundary_enhancement(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return empirical velocity boosts for Gulf Stream and Canary branches."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    gulf_core = np.exp(-((lon + 74.0) / 6.0) ** 2) * np.exp(-((lat - 32.0) / 5.0) ** 2)
    canary_core = np.exp(-((lon + 19.0) / 5.0) ** 2) * np.exp(-((lat - 32.0) / 6.0) ** 2)
    u_boost = 0.22 * gulf_core - 0.16 * canary_core
    v_boost = 0.18 * gulf_core - 0.12 * canary_core
    return u_boost, v_boost


def _coriolis_parameter(lat: np.ndarray) -> np.ndarray:
    """Return the Coriolis parameter f (s^-1) for the provided latitudes."""
    lat_rad = np.deg2rad(lat)
    return 2.0 * EARTH_ROTATION_RATE * np.sin(lat_rad)


def _ekman_drift(
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: float | np.ndarray,
    config: GyreFieldConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a first-order Ekman surface drift using synthetic winds."""
    wind_cfg = WindFieldConfig()
    tau_x: np.ndarray
    tau_y: np.ndarray
    u10, v10 = seasonal_wind_field(lon, lat, time_days, config=wind_cfg)
    speed = np.hypot(u10, v10)
    tau_x = config.ekman_air_density * config.ekman_drag_coeff * speed * u10
    tau_y = config.ekman_air_density * config.ekman_drag_coeff * speed * v10
    f = _coriolis_parameter(lat)
    safe_f = np.where(np.abs(f) < 1e-6, np.sign(f) * 1e-6, f)
    denom = config.ekman_water_density * safe_f
    u_ek = config.ekman_alpha * (tau_y / denom)
    v_ek = -config.ekman_alpha * (tau_x / denom)
    u_ek = np.where(np.isfinite(u_ek), u_ek, 0.0)
    v_ek = np.where(np.isfinite(v_ek), v_ek, 0.0)
    return u_ek, v_ek


def streamfunction(
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: float | np.ndarray = 0.0,
    config: GyreFieldConfig | None = None,
) -> np.ndarray:
    """Return the analytic streamfunction ψ for the configured gyres.

    Args:
        lon: Longitudes in degrees east.
        lat: Latitudes in degrees north.
        time_days: Time since simulation start in days.
        config: Optional overriding configuration.

    Returns:
        Streamfunction values with shape broadcast from inputs.

    Example:
        >>> lon = np.linspace(-80, -10, 5)
        >>> lat = np.linspace(10, 60, 5)
        >>> psi = streamfunction(lon[:, None], lat[None, :])
        >>> psi.shape
        (5, 5)
    """
    config = config or GyreFieldConfig()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_b, lat_b = np.broadcast_arrays(lon, lat)
    time_days = np.asarray(time_days, dtype=float)
    envelope_terms = []
    seasonal = np.asarray(_seasonal_factor(time_days, config), dtype=float)

    for spec in (config.subtropical, config.subpolar):
        radius_rad = np.deg2rad(spec.radius)
        r2 = _angular_distance(lon_b, lat_b, spec)
        envelope = np.exp(-0.5 * r2 / (radius_rad**2))
        envelope_terms.append(spec.amplitude * envelope)

    envelope_sum = sum(envelope_terms)
    psi = np.asarray(envelope_sum) * np.broadcast_to(seasonal, envelope_sum.shape)
    return psi


def gyre_velocity_field(
    lon: np.ndarray,
    lat: np.ndarray,
    time_days: float | np.ndarray = 0.0,
    config: GyreFieldConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return zonal (u) and meridional (v) velocities derived from ψ.

    The conversion uses the relations on a sphere:
        u = -(1 / (a cos φ)) ∂ψ/∂φ,  v = (1 / a) ∂ψ/∂λ,
    where ``a`` is the Earth's radius and (λ, φ) are longitude and latitude.

    Args:
        lon: Longitudes in degrees east (array-like, broadcast friendly).
        lat: Latitudes in degrees north (array-like, broadcast friendly).
        time_days: Time since start in days.
        config: Optional overriding configuration values.

    Returns:
        Tuple of ``(u, v)`` velocities in meters per second.
    """
    config = config or GyreFieldConfig()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_b, lat_b = np.broadcast_arrays(lon, lat)
    lon_rad = np.deg2rad(lon_b)
    lat_rad = np.deg2rad(lat_b)
    seasonal = np.asarray(_seasonal_factor(time_days, config), dtype=float)

    dpsi_dlambda = np.zeros(lon_rad.shape, dtype=float)
    dpsi_dphi = np.zeros_like(dpsi_dlambda)

    for spec in (config.subtropical, config.subpolar):
        radius_rad = np.deg2rad(spec.radius)
        lon_c = math.radians(spec.center_lon)
        lat_c = math.radians(spec.center_lat)
        cos_mid = np.cos(0.5 * (lat_rad + lat_c))
        dlambda = lon_rad - lon_c
        dphi = lat_rad - lat_c
        dx = cos_mid * dlambda
        r2 = dx**2 + dphi**2
        envelope = np.exp(-0.5 * r2 / (radius_rad**2))
        common = (
            spec.amplitude
            * np.broadcast_to(seasonal, envelope.shape)
            * envelope
            / (radius_rad**2)
        )
        dpsi_dlambda += -common * dx * cos_mid
        dpsi_dphi += -common * dphi

    a = EARTH_RADIUS_KM * 1000.0
    cos_lat = np.clip(np.cos(lat_rad), 1e-6, None)
    u = -(1.0 / (a * cos_lat)) * dpsi_dphi
    v = (1.0 / a) * dpsi_dlambda
    bias_u, bias_v = _western_boundary_enhancement(lon_b, lat_b)
    u += bias_u
    v += bias_v

    if config.ekman_enabled and config.ekman_alpha > 0.0:
        u_ek, v_ek = _ekman_drift(lon_b, lat_b, time_days, config=config)
        u += u_ek
        v += v_ek
    return u, v
