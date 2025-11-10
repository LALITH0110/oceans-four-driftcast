# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Supplies integration parameter dataclass and factory to obtain stepping kernels.
- Implements deterministic Euler and RK4 variants plus stochastic Euler-Maruyama.
- Consumed by driftcast.sim.runner when advancing particle ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

from driftcast import logger
from driftcast.core.units import deg_lat_per_meter, deg_lon_per_meter
from driftcast.particles.physics import (
    MaskFunction,
    ParticleState,
    VelocityField,
    euler_maruyama_step,
)

IntegratorFn = Callable[..., float]


@dataclass
class IntegrationParameters:
    """Configuration for the particle integrator."""

    method: Literal["euler", "euler_maruyama", "rk4"] = "euler_maruyama"
    dt_minutes: float = 30.0
    grid_spacing_deg: float = 0.25

    @property
    def dt_seconds(self) -> float:
        """Return the fixed time step in seconds."""
        return self.dt_minutes * 60.0


def _composite_velocity(
    velocity_fn: VelocityField,
    wind_fn: Optional[VelocityField],
    stokes_fn: Optional[VelocityField],
    windage_coeff: float,
    stokes_coeff: float,
) -> Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """Summation of base, wind, and Stokes velocities evaluated lazily."""

    def closure(lon: np.ndarray, lat: np.ndarray, time_days: float) -> Tuple[np.ndarray, np.ndarray]:
        u, v = velocity_fn(lon, lat, time_days)
        total_u = np.array(u, dtype=float)
        total_v = np.array(v, dtype=float)
        if wind_fn is not None and windage_coeff > 0.0:
            wu, wv = wind_fn(lon, lat, time_days)
            total_u += windage_coeff * np.asarray(wu, dtype=float)
            total_v += windage_coeff * np.asarray(wv, dtype=float)
        if stokes_fn is not None and stokes_coeff > 0.0:
            su, sv = stokes_fn(lon, lat, time_days)
            total_u += stokes_coeff * np.asarray(su, dtype=float)
            total_v += stokes_coeff * np.asarray(sv, dtype=float)
        return total_u, total_v

    return closure


def _euler_step(
    state: ParticleState,
    time_days: float,
    dt_seconds: float,
    velocity_fn: VelocityField,
    rng: np.random.Generator,
    diffusivity: float,
    windage_coeff: float,
    wind_fn: Optional[VelocityField],
    stokes_coeff: float,
    stokes_fn: Optional[VelocityField],
    land_mask: Optional[MaskFunction],
    beaching,
    grid_spacing_deg: Optional[float],
    cfl_state: Optional[Dict[str, bool]],
) -> float:
    """Deterministic Euler step with optional diffusion."""
    return euler_maruyama_step(
        state=state,
        time_days=time_days,
        dt_seconds=dt_seconds,
        velocity_fn=velocity_fn,
        rng=rng,
        diffusivity=diffusivity,
        windage_coeff=windage_coeff,
        wind_fn=wind_fn,
        stokes_coeff=stokes_coeff,
        stokes_fn=stokes_fn,
        land_mask=land_mask,
        beaching=beaching,
        grid_spacing_deg=grid_spacing_deg,
        cfl_state=cfl_state,
    )


def _rk4_step(
    state: ParticleState,
    time_days: float,
    dt_seconds: float,
    velocity_fn: VelocityField,
    rng: np.random.Generator,
    diffusivity: float,
    windage_coeff: float,
    wind_fn: Optional[VelocityField],
    stokes_coeff: float,
    stokes_fn: Optional[VelocityField],
    land_mask: Optional[MaskFunction],
    beaching,
    grid_spacing_deg: Optional[float],
    cfl_state: Optional[Dict[str, bool]],
) -> float:
    """Fourth-order Runge-Kutta for deterministic part plus additive diffusion."""
    indices = state.active_indices()
    if indices.size == 0:
        return dt_seconds

    composite = _composite_velocity(
        velocity_fn=velocity_fn,
        wind_fn=wind_fn,
        stokes_fn=stokes_fn,
        windage_coeff=windage_coeff,
        stokes_coeff=stokes_coeff,
    )
    lon0 = state.lon[indices].copy()
    lat0 = state.lat[indices].copy()
    dt_days = dt_seconds / 86400.0

    u1, v1 = composite(lon0, lat0, time_days)
    lon2 = lon0 + 0.5 * dt_seconds * u1 * deg_lon_per_meter(lat0)
    lat2 = lat0 + 0.5 * dt_seconds * v1 * deg_lat_per_meter(lat0)

    u2, v2 = composite(lon2, lat2, time_days + 0.5 * dt_days)
    lon3 = lon0 + 0.5 * dt_seconds * u2 * deg_lon_per_meter(lat2)
    lat3 = lat0 + 0.5 * dt_seconds * v2 * deg_lat_per_meter(lat2)

    u3, v3 = composite(lon3, lat3, time_days + 0.5 * dt_days)
    lon4 = lon0 + dt_seconds * u3 * deg_lon_per_meter(lat3)
    lat4 = lat0 + dt_seconds * v3 * deg_lat_per_meter(lat3)

    u4, v4 = composite(lon4, lat4, time_days + dt_days)

    mean_u = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
    mean_v = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

    step_dt = float(dt_seconds)
    if grid_spacing_deg is not None and indices.size:
        deg_per_meter_lon = deg_lon_per_meter(lat0)
        deg_per_meter_lat = deg_lat_per_meter(lat0)
        adv_lon = np.abs(mean_u) * step_dt * deg_per_meter_lon
        adv_lat = np.abs(mean_v) * step_dt * deg_per_meter_lat
        max_adv = float(np.max(np.maximum(adv_lon, adv_lat)))
        if np.isfinite(max_adv) and max_adv > grid_spacing_deg:
            scale = max(grid_spacing_deg / (max_adv + 1e-12), 0.05)
            step_dt = dt_seconds * scale
            if cfl_state is not None and not cfl_state.get("warned", False):
                logger.warning(
                    "CFL safety triggered: reducing dt from %.2f s to %.2f s "
                    "(max drift %.3f deg exceeds grid %.3f deg).",
                    dt_seconds,
                    step_dt,
                    max_adv,
                    grid_spacing_deg,
                )
                cfl_state["warned"] = True

    lon_update = lon0 + step_dt * mean_u * deg_lon_per_meter(lat0)
    lat_update = lat0 + step_dt * mean_v * deg_lat_per_meter(lat0)

    state.lon[indices] = (lon_update + 180.0) % 360.0 - 180.0
    state.lat[indices] = np.clip(lat_update, -89.9, 89.9)
    state.age_days[indices] += step_dt / 86400.0

    if diffusivity > 0.0:
        euler_maruyama_step(
            state=state,
            time_days=time_days,
            dt_seconds=step_dt,
            velocity_fn=lambda lon, lat, t: (np.zeros_like(lon), np.zeros_like(lat)),
            rng=rng,
            diffusivity=diffusivity,
            windage_coeff=0.0,
            wind_fn=None,
            stokes_coeff=0.0,
            stokes_fn=None,
            land_mask=land_mask,
            beaching=beaching,
            grid_spacing_deg=None,
            cfl_state=None,
        )

    return step_dt


def build_integrator(params: IntegrationParameters) -> IntegratorFn:
    """Return the integration function corresponding to ``params.method``."""
    if params.method == "rk4":
        return partial(_rk4_step, grid_spacing_deg=params.grid_spacing_deg)
    if params.method == "euler":
        return partial(_euler_step, grid_spacing_deg=params.grid_spacing_deg)
    return partial(_euler_step, grid_spacing_deg=params.grid_spacing_deg)
