# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Implements particle state containers and motion integrators for driftcast.
- Combines advection, diffusion, windage, Stokes drift, and beaching toggles.
- Coupled with driftcast.sim.runner to step ensembles over simulation windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from driftcast import logger
from driftcast.config import BeachingConfig
from driftcast.core.units import deg_lat_per_meter, deg_lon_per_meter

VelocityField = Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
MaskFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ParticleState:
    """Hold particle position and status arrays."""

    particle_id: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    age_days: np.ndarray
    beached: np.ndarray
    beached_time_days: np.ndarray
    class_name: np.ndarray
    source_name: np.ndarray

    def active_indices(self) -> np.ndarray:
        """Indices of particles currently in the water."""
        return np.where(~self.beached)[0]


def initialize_state(
    lon: Iterable[float],
    lat: Iterable[float],
    class_name: Iterable[str],
    source_name: Iterable[str],
    ids: Optional[Iterable[int]] = None,
) -> ParticleState:
    """Create an initial particle state object from iterables."""
    lon_arr = np.asarray(list(lon), dtype=float)
    lat_arr = np.asarray(list(lat), dtype=float)
    class_arr = np.asarray(list(class_name), dtype=object)
    source_arr = np.asarray(list(source_name), dtype=object)
    n = lon_arr.size
    if not (lat_arr.size == class_arr.size == source_arr.size == n):
        raise ValueError("All input arrays must share the same length")
    if ids is None:
        particle_id = np.arange(n, dtype=int)
    else:
        particle_id = np.asarray(list(ids), dtype=int)
    return ParticleState(
        particle_id=particle_id,
        lon=lon_arr,
        lat=lat_arr,
        age_days=np.zeros(n, dtype=float),
        beached=np.zeros(n, dtype=bool),
        beached_time_days=np.full(n, np.nan),
        class_name=class_arr,
        source_name=source_arr,
    )


def euler_maruyama_step(
    state: ParticleState,
    time_days: float,
    dt_seconds: float,
    velocity_fn: VelocityField,
    rng: np.random.Generator,
    diffusivity: float,
    windage_coeff: float,
    wind_fn: Optional[VelocityField] = None,
    stokes_coeff: float = 0.0,
    stokes_fn: Optional[VelocityField] = None,
    land_mask: Optional[MaskFunction] = None,
    beaching: Optional[BeachingConfig] = None,
    grid_spacing_deg: Optional[float] = None,
    cfl_state: Optional[Dict[str, bool]] = None,
) -> float:
    """Advance particle positions one Euler-Maruyama step in-place.

    Returns:
        The actual time step applied (seconds) after CFL safety adjustments.
    """
    active = state.active_indices()
    if active.size == 0:
        return dt_seconds
    lon = state.lon[active]
    lat = state.lat[active]
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

    deg_per_meter_lon = deg_lon_per_meter(lat)
    deg_per_meter_lat = deg_lat_per_meter(lat)

    step_dt = float(dt_seconds)
    if grid_spacing_deg is not None and active.size:
        adv_lon = np.abs(total_u) * step_dt * deg_per_meter_lon
        adv_lat = np.abs(total_v) * step_dt * deg_per_meter_lat
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

    delta_lon = total_u * step_dt * deg_per_meter_lon
    delta_lat = total_v * step_dt * deg_per_meter_lat

    if diffusivity > 0.0:
        sigma = np.sqrt(2.0 * diffusivity * step_dt)
        xi_lon = rng.normal(size=active.size)
        xi_lat = rng.normal(size=active.size)
        delta_lon += sigma * xi_lon * deg_per_meter_lon
        delta_lat += sigma * xi_lat * deg_per_meter_lat

    new_lon = (lon + delta_lon + 180.0) % 360.0 - 180.0
    new_lat = np.clip(lat + delta_lat, -89.9, 89.9)
    state.lon[active] = new_lon
    state.lat[active] = new_lat
    state.age_days[active] += step_dt / 86400.0

    if land_mask is not None and beaching is not None:
        apply_beaching(
            state,
            indices=active,
            land_mask=land_mask,
            beaching=beaching,
            rng=rng,
            time_days=time_days,
            dt_days=step_dt / 86400.0,
        )

    return step_dt


def apply_beaching(
    state: ParticleState,
    indices: np.ndarray,
    land_mask: MaskFunction,
    beaching: BeachingConfig,
    rng: np.random.Generator,
    time_days: float,
    dt_days: float,
) -> None:
    """Resolve shoreline interactions for the provided indices."""
    lon = state.lon[indices]
    lat = state.lat[indices]
    on_land = land_mask(lon, lat)
    if np.any(on_land):
        beach_draw = rng.uniform(size=np.count_nonzero(on_land))
        to_beach = beach_draw < beaching.probability
        land_indices = indices[on_land]
        newly_beached = land_indices[to_beach]
        state.beached[newly_beached] = True
        state.beached_time_days[newly_beached] = time_days
        # Slightly nudge beached particles back toward coastline to avoid NaNs.
        state.lat[newly_beached] = np.clip(state.lat[newly_beached], -89.0, 89.0)

    # Handle resuspension as a Poisson process.
    if beaching.resuspension_days is None:
        return
    beached_idxs = np.where(state.beached)[0]
    if beached_idxs.size == 0:
        return
    dt_days = max(dt_days, 1e-6)
    release_prob = 1.0 - np.exp(-dt_days / beaching.resuspension_days)
    release_draw = rng.uniform(size=beached_idxs.size)
    to_release = beached_idxs[release_draw < release_prob]
    if to_release.size:
        state.beached[to_release] = False
        state.beached_time_days[to_release] = np.nan


def append_particles(
    state: ParticleState,
    particle_id: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    class_names: Sequence[str],
    source_names: Sequence[str],
) -> None:
    """Append new particles to the state in-place."""
    if lon.size == 0:
        return
    state.particle_id = np.concatenate([state.particle_id, particle_id.astype(int)])
    state.lon = np.concatenate([state.lon, lon.astype(float)])
    state.lat = np.concatenate([state.lat, lat.astype(float)])
    state.age_days = np.concatenate([state.age_days, np.zeros(lon.size)])
    state.beached = np.concatenate([state.beached, np.zeros(lon.size, dtype=bool)])
    state.beached_time_days = np.concatenate(
        [state.beached_time_days, np.full(lon.size, np.nan)]
    )
    state.class_name = np.concatenate(
        [state.class_name, np.asarray(class_names, dtype=object)]
    )
    state.source_name = np.concatenate(
        [state.source_name, np.asarray(source_names, dtype=object)]
    )
