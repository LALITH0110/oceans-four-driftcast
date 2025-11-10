# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Coordinates particle seeding, advection, and output for a single simulation run.
- Wires configuration, field generators, and integrators into a cohesive workflow.
- Returns an xarray.Dataset and optionally writes NetCDF/Zarr diagnostics.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json

import numpy as np
import pandas as pd
import xarray as xr
from driftcast import logger

from driftcast.config import SimulationConfig
from driftcast.core.run_manifest import build_manifest, update_dataset_attrs, write_manifest
from driftcast.fields.gyres import GyreFieldConfig, gyre_velocity_field
from driftcast.fields.stokes import StokesConfig, stokes_drift_velocity
from driftcast.fields.winds import WindFieldConfig, seasonal_wind_field
from driftcast.particles.physics import append_particles, initialize_state
from driftcast.sim.integrators import IntegrationParameters, build_integrator
from driftcast.sources import BaseSource, REGISTRY


def _build_land_mask(config: SimulationConfig) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    dom = config.domain

    def mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return (
            (lon <= dom.lon_min)
            | (lon >= dom.lon_max)
            | (lat <= dom.lat_min)
            | (lat >= dom.lat_max)
        )

    return mask


def _initialize_sources(config: SimulationConfig, seed_seq: np.random.SeedSequence) -> List[BaseSource]:
    child_seqs = seed_seq.spawn(len(config.sources))
    sources: List[BaseSource] = []
    for idx, source_cfg in enumerate(config.sources):
        cls = REGISTRY[source_cfg.type]
        rng = np.random.default_rng(child_seqs[idx])
        sources.append(cls(source_cfg, rng))
    return sources


def run_simulation(
    config: SimulationConfig,
    output_path: Optional[Path] = None,
    seed: Optional[int] = None,
    write_manifest_sidecar: bool = True,
) -> xr.Dataset:
    """Execute a single simulation defined by ``config``."""
    logger.info("Starting simulation %s -> %s", config.time.start, config.time.end)
    time_cfg = config.time
    physics = config.physics
    params = IntegrationParameters(
        method="euler_maruyama",
        dt_minutes=time_cfg.dt_minutes,
        grid_spacing_deg=config.domain.resolution_deg,
    )
    base_dt_seconds = params.dt_seconds
    integrator = build_integrator(params)

    base_seed = seed if seed is not None else int(time_cfg.start.timestamp()) & 0xFFFFFFFF
    master_seq = np.random.SeedSequence(base_seed)
    step_rng = np.random.default_rng(master_seq.spawn(1)[0])
    sources = _initialize_sources(config, master_seq)

    state = initialize_state([], [], [], [], ids=[])
    next_id = 0

    land_mask = _build_land_mask(config)
    seasonal_cfg = physics.seasonal
    ekman_cfg = physics.ekman
    gyre_cfg = GyreFieldConfig(
        seasonal_enabled=seasonal_cfg.enabled,
        seasonal_amp=seasonal_cfg.amplitude,
        base_time_days=seasonal_cfg.phase_day,
        ekman_enabled=ekman_cfg.enabled,
        ekman_alpha=ekman_cfg.alpha,
        ekman_drag_coeff=ekman_cfg.drag_coefficient,
        ekman_air_density=ekman_cfg.air_density,
        ekman_water_density=ekman_cfg.water_density,
    )
    wind_cfg = WindFieldConfig()
    stokes_cfg = StokesConfig()

    def velocity_fn(lon: np.ndarray, lat: np.ndarray, time_days: float):
        return gyre_velocity_field(lon, lat, time_days, config=gyre_cfg)

    def wind_fn(lon: np.ndarray, lat: np.ndarray, time_days: float):
        return seasonal_wind_field(lon, lat, time_days, config=wind_cfg)

    def stokes_fn(lon: np.ndarray, lat: np.ndarray, time_days: float):
        return stokes_drift_velocity(lon, lat, time_days, config=stokes_cfg)

    snapshots: List[dict] = []
    snapshot_times: List[pd.Timestamp] = []
    output_interval_seconds = time_cfg.output_interval_hours * 3600.0
    seconds_since_last_save = output_interval_seconds

    current_time = time_cfg.start
    end_time = time_cfg.end
    elapsed_seconds = 0.0
    cfl_state: Dict[str, bool] = {"warned": False}

    while current_time < end_time:
        remaining = (end_time - current_time).total_seconds()
        target_dt = min(base_dt_seconds, remaining)
        time_days = elapsed_seconds / 86400.0
        dt_days = target_dt / 86400.0

        for src in sources:
            lon, lat, class_names, source_names = src.emit(time_days, dt_days)
            if lon.size:
                ids = np.arange(next_id, next_id + lon.size, dtype=int)
                append_particles(state, ids, lon, lat, class_names, source_names)
                next_id += lon.size

        actual_dt = target_dt
        if state.lon.size:
            actual_dt = integrator(
                state=state,
                time_days=time_days,
                dt_seconds=target_dt,
                velocity_fn=velocity_fn,
                rng=step_rng,
                diffusivity=physics.diffusivity_m2s,
                windage_coeff=physics.windage_coeff,
                wind_fn=wind_fn,
                stokes_coeff=physics.stokes_coeff,
                stokes_fn=stokes_fn,
                land_mask=land_mask,
                beaching=physics.beaching,
                cfl_state=cfl_state,
            )

        elapsed_seconds += actual_dt
        current_time += timedelta(seconds=actual_dt)
        seconds_since_last_save += actual_dt

        if seconds_since_last_save >= output_interval_seconds or current_time >= end_time:
            snapshots.append(
                {
                    "particle_id": state.particle_id.copy(),
                    "lon": state.lon.copy(),
                    "lat": state.lat.copy(),
                    "age_days": state.age_days.copy(),
                    "beached": state.beached.copy(),
                    "class_name": state.class_name.copy(),
                    "source_name": state.source_name.copy(),
                }
            )
            snapshot_times.append(pd.Timestamp(current_time))
            seconds_since_last_save = 0.0

    if snapshots:
        n_particles = next_id
        n_times = len(snapshots)
        lon_data = np.full((n_times, n_particles), np.nan, dtype=float)
        lat_data = np.full_like(lon_data, np.nan)
        age_data = np.full_like(lon_data, np.nan)
        beached_data = np.zeros((n_times, n_particles), dtype=bool)
        class_per_particle = np.full(n_particles, "", dtype=object)
        source_per_particle = np.full(n_particles, "", dtype=object)

        for t_idx, snap in enumerate(snapshots):
            ids = snap["particle_id"]
            lon_data[t_idx, ids] = snap["lon"]
            lat_data[t_idx, ids] = snap["lat"]
            age_data[t_idx, ids] = snap["age_days"]
            beached_data[t_idx, ids] = snap["beached"]
            for pid, cname, sname in zip(ids, snap["class_name"], snap["source_name"]):
                if not class_per_particle[pid]:
                    class_per_particle[pid] = cname
                if not source_per_particle[pid]:
                    source_per_particle[pid] = sname

        class_coord = np.array(class_per_particle, dtype="U32")
        source_coord = np.array(source_per_particle, dtype="U32")

        time_values = [
            ts.tz_convert("UTC").tz_localize(None) if getattr(ts, "tzinfo", None) else ts
            for ts in snapshot_times
        ]
        time_coord = np.array(time_values, dtype="datetime64[ns]")

        dataset = xr.Dataset(
            data_vars={
                "lon": (("time", "particle"), lon_data),
                "lat": (("time", "particle"), lat_data),
                "age_days": (("time", "particle"), age_data),
                "beached": (("time", "particle"), beached_data),
            },
            coords={
                "time": time_coord,
                "particle": np.arange(n_particles),
                "class_name": ("particle", class_coord),
                "source_name": ("particle", source_coord),
            },
        )
    else:
        dataset = xr.Dataset()

    if config.output.chunks:
        dataset = dataset.chunk(config.output.chunks)

    dataset.attrs["gyre_box"] = json.dumps(
        {
            "lon_min": config.gyre_box.lon_min,
            "lon_max": config.gyre_box.lon_max,
            "lat_min": config.gyre_box.lat_min,
            "lat_max": config.gyre_box.lat_max,
        }
    )

    if output_path is None:
        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.output.format == "netcdf":
            output_path = output_dir / "simulation.nc"
        else:
            output_path = output_dir / "simulation.zarr"

    particle_counts = {
        "emitted": int(next_id),
        "active": int((~state.beached).sum()),
        "beached": int(state.beached.sum()),
    }
    manifest_payload = build_manifest(
        config=config,
        output_path=output_path,
        seed=base_seed,
        particle_counts=particle_counts,
        extra_checks={"cfl_reduction": cfl_state["warned"]},
    )
    update_dataset_attrs(dataset, manifest_payload)

    if config.output.format == "netcdf":
        dataset.to_netcdf(output_path)
    else:
        dataset.to_zarr(output_path, mode="w")

    if write_manifest_sidecar:
        write_manifest(manifest_payload)
        logger.info("Wrote manifest to %s", manifest_payload.path)
    logger.info("Simulation outputs written to %s", output_path)
    return dataset
