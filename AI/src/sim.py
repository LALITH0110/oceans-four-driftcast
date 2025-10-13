from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from .config import Config
from .data import sample_vector

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEG_LAT = (math.pi / 180.0) * EARTH_RADIUS_M

@dataclass
class State:
    lat: np.ndarray  # shape (N,)
    lon: np.ndarray  # shape (N,)

def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0

def init_particles(cfg: Config, n: int, start_points: Optional[np.ndarray] = None, seed: Optional[int] = None) -> State:
    rng = np.random.default_rng(seed)
    if start_points is not None and len(start_points) >= n:
        pts = np.asarray(start_points[:n], dtype=np.float32)
        lat = pts[:, 0].astype(np.float32)
        lon = pts[:, 1].astype(np.float32)
    else:
        lat = rng.uniform(cfg.region_south, cfg.region_north, size=n).astype(np.float32)
        lon = rng.uniform(cfg.region_west, cfg.region_east, size=n).astype(np.float32)
    return State(lat=lat, lon=lon)

def step(
    cfg: Config,
    currents: xr.Dataset,
    winds: xr.Dataset,
    time,
    state: State
) -> State:
    lat = state.lat
    lon = state.lon
    # Interpolate fields at particle locations
    uo, vo = sample_vector(currents,
                           ("uo", "eastward_sea_water_velocity", "u"),
                           ("vo", "northward_sea_water_velocity", "v"),
                           time, lat, lon)
    u10, v10 = sample_vector(winds,
                             ("u10", "u_component_of_wind"),
                             ("v10", "v_component_of_wind"),
                             time, lat, lon)
    # total velocity (m/s)
    u_tot = uo + cfg.alpha_wind * u10 + cfg.stokes_scale * u10
    v_tot = vo + cfg.alpha_wind * v10 + cfg.stokes_scale * v10
    dt = cfg.dt_hours * 3600.0
    # advection
    dlat = (v_tot * dt) / METERS_PER_DEG_LAT
    coslat = np.cos(np.deg2rad(lat)).clip(1e-3, None)
    meters_per_deg_lon = METERS_PER_DEG_LAT * coslat
    dlon = (u_tot * dt) / meters_per_deg_lon
    # diffusion
    sigma = np.sqrt(2.0 * cfg.diffusion_kappa * dt) / METERS_PER_DEG_LAT
    turb_lat = sigma * np.random.normal(size=lat.shape).astype(np.float32)
    turb_lon = sigma * np.random.normal(size=lon.shape).astype(np.float32)
    lat_new = lat + dlat + turb_lat
    lon_new = wrap_lon(lon + dlon + turb_lon)
    return State(lat=lat_new.astype(np.float32), lon=lon_new.astype(np.float32))

def run_simulation(
    cfg: Config,
    currents: xr.Dataset,
    winds: xr.Dataset,
    start_time: datetime,
    end_time: datetime,
    state: State
) -> Dict[str, np.ndarray]:
    times = []
    t = start_time
    while t <= end_time + timedelta(seconds=1):
        times.append(t)
        t += timedelta(hours=cfg.dt_hours)
    lat_hist = [state.lat.copy()]
    lon_hist = [state.lon.copy()]
    for tt in times[1:]:
        state = step(cfg, currents, winds, tt, state)
        lat_hist.append(state.lat.copy())
        lon_hist.append(state.lon.copy())
    return {
        "time": np.array(times, dtype="datetime64[ns]"),
        "lat": np.stack(lat_hist),     # shape (T, N)
        "lon": np.stack(lon_hist),     # shape (T, N)
    }
