from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict

import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .sim import init_particles, run_simulation

@dataclass
class EnsembleResult:
    csv: pd.DataFrame
    lat_mean: np.ndarray
    lon_mean: np.ndarray
    final_lat_all: np.ndarray
    final_lon_all: np.ndarray
    message_fn: Callable[[datetime], str]

def _km_radius(samples_lat: np.ndarray, samples_lon: np.ndarray) -> float:
    lat0 = float(np.mean(samples_lat))
    # convert to km at that latitude
    dx = (samples_lon - np.mean(samples_lon)) * 111.0 * np.cos(np.deg2rad(lat0))
    dy = (samples_lat - np.mean(samples_lat)) * 111.0
    r = np.sqrt(dx * dx + dy * dy)
    return float(np.percentile(r, 90))

def _radius_at_percentile(samples_lat: np.ndarray, samples_lon: np.ndarray, pct: float) -> float:
    lat0 = float(np.mean(samples_lat))
    dx = (samples_lon - np.mean(samples_lon)) * 111.0 * np.cos(np.deg2rad(lat0))
    dy = (samples_lat - np.mean(samples_lat)) * 111.0
    r = np.sqrt(dx * dx + dy * dy)
    return float(np.percentile(r, pct))

def run_ensemble_forecast(
    cfg: Config,
    currents: xr.Dataset,
    winds: xr.Dataset,
    start: datetime,
    end: datetime,
    start_lat: float,
    start_lon: float,
    n_particles: int = 100,
    runs: int = 50,
    perturbation_scale: float = 0.1,
) -> Dict[str, object]:
    rows = []
    means_lat = []
    means_lon = []
    finals_lat = []
    finals_lon = []
    for i in range(runs):
        # perturb physics
        cfg_i = Config(
            region_north=cfg.region_north,
            region_south=cfg.region_south,
            region_west=cfg.region_west,
            region_east=cfg.region_east,
            dt_hours=cfg.dt_hours,
            diffusion_kappa=cfg.diffusion_kappa * (1.0 + np.random.uniform(-perturbation_scale, perturbation_scale)),
            alpha_wind=cfg.alpha_wind * (1.0 + np.random.uniform(-perturbation_scale, perturbation_scale)),
            stokes_scale=cfg.stokes_scale,
        )
        state = init_particles(cfg_i, n=n_particles, start_points=np.array([[start_lat, start_lon]] * n_particles))
        traj = run_simulation(cfg_i, currents, winds, start, end, state)
        # collect
        T, N = traj["lat"].shape
        means_lat.append(traj["lat"].mean(axis=1))  # shape (T,)
        means_lon.append(traj["lon"].mean(axis=1))
        finals_lat.append(traj["lat"][-1])
        finals_lon.append(traj["lon"][-1])
        df = pd.DataFrame({
            "time": np.repeat(traj["time"][-1], N),
            "run": i,
            "lat": traj["lat"][-1],
            "lon": traj["lon"][-1],
        })
        rows.append(df)

    csv = pd.concat(rows, ignore_index=True)
    lat_mean = np.stack(means_lat)  # (runs, T)
    lon_mean = np.stack(means_lon)  # (runs, T)
    final_lat_all = np.concatenate(finals_lat)  # (runs*N,)
    final_lon_all = np.concatenate(finals_lon)

    def message_fn(target_time: datetime) -> str:
        # nearest mean at target_time is the same as end here, so report end
        lat_c = float(np.mean(final_lat_all))
        lon_c = float(np.mean(final_lon_all))
        r50 = _radius_at_percentile(final_lat_all, final_lon_all, 50.0)
        r90 = _radius_at_percentile(final_lat_all, final_lon_all, 90.0)
        return (
            f"At {target_time.isoformat()} UTC the predicted center is near "
            f"{lat_c:.2f}°, {lon_c:.2f}°, with ~{r50:.1f} km (50%) and ~{r90:.1f} km (90%) radii."
        )

    return {
        "csv": csv,
        "lat_mean": lat_mean.mean(axis=0),   # average across runs, shape (T,)
        "lon_mean": lon_mean.mean(axis=0),
        "final_lat_all": final_lat_all,
        "final_lon_all": final_lon_all,
        "message_fn": message_fn,
    }
