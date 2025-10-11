# DriftCast smoke tests covering loaders, simulator, RL envs, and uncertainty utils.
# Keeps fixtures lightweight so CI can run quickly on CPU-only machines.
# Skips gracefully if optional dependencies (cartopy, imageio) are absent.

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import DataConfig
from src.data_loader import build_forcing_reader, load_currents, load_drifters, load_winds
from src.error_utils import compute_error, monte_carlo_ensemble
from src.rl_cleanup import CleanupEnv
from src.rl_drift_correction import DriftCorrectionEnv
from src.simulator import init_particles, run_simulation


@pytest.fixture()
def sample_config(tmp_path: Path) -> DataConfig:
    hycom = tmp_path / "hycom"
    era5 = tmp_path / "era5"
    drifters_dir = tmp_path / "drifters"
    for directory in (hycom, era5, drifters_dir):
        directory.mkdir(parents=True, exist_ok=True)

    time = np.array(["2024-07-01T00:00:00", "2024-07-01T06:00:00"], dtype="datetime64[ns]")
    lat = np.linspace(10, 20, 4)
    lon = np.linspace(-60, -50, 4)
    ds_curr = xr.Dataset(
        {
            "uo": (("time", "lat", "lon"), np.ones((2, 4, 4))),
            "vo": (("time", "lat", "lon"), np.zeros((2, 4, 4))),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds_curr.to_netcdf(hycom / "currents.nc")

    ds_wind = xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), np.full((2, 4, 4), 5.0)),
            "v10": (("time", "lat", "lon"), np.full((2, 4, 4), -2.0)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds_wind.to_netcdf(era5 / "winds.nc")

    drifter_df = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01", periods=8, freq="H", tz="UTC"),
            "latitude": np.linspace(12, 18, 8),
            "longitude": np.linspace(-58, -54, 8),
            "id": 0,
        }
    )
    drifter_df.to_csv(drifters_dir / "drifters.csv", index=False)

    return DataConfig(
        hycom_root=hycom,
        era5_root=era5,
        waves_root=tmp_path / "waves",
        drifters_root=drifters_dir,
        region_north=20,
        region_south=10,
        region_west=-60,
        region_east=-50,
        dt_hours=1.0,
        n_particles_default=200,
    )


def test_loaders(sample_config: DataConfig):
    currents = load_currents(sample_config)
    winds = load_winds(sample_config)
    drifters = load_drifters(sample_config)
    assert currents is not None and "uo" in currents
    assert winds is not None and "u10" in winds
    assert not drifters.empty


def test_simulator_step(sample_config: DataConfig):
    reader = build_forcing_reader(sample_config)
    state = init_particles(sample_config, n=8)
    start = datetime(2024, 7, 1)
    traj = run_simulation(reader, sample_config, start, start + timedelta(hours=6), state)
    assert traj["lat"].shape[0] >= 2
    assert traj["lon"].shape == traj["lat"].shape


def test_rl_envs(sample_config: DataConfig):
    drift_env = DriftCorrectionEnv(sample_config, horizon_hours=6, dt_hours=1.0)
    obs, _ = drift_env.reset()
    assert drift_env.observation_space.contains(obs)
    action = drift_env.action_space.sample()
    obs, reward, terminated, truncated, info = drift_env.step(action)
    assert drift_env.observation_space.contains(obs)
    assert isinstance(reward, float)
    cleanup_env = CleanupEnv(sample_config, n_assets=1, grid_shape=(8, 8))
    obs, _ = cleanup_env.reset()
    assert cleanup_env.observation_space.contains(obs)
    action = cleanup_env.action_space.sample()
    obs, reward, terminated, truncated, info = cleanup_env.step(action)
    assert cleanup_env.observation_space.contains(obs)
    assert isinstance(info["plastic_removed"], float)


def test_error_metrics(sample_config: DataConfig):
    reader = build_forcing_reader(sample_config)
    state = init_particles(sample_config, n=5)
    now = datetime(2024, 7, 1)
    ensemble = monte_carlo_ensemble(sample_config, now, now + timedelta(hours=6), sample_config.dt_hours, state, n_runs=3)
    assert ensemble.shape[0] == 3
    metrics = compute_error(np.zeros((5, 2)), np.ones((5, 2)))
    assert pytest.approx(metrics["rmse"]) == np.sqrt(2.0)

