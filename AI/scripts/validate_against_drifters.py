# DriftCast validation CLI.
# Compares simulated trajectories against observed drifters and reports RMSE metrics.
# Ideal for quick sanity checks after training or data updates.

"""
Validation helper comparing simulator trajectories against drifter observations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src import data_loader, error_utils, simulator

LOGGER = logging.getLogger("validate")


def _prepare_env(currents: str | None, winds: str | None) -> Dict[str, object]:
    env: Dict[str, object] = {}
    if currents and Path(currents).exists():
        env["currents"] = data_loader.load_currents(currents)
    else:
        env["currents"] = data_loader.synthetic_currents()
    if winds and Path(winds).exists():
        env["winds"] = data_loader.load_winds(winds)
    else:
        env["winds"] = data_loader.synthetic_winds()
    env["windage_coeff"] = 0.02
    env["diffusivity"] = 40.0
    return env


def _group_drifters(drifters: data_loader.DrifterTrajectories) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    ids = np.asarray(drifters.ids)
    lats = np.asarray(drifters.lat)
    lons = np.asarray(drifters.lon)
    unique_ids = np.unique(ids)
    for drifter_id in unique_ids:
        mask = ids == drifter_id
        if mask.sum() < 2:
            continue
        yield int(drifter_id), lats[mask], lons[mask]


def validate(drifters_csv: str, currents: str | None, winds: str | None, dt_hours: float = 1.0) -> float:
    drifters = data_loader.load_drifters(drifters_csv)
    env = _prepare_env(currents, winds)

    observed_tracks = []
    predicted_tracks = []

    for drifter_id, lat_obs, lon_obs in _group_drifters(drifters):
        state = simulator.init_particles(n=1, start_pos=(float(lat_obs[0]), float(lon_obs[0])))
        positions, _ = simulator.run_simulation(state, env, steps=len(lat_obs) - 1, dt_hours=dt_hours)
        pred_lat = np.asarray(positions["lat"][:, 0])
        pred_lon = np.asarray(positions["lon"][:, 0])
        observed_tracks.extend(np.stack([lat_obs, lon_obs], axis=1))
        predicted_tracks.extend(np.stack([pred_lat, pred_lon], axis=1))
        LOGGER.debug("Drifter %s processed (%d steps).", drifter_id, len(lat_obs))

    observed_arr = np.asarray(observed_tracks)
    predicted_arr = np.asarray(predicted_tracks)

    metrics = error_utils.compute_error(observed_arr, predicted_arr)
    rmse_km = metrics["rmse"]
    steps = max(1, len(observed_tracks))
    days = (steps * dt_hours) / 24.0
    rmse_per_day = rmse_km / days
    LOGGER.info("Model error: %.2f km/day (RMSE %.2f km).", rmse_per_day, rmse_km)
    return rmse_per_day


def main():
    parser = argparse.ArgumentParser(description="Validate DriftCast simulator against drifter data.")
    parser.add_argument("--drifters", required=True, help="Path to drifter CSV file.")
    parser.add_argument("--currents", help="Optional NetCDF currents file.", default=None)
    parser.add_argument("--winds", help="Optional NetCDF winds file.", default=None)
    parser.add_argument("--dt-hours", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate(args.drifters, args.currents, args.winds, dt_hours=args.dt_hours)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
