# DriftCast uncertainty and error analysis helpers.
# Provides Monte Carlo ensembles, RMSE/ellipse metrics, and textual forecast summaries.
# Shared across CLI commands, pipeline evaluations, and reporting utilities.

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from .data_loader import ForcingReader, build_forcing_reader
import jax.numpy as jnp

from .config import DataConfig
from .simulator import ParticleState, init_particles, run_simulation

LOGGER = logging.getLogger(__name__)


def monte_carlo_ensemble(
    cfg: DataConfig,
    start_time: datetime,
    end_time: datetime,
    dt_hours: float,
    base_state: ParticleState,
    n_runs: int = 20,
    perturbation_scale: float = 0.1,
) -> np.ndarray:
    """Run multiple simulator realisations under perturbed physics parameters."""
    reader = build_forcing_reader(cfg)
    trajectories = []
    for _ in range(n_runs):
        scaled_cfg = cfg.copy(
            update={
                "alpha_wind": cfg.alpha_wind * (1.0 + np.random.uniform(-perturbation_scale, perturbation_scale)),
                "diffusion_kappa": cfg.diffusion_kappa * (1.0 + np.random.uniform(-perturbation_scale, perturbation_scale)),
            }
        )
        perturb_state = ParticleState(
            lat=base_state.lat.copy(),
            lon=base_state.lon.copy(),
            zmix=base_state.zmix.copy(),
            key=jnp.array(np.random.randint(0, 2**16), dtype=jnp.uint32),
        )
        traj = run_simulation(reader, scaled_cfg, start_time, end_time, perturb_state)
        trajectories.append(np.stack([np.asarray(traj["lat"]), np.asarray(traj["lon"])], axis=-1))
    return np.stack(trajectories)


def compute_error(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, np.ndarray]:
    """Return RMSE, mean residual, and covariance between observed/predicted tracks."""
    obs = np.asarray(observed)
    pred = np.asarray(predicted)
    n = min(len(obs), len(pred))
    if n == 0:
        return {
            "rmse": np.array(np.nan),
            "mean": np.array([np.nan, np.nan]),
            "covariance": np.full((2, 2), np.nan),
        }
    residual = pred[:n] - obs[:n]
    mse = np.mean(np.sum(residual**2, axis=1))
    cov = np.cov(residual.T)
    return {"rmse": np.array(np.sqrt(mse)), "mean": residual.mean(axis=0), "covariance": cov}


def predict_location(
    cfg: DataConfig,
    start_lat_lon: Tuple[float, float],
    timestamp: datetime,
    hours_ahead: int,
    n_runs: int = 20,
    perturbation_scale: float = 0.1,
) -> str:
    """Generate a textual forecast with an 80th-percentile error bound."""
    state = init_particles(cfg, n=1, seed=int(timestamp.timestamp()), start_points=np.array([start_lat_lon]))
    end_time = timestamp + timedelta(hours=hours_ahead)
    reader = build_forcing_reader(cfg)
    ensemble = monte_carlo_ensemble(cfg, timestamp, end_time, cfg.dt_hours, state, n_runs, perturbation_scale, reader)
    final = ensemble[:, -1, 0, :]
    mean = final.mean(axis=0)
    diffs = (final - mean) * np.array([111_000, 111_000 * np.cos(np.deg2rad(mean[0]))])
    spread_km = np.percentile(np.linalg.norm(diffs, axis=1), 80) / 1000.0
    return (
        f"Forecast position {mean[0]:.2f} deg, {mean[1]:.2f} deg +/- {spread_km:.1f} km "
        f"at {end_time.isoformat()}."
    )


def plot_uncertainty_cloud(ensemble: np.ndarray, output_path: Path) -> Path:
    """Scatter plot of ensemble end points."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover
        LOGGER.info("matplotlib not installed; skipping uncertainty plot.")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end_points = ensemble[:, -1, 0, :]
    plt.figure(figsize=(5, 4))
    plt.scatter(end_points[:, 1], end_points[:, 0], s=20, alpha=0.4)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ensemble spread")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = DataConfig()
    state = init_particles(cfg, n=1, start_points=np.array([[30.0, -50.0]]))
    now = datetime.utcnow()
    reader = build_forcing_reader(cfg)
    ensemble = monte_carlo_ensemble(cfg, now, now + timedelta(hours=12), cfg.dt_hours, state, n_runs=5, reader=reader)
    plot_uncertainty_cloud(ensemble, Path("docs/plots/uncertainty_demo.png"))
