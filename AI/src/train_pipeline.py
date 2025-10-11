# DriftCast end-to-end orchestration for simulations and PPO training.
# Used by CLI commands to scan data, run baseline sims, train/evaluate Tasks A/B, and build reports.
# Keeps operations lightweight so short experiments run quickly on local hardware.

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import DataConfig, data_availability_report
from .error_utils import monte_carlo_ensemble, plot_uncertainty_cloud
from .rl_cleanup import evaluate_cleanup, train_cleanup
from .rl_drift_correction import evaluate_correction, train_agent
from .simulator import init_particles, quick_simulation
from .viz import plot_trajectories

LOGGER = logging.getLogger(__name__)


def preprocess_data(cfg: Optional[DataConfig] = None) -> str:
    cfg = cfg or DataConfig()
    report = data_availability_report(cfg)
    LOGGER.info("\n%s", report)
    return report


def run_baseline(cfg: DataConfig, hours: int, n_particles: int) -> Path:
    trajectory = quick_simulation(cfg, hours=hours, n_particles=n_particles)
    lat = np.array(trajectory["lat"])
    lon = np.array(trajectory["lon"])
    baseline = np.stack([lat, lon], axis=-1)
    output = Path("docs/plots/baseline_trajectories.png")
    plot_trajectories(baseline, None, output, cfg.bbox)
    LOGGER.info("Saved baseline trajectories to %s", output)
    return output


def train_task_a(cfg: Optional[DataConfig] = None, timesteps: int = 100_000) -> Path:
    cfg = cfg or DataConfig()
    model_path = Path("models/correction.zip")
    model = train_agent(cfg, timesteps=timesteps, save_path=str(model_path))
    metrics = evaluate_correction(model, episodes=3, cfg=cfg)
    LOGGER.info("Task A RMSE: %.3f deg", metrics["rmse"])
    return model_path


def train_task_b(cfg: Optional[DataConfig] = None, episodes: int = 300) -> Path:
    cfg = cfg or DataConfig()
    model_path = Path("models/cleanup.zip")
    model = train_cleanup(cfg, episodes=episodes, save_path=str(model_path))
    metrics = evaluate_cleanup(model, episodes=5, cfg=cfg)
    LOGGER.info("Task B mean reward: %.2f", metrics["mean_reward"])
    return model_path


def build_uncertainty_plot(cfg: Optional[DataConfig] = None, hours_ahead: int = 24, runs: int = 15) -> Path:
    cfg = cfg or DataConfig()
    state = init_particles(cfg, n=cfg.n_particles_default // 4)
    start = datetime.utcnow()
    ensemble = monte_carlo_ensemble(cfg, start, start + timedelta(hours=hours_ahead), cfg.dt_hours, state, runs)
    output = Path("docs/plots/uncertainty_cloud.png")
    plot_uncertainty_cloud(ensemble, output)
    LOGGER.info("Saved uncertainty plot to %s", output)
    return output


def orchestrate(
    cfg: Optional[DataConfig] = None,
    timesteps_a: int = 100_000,
    episodes_b: int = 300,
) -> Dict[str, Path]:
    cfg = cfg or DataConfig()
    preprocess_data(cfg)
    artefacts: Dict[str, Path] = {}
    artefacts["baseline_plot"] = run_baseline(cfg, hours=24, n_particles=cfg.n_particles_default)
    artefacts["task_a_model"] = train_task_a(cfg, timesteps=timesteps_a)
    artefacts["task_b_model"] = train_task_b(cfg, episodes=episodes_b)
    artefacts["uncertainty_plot"] = build_uncertainty_plot(cfg)
    return artefacts
