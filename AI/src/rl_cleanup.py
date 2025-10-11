# DriftCast Task B cleanup environment and PPO training utilities.
# Evolves a particle-based plastic cloud with the simulator and rewards removal by mobile assets.
# Called by the CLI and pipeline to train/evaluate cleanup policies.

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd
from gymnasium import spaces
from .data_loader import build_forcing_reader, load_drifters
from .simulator import ParticleState, init_particles, _step_core
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import DataConfig
from .data_loader import build_forcing_reader, load_drifters
from .simulator import ParticleState, init_particles, _step_core

LOGGER = logging.getLogger(__name__)


def _particles_from_map(cfg: DataConfig, df: pd.DataFrame, n: int) -> np.ndarray:
    if df.empty:
        rng = np.random.default_rng(cfg.random_seed)
        lat = rng.uniform(cfg.region_south, cfg.region_north, size=n)
        lon = rng.uniform(cfg.region_west, cfg.region_east, size=n)
        return np.stack([lat, lon], axis=1)
    weights = np.ones(len(df)) / len(df)
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(len(df), size=n, replace=True, p=weights)
    sample = df.iloc[idx][["lat", "lon"]].to_numpy()
    jitter = rng.normal(scale=0.25, size=sample.shape)
    sample[:, 0] = np.clip(sample[:, 0] + jitter[:, 0], cfg.region_south, cfg.region_north)
    sample[:, 1] = np.clip(sample[:, 1] + jitter[:, 1], cfg.region_west, cfg.region_east)
    return sample


def _grid_histogram(cfg: DataConfig, particles: ParticleState, grid_shape: Tuple[int, int]) -> np.ndarray:
    lat = np.asarray(particles.lat)
    lon = np.asarray(particles.lon)
    lat_bins = np.linspace(cfg.region_south, cfg.region_north, grid_shape[0] + 1)
    lon_bins = np.linspace(cfg.region_west, cfg.region_east, grid_shape[1] + 1)
    hist, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins])
    if hist.max() > 0:
        hist /= hist.max()
    return hist.astype(np.float32)


def _step_particles(state: ParticleState, fields, cfg: DataConfig, dt_hours: float):
    dt_seconds = dt_hours * 3600.0
    lat, lon, zmix, key = _step_core(
        state.lat,
        state.lon,
        state.zmix,
        state.key,
        fields["currents"][0],
        fields["currents"][1],
        fields["winds"][0],
        fields["winds"][1],
        fields["stokes"][0],
        fields["stokes"][1],
        cfg.alpha_wind,
        cfg.diffusion_kappa,
        dt_seconds,
    )
    return lat, lon, zmix, key

    metadata = {"render_modes": []}

class CleanupEnv(gym.Env):

    def __init__(
        self,
        cfg: DataConfig,
        n_assets: int = 1,
        grid_shape: Tuple[int, int] = (24, 24),
        dt_hours: float = 6.0,
        cleanup_radius_deg: float = 1.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.reader = build_forcing_reader(cfg)
        self.drifters = load_drifters(cfg)
        self.grid_shape = grid_shape
        self.dt_hours = dt_hours
        self.cleanup_radius = cleanup_radius_deg
        self.n_assets = n_assets
        self.current_time = datetime.utcnow()
        map_size = grid_shape[0] * grid_shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(map_size + 2 * n_assets,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * n_assets,), dtype=np.float32)
        self.cloud: ParticleState = None  # type: ignore
        self.assets = np.zeros((n_assets, 2), dtype=np.float32)
        self.plastic_map = np.zeros(grid_shape, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        seed_val = seed if seed is not None else self.cfg.random_seed
        particle_points = _particles_from_map(self.cfg, self.drifters, n=800)
        self.cloud = init_particles(self.cfg, n=particle_points.shape[0], seed=seed_val, start_points=particle_points)
        self.current_time = self.drifters["time"].min().to_pydatetime() if not self.drifters.empty else datetime.utcnow()
        self.plastic_map = _grid_histogram(self.cfg, self.cloud, self.grid_shape)
        self.assets = self._initial_asset_positions()
        return self._observation(), {}

    def step(self, action):
        action = action.reshape(self.n_assets, 2)
        self._advance_particles()
        removed = self._apply_cleanup(action)
        reward = removed - 0.01 * float(np.linalg.norm(action, axis=1).sum())
        terminated = float(self.plastic_map.sum()) < 1e-3
        return self._observation(), float(reward), bool(terminated), False, {"plastic_removed": float(removed)}

    def _advance_particles(self):
        next_time = self.current_time + timedelta(hours=self.dt_hours)
        fields = self.reader.sample(next_time, np.asarray(self.cloud.lat), np.asarray(self.cloud.lon))
        lat, lon, zmix, key = _step_particles(self.cloud, fields, self.cfg, self.dt_hours)
        self.cloud = ParticleState(lat=lat, lon=lon, zmix=zmix, key=key)
        self.current_time = next_time
        self.plastic_map = _grid_histogram(self.cfg, self.cloud, self.grid_shape)

    def _apply_cleanup(self, action: np.ndarray) -> float:
        self.assets += action * 0.5  # scale movement
        self.assets[:, 0] = np.clip(self.assets[:, 0], self.cfg.region_west, self.cfg.region_east)
        self.assets[:, 1] = np.clip(self.assets[:, 1], self.cfg.region_south, self.cfg.region_north)
        removed = 0.0
        lat_bins = np.linspace(self.cfg.region_south, self.cfg.region_north, self.grid_shape[0])
        lon_bins = np.linspace(self.cfg.region_west, self.cfg.region_east, self.grid_shape[1])
        lat_np = np.asarray(self.cloud.lat)
        lon_np = np.asarray(self.cloud.lon)
        for lon_asset, lat_asset in self.assets:
            dist = np.sqrt((lat_bins[:, None] - lat_asset) ** 2 + (lon_bins[None, :] - lon_asset) ** 2)
            mask = dist <= self.cleanup_radius
            removed += self.plastic_map[mask].sum()
            self.plastic_map[mask] *= 0.3
            particle_dist = np.sqrt((lat_np - lat_asset) ** 2 + (lon_np - lon_asset) ** 2)
            close_idx = np.where(particle_dist <= self.cleanup_radius)[0]
            if close_idx.size:
                reroll = _particles_from_map(self.cfg, self.drifters, close_idx.size)
                lat_np[close_idx] = reroll[:, 0]
                lon_np[close_idx] = reroll[:, 1]
        self.cloud = ParticleState(lat=jnp.asarray(lat_np), lon=jnp.asarray(lon_np), zmix=self.cloud.zmix, key=self.cloud.key)
        return removed

    def _initial_asset_positions(self) -> np.ndarray:
        lats = np.linspace(self.cfg.region_south, self.cfg.region_north, self.n_assets + 2)[1:-1]
        lons = np.linspace(self.cfg.region_west, self.cfg.region_east, self.n_assets + 2)[1:-1]
        return np.stack([lons, lats], axis=1).astype(np.float32)

    def _observation(self) -> np.ndarray:
        return np.concatenate([self.plastic_map.flatten(), self.assets.flatten()]).astype(np.float32)


def train_cleanup(
    cfg: Optional[DataConfig] = None,
    episodes: int = 300,
    save_path: str = "models/cleanup.zip",
) -> PPO:
    cfg = cfg or DataConfig()

    def make_env():
        return CleanupEnv(cfg)

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=episodes * 128)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    env.close()
    LOGGER.info("Saved Task B model to %s", save_path)
    return model


def evaluate_cleanup(model: PPO, episodes: int = 5, cfg: Optional[DataConfig] = None) -> Dict[str, float]:
    cfg = cfg or DataConfig()
    env = CleanupEnv(cfg)
    rewards: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return {"mean_reward": float(np.mean(rewards)) if rewards else float("nan")}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = DataConfig()
    model = train_cleanup(cfg, episodes=10, save_path="models/debug_cleanup.zip")
    metrics = evaluate_cleanup(model, episodes=2, cfg=cfg)
    LOGGER.info("Cleanup mean reward: %.2f", metrics["mean_reward"])
