# DriftCast Task A Gymnasium environment and PPO training helpers.
# Wraps the JAX simulator and local drifter trajectories to learn velocity corrections.
# Provides train_agent / evaluate_correction entry points used by the CLI and pipeline.

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.neighbors import KDTree
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import DataConfig
from .data_loader import build_forcing_reader, load_drifters
from .simulator import ParticleState, _step_core, init_particles


def _drifter_kdtree(df: pd.DataFrame) -> KDTree:
    if df.empty:
        raise ValueError("No drifter data available for Task A.")
    coords = np.stack(
        [
            df["time"].astype("int64").to_numpy() / 1e9,  # seconds
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
        ],
        axis=1,
    )
    return KDTree(coords)


def _nearest_drifter(tree: KDTree, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dists, idx = tree.query(points, k=1)
    return dists[:, 0], idx[:, 0]


class DriftDataset:
    def __init__(self, drifters: pd.DataFrame, kdtree: KDTree):
        self.drifters = drifters
        self.kdtree = kdtree


import gymnasium as gym


class DriftCorrectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: DataConfig,
        horizon_hours: int = 24,
        dt_hours: float = 1.0,
        max_particles: int = 1000,
    ):
        super().__init__()
        self.cfg = cfg
        self.dt_hours = dt_hours
        self.reader = build_forcing_reader(cfg)
        self.drifters = load_drifters(cfg)
        if self.drifters.empty:
            raise RuntimeError("Task A requires at least one drifter track.")
        self.drift_ds = DriftDataset(self.drifters, _drifter_kdtree(self.drifters))
        self.horizon_steps = max(1, int(horizon_hours / dt_hours))
        self.max_particles = max_particles
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.track_segment: Optional[pd.DataFrame] = None
        self.state: Optional[ParticleState] = None
        self.step_index = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        track_id = rng.integers(0, self.drifters["id"].max() + 1)
        track = self.drifters[self.drifters["id"] == track_id].sort_values("time")
        if track.shape[0] < 6:
            track = self.drifters.sort_values("time").head(self.horizon_steps + 1)
        self.track_segment = track.head(self.horizon_steps + 1).reset_index(drop=True)
        start_point = self.track_segment.loc[0, ["lat", "lon"]].to_numpy()[None, :]
        bbox = self.cfg.bbox
        self.state = init_particles(self.cfg, n=1, start_points=start_point)
        self.step_index = 0
        obs = self._build_observation(self.track_segment.loc[0, "time"].to_pydatetime())
        return obs, {}

    def step(self, action):
        assert self.track_segment is not None and self.state is not None
        next_row = self.track_segment.loc[self.step_index + 1]
        time = next_row["time"].to_pydatetime()
        fields = self.reader.sample(time, np.asarray(self.state.lat), np.asarray(self.state.lon))
        fields["currents"] = (
            fields["currents"][0] + jnp.array([action[0]], dtype=jnp.float32),
            fields["currents"][1] + jnp.array([action[1]], dtype=jnp.float32),
        )
        lat_new, lon_new, zmix_new, key_new = _step_core_wrapper(
            self.state,
            fields,
            self.cfg,
            self.dt_hours,
        )
        self.state = ParticleState(lat=lat_new, lon=lon_new, zmix=zmix_new, key=key_new)
        self.step_index += 1
        obs = self._build_observation(time)
        reward = self._compute_reward(time)
        terminated = self.step_index >= len(self.track_segment) - 1
        return obs, reward, bool(terminated), False, {"mse": -reward}

    def _compute_reward(self, time: datetime) -> float:
        assert self.state is not None and self.track_segment is not None
        coord = np.array([[time.timestamp(), float(self.state.lat[0]), float(self.state.lon[0])]])
        dists, idx = _nearest_drifter(self.kdtree, coord)
        weight = np.exp(-dists[0] / 3600.0)  # seconds -> weight
        nearest = self.drifters.iloc[idx[0]]
        mse = (float(self.state.lat[0]) - nearest["lat"]) ** 2 + (float(self.state.lon[0]) - nearest["lon"]) ** 2
        return float(-weight * mse)

    def _build_observation(self, time: datetime) -> np.ndarray:
        assert self.state is not None
        fields = self.reader.sample(time, np.asarray(self.state.lat), np.asarray(self.state.lon))
        lat = float(self.state.lat[0])
        lon = float(self.state.lon[0])
        u_curr, v_curr = [f[0] for f in fields["currents"]], [f[0] for f in fields["currents"]]
        u_curr_val = float(fields["currents"][0][0])
        v_curr_val = float(fields["currents"][1][0])
        u_wind_val = float(fields["winds"][0][0])
        v_wind_val = float(fields["winds"][1][0])
        u_stokes_val = float(fields["stokes"][0][0])
        v_stokes_val = float(fields["stokes"][1][0])
        vort = 0.0  # TODO: add coastline/vorticity product
        obs = np.array(
            [
                lat,
                lon,
                u_curr_val,
                v_curr_val,
                u_wind_val,
                v_wind_val,
                u_stokes_val,
                v_stokes_val,
                vort,
                0.0,  # distance to coast placeholder
            ],
            dtype=np.float32,
        )
        return obs


def _step_core_wrapper(state: ParticleState, fields, cfg: DataConfig, dt_hours: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
    from .simulator import _step_core  # avoid circular import

    dt_seconds = dt_hours * 3600.0
    return _step_core(
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


def train_agent(
    cfg: Optional[DataConfig] = None,
    timesteps: int = 100_000,
    save_path: str = "models/correction.zip",
    eval_freq: int = 10_000,
) -> PPO:
    cfg = cfg or DataConfig()

    def make_env():
        return DriftCorrectionEnv(cfg)

    env = DummyVecEnv([make_env])
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env.training = False

    callbacks = [
        CheckpointCallback(save_freq=eval_freq // 2, save_path="models/task_a_checkpoints", name_prefix="ppo_drift"),
        EvalCallback(
            eval_env,
            best_model_save_path="models",
            eval_freq=eval_freq,
            n_eval_episodes=5,
            callbacks=[StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=2)],
        ),
    ]

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callbacks)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    vec_env.save(Path(save_path).with_suffix(".vecnorm"))
    LOGGER.info("Saved Task A PPO model to %s", save_path)
    env.close()
    eval_env.close()
    return model


def evaluate_correction(model: PPO, episodes: int = 5, cfg: Optional[DataConfig] = None) -> Dict[str, float]:
    cfg = cfg or DataConfig()
    env = DriftCorrectionEnv(cfg)
    mse_vals: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += info.get("mse", 0.0)
            steps += 1
        if steps:
            mse_vals.append(total / steps)
    env.close()
    rmse = float(np.sqrt(np.mean(mse_vals))) if mse_vals else float("nan")
    LOGGER.info("Task A evaluation RMSE: %.4f deg", rmse)
    return {"rmse": rmse}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = DataConfig()
    model = train_agent(cfg, timesteps=10_000, save_path="models/debug_correction.zip")
    evaluate_correction(model, episodes=2, cfg=cfg)
