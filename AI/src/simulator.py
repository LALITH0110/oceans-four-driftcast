# DriftCast JAX advection–diffusion simulator for surface plastics.
# Wraps ForcingReader interpolation with jit/vmap-powered stepping while staying I/O aware.
# Used by CLI simulations, RL environments, and Monte Carlo ensemble utilities.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import DataConfig
from .data_loader import ForcingReader, build_forcing_reader

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEG_LAT = (math.pi / 180.0) * EARTH_RADIUS_M


@dataclass
class ParticleState:
    lat: jnp.ndarray
    lon: jnp.ndarray
    zmix: jnp.ndarray
    key: jax.Array


@jax.jit
def _step_core(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    zmix: jnp.ndarray,
    key: jax.Array,
    u_curr: jnp.ndarray,
    v_curr: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    u_stokes: jnp.ndarray,
    v_stokes: jnp.ndarray,
    alpha: float,
    kappa: float,
    dt_seconds: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
    rng, turb_key, mix_key = jax.random.split(key, 3)
    u_total = u_curr + alpha * u_wind + u_stokes
    v_total = v_curr + alpha * v_wind + v_stokes
    dlat = (v_total * dt_seconds) / METERS_PER_DEG_LAT
    cos_lat = jnp.cos(jnp.deg2rad(lat))
    meters_per_deg_lon = METERS_PER_DEG_LAT * jnp.maximum(jnp.abs(cos_lat), 1e-3)
    dlon = (u_total * dt_seconds) / meters_per_deg_lon
    sigma = jnp.sqrt(2.0 * kappa * dt_seconds) / METERS_PER_DEG_LAT
    turb_lat = sigma * jax.random.normal(turb_key, lat.shape)
    turb_lon = sigma * jax.random.normal(turb_key, lon.shape)
    zmix_new = 0.85 * zmix + 0.15 * jax.random.normal(mix_key, zmix.shape)
    mix_factor = jnp.clip(1.0 + 0.2 * zmix_new, 0.5, 1.5)
    lat_new = lat + dlat + mix_factor * turb_lat
    lon_new = wrap_longitude(lon + dlon + mix_factor * turb_lon)
    return lat_new, lon_new, zmix_new, rng


def init_particles(
    cfg: DataConfig,
    n: int,
    seed: Optional[int] = None,
    start_points: Optional[np.ndarray] = None,
) -> ParticleState:
    north, south, west, east = cfg.bbox
    key = jax.random.PRNGKey(seed if seed is not None else cfg.random_seed)
    if start_points is not None and len(start_points) >= n:
        pts = jnp.asarray(start_points[:n])
        lat = pts[:, 0]
        lon = pts[:, 1]
    else:
        key, lat_key, lon_key = jax.random.split(key, 3)
        lat = jax.random.uniform(lat_key, (n,), minval=south, maxval=north)
        lon = jax.random.uniform(lon_key, (n,), minval=west, maxval=east)
    zmix = jnp.zeros((n,), dtype=jnp.float32)
    return ParticleState(lat=lat, lon=lon, zmix=zmix, key=key)


def sample_fields(reader: ForcingReader, time: datetime, state: ParticleState) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    lat_np = np.asarray(state.lat, dtype=np.float32)
    lon_np = np.asarray(state.lon, dtype=np.float32)
    return reader.sample(time, lat_np, lon_np)


def run_simulation(
    reader: ForcingReader,
    cfg: DataConfig,
    start_time: datetime,
    end_time: datetime,
    state: ParticleState,
) -> Dict[str, jnp.ndarray]:
    dt_seconds = cfg.dt_hours * 3600.0
    times: List[datetime] = []
    t = start_time
    while t <= end_time + timedelta(seconds=1):
        times.append(t)
        t += timedelta(hours=cfg.dt_hours)
    lat_hist = [state.lat]
    lon_hist = [state.lon]
    for step_time in times[1:]:
        fields = sample_fields(reader, step_time, state)
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
        state = ParticleState(lat=lat, lon=lon, zmix=zmix, key=key)
        lat_hist.append(lat)
        lon_hist.append(lon)
    return {
        "time": jnp.asarray(np.array(times, dtype="datetime64[ns]").astype(np.int64)),
        "lat": jnp.stack(lat_hist),
        "lon": jnp.stack(lon_hist),
    }


def wrap_longitude(lon: jnp.ndarray) -> jnp.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


def quick_simulation(cfg: DataConfig, hours: int, n_particles: int) -> Dict[str, jnp.ndarray]:
    reader = build_forcing_reader(cfg)
    state = init_particles(cfg, n_particles)
    now = datetime.utcnow()
    return run_simulation(reader, cfg, now, now + timedelta(hours=hours), state)


def save_quicklook(traj: Dict[str, jnp.ndarray], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover
        LOGGER.info("matplotlib missing; skipping quicklook plot.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(traj["lat"])
    lon = np.asarray(traj["lon"])
    plt.figure(figsize=(6, 4))
    plt.plot(lon[0], lat[0], "k.", alpha=0.4, label="start")
    plt.plot(lon[-1], lat[-1], "r.", alpha=0.4, label="end")
    plt.plot(lon.mean(axis=1), lat.mean(axis=1), "b-", lw=1.5, label="centroid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    config = DataConfig()
    traj = quick_simulation(config, hours=24, n_particles=500)
    save_quicklook(traj, Path("docs/plots/quick_sim.png"))
    LOGGER.info("Quick simulation stored at docs/plots/quick_sim.png")
