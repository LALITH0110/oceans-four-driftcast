"""
File Summary:
- Implements Poisson-distributed riverine particle injections along coastlines.
- Samples from configured river mouths and particle class compositions.
- Called by driftcast.sim.runner on each integration step.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from driftcast.config import ParticleClassConfig, SourceConfig
from driftcast.particles.classes import ParticleClass
from . import BaseSource, ParticleEmission


def _prepare_composition(config: SourceConfig) -> Tuple[Sequence[str], np.ndarray]:
    if not config.composition:
        defaults = [
            ParticleClassConfig(
                name="microfiber",
                density_kgm3=1040.0,
                diameter_mm=0.3,
                settling_velocity_mps=-2e-4,
                fraction=0.4,
            ),
            ParticleClassConfig(
                name="fragment",
                density_kgm3=950.0,
                diameter_mm=1.2,
                settling_velocity_mps=-8e-5,
                fraction=0.4,
            ),
            ParticleClassConfig(
                name="pellet",
                density_kgm3=910.0,
                diameter_mm=3.0,
                settling_velocity_mps=0.0,
                fraction=0.2,
            ),
        ]
        comp = defaults
    else:
        comp = config.composition
    names = [c.name for c in comp]
    weights = np.array([c.fraction for c in comp], dtype=float)
    weights = weights / weights.sum()
    return names, weights


class RiverSource(BaseSource):
    """Particle emissions localized at river mouths."""

    def __init__(self, config: SourceConfig, rng: np.random.Generator):
        super().__init__(config, rng)
        locations = config.params.get(
            "locations",
            [
                {"name": "hudson", "lon": -74.0, "lat": 40.7},
                {"name": "mississippi", "lon": -89.4, "lat": 30.0},
                {"name": "seine", "lon": 2.0, "lat": 49.0},
            ],
        )
        self.lons = np.array([loc["lon"] for loc in locations], dtype=float)
        self.lats = np.array([loc["lat"] for loc in locations], dtype=float)
        self.weights = np.array(
            [loc.get("weight", 1.0) for loc in locations], dtype=float
        )
        self.weights = self.weights / self.weights.sum()
        self.class_names, self.class_weights = _prepare_composition(config)

    def emit(self, time_days: float, dt_days: float) -> ParticleEmission:
        lam = self.config.rate_per_day * dt_days
        n_new = self.rng.poisson(lam)
        if n_new == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty, [], []
        location_idx = self.rng.choice(len(self.lons), size=n_new, p=self.weights)
        lon = self.lons[location_idx]
        lat = self.lats[location_idx]
        jitter = self.config.params.get("jitter_deg", 0.2)
        lon += self.rng.normal(scale=jitter, size=n_new)
        lat += self.rng.normal(scale=jitter / 2.0, size=n_new)
        class_choices = self.rng.choice(
            self.class_names, size=n_new, p=self.class_weights
        )
        sources = [self.config.name] * n_new
        return lon, lat, class_choices, sources
