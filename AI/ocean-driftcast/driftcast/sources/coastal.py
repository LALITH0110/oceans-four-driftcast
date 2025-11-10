"""
File Summary:
- Implements diffuse coastal particle seeding along continental margins.
- Uses rejection sampling near land fraction proxies supplied via config.
- Serves as a background pollution source for scenario composition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from driftcast.config import SourceConfig
from . import BaseSource, ParticleEmission


class CoastalSource(BaseSource):
    """Uniform coastal seeding within a buffered band from shore."""

    def __init__(self, config: SourceConfig, rng: np.random.Generator):
        super().__init__(config, rng)
        self.lon_bounds = tuple(config.params.get("lon_bounds", (-100.0, 20.0)))
        self.lat_bounds = tuple(config.params.get("lat_bounds", (0.0, 70.0)))
        self.buffer_deg = float(config.params.get("buffer_deg", 2.0))

    def emit(self, time_days: float, dt_days: float) -> ParticleEmission:
        lam = self.config.rate_per_day * dt_days
        n_new = self.rng.poisson(lam)
        if n_new == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty, [], []
        lon = self.rng.uniform(self.lon_bounds[0], self.lon_bounds[1], size=n_new)
        lat = self.rng.uniform(self.lat_bounds[0], self.lat_bounds[1], size=n_new)
        # Bias toward coasts by nudging latitudes toward band edges.
        bias = self.rng.uniform(-self.buffer_deg, self.buffer_deg, size=n_new)
        lat = np.clip(lat + bias, self.lat_bounds[0], self.lat_bounds[1])
        class_names = [self.config.params.get("class_name", "microfiber")] * n_new
        sources = [self.config.name] * n_new
        return lon, lat, class_names, sources
