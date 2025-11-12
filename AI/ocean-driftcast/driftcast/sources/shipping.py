"""
File Summary:
- Simulates leakage from major North Atlantic shipping lanes.
- Samples particle positions along configured great-circle style segments.
- Leveraged within driftcast.sim.runner for scenario construction.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from driftcast.config import SourceConfig
from . import BaseSource, ParticleEmission


def _default_routes() -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    return [
        ((-80.0, 26.0), (-10.0, 50.0)),  # Miami to English Channel
        ((-75.0, 40.0), (-5.0, 55.0)),  # New York to North Sea
        ((-90.0, 30.0), (-20.0, 35.0)),  # Gulf to Iberia
    ]


class ShippingSource(BaseSource):
    """Particles emitted along shipping lanes with Gaussian cross-track spread."""

    def __init__(self, config: SourceConfig, rng: np.random.Generator):
        super().__init__(config, rng)
        routes_cfg = config.params.get("routes")
        if routes_cfg:
            self.routes = [
                ((float(r["start_lon"]), float(r["start_lat"])), (float(r["end_lon"]), float(r["end_lat"])))
                for r in routes_cfg
            ]
        else:
            self.routes = _default_routes()
        self.width_deg = float(config.params.get("width_deg", 1.5))

    def emit(self, time_days: float, dt_days: float) -> ParticleEmission:
        lam = self.config.rate_per_day * dt_days
        n_new = self.rng.poisson(lam)
        if n_new == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty, [], []
        route_idx = self.rng.choice(len(self.routes), size=n_new)
        lon = np.zeros(n_new, dtype=float)
        lat = np.zeros(n_new, dtype=float)
        frac = self.rng.uniform(size=n_new)
        cross = self.rng.normal(scale=self.width_deg, size=n_new)
        for i, idx in enumerate(route_idx):
            (lon0, lat0), (lon1, lat1) = self.routes[idx]
            lon[i] = lon0 + frac[i] * (lon1 - lon0) + cross[i] * np.cos(np.deg2rad(lat0))
            lat[i] = lat0 + frac[i] * (lat1 - lat0) + cross[i] * np.sin(np.deg2rad(lat0))
        class_name = [self.config.params.get("class_name", "fragment")] * n_new
        sources = [self.config.name] * n_new
        return lon, lat, class_name, sources
