"""
File Summary:
- Encapsulates particle seeding strategies for rivers, shipping, and coasts.
- Provides a BaseSource interface and registry for config-driven instantiation.
- Used by driftcast.sim.runner when emitting particles at each time step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from driftcast.config import SourceConfig

ParticleEmission = Tuple[np.ndarray, np.ndarray, Sequence[str], Sequence[str]]


class BaseSource(ABC):
    """Abstract base class for particle sources."""

    def __init__(self, config: SourceConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    @abstractmethod
    def emit(self, time_days: float, dt_days: float) -> ParticleEmission:
        """Return (lon, lat, class_names, source_names) for this step."""


from .coastal import CoastalSource
from .rivers import RiverSource
from .shipping import ShippingSource

REGISTRY: Dict[str, type] = {
    "rivers": RiverSource,
    "shipping": ShippingSource,
    "coastal": CoastalSource,
}

__all__ = ["BaseSource", "CoastalSource", "RiverSource", "ShippingSource", "REGISTRY"]
