"""
File Summary:
- Defines microplastic particle classes with density and settling attributes.
- Supplies factory helpers for default surface-oriented class catalogues.
- Referenced by driftcast.sources.* modules when sampling new particles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ParticleClass:
    """Describe an individual microplastic class."""

    name: str
    density_kgm3: float
    diameter_mm: float
    settling_velocity_mps: float


def default_classes() -> List[ParticleClass]:
    """Return a default suite of three representative particle classes."""
    return [
        ParticleClass("microfiber", 1040.0, 0.3, -2e-4),
        ParticleClass("fragment", 950.0, 1.5, -5e-5),
        ParticleClass("nurdle", 910.0, 4.0, 0.0),
    ]


def expand_composition(
    classes: Iterable[ParticleClass],
    fractions: Iterable[float],
    n_total: int,
) -> List[ParticleClass]:
    """Generate a list of classes sampled according to fractional weights."""
    catalogue = list(classes)
    probs = list(fractions)
    if len(catalogue) != len(probs):
        raise ValueError("classes and fractions must be same length")
    total = sum(probs)
    if total == 0:
        raise ValueError("fractions must sum to > 0")
    normalized = [p / total for p in probs]
    cumulative = [sum(normalized[: i + 1]) for i in range(len(normalized))]
    draws = []
    for idx in range(n_total):
        r = (idx + 0.5) / n_total
        for j, threshold in enumerate(cumulative):
            if r <= threshold:
                draws.append(catalogue[j])
                break
    return draws
