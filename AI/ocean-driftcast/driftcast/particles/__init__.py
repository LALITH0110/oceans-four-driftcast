"""
File Summary:
- Provides particle class metadata and physics stepping utilities.
- Re-exports high-level APIs for creating ensembles and advancing positions.
- See driftcast.particles.physics for integrator hooks used during simulation.
"""

from .classes import ParticleClass, default_classes
from .physics import (
    ParticleState,
    append_particles,
    apply_beaching,
    euler_maruyama_step,
    initialize_state,
)

__all__ = [
    "ParticleClass",
    "default_classes",
    "ParticleState",
    "initialize_state",
    "euler_maruyama_step",
    "apply_beaching",
    "append_particles",
]
