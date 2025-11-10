"""
File Summary:
- Provides post-processing utilities including density rasters and hotspot metrics.
- Re-exports helper functions for CLI and notebook usage.
- Refer to individual modules for methodological details.
"""

from .density import particle_density
from .metrics import hotspot_scores

__all__ = ["particle_density", "hotspot_scores"]
