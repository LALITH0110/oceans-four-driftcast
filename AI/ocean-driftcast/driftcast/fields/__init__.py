"""
File Summary:
- Groups synthetic velocity field generators for ocean drift simulations.
- Re-exports gyre, wind, and Stokes drift utilities for convenient access.
- See individual modules for parameter documentation and implementation details.
"""

from .gyres import GyreFieldConfig, gyre_velocity_field, streamfunction
from .stokes import stokes_drift_velocity
from .winds import seasonal_wind_field

__all__ = [
    "GyreFieldConfig",
    "gyre_velocity_field",
    "seasonal_wind_field",
    "stokes_drift_velocity",
]
