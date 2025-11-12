"""
File Summary:
- Provides helper to set up a cartopy basemap matching the driftcast domain.
- Adds coastlines, land shading, and optional graticule lines.
- Used by animation pipeline and static hero frame rendering.
"""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt

from driftcast.config import DomainConfig


def make_basemap(
    domain: DomainConfig,
    figsize: tuple[float, float] = (10, 6),
    draw_grid: bool = True,
):
    """Create a matplotlib figure and cartopy GeoAxes for the domain."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max], ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature("physical", "land", "50m")
    ax.add_feature(land, facecolor="#3b4b57", edgecolor="#cdd1c4", linewidth=0.6)
    ax.coastlines(resolution="50m", color="#cdd1c4", linewidth=0.6)
    if draw_grid:
        gl = ax.gridlines(draw_labels=True, color="#45586a", linewidth=0.3, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    return fig, ax
