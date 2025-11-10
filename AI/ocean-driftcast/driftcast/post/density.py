"""
File Summary:
- Offers routines to aggregate particle positions into gridded density rasters.
- Supports histogram-based and Gaussian kernel smoothing for quick diagnostics.
- Used within tests to verify particle conservation across rasterization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from driftcast.config import DomainConfig


def particle_density(
    lon: np.ndarray,
    lat: np.ndarray,
    domain: DomainConfig,
    resolution_deg: Optional[float] = None,
    smooth_sigma: Optional[float] = None,
) -> xr.DataArray:
    """Compute a 2-D histogram of particles over the domain."""
    res = resolution_deg or domain.resolution_deg
    if lon.size == 0:
        grid_lon = np.arange(domain.lon_min, domain.lon_max, res)
        grid_lat = np.arange(domain.lat_min, domain.lat_max, res)
        return xr.DataArray(
            np.zeros((grid_lat.size - 1, grid_lon.size - 1)),
            coords={"lat": grid_lat[:-1], "lon": grid_lon[:-1]},
            dims=("lat", "lon"),
            name="density",
        )

    lon_bins = np.arange(domain.lon_min, domain.lon_max + res, res)
    lat_bins = np.arange(domain.lat_min, domain.lat_max + res, res)

    counts, lon_edges, lat_edges = np.histogram2d(
        lon,
        lat,
        bins=[lon_bins, lat_bins],
    )
    density = counts.T  # lat first

    if smooth_sigma:
        from scipy.ndimage import gaussian_filter

        density = gaussian_filter(density, sigma=smooth_sigma, mode="nearest")

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    return xr.DataArray(
        density,
        coords={"lat": lat_centers, "lon": lon_centers},
        dims=("lat", "lon"),
        name="density",
    )
