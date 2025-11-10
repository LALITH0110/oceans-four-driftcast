"""
File Summary:
- Implements basic metrics derived from gridded particle densities.
- Provides hotspot ranking and residence-time style summaries.
- Designed to accompany density rasters produced in driftcast.post.density.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr


def hotspot_scores(density: xr.DataArray, top_n: int = 5) -> pd.DataFrame:
    """Return the top-N grid cells ranked by density."""
    stacked = density.stack(point=("lat", "lon"))
    sorted_stack = stacked.sortby(stacked, ascending=False)
    top = sorted_stack.isel(point=slice(0, top_n))
    frame = top.to_series().to_frame(name="count").reset_index()
    total = float(density.sum())
    frame["fraction"] = frame["count"] / total if total else 0.0
    return frame
