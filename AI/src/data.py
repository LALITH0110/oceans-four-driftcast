from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import Config

LOGGER = logging.getLogger(__name__)

def _find_files(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(Path().glob(pat)))
    return files

def _get_lon_lat_names(ds: xr.Dataset) -> Tuple[str, str]:
    lon_candidates = ("lon", "longitude", "LONGITUDE", "x")
    lat_candidates = ("lat", "latitude", "LATITUDE", "y")
    lon_name = next((n for n in lon_candidates if n in ds.coords), None)
    lat_name = next((n for n in lat_candidates if n in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Dataset missing lon/lat coords, found {list(ds.coords)}")
    return lon_name, lat_name

def _get_time_name(ds: xr.Dataset) -> Optional[str]:
    for c in ("time", "Time", "datetime"):
        if c in ds.coords:
            return c
    return None

def _normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
    lon_name, _ = _get_lon_lat_names(ds)
    lon = ds[lon_name]
    if float(lon.min()) >= 0.0 and float(lon.max()) > 180.0:
        ds = ds.assign_coords({lon_name: ((lon + 180) % 360) - 180}).sortby(lon_name)
    return ds

def _subset_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    north, south, west, east = bbox
    lon_name, lat_name = _get_lon_lat_names(ds)
    lat_min, lat_max = sorted((south, north))
    lon_min, lon_max = sorted((west, east))
    return ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

def _select_surface(ds: xr.Dataset) -> xr.Dataset:
    depth_dim = next((d for d in ("depth", "Depth", "lev", "z", "deptht") if d in ds.dims), None)
    if depth_dim is not None:
        ds = ds.isel({depth_dim: 0})
    return ds

def _open_mf(paths: List[Path], bbox: Tuple[float, float, float, float]) -> Optional[xr.Dataset]:
    if not paths:
        return None

    def _pre(ds: xr.Dataset) -> xr.Dataset:
        # normalize, slice surface, crop bbox
        ds = _normalize_longitudes(ds)
        ds = _select_surface(ds)
        ds = _subset_bbox(ds, bbox)

        # enforce monotonic coords
        lon_name, lat_name = _get_lon_lat_names(ds)
        ds = ds.sortby(lat_name)
        ds = ds.sortby(lon_name)

        # drop duplicate coord values if any
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        _, lat_idx = np.unique(lat_vals, return_index=True)
        _, lon_idx = np.unique(lon_vals, return_index=True)
        if len(lat_idx) != len(lat_vals):
            ds = ds.isel({lat_name: np.sort(lat_idx)})
        if len(lon_idx) != len(lon_vals):
            ds = ds.isel({lon_name: np.sort(lon_idx)})

        return ds

    # Open each file, preprocess, then combine
    dsets: List[xr.Dataset] = []
    for p in paths:
        ds = xr.open_dataset(str(p), engine="netcdf4", decode_times=True)
        dsets.append(_pre(ds))

    # Prefer concatenation along time if present, else combine by coords
    tname = _get_time_name(dsets[0])
    if tname and all(tname in ds.coords for ds in dsets):
        ds_out = xr.concat(
            dsets, dim=tname,
            data_vars="minimal", coords="minimal", compat="override", join="outer"
        )
    else:
        ds_out = xr.combine_by_coords(
            dsets, combine_attrs="drop_conflicts", join="outer"
        )

    return ds_out


def load_cmems_currents(cfg: Config) -> Optional[xr.Dataset]:
    files = _find_files(["data/cmems/*.nc"])
    ds = _open_mf(files, cfg.bbox())
    if ds is not None:
        # keep only current variables if present
        keep = [v for v in ("uo", "vo", "eastward_sea_water_velocity", "northward_sea_water_velocity") if v in ds.data_vars]
        return ds if not keep else ds[keep]
    return None

def load_era5_winds(cfg: Config) -> Optional[xr.Dataset]:
    files = _find_files(["data/era5/*.nc", "data/raw/era5/*.nc", "data/raw/era5/nc/*.nc"])
    ds = _open_mf(files, cfg.bbox())
    if ds is not None:
        keep = [v for v in ("u10", "v10", "u_component_of_wind", "v_component_of_wind") if v in ds.data_vars]
        return ds if not keep else ds[keep]
    return None

def load_drifters_csv(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = Path("data/drifters/natl_gdp_drifters_2015_2024.csv")
    if not path.exists():
        # optional synthetic for quick tests
        synth = Path("data/synthetic_drifters.csv")
        if synth.exists():
            df = pd.read_csv(synth)
            df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df["id"] = df["id"].astype(str)
            return df[["time", "lat", "lon", "id"]], {"dups_removed": 0, "currents_files": [], "winds_files": []}
        return pd.DataFrame(columns=["time", "lat", "lon", "id"]), {"dups_removed": 0, "currents_files": [], "winds_files": []}

    # detect the units row on line 2
    skiprows = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        second = f.readline()
        if second and ("degrees_north" in second or "degrees_east" in second or "UTC" in second):
            skiprows = [1]

    df = pd.read_csv(path, skiprows=skiprows)
    # Normalize column names
    df = df.rename(columns={
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "ID": "id",
        "ve": "u",
        "vn": "v",
        "sst": "sea_surface_temperature",
    })
    # Clean types
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["id"] = df["id"].astype(str)
    for c in ("lat", "lon", "u", "v", "sea_surface_temperature"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop bad rows
    df = df.dropna(subset=["time", "lat", "lon"])
    # Remove sentinel -999999 velocities
    for c in ("u", "v"):
        if c in df.columns:
            df.loc[df[c] <= -1e5, c] = np.nan
    # De-duplicate by (id, time) then by (time, lat, lon)
    before = len(df)
    df = df.sort_values(["id", "time"])
    df = df.drop_duplicates(subset=["id", "time"], keep="first")
    df = df.drop_duplicates(subset=["time", "lat", "lon"], keep="first")
    dups_removed = before - len(df)
    return df[["time", "lat", "lon", "id"]], {
        "dups_removed": int(dups_removed),
        "currents_files": [p.name for p in _find_files(["data/cmems/*.nc"])],
        "winds_files": [p.name for p in _find_files(["data/era5/*.nc", "data/raw/era5/*.nc", "data/raw/era5/nc/*.nc"])],
    }

def sample_vector(
    ds: Optional[xr.Dataset],
    var_u_candidates: Sequence[str],
    var_v_candidates: Sequence[str],
    time,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if ds is None:
        return np.zeros_like(lat, dtype=np.float32), np.zeros_like(lat, dtype=np.float32)
    # pick variables
    var_u = next((v for v in var_u_candidates if v in ds.data_vars), None)
    var_v = next((v for v in var_v_candidates if v in ds.data_vars), None)
    if var_u is None or var_v is None:
        return np.zeros_like(lat, dtype=np.float32), np.zeros_like(lat, dtype=np.float32)
    # time interp, then spatial interp at points
    tname = _get_time_name(ds)
    ds_t = ds if tname is None else ds.interp({tname: np.array(time, dtype="datetime64[ns]")}, kwargs={"fill_value": "extrapolate"})
    lon_name, lat_name = _get_lon_lat_names(ds_t)
    sample = ds_t[[var_u, var_v]].interp({lat_name: ("points", lat), lon_name: ("points", lon)}, kwargs={"fill_value": "extrapolate"})
    u = np.asarray(sample[var_u].values, dtype=np.float32)
    v = np.asarray(sample[var_v].values, dtype=np.float32)
    if sample[var_u].dims[0] == "points":
        return u, v
    return u[0], v[0]
