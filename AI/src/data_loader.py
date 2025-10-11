# DriftCast data-access utilities for local HYCOM/WAVES/ERA5/GDP datasets.
# Provides lazy xarray loaders, time/space interpolation helpers, and a ForcingReader wrapper.
# Imported by the simulator, RL environments, and pipeline orchestration.

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from .config import DataConfig, get_lon_lat_names, get_time_name, index_datasets, normalize_longitudes, subset_bbox

LOGGER = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _resolve_files(root_or_paths: Union[PathLike, Iterable[PathLike], None], suffix: str) -> List[Path]:
    if root_or_paths is None:
        return []
    if isinstance(root_or_paths, (str, Path)):
        root = Path(root_or_paths)
        if root.is_dir():
            files = sorted(root.glob(f"*{suffix}"))
        else:
            files = [root]
    else:
        files = [Path(p) for p in root_or_paths]
    existing = [f for f in files if f.exists()]
    for missing in set(files) - set(existing):
        LOGGER.warning("Skipping missing file %s", missing)
    return existing


def open_dataset(
    paths: List[Path],
    bbox: Tuple[float, float, float, float],
    drop_vars: Optional[Sequence[str]] = None,
) -> Optional[xr.Dataset]:
    if not paths:
        return None
    ds = xr.open_mfdataset(
        paths,
        combine="by_coords",
        engine="netcdf4",
        decode_times=True,
        chunks="auto",
    )
    ds = normalize_longitudes(ds)
    ds = subset_bbox(ds, bbox)
    if drop_vars:
        ds = ds.drop_vars([v for v in drop_vars if v in ds])
    return ds


def load_currents(cfg: DataConfig, months: Optional[List[str]] = None) -> Optional[xr.Dataset]:
    index = index_datasets(cfg)
    selected = months if months is not None else index.mandatory_intersection()
    if not selected:  # fallback
        selected = sorted(index.currents.keys())
    files = [p for m in selected for p in index.currents.get(m, [])]
    if not files:
        LOGGER.warning("No current NetCDF files found.")
        return None
    return open_dataset(files, cfg.bbox)


def load_winds(cfg: DataConfig, months: Optional[List[str]] = None) -> Optional[xr.Dataset]:
    index = index_datasets(cfg)
    selected = months if months is not None else index.mandatory_intersection()
    if not selected:  # fallback
        selected = sorted(index.winds.keys())
    files = [p for m in selected for p in index.winds.get(m, [])]
    if not files:
        LOGGER.warning("No ERA5 wind NetCDF files found.")
        return None
    return open_dataset(files, cfg.bbox)


def load_stokes(cfg: DataConfig, months: Optional[List[str]] = None) -> Optional[xr.Dataset]:
    index = index_datasets(cfg)
    selected = months if months is not None else index.waves_intersection()
    if not selected:  # fallback
        selected = sorted(index.waves.keys())
    files = [p for m in selected for p in index.waves.get(m, [])]
    if not files:
        LOGGER.info("No Stokes drift NetCDF files present; falling back to wind-derived drift.")
        return None
    return open_dataset(files, cfg.bbox)


def load_drifters(cfg: DataConfig, months: Optional[List[str]] = None) -> pd.DataFrame:
    index = index_datasets(cfg)
    months = months or sorted(index.drifters.keys())
    frames: List[pd.DataFrame] = []

    for month in months:
        for path in index.drifters.get(month, []):
            try:
                # detect and skip a units row on line 2
                skiprows = None
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    _ = f.readline()
                    second = f.readline()
                    if second and (
                        "degrees_north" in second
                        or "degrees_east" in second
                        or second.strip().startswith(("UTC", "seconds since"))
                    ):
                        skiprows = [1]

                df = pd.read_csv(
                    path,
                    comment="#",
                    engine="python",
                    on_bad_lines="skip",
                    na_values=["NaN", "nan", "", "-999999", "-999999.0"],
                    skiprows=skiprows,
                    parse_dates=["time"],
                    date_format="ISO8601",  # pandas 2.x fast path
                )
            except Exception as exc:
                LOGGER.warning("Skipping %s: %s", path, exc)
                continue

            df = df.rename(
                columns={
                    "latitude": "lat",
                    "longitude": "lon",
                    "Latitude": "lat",
                    "Longitude": "lon",
                    "ID": "id",
                    "trajectory": "id",
                    "ve": "u",
                    "vn": "v",
                    "sst": "sea_surface_temperature",
                }
            )
            if "id" in df.columns:
                df["id"] = df["id"].astype(str)
            for c in ("lat", "lon", "u", "v", "sea_surface_temperature"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=[c for c in ("time", "lat", "lon") if c in df.columns])
            frames.append(df[[c for c in ("time", "lat", "lon", "id", "u", "v", "sea_surface_temperature") if c in df.columns]])

    if not frames:
        return pd.DataFrame(columns=["time", "lat", "lon", "id"])

    return pd.concat(frames, ignore_index=True)


def time_nearest_or_interp(ds: xr.Dataset, target: datetime) -> xr.Dataset:
    time_name = get_time_name(ds)
    if time_name is None:
        return ds
    target_ts = np.array(target, dtype="datetime64[ns]")
    try:
        result = ds.interp({time_name: target_ts}, kwargs={"fill_value": "extrapolate"})
    except ValueError:
        result = ds.sel({time_name: target_ts}, method="nearest")
    return result


def _select_variable(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def interpolate_vector(
    ds: Optional[xr.Dataset],
    var_u_candidates: Sequence[str],
    var_v_candidates: Sequence[str],
    time: datetime,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if ds is None:
        return np.zeros_like(lat, dtype=np.float32), np.zeros_like(lat, dtype=np.float32)
    var_u = _select_variable(ds, var_u_candidates)
    var_v = _select_variable(ds, var_v_candidates)
    if var_u is None or var_v is None:
        return np.zeros_like(lat, dtype=np.float32), np.zeros_like(lat, dtype=np.float32)
    ds_t = time_nearest_or_interp(ds, time)
    lon_name, lat_name = get_lon_lat_names(ds_t)
    sample = ds_t[[var_u, var_v]].interp(
        {lat_name: ("points", lat), lon_name: ("points", lon)},
        kwargs={"fill_value": "extrapolate"},
    ).compute()
    u = np.asarray(sample[var_u]).astype(np.float32)
    v = np.asarray(sample[var_v]).astype(np.float32)
    if sample[var_u].dims[0] == "points":
        return u, v
    return u[0], v[0]


def to_jax(array_like) -> jnp.ndarray:
    if isinstance(array_like, xr.DataArray):
        array_like = array_like.data
    elif isinstance(array_like, pd.Series):
        array_like = array_like.to_numpy()
    elif isinstance(array_like, pd.DataFrame):
        array_like = array_like.to_numpy()
    return jnp.asarray(array_like)


@dataclass
class ForcingReader:
    cfg: DataConfig
    currents: Optional[xr.Dataset]
    winds: Optional[xr.Dataset]
    stokes: Optional[xr.Dataset]

    def sample(self, time: datetime, lat: np.ndarray, lon: np.ndarray) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        u_curr, v_curr = interpolate_vector(
            self.currents,
            ("uo", "eastward_current", "u"),
            ("vo", "northward_current", "v"),
            time,
            lat,
            lon,
        )
        u_wind, v_wind = interpolate_vector(self.winds, ("u10", "u_component"), ("v10", "v_component"), time, lat, lon)
        if self.stokes is not None:
            u_stokes, v_stokes = interpolate_vector(self.stokes, ("vsdx",), ("vsdy",), time, lat, lon)
        else:
            scale = self.cfg.stokes_scale
            u_stokes = scale * u_wind
            v_stokes = scale * v_wind
        return {
            "currents": (to_jax(u_curr), to_jax(v_curr)),
            "winds": (to_jax(u_wind), to_jax(v_wind)),
            "stokes": (to_jax(u_stokes), to_jax(v_stokes)),
        }


def build_forcing_reader(cfg: DataConfig, months: Optional[List[str]] = None) -> ForcingReader:
    return ForcingReader(
        cfg=cfg,
        currents=load_currents(cfg, months),
        winds=load_winds(cfg, months),
        stokes=load_stokes(cfg, months),
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = DataConfig()
    reader = build_forcing_reader(cfg)
    info = load_drifters(cfg)
    LOGGER.info("Drifters rows: %d", len(info))
