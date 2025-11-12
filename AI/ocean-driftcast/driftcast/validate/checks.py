"""
File Summary:
- Computes driftcast validation "golden numbers" for regression sanity checks.
- Provides utilities to assert ranges and emit JSON validation reports.
- Designed to operate on existing NetCDF outputs without mutating them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple
import json

import numpy as np
import xarray as xr

GYRE_BOX_DEFAULT = {
    "lon_min": -70.0,
    "lon_max": -30.0,
    "lat_min": 20.0,
    "lat_max": 40.0,
}

DEFAULT_THRESHOLDS = {
    "final_gyre_fraction_min": 0.05,
    "final_gyre_fraction_max": 0.95,
    "percent_beached_max": 75.0,
    "mean_speed_min": 0.001,
    "mean_speed_max": 1.5,
    "curvature_index_mean_min": 0.001,
}


@dataclass(frozen=True)
class RunContext:
    dataset: xr.Dataset
    manifest: Dict[str, object]
    dataset_path: Path
    gyre_box: Dict[str, float]


def _locate_dataset(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path
    candidates = [
        run_path / "simulation.nc",
        run_path / "simulation.zarr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    nc_files = sorted(run_path.glob("*.nc"))
    if nc_files:
        return nc_files[0]
    raise FileNotFoundError(f"Could not locate dataset under {run_path}")


def _load_manifest(dataset_path: Path) -> Dict[str, object]:
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf8"))
    alt_path = dataset_path.parent / "manifest.json"
    if alt_path.exists():
        return json.loads(alt_path.read_text(encoding="utf8"))
    return {}


def _load_context(run_path: Path | str) -> RunContext:
    dataset_path = _locate_dataset(Path(run_path))
    manifest = _load_manifest(dataset_path)
    with xr.open_dataset(dataset_path) as ds:
        dataset = ds.load()
    gyre_box = (
        manifest.get("metrics", {}).get("gyre_box")  # type: ignore[assignment]
        if isinstance(manifest.get("metrics"), dict)
        else None
    )
    if not gyre_box:
        gyre_box = manifest.get("gyre_box") or {}
    resolved_box = {**GYRE_BOX_DEFAULT, **(gyre_box if isinstance(gyre_box, dict) else {})}
    return RunContext(dataset=dataset, manifest=manifest, dataset_path=dataset_path, gyre_box=resolved_box)


def _haversine(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c


def _gyre_mask(lon: np.ndarray, lat: np.ndarray, box: Dict[str, float]) -> np.ndarray:
    return (
        (lon >= box["lon_min"])
        & (lon <= box["lon_max"])
        & (lat >= box["lat_min"])
        & (lat <= box["lat_max"])
    )


def _curvature_index_per_particle(lon_traj: np.ndarray, lat_traj: np.ndarray) -> float:
    mask = np.isfinite(lon_traj) & np.isfinite(lat_traj)
    if mask.sum() < 3:
        return np.nan
    lon = lon_traj[mask]
    lat = lat_traj[mask]
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    dlon = np.diff(lon_rad)
    mean_lat = 0.5 * (lat_rad[1:] + lat_rad[:-1])
    dx = dlon * np.cos(mean_lat)
    dy = np.diff(lat_rad)
    headings = np.arctan2(dy, dx)
    if headings.size < 3:
        return np.nan
    turning = np.diff(np.unwrap(headings))
    dist_km = _haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    step_dist = np.maximum(dist_km[1:], 1e-3)
    curvature = np.rad2deg(np.abs(turning)) / step_dist
    if curvature.size == 0:
        return np.nan
    return float(np.nanmean(curvature))


def _mean_speed(dataset: xr.Dataset) -> float:
    if dataset.sizes.get("time", 0) < 2:
        return float("nan")
    lon = dataset["lon"].values
    lat = dataset["lat"].values
    time = dataset["time"].values
    mask = np.isfinite(lon) & np.isfinite(lat)
    lon_prev = lon[:-1]
    lon_next = lon[1:]
    lat_prev = lat[:-1]
    lat_next = lat[1:]
    mask_pair = mask[:-1] & mask[1:]
    distances_km = _haversine(lon_prev, lat_prev, lon_next, lat_next)
    distances_m = distances_km * 1000.0
    dt_seconds = np.diff(time.astype("datetime64[ns]").astype("float64")) / 1e9
    dt_array = dt_seconds[:, None]
    valid = (dt_array > 0) & mask_pair
    if not np.any(valid):
        return float("nan")
    speeds = np.where(valid, distances_m / dt_array, np.nan)
    return float(np.nanmean(speeds))


def compute_golden_numbers(run_path: Path | str) -> Dict[str, float]:
    """Compute golden-number diagnostics for a completed simulation."""
    ctx = _load_context(run_path)
    ds = ctx.dataset
    box = ctx.gyre_box
    lon = ds["lon"].values
    lat = ds["lat"].values
    beached = ds["beached"].values
    mask = np.isfinite(lon) & np.isfinite(lat)

    # Final snapshot metrics
    final_lon = lon[-1]
    final_lat = lat[-1]
    final_mask = mask[-1]
    afloat_mask = final_mask & ~beached[-1]
    total_afloat = np.count_nonzero(afloat_mask)
    inside_gyre = np.count_nonzero(afloat_mask & _gyre_mask(final_lon, final_lat, box))
    final_fraction = inside_gyre / total_afloat if total_afloat else 0.0
    percent_beached = float(np.count_nonzero(beached[-1]) / beached.shape[1] * 100.0)

    # Residence time statistics
    ages = ds["age_days"].values[-1]
    active_ages = ages[afloat_mask]
    median_residence = float(np.nanmedian(active_ages)) if active_ages.size else float("nan")

    # Curvature indices per particle
    curvature_values = []
    for particle in range(lon.shape[1]):
        curvature = _curvature_index_per_particle(lon[:, particle], lat[:, particle])
        if np.isfinite(curvature):
            curvature_values.append(curvature)
    curvature_arr = np.asarray(curvature_values) if curvature_values else np.array([np.nan])
    curvature_mean = float(np.nanmean(curvature_arr))
    curvature_p95 = float(np.nanpercentile(curvature_arr, 95)) if np.isfinite(curvature_arr).any() else float("nan")

    # Mean speed across the run
    mean_speed = _mean_speed(ds)

    metrics = {
        "final_gyre_fraction": float(final_fraction),
        "mean_speed": float(mean_speed),
        "percent_beached": float(percent_beached),
        "median_residence_days": float(median_residence),
        "curvature_index_mean": curvature_mean,
        "curvature_index_p95": curvature_p95,
    }
    return metrics


def assert_sane(run_path: Path | str, thresholds: Dict[str, float] | None = None) -> None:
    """Raise an AssertionError if golden numbers violate the provided thresholds."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    metrics = compute_golden_numbers(run_path)
    violations = []
    final_fraction = metrics["final_gyre_fraction"]
    if final_fraction < limits["final_gyre_fraction_min"] or final_fraction > limits["final_gyre_fraction_max"]:
        violations.append(f"final_gyre_fraction={final_fraction:.3f}")
    percent_beached = metrics["percent_beached"]
    if percent_beached > limits["percent_beached_max"]:
        violations.append(f"percent_beached={percent_beached:.1f}")
    mean_speed = metrics["mean_speed"]
    if (not np.isnan(mean_speed)) and (
        mean_speed < limits["mean_speed_min"] or mean_speed > limits["mean_speed_max"]
    ):
        violations.append(f"mean_speed={mean_speed:.4f}")
    curvature_mean = metrics["curvature_index_mean"]
    if not np.isnan(curvature_mean) and curvature_mean < limits["curvature_index_mean_min"]:
        violations.append(f"curvature_index_mean={curvature_mean:.4f}")
    if violations:
        raise AssertionError(
            "Validation sanity check failed: " + ", ".join(violations)
        )


def write_validation_report(
    run_path: Path | str,
    out: Path | str = Path("results/validation/report.json"),
) -> Path:
    """Write a JSON validation report containing manifest echo and golden numbers."""
    ctx = _load_context(run_path)
    metrics = compute_golden_numbers(run_path)
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": str(ctx.dataset_path.resolve()),
        "gyre_box": ctx.gyre_box,
        "metrics": metrics,
        "manifest": ctx.manifest,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    return out_path
