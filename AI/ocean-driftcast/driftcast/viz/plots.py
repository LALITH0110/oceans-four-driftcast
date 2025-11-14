"""
File Summary:
- Provides publication-grade static visualizations for Driftcast runs and configs.
- Loads simulation datasets or manifests, computes diagnostics, and saves PNG/SVG outputs.
- Exposed plot_* functions return Matplotlib figures while persisting artifacts under results/ and docs/.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm, colors, dates as mdates, pyplot as plt

from driftcast.config import DomainConfig, SimulationConfig, load_config
from driftcast.fields.gyres import GyreFieldConfig, gyre_velocity_field, streamfunction
from driftcast.post.density import particle_density
from driftcast.viz.map import make_basemap
from driftcast.viz.style import apply_style


FIGURE_WIDTH_PX = 1600
FIGURE_HEIGHT_PX = 900
HIGHRES_WIDTH_PX = 2400
HIGHRES_HEIGHT_PX = 1350
FIGURE_DPI = 300
RESULTS_FIG_DIR = Path("results/figures")
DOCS_ASSETS_DIR = Path("docs/assets")
DEFAULT_GYRE_CFG = GyreFieldConfig()
DEFAULT_GYRE_CENTER = (
    DEFAULT_GYRE_CFG.subtropical.center_lon,
    DEFAULT_GYRE_CFG.subtropical.center_lat,
)
GYRE_BOX = {"lon_min": -70.0, "lon_max": -30.0, "lat_min": 20.0, "lat_max": 40.0}


@dataclass(frozen=True)
class RunContext:
    """Container bundling dataset, manifest metadata, and convenience properties."""

    dataset: xr.Dataset
    manifest: Dict[str, object]
    dataset_path: Path
    run_label: str
    domain: DomainConfig

    @property
    def time_index(self) -> pd.DatetimeIndex:
        raw = self.dataset.coords.get("time")
        if raw is None:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(raw.values))


def _figure_size(width_px: int = FIGURE_WIDTH_PX, height_px: int = FIGURE_HEIGHT_PX) -> Tuple[float, float]:
    """Return matplotlib figsize tuple for the requested pixel dimensions."""
    return width_px / FIGURE_DPI, height_px / FIGURE_DPI


def _run_label_from_manifest(dataset_path: Path, manifest: Dict[str, object]) -> str:
    run_id = manifest.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id[:8]
    return dataset_path.stem.replace("simulation", "run")


def _load_dataset(dataset_path: Path) -> xr.Dataset:
    with xr.open_dataset(dataset_path) as ds:
        return ds.load()


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
    raise FileNotFoundError(f"Could not find dataset under {run_path}")


def _load_manifest(dataset_path: Path) -> Dict[str, object]:
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf8"))
    alt = dataset_path.parent / "manifest.json"
    if alt.exists():
        return json.loads(alt.read_text(encoding="utf8"))
    return {}


def _domain_from_manifest(manifest: Dict[str, object]) -> DomainConfig:
    payload = manifest.get("domain") or {}
    try:
        return DomainConfig(**payload)  # type: ignore[arg-type]
    except Exception:
        return DomainConfig()


def _prepare_run_context(run_path: Path | str) -> RunContext:
    dataset_path = _locate_dataset(Path(run_path))
    manifest = _load_manifest(dataset_path)
    dataset = _load_dataset(dataset_path)
    domain = _domain_from_manifest(manifest)
    label = _run_label_from_manifest(dataset_path, manifest)
    return RunContext(dataset=dataset, manifest=manifest, dataset_path=dataset_path, run_label=label, domain=domain)


def _ensure_output_dirs() -> None:
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, stem: str, save_svg: bool = False, dpi: int = FIGURE_DPI) -> Path:
    """Persist a figure to PNG (and optionally SVG) in both results/ and docs/ assets."""
    _ensure_output_dirs()
    filename = f"{stem}.png"
    png_path = RESULTS_FIG_DIR / filename
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", metadata={"Software": "driftcast"})
    docs_path = DOCS_ASSETS_DIR / filename
    if png_path.resolve() != docs_path.resolve():
        shutil.copyfile(png_path, docs_path)
    if save_svg:
        svg_name = f"{stem}.svg"
        svg_path = RESULTS_FIG_DIR / svg_name
        fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches="tight")
        shutil.copyfile(svg_path, DOCS_ASSETS_DIR / svg_name)
    return png_path


def _flatten_positions(lon: np.ndarray, lat: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    lon_flat = lon.reshape(-1)
    lat_flat = lat.reshape(-1)
    valid = np.isfinite(lon_flat) & np.isfinite(lat_flat)
    if mask is not None:
        valid &= mask.reshape(-1)
    return lon_flat[valid], lat_flat[valid]


def _histogram2d(
    lon: np.ndarray,
    lat: np.ndarray,
    domain: DomainConfig,
    bins: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lon.size == 0:
        lon_edges = np.linspace(domain.lon_min, domain.lon_max, bins[0] + 1)
        lat_edges = np.linspace(domain.lat_min, domain.lat_max, bins[1] + 1)
        return np.zeros((bins[1], bins[0])), lon_edges, lat_edges
    lon_edges = np.linspace(domain.lon_min, domain.lon_max, bins[0] + 1)
    lat_edges = np.linspace(domain.lat_min, domain.lat_max, bins[1] + 1)
    hist, _, _ = np.histogram2d(lon, lat, bins=[lon_edges, lat_edges])
    return hist.T, lon_edges, lat_edges


def _gyre_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return (
        (lon >= GYRE_BOX["lon_min"])
        & (lon <= GYRE_BOX["lon_max"])
        & (lat >= GYRE_BOX["lat_min"])
        & (lat <= GYRE_BOX["lat_max"])
    )


def _time_series_counts(ctx: RunContext) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    beached = ctx.dataset["beached"].values
    valid = np.isfinite(lon) & np.isfinite(lat)
    afloat = np.sum(valid & ~beached, axis=1)
    beached_count = np.sum(beached, axis=1)
    gyre = np.sum(valid & ~beached & _gyre_mask(lon, lat), axis=1)
    return ctx.time_index, afloat, beached_count, gyre


def _final_snapshot(ctx: RunContext) -> xr.Dataset:
    if ctx.dataset.sizes.get("time", 0) == 0:
        return ctx.dataset
    return ctx.dataset.isel(time=-1)


def _compute_curvature(lon_track: np.ndarray, lat_track: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(lon_track) & np.isfinite(lat_track)
    if mask.sum() < 3:
        return np.array([]), np.array([]), np.array([])
    lon_valid = lon_track[mask]
    lat_valid = lat_track[mask]
    lon_rad = np.deg2rad(lon_valid)
    lat_rad = np.deg2rad(lat_valid)
    dx = np.diff(lon_rad) * np.cos(0.5 * (lat_rad[1:] + lat_rad[:-1]))
    dy = np.diff(lat_rad)
    headings = np.arctan2(dy, dx)
    if headings.size < 2:
        return np.array([]), np.array([]), np.array([])
    delta_heading = np.diff(np.unwrap(headings))
    curvature = np.abs(np.rad2deg(delta_heading))
    sample_lon = lon_valid[1:-1]
    sample_lat = lat_valid[1:-1]
    return sample_lon, sample_lat, curvature


def _haversine_km(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float | np.ndarray,
    lat0: float | np.ndarray,
) -> np.ndarray:
    """Great-circle separation (km) supporting scalar or array reference points."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_arr = np.asarray(lon0, dtype=float)
    lat0_arr = np.asarray(lat0, dtype=float)
    lon0_rad = np.deg2rad(lon0_arr)
    lat0_rad = np.deg2rad(lat0_arr)
    lon0_b, lat0_b = np.broadcast_arrays(lon0_rad, lat0_rad, lon_rad)
    dlon = lon_rad - lon0_b
    dlat = lat_rad - lat0_b
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat0_b) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c


def _density_radial_profile(density: xr.DataArray, center: Tuple[float, float], radial_bins_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(density.coords["lon"].values, density.coords["lat"].values)
    distances = _haversine_km(lon_grid, lat_grid, center[0], center[1])
    distances_flat = distances.ravel()
    weights = density.values.ravel()
    counts, bin_edges = np.histogram(distances_flat, bins=radial_bins_km, weights=weights)
    cell_counts, _ = np.histogram(distances_flat, bins=radial_bins_km, weights=np.ones_like(weights))
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(cell_counts > 0, counts / cell_counts, 0.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, profile


def _gyre_fraction_series(ctx: RunContext) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    time_index, afloat, _, gyre = _time_series_counts(ctx)
    fractions = np.zeros_like(afloat, dtype=float)
    valid = afloat > 0
    fractions[valid] = gyre[valid] / afloat[valid]
    return time_index, fractions


def _logistic_fit(days: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    if fractions.size == 0:
        return fractions
    clipped = np.clip(fractions, 1e-4, 1.0 - 1e-4)
    valid = (clipped > 0.0) & (clipped < 1.0)
    if np.count_nonzero(valid) < 2:
        return clipped
    logit = np.log(clipped[valid] / (1.0 - clipped[valid]))
    slope, intercept = np.polyfit(days[valid], logit, 1)
    fitted = slope * days + intercept
    return 1.0 / (1.0 + np.exp(-fitted))


def _curvature_index_samples(ctx: RunContext) -> np.ndarray:
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    samples: List[float] = []
    for pid in range(lon.shape[1]):
        curvature = _curvature_index_per_particle(lon[:, pid], lat[:, pid])
        if np.isfinite(curvature):
            samples.append(curvature)
    return np.asarray(samples) if samples else np.array([])


def _curvature_index_per_particle(lon_track: np.ndarray, lat_track: np.ndarray) -> float:
    mask = np.isfinite(lon_track) & np.isfinite(lat_track)
    if mask.sum() < 3:
        return float("nan")
    lon = lon_track[mask]
    lat = lat_track[mask]
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    dlon = np.diff(lon_rad)
    mid_lat = 0.5 * (lat_rad[1:] + lat_rad[:-1])
    dx = dlon * np.cos(mid_lat)
    dy = np.diff(lat_rad)
    headings = np.arctan2(dy, dx)
    if headings.size < 3:
        return float("nan")
    turning = np.diff(np.unwrap(headings))
    dist_km = _haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    step = np.maximum(dist_km[1:], 1e-3)
    curvature = np.rad2deg(np.abs(turning)) / step
    if curvature.size == 0:
        return float("nan")
    return float(np.nanmean(curvature))


# --------------------------------------------------------------------------- #
# Plot functions
# --------------------------------------------------------------------------- #


def plot_accumulation_heatmap(
    run_path: Path | str,
    bins: Tuple[int, int] = (300, 200),
    cmap: str = "viridis",
) -> plt.Figure:
    """2-D histogram of particle positions over the entire run."""
    apply_style()
    ctx = _prepare_run_context(Path(run_path))
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    lon_flat, lat_flat = _flatten_positions(lon, lat)
    hist, lon_edges, lat_edges = _histogram2d(lon_flat, lat_flat, ctx.domain, bins)
    fig, ax = make_basemap(ctx.domain, figsize=_figure_size())
    vmin = 1.0 if np.any(hist > 0) else 0.1
    vmax = float(hist.max()) if np.any(hist) else 1.0
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        hist,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        norm=colors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
    )
    center_lon, center_lat = DEFAULT_GYRE_CENTER
    ax.scatter(center_lon, center_lat, s=120, marker="x", color="#f8d568", linewidths=2.0, transform=ccrs.PlateCarree())
    ax.text(center_lon + 2.0, center_lat + 1.0, "Gyre center", color="#f8d568", fontsize=9, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Particle visits (log scale)")
    ax.set_title("Accumulation Heatmap")
    stem = f"accumulation_heatmap_{ctx.run_label}"
    _save_figure(fig, stem, save_svg=True)
    return fig


def plot_streamfunction_contours(
    config_path: Path | str,
    t: float,
) -> plt.Figure:
    """Visualize analytic streamfunction and velocity quivers."""
    apply_style()
    cfg: SimulationConfig = load_config(config_path)
    domain = cfg.domain
    lon = np.linspace(domain.lon_min, domain.lon_max, 180)
    lat = np.linspace(domain.lat_min, domain.lat_max, 120)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    psi = streamfunction(lon_grid, lat_grid, time_days=t, config=DEFAULT_GYRE_CFG)
    u, v = gyre_velocity_field(lon_grid, lat_grid, time_days=t, config=DEFAULT_GYRE_CFG)
    speed = np.hypot(u, v)
    fig, ax = make_basemap(domain, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    contour_levels = np.linspace(np.min(psi), np.max(psi), 15)
    cs = ax.contour(lon_grid, lat_grid, psi, levels=contour_levels, colors="#4e79a7", linewidths=0.6, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fmt="%.0f", fontsize=6)
    stride = (slice(None, None, 8), slice(None, None, 8))
    ax.quiver(
        lon_grid[stride],
        lat_grid[stride],
        u[stride],
        v[stride],
        speed[stride],
        transform=ccrs.PlateCarree(),
        cmap="plasma",
        scale=400.0,
        width=0.002,
    )
    labels = {
        "Gulf Stream": (-70.0, 35.0),
        "North Atlantic Current": (-45.0, 45.0),
        "Azores Current": (-35.0, 30.0),
        "Canary Current": (-20.0, 30.0),
    }
    for name, (lon_pt, lat_pt) in labels.items():
        ax.text(lon_pt, lat_pt, name, color="#f1f0ea", fontsize=8, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=speed.min(), vmax=speed.max()), cmap="plasma"), ax=ax, orientation="vertical", shrink=0.65)
    cbar.set_label("Speed (m s$^{-1}$)")
    ax.set_title(f"Streamfunction & Currents (t={t:.1f} days)")
    stem = f"streamfunction_contours_{Path(config_path).stem}"
    _save_figure(fig, stem, save_svg=True)
    return fig


def plot_source_mix_pie(run_path: Path | str) -> plt.Figure:
    """Pie chart of final composition by source categories."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    final = _final_snapshot(ctx)
    sources = ctx.dataset.coords.get("source_name")
    if sources is None or sources.size == 0:
        labels = ["no-data"]
        counts = np.array([1])
    else:
        source_array = np.array(sources.values, dtype=str)
        afloat_mask = np.isfinite(final["lon"].values) & np.isfinite(final["lat"].values) & ~final["beached"].values
        counts_series = pd.Series(source_array[afloat_mask]).value_counts()
        if counts_series.empty:
            counts_series = pd.Series(source_array).value_counts()
        labels = counts_series.index.tolist()
        counts = counts_series.values.astype(float)
    fig, ax = plt.subplots(figsize=_figure_size(), subplot_kw={"aspect": "equal"})
    colors_cycle = plt.colormaps["Set2"](np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[label.replace("_", " ") for label in labels],
        autopct="%1.1f%%",
        colors=colors_cycle,
        startangle=90,
        pctdistance=0.8,
    )
    for wedge in wedges:
        wedge.set_edgecolor("#0b1d3a")
    ax.set_title("Final Composition by Source (afloat)")
    stem = f"source_mix_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_source_contribution_map(run_path: Path | str) -> plt.Figure:
    """Small-multiple density maps per source with shared colorbar."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    sources = ctx.dataset.coords.get("source_name")
    if sources is None or sources.size == 0:
        unique_sources = ["all"]
    else:
        unique_sources = sorted({str(val) for val in sources.values})
    panels = min(len(unique_sources), 3)
    total_panels = panels + 1 if panels < 4 else panels
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.ravel()
    density_maps: List[np.ndarray] = []
    lon_edges = np.linspace(ctx.domain.lon_min, ctx.domain.lon_max, 181)
    lat_edges = np.linspace(ctx.domain.lat_min, ctx.domain.lat_max, 121)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    if len(unique_sources) == 1:
        masks = [np.ones(lon.shape[1], dtype=bool)]
        labels = ["All sources"]
    else:
        source_names = np.array(sources.values, dtype=str)
        masks = [(source_names == name) for name in unique_sources[:panels]]
        labels = [name.replace("_", " ") for name in unique_sources[:panels]]
    for mask in masks:
        sub_lon = lon[:, mask]
        sub_lat = lat[:, mask]
        lon_flat, lat_flat = _flatten_positions(sub_lon, sub_lat)
        hist, _, _ = _histogram2d(lon_flat, lat_flat, ctx.domain, bins=(180, 120))
        density_maps.append(hist)
    combined_hist, _, _ = _histogram2d(*_flatten_positions(lon, lat), ctx.domain, bins=(180, 120))
    density_maps.append(combined_hist)
    labels.append("All sources")
    vmax = max([np.nanmax(hist) for hist in density_maps]) or 1.0
    for ax, hist, label in zip(axes, density_maps, labels):
        ax.set_extent([ctx.domain.lon_min, ctx.domain.lon_max, ctx.domain.lat_min, ctx.domain.lat_max], ccrs.PlateCarree())
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            hist,
            transform=ccrs.PlateCarree(),
            cmap="magma",
            norm=colors.LogNorm(vmin=1.0, vmax=vmax),
        )
        ax.coastlines(resolution="50m", color="#d6d4c6", linewidth=0.4)
        ax.set_title(label)
    for idx in range(len(density_maps), axes.size):
        axes[idx].axis("off")
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=colors.LogNorm(vmin=1.0, vmax=vmax), cmap="magma"),
        ax=axes.tolist(),
        location="bottom",
        fraction=0.05,
        pad=0.05,
    )
    cbar.set_label("Particle visits (log scale)")
    fig.suptitle("Source Contribution Density Maps", fontsize=14)
    stem = f"source_contribution_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_beaching_hotspots(run_path: Path | str) -> plt.Figure:
    """Choropleth-style depiction of beached particle counts by coastal bins."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    final = _final_snapshot(ctx)
    lon = np.array(final["lon"].values)
    lat = np.array(final["lat"].values)
    beached = np.array(final["beached"].values)
    beach_lon = lon[beached & np.isfinite(lon)]
    beach_lat = lat[beached & np.isfinite(lat)]
    if beach_lon.size == 0:
        beach_lon = np.array([ctx.domain.lon_min])
        beach_lat = np.array([ctx.domain.lat_min])
    hist, lon_edges, lat_edges = _histogram2d(beach_lon, beach_lat, ctx.domain, bins=(90, 60))
    fig, ax = make_basemap(ctx.domain, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        hist,
        cmap="inferno",
        transform=ccrs.PlateCarree(),
        norm=colors.LogNorm(vmin=1.0, vmax=hist.max() or 1.0),
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    cbar.set_label("Beached particle count (log scale)")
    ax.set_title("Beaching Hotspots by Coastal Segment")
    stem = f"beaching_hotspots_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_residence_time(run_path: Path | str) -> plt.Figure:
    """Average in-water residence time per spatial bin."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    age = ctx.dataset["age_days"].values
    lon_flat, lat_flat = _flatten_positions(lon, lat)
    age_flat = age.reshape(-1)
    age_flat = age_flat[np.isfinite(lon_flat) & np.isfinite(lat_flat)]
    hist_counts, lon_edges, lat_edges = _histogram2d(lon_flat, lat_flat, ctx.domain, bins=(160, 100))
    if hist_counts.size == 0:
        mean_age = hist_counts
    else:
        weighted_sum, _, _ = np.histogram2d(
            lon_flat,
            lat_flat,
            bins=[lon_edges, lat_edges],
            weights=age_flat,
        )
        weighted_sum = weighted_sum.T
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_age = np.where(hist_counts > 0, weighted_sum / hist_counts, 0.0)
    fig, ax = make_basemap(ctx.domain, figsize=_figure_size())
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        mean_age,
        cmap="cividis",
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean residence time (days)")
    ax.set_title("Residence Time Map")
    stem = f"residence_time_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_age_histogram(run_path: Path | str) -> plt.Figure:
    """Histogram of particle ages highlighting beached vs afloat populations."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    final = _final_snapshot(ctx)
    age = np.array(final["age_days"].values)
    beached = np.array(final["beached"].values)
    afloat = age[~beached & np.isfinite(age)]
    stranded = age[beached & np.isfinite(age)]
    bins = np.linspace(0, max(float(np.nanmax(age)) if np.isfinite(age).any() else 1.0, 5.0), 30)
    fig, ax = plt.subplots(figsize=_figure_size())
    ax.hist(afloat, bins=bins, alpha=0.7, label="Afloat", color="#4e79a7")
    ax.hist(stranded, bins=bins, alpha=0.7, label="Beached", color="#e15759")
    ax.set_xlabel("Particle age (days)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Final Particle Age Distribution")
    stem = f"age_histogram_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_time_series(run_path: Path | str) -> plt.Figure:
    """Time series of afloat, beached, and gyre-contained particle counts."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    time_index, afloat, beached_count, gyre = _time_series_counts(ctx)
    fig, ax = plt.subplots(figsize=_figure_size())
    if afloat.size == 0:
        ax.text(0.5, 0.5, "No particle records", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.plot(time_index, afloat, label="Afloat", color="#4e79a7")
        ax.plot(time_index, beached_count, label="Beached", color="#e15759")
        ax.plot(time_index, gyre, label="Inside gyre box", color="#f28e2b")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel("Particle count")
    ax.legend()
    ax.set_title("Particle Status Time Series")
    stem = f"time_series_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_hovmoller_lat_density(run_path: Path | str) -> plt.Figure:
    """Latitude-time Hovmöller diagram of zonal-mean density."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    times = ctx.time_index
    lat_bins = np.linspace(ctx.domain.lat_min, ctx.domain.lat_max, 120)
    hov = []
    for t_idx in range(lon.shape[0]):
        lon_t = lon[t_idx]
        lat_t = lat[t_idx]
        mask = np.isfinite(lon_t) & np.isfinite(lat_t)
        counts, _ = np.histogram(lat_t[mask], bins=lat_bins)
        hov.append(counts)
    hov_array = np.array(hov) if hov else np.zeros((1, lat_bins.size - 1))
    fig, ax = plt.subplots(figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    time_numeric = np.arange(hov_array.shape[0])
    mesh = ax.pcolormesh(time_numeric, lat_centers, hov_array.T, cmap="inferno", shading="auto")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlabel("Time")
    if len(times):
        ax.set_xticks(time_numeric)
        ax.set_xticklabels(pd.to_datetime(times).strftime("%b %d"), rotation=45, ha="right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Particle count")
    ax.set_title("Hovmöller (Latitude-Time) Density")
    stem = f"hovmoller_lat_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_parameter_sweep_matrix(
    sweep_dir: Path | str,
    metric: str = "gyre_fraction",
) -> plt.Figure:
    """Heatmap of metric values across windage and diffusivity parameter sweeps."""
    apply_style()
    sweep_path = Path(sweep_dir)
    nc_files = sorted(sweep_path.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF outputs discovered in {sweep_path}")
    records: List[Dict[str, float]] = []
    for nc_file in nc_files:
        manifest = _load_manifest(nc_file)
        physics = manifest.get("physics") or {}
        windage = float(physics.get("windage_coeff", np.nan))
        diffusivity = float(physics.get("diffusivity_m2s", np.nan))
        ctx = _prepare_run_context(nc_file)
        final = _final_snapshot(ctx)
        lon = np.array(final["lon"].values)
        lat = np.array(final["lat"].values)
        beached = np.array(final["beached"].values)
        mask = np.isfinite(lon) & np.isfinite(lat) & ~beached
        total = mask.sum()
        gyre_total = np.sum(mask & _gyre_mask(lon, lat))
        fraction = (gyre_total / total) if total else 0.0
        metric_value = fraction if metric == "gyre_fraction" else fraction
        records.append({"windage": windage, "diffusivity": diffusivity, "value": metric_value})
    frame = pd.DataFrame(records).dropna()
    if frame.empty:
        raise ValueError("Sweep manifests missing physics metadata; cannot build matrix.")
    pivot = frame.pivot_table(index="diffusivity", columns="windage", values="value")
    fig, ax = plt.subplots(figsize=_figure_size())
    im = ax.imshow(pivot.values, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{val:.3f}" for val in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{val:.0f}" for val in pivot.index])
    ax.set_xlabel("Windage coefficient")
    ax.set_ylabel("Diffusivity Kh (m²/s)")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            ax.text(x, y, f"{pivot.values[y, x]:.2f}", ha="center", va="center", color="#0b1d3a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction inside gyre box")
    ax.set_title("Parameter Sweep Gyre Fraction")
    stem = f"parameter_sweep_{sweep_path.name}"
    _save_figure(fig, stem, save_svg=True)
    return fig


def plot_traj_bundle(run_path: Path | str, n: int = 500) -> plt.Figure:
    """Subsample trajectories with curvature-based colouring to highlight swirling."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    n_particles = lon.shape[1] if lon.ndim == 2 else 0
    fig, ax = make_basemap(ctx.domain, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    if n_particles == 0:
        ax.text(0.5, 0.5, "No trajectories available", transform=ax.transAxes, ha="center", va="center")
        stem = f"traj_bundle_{ctx.run_label}"
        _save_figure(fig, stem)
        return fig
    rng = np.random.default_rng(0)
    choices = rng.choice(n_particles, size=min(n, n_particles), replace=False)
    cmap = plt.colormaps["magma"]
    curvatures: List[float] = []
    for idx in choices:
        sample_lon, sample_lat, curvature = _compute_curvature(lon[:, idx], lat[:, idx])
        if curvature.size == 0:
            continue
        mean_curv = float(np.nanmean(curvature))
        curvatures.append(mean_curv)
        color = cmap(min(mean_curv / 15.0, 1.0))
        ax.plot(
            lon[:, idx],
            lat[:, idx],
            color=color,
            linewidth=0.5,
            alpha=0.65,
            transform=ccrs.Geodetic(),
        )
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=max(curvatures or [1.0])), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    cbar.set_label("Mean heading change per step (degrees)")
    ax.set_title("Trajectory Bundle with Curvature Shading")
    stem = f"traj_bundle_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_curvature_map(run_path: Path | str) -> plt.Figure:
    """Spatial grid of mean curvature to highlight gyre swirling vs straight tracks."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    bins = (160, 120)
    lon_edges = np.linspace(ctx.domain.lon_min, ctx.domain.lon_max, bins[0] + 1)
    lat_edges = np.linspace(ctx.domain.lat_min, ctx.domain.lat_max, bins[1] + 1)
    accum = np.zeros((bins[1], bins[0]))
    counts = np.zeros_like(accum)
    for idx in range(lon.shape[1]):
        sample_lon, sample_lat, curvature = _compute_curvature(lon[:, idx], lat[:, idx])
        if curvature.size == 0:
            continue
        hist, _, _ = _histogram2d(sample_lon, sample_lat, ctx.domain, bins=bins)
        accum += hist * (np.nanmean(curvature) if np.isfinite(curvature).any() else 0.0)
        counts += hist
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature_mean = np.where(counts > 0, accum / counts, 0.0)
    fig, ax = make_basemap(ctx.domain, figsize=_figure_size())
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        curvature_mean,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        vmin=0.0,
        vmax=max(float(curvature_mean.max()), 1.0),
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label("Mean curvature (deg)")
    ax.set_title("Trajectory Curvature Map")
    stem = f"curvature_map_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_release_schedule(config_path: Path | str) -> plt.Figure:
    """Timeline of seeding by scenario derived from configuration file."""
    apply_style()
    cfg = load_config(config_path)
    start = cfg.time.start
    end = cfg.time.end
    days = pd.date_range(start=start, end=end, freq="D")
    if days.empty:
        days = pd.date_range(start=start, periods=1, freq="D")
    frame = pd.DataFrame(index=days)
    for source in cfg.sources:
        src_start = source.start or cfg.time.start
        src_end = source.end or cfg.time.end
        src_days = pd.date_range(start=src_start, end=src_end, freq="D")
        if src_days.empty:
            src_days = pd.date_range(start=src_start, periods=1, freq="D")
        series = pd.Series(source.rate_per_day, index=src_days)
        frame[source.name] = series.reindex(frame.index, fill_value=0.0)
    frame.fillna(0.0, inplace=True)
    fig, ax = plt.subplots(figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    if frame.empty or frame.sum().sum() == 0:
        ax.text(0.5, 0.5, "No sources configured", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.stackplot(frame.index, frame.T.values, labels=[name.replace("_", " ") for name in frame.columns])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    ax.set_ylabel("Expected particles per day")
    ax.set_title(f"Release Schedule – {Path(config_path).stem}")
    if frame.columns.size:
        ax.legend(loc="upper right")
    stem = f"release_schedule_{Path(config_path).stem}"
    _save_figure(fig, stem)
    return fig


def plot_density_vs_distance_to_gyre_center(run_path: Path | str) -> plt.Figure:
    """Radial density profile relative to the subtropical gyre center."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    final = _final_snapshot(ctx)
    lon = final["lon"].values
    lat = final["lat"].values
    mask = np.isfinite(lon) & np.isfinite(lat)
    density = particle_density(lon[mask], lat[mask], ctx.domain, resolution_deg=ctx.domain.resolution_deg)
    radial_bins = np.linspace(0, 2500, 40)
    centers, profile = _density_radial_profile(density, DEFAULT_GYRE_CENTER, radial_bins)
    fig, ax = plt.subplots(figsize=_figure_size())
    ax.plot(centers, profile, color="#4e79a7")
    ax.fill_between(centers, profile, alpha=0.3, color="#4e79a7")
    ax.axvline(700, color="#f28e2b", linestyle="--", linewidth=1.0, label="Gyre core ~700 km")
    ax.set_xlabel("Distance from gyre center (km)")
    ax.set_ylabel("Mean particle density")
    ax.set_title("Density vs Distance to Gyre Center")
    ax.legend()
    stem = f"density_vs_distance_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_hotspot_rank(run_path: Path | str) -> plt.Figure:
    """Zipf-like rank plot of the top accumulation cells."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    lon = ctx.dataset["lon"].values
    lat = ctx.dataset["lat"].values
    lon_flat, lat_flat = _flatten_positions(lon, lat)
    density = particle_density(lon_flat, lat_flat, ctx.domain, resolution_deg=ctx.domain.resolution_deg)
    stacked = density.stack(cell=("lat", "lon"))
    sorted_counts = stacked.sortby(stacked, ascending=False).values[:50]
    ranks = np.arange(1, sorted_counts.size + 1)
    fig, ax = plt.subplots(figsize=_figure_size())
    ax.plot(ranks, sorted_counts, marker="o", color="#f28e2b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cell visits")
    ax.set_title("Hotspot Rank Distribution (Top 50)")
    stem = f"hotspot_rank_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_compare_presets(run_paths: Sequence[Path | str]) -> plt.Figure:
    """Multi-panel comparison of accumulation heatmaps for different presets."""
    apply_style()
    run_paths = list(run_paths)
    if not run_paths:
        raise ValueError("plot_compare_presets requires at least one run path.")
    contexts = [_prepare_run_context(path) for path in run_paths]
    fig, axes = plt.subplots(1, len(contexts), figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX), subplot_kw={"projection": ccrs.PlateCarree()})
    if len(contexts) == 1:
        axes = [axes]  # type: ignore[list-item]
    vmax = 0.0
    histograms: List[np.ndarray] = []
    lon_edges = lat_edges = None
    for ctx in contexts:
        lon_flat, lat_flat = _flatten_positions(ctx.dataset["lon"].values, ctx.dataset["lat"].values)
        hist, lon_edges, lat_edges = _histogram2d(lon_flat, lat_flat, ctx.domain, bins=(200, 140))
        histograms.append(hist)
        vmax = max(vmax, float(hist.max()))
    for ax, ctx, hist in zip(axes, contexts, histograms):
        ax.set_extent([ctx.domain.lon_min, ctx.domain.lon_max, ctx.domain.lat_min, ctx.domain.lat_max], ccrs.PlateCarree())
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            hist,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            norm=colors.LogNorm(vmin=1.0, vmax=vmax or 1.0),
        )
        ax.coastlines(resolution="50m", color="#d6d4c6", linewidth=0.4)
        title = (
            ctx.dataset.attrs.get("preset_name")
            or ctx.manifest.get("preset_name")
            or ctx.dataset_path.parent.name
            or ctx.run_label
        )
        ax.set_title(str(title).replace("_", " "))
    fig.suptitle("Preset Comparison – Accumulation Heatmaps", fontsize=14)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=colors.LogNorm(vmin=1.0, vmax=vmax or 1.0), cmap="viridis"),
        ax=axes,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Particle visits (log scale)")
    stem = "compare_presets_" + "_".join(ctx.run_label for ctx in contexts)
    _save_figure(fig, stem, save_svg=True)
    return fig


def plot_gyre_fraction_curve(run_path: Path | str) -> plt.Figure:
    """Fraction of particles inside the gyre box over time with logistic fit."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    time_index, fractions = _gyre_fraction_series(ctx)
    if fractions.size == 0:
        fractions = np.array([0.0])
    days = (
        (time_index - time_index[0]).total_seconds().to_numpy() / 86400.0
        if time_index.size
        else np.arange(fractions.size, dtype=float)
    )
    logistic = _logistic_fit(days.astype(float), fractions.astype(float))
    fig, ax = plt.subplots(figsize=_figure_size())
    ax.plot(time_index, fractions, color="#4e79a7", label="Observed fraction", marker="o")
    ax.plot(time_index, logistic, color="#f28e2b", linestyle="--", label="Logistic fit")
    ax.set_ylabel("Fraction inside gyre")
    ax.set_xlabel("Time")
    if time_index.size:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.set_title("Gyre Occupancy Trajectory")
    stem = f"gyre_fraction_curve_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_curvature_cdf(run_path: Path | str) -> plt.Figure:
    """Cumulative distribution of curvature indices to highlight swirling motion."""
    apply_style()
    ctx = _prepare_run_context(run_path)
    samples = np.sort(_curvature_index_samples(ctx))
    if samples.size == 0:
        samples = np.array([0.0])
    cdf = np.linspace(0.0, 1.0, samples.size, endpoint=False) + (0.5 / samples.size)
    fig, ax = plt.subplots(figsize=_figure_size())
    ax.plot(samples, cdf, color="#af7aa1", linewidth=2.0)
    ax.set_xlabel("Curvature index (deg km$^{-1}$)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(color="#45586a", linestyle="--", linewidth=0.3)
    ax.set_title("Curvature Distribution")
    stem = f"curvature_cdf_{ctx.run_label}"
    _save_figure(fig, stem)
    return fig


def plot_ekman_vs_noekman(run_paths: Sequence[Path | str]) -> plt.Figure:
    """Side-by-side comparison of accumulation heatmaps with and without Ekman drift."""
    if len(run_paths) != 2:
        raise ValueError("plot_ekman_vs_noekman expects exactly two run paths.")
    apply_style()
    contexts = [_prepare_run_context(path) for path in run_paths]
    fig, axes = plt.subplots(1, 2, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX), subplot_kw={"projection": ccrs.PlateCarree()})
    heatmaps: List[np.ndarray] = []
    lon_edges = lat_edges = None
    for ctx in contexts:
        lon_flat, lat_flat = _flatten_positions(ctx.dataset["lon"].values, ctx.dataset["lat"].values)
        hist, lon_edges, lat_edges = _histogram2d(lon_flat, lat_flat, ctx.domain, bins=(200, 140))
        heatmaps.append(hist)
    vmax = max(float(hist.max()) for hist in heatmaps) if heatmaps else 1.0
    if vmax > 1.0:
        shared_norm: colors.Normalize = colors.LogNorm(vmin=1.0, vmax=vmax)
    else:
        shared_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    for ax, ctx, label, hist in zip(
        axes,
        contexts,
        ["Ekman OFF", "Ekman ON"],
        heatmaps,
    ):
        ax.pcolormesh(
            lon_edges,
            lat_edges,
            hist,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            norm=shared_norm,
        )
        ax.coastlines(resolution="50m", color="#d6d4c6", linewidth=0.4)
        ax.set_title(label)
    fig.colorbar(
        cm.ScalarMappable(norm=shared_norm, cmap="viridis"),
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Particle visits (log scale)" if vmax > 1.0 else "Particle visits",
    )
    fig.suptitle("Ekman Drift Impact on Accumulation", fontsize=14)
    stem = "ekman_vs_noekman_" + "_".join(ctx.run_label for ctx in contexts)
    _save_figure(fig, stem)
    return fig


def plot_seasonal_ramp_effect(run_paths: Sequence[Path | str]) -> plt.Figure:
    """Compare gyre fraction curves with seasonal ramp disabled/enabled."""
    if len(run_paths) != 2:
        raise ValueError("plot_seasonal_ramp_effect expects exactly two run paths.")
    apply_style()
    contexts = [_prepare_run_context(path) for path in run_paths]
    fig, axes = plt.subplots(1, 2, figsize=_figure_size(HIGHRES_WIDTH_PX, HIGHRES_HEIGHT_PX))
    labels = ["Seasonal Ramp OFF", "Seasonal Ramp ON"]
    for ax, ctx, label in zip(axes, contexts, labels):
        time_index, fractions = _gyre_fraction_series(ctx)
        days = (
            (time_index - time_index[0]).total_seconds().to_numpy() / 86400.0
            if time_index.size
            else np.arange(fractions.size, dtype=float)
        )
        logistic = _logistic_fit(days.astype(float), fractions.astype(float))
        ax.plot(time_index, fractions, color="#4e79a7", marker="o", label="Observed")
        ax.plot(time_index, logistic, color="#f28e2b", linestyle="--", label="Logistic")
        if time_index.size:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_horizontalalignment("right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(label)
        ax.set_ylabel("Fraction inside gyre")
        ax.set_xlabel("Time")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Seasonal Forcing Effect on Gyre Occupancy", fontsize=14)
    stem = "seasonal_ramp_effect_" + "_".join(ctx.run_label for ctx in contexts)
    _save_figure(fig, stem)
    return fig
