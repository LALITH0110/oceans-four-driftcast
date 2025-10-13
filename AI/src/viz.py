# # DriftCast plotting helpers for trajectories, plastic heatmaps, and animations.
# # Works with numpy/jax arrays and gracefully falls back when cartopy/imageio are absent.
# # Invoked by the CLI simulate/predict commands and during pipeline reporting.

# from __future__ import annotations

# import logging
# from pathlib import Path
# from typing import Iterable, Optional, Sequence

# import numpy as np

# LOGGER = logging.getLogger(__name__)


# def _get_matplotlib():
#     import matplotlib.pyplot as plt  # type: ignore

#     return plt


# def _get_cartopy():
#     try:
#         import cartopy.crs as ccrs  # type: ignore
#         import cartopy.feature as cfeature  # type: ignore
#         return ccrs, cfeature
#     except ImportError:  # pragma: no cover
#         return None, None


# def plot_trajectories(
#     baseline: np.ndarray,
#     corrected: Optional[np.ndarray],
#     output_path: Path,
#     bbox: Sequence[float],
# ) -> Path:
#     plt = _get_matplotlib()
#     ccrs, cfeature = _get_cartopy()
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     if ccrs:
#         ax = plt.axes(projection=ccrs.PlateCarree())
#         ax.coastlines(resolution="110m")
#         ax.add_feature(cfeature.LAND, facecolor="lightgray")
#         ax.set_extent([bbox[2], bbox[3], bbox[1], bbox[0]])
#     else:
#         fig, ax = plt.subplots(figsize=(6, 4))

#     ax.plot(baseline[:, :, 1], baseline[:, :, 0], "r-", alpha=0.5, label="Baseline")
#     if corrected is not None:
#         ax.plot(corrected[:, :, 1], corrected[:, :, 0], "b-", alpha=0.8, label="Corrected")
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     ax.legend()
#     ax.set_title("Particle trajectories")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150)
#     plt.close()
#     return output_path


# def plot_heatmap(
#     grid: np.ndarray,
#     output_path: Path,
#     title: str = "Plastic concentration",
# ) -> Path:
#     plt = _get_matplotlib()
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     im = ax.imshow(grid, origin="lower", cmap="viridis")
#     plt.colorbar(im, ax=ax, label="Normalized density")
#     ax.set_title(title)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150)
#     plt.close(fig)
#     return output_path


# def animate_simulation(
#     positions: np.ndarray,
#     output_path: Path,
#     step_stride: int = 1,
# ) -> Optional[Path]:
#     try:
#         import imageio  # type: ignore
#     except ImportError:  # pragma: no cover
#         LOGGER.warning("imageio not installed; skipping animation.")
#         return None

#     frames = []
#     path = output_path.with_suffix(".gif")
#     for idx in range(0, positions.shape[0], step_stride):
#         plt = _get_matplotlib()
#         fig, ax = plt.subplots(figsize=(5, 4))
#         ax.scatter(positions[idx, :, 1], positions[idx, :, 0], s=10, alpha=0.6)
#         ax.set_xlim(positions[..., 1].min(), positions[..., 1].max())
#         ax.set_ylim(positions[..., 0].min(), positions[..., 0].max())
#         ax.set_title(f"Step {idx}")
#         buf = Path("__tmp_frame.png")
#         fig.savefig(buf, dpi=100)
#         plt.close(fig)
#         frames.append(imageio.imread(buf))
#         buf.unlink(missing_ok=True)
#     imageio.mimsave(path, frames, fps=4)
#     return path


from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(lat: np.ndarray, lon: np.ndarray, output: Path, bbox: Sequence[float]) -> None:
    # lat, lon shapes: (T, N) for baseline, or (T,) for mean
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if lat.ndim == 2:
        # show centroid path and start/end
        plt.plot(lon.mean(axis=1), lat.mean(axis=1), "-", lw=1.5, label="centroid")
        plt.plot(lon[0], lat[0], ".", label="start", alpha=0.5)
        plt.plot(lon[-1], lat[-1], ".", label="end", alpha=0.5)
    else:
        plt.plot(lon, lat, "-", lw=1.5, label="mean track")
    plt.xlim(bbox[2], bbox[3])
    plt.ylim(bbox[1], bbox[0])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

def plot_prob_heatmap(final_lat: np.ndarray, final_lon: np.ndarray, output: Path, bbox: Sequence[float], bins: int = 80) -> None:
    # simple 2D histogram at final time
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    H, xedges, yedges = np.histogram2d(final_lat, final_lon,
                                       bins=[bins, bins],
                                       range=[[bbox[1], bbox[0]], [bbox[2], bbox[3]]])
    H = H.T  # x=lon, y=lat for imshow later
    plt.imshow(H, origin="lower",
               extent=[bbox[2], bbox[3], bbox[1], bbox[0]],
               aspect="auto")
    plt.colorbar(label="counts")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Final position probability")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
