# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Builds provenance manifests capturing simulation configuration and runtime context.
- Serializes metadata sidecars that accompany NetCDF/Zarr outputs for auditing.
- Integrates git commit, library versions, environment checks, and random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import json
import platform
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from driftcast.config import SimulationConfig


@dataclass(frozen=True)
class ManifestPayload:
    """Structured manifest bundle prior to JSON emission."""

    metadata: Dict[str, Any]
    path: Path


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result or None


def _config_hash(config: SimulationConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, default=str).encode("utf8")
    return sha256(payload).hexdigest()


def _library_versions() -> Dict[str, str]:
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xarray": xr.__version__,
    }
    try:
        import dask  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        versions["dask"] = dask.__version__
    return versions


def _environment_checks() -> Dict[str, Any]:
    try:
        import matplotlib
        from matplotlib import animation as mpl_animation
    except ImportError:  # pragma: no cover - matplotlib optional for manifest
        ffmpeg_path = shutil.which("ffmpeg")
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "ffmpeg_available": ffmpeg_path is not None,
            "ffmpeg_path": ffmpeg_path,
        }

    ffmpeg_path = matplotlib.rcParams.get("animation.ffmpeg_path") or shutil.which("ffmpeg")
    try:
        ffmpeg_available = mpl_animation.writers.is_available("ffmpeg")
    except Exception:  # pragma: no cover - robustness for stripped builds
        ffmpeg_available = False

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_path": ffmpeg_path,
    }
def build_manifest(
    config: SimulationConfig,
    output_path: Path,
    seed: int,
    particle_counts: Dict[str, int],
    extra_checks: Optional[Dict[str, Any]] = None,
) -> ManifestPayload:
    """Assemble manifest metadata and destination path for serialization."""
    run_id = str(uuid4())
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(config),
        "random_seeds": {
            "base_seed": seed,
        },
        "library_versions": _library_versions(),
        "environment_checks": _environment_checks(),
        "domain": {
            "lon_min": config.domain.lon_min,
            "lon_max": config.domain.lon_max,
            "lat_min": config.domain.lat_min,
            "lat_max": config.domain.lat_max,
            "resolution_deg": config.domain.resolution_deg,
        },
        "time_span": {
            "start": config.time.start.isoformat(),
            "end": config.time.end.isoformat(),
            "dt_minutes": config.time.dt_minutes,
        },
        "physics": {
            "diffusivity_m2s": config.physics.diffusivity_m2s,
            "windage_coeff": config.physics.windage_coeff,
            "stokes_coeff": config.physics.stokes_coeff,
            "beaching": {
                "probability": config.physics.beaching.probability,
                "resuspension_days": config.physics.beaching.resuspension_days,
                "sticky_buffer_km": config.physics.beaching.sticky_coastline_buffer_km,
            },
        },
        "particle_counts": particle_counts,
        "outputs": {
            "path": str(output_path.resolve()),
            "format": config.output.format,
        },
    }
    manifest["metrics"] = {
        "gyre_box": {
            "lon_min": config.gyre_box.lon_min,
            "lon_max": config.gyre_box.lon_max,
            "lat_min": config.gyre_box.lat_min,
            "lat_max": config.gyre_box.lat_max,
        }
    }
    if extra_checks:
        manifest["environment_checks"].update(extra_checks)
    manifest_path = _manifest_path_for_output(output_path)
    return ManifestPayload(metadata=manifest, path=manifest_path)


def _manifest_path_for_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(output_path.suffix + ".manifest.json")
    return output_path / "manifest.json"


def write_manifest(payload: ManifestPayload) -> None:
    """Persist the manifest JSON to disk."""
    payload.path.parent.mkdir(parents=True, exist_ok=True)
    payload.path.write_text(json.dumps(payload.metadata, indent=2), encoding="utf8")


def update_dataset_attrs(dataset: xr.Dataset, manifest: ManifestPayload) -> None:
    """Inject a subset of manifest metadata into the dataset attributes."""
    attrs = dataset.attrs
    attrs["run_id"] = manifest.metadata["run_id"]
    attrs["git_commit"] = manifest.metadata["git_commit"]
    attrs["config_hash"] = manifest.metadata["config_hash"]
    attrs["base_seed"] = manifest.metadata["random_seeds"]["base_seed"]
    attrs["generated_at"] = manifest.metadata["timestamp"]
    dataset.attrs = attrs
