# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Centralizes structured configuration loading and validation for driftcast.
- Uses Pydantic v2 models to deserialize YAML scenario files into typed objects.
- Downstream modules consume SimulationConfig for running single or batch jobs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class DomainConfig(BaseModel):
    """Define the geographic bounds for a simulation domain."""

    lon_min: float = Field(-100.0, description="Western boundary in degrees east.")
    lon_max: float = Field(20.0, description="Eastern boundary in degrees east.")
    lat_min: float = Field(0.0, description="Southern boundary in degrees north.")
    lat_max: float = Field(70.0, description="Northern boundary in degrees north.")
    resolution_deg: float = Field(
        0.25,
        description="Grid resolution in degrees for diagnostics and visualization.",
    )

    @field_validator("lon_max")
    @classmethod
    def _check_lon(cls, v: float, info: ValidationInfo) -> float:
        lon_min = info.data.get("lon_min")
        if lon_min is not None and v <= lon_min:
            raise ValueError("lon_max must be greater than lon_min")
        return v

    @field_validator("lat_max")
    @classmethod
    def _check_lat(cls, v: float, info: ValidationInfo) -> float:
        lat_min = info.data.get("lat_min")
        if lat_min is not None and v <= lat_min:
            raise ValueError("lat_max must be greater than lat_min")
        return v


class GyreBoxConfig(BaseModel):
    """Define the gyre monitoring box for metrics and overlays."""

    lon_min: float = Field(-70.0, description="Western boundary of the gyre box.")
    lon_max: float = Field(-30.0, description="Eastern boundary of the gyre box.")
    lat_min: float = Field(20.0, description="Southern boundary of the gyre box.")
    lat_max: float = Field(40.0, description="Northern boundary of the gyre box.")


class TimeConfig(BaseModel):
    """Define start/end times and integration step for a simulation."""

    start: datetime = Field(..., description="Simulation start timestamp (UTC).")
    end: datetime = Field(..., description="Simulation end timestamp (UTC).")
    dt_minutes: float = Field(30.0, description="Fixed time step in minutes.")
    output_interval_hours: float = Field(
        6.0, description="Cadence for writing diagnostic outputs."
    )

    @field_validator("end")
    @classmethod
    def _check_chronology(cls, v: datetime, info: ValidationInfo) -> datetime:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be greater than start")
        return v

    @property
    def duration_days(self) -> float:
        """Return the total duration in days."""
        return (self.end - self.start).total_seconds() / 86400.0


class BeachingConfig(BaseModel):
    """Parameters governing shoreline interactions."""

    probability: float = Field(
        0.02, ge=0.0, le=1.0, description="Probability of beaching on land contact."
    )
    resuspension_days: Optional[float] = Field(
        500.0,
        ge=0.0,
        description="Mean waiting time before resuspension, None disables.",
    )
    sticky_coastline_buffer_km: float = Field(
        15.0, description="Buffer distance around coastlines for beach interactions."
    )


class SeasonalRampConfig(BaseModel):
    """Seasonal modulation applied to the synthetic gyre strength."""

    enabled: bool = Field(False, description="Toggle seasonal amplitude ramp.")
    amplitude: float = Field(0.1, ge=0.0, description="Fractional seasonal amplitude applied when enabled.")
    phase_day: float = Field(0.0, description="Phase offset (days) for the cosine seasonal cycle.")


class EkmanConfig(BaseModel):
    """Configuration for the optional Ekman surface drift approximation."""

    enabled: bool = Field(False, description="Toggle Ekman surface drift contribution.")
    alpha: float = Field(0.03, ge=0.0, description="Scaling applied to the Ekman solution.")
    drag_coefficient: float = Field(1.4e-3, ge=0.0, description="Quadratic drag coefficient for wind stress.")
    air_density: float = Field(1.225, ge=0.0, description="Air density used in wind stress (kg/m^3).")
    water_density: float = Field(1025.0, ge=0.0, description="Water density used for Ekman velocity (kg/m^3).")


class PhysicsConfig(BaseModel):
    """Aggregate physics toggles and coefficients for particle motion."""

    diffusivity_m2s: float = Field(30.0, ge=0.0, description="Horizontal Kh.")
    windage_coeff: float = Field(
        0.002, ge=0.0, description="Fraction of 10 m winds added to drift."
    )
    stokes_coeff: float = Field(
        0.05, ge=0.0, description="Fraction of Stokes drift added to drift."
    )
    vertical_enabled: bool = Field(
        False, description="Placeholder flag for future 3-D vertical mixing."
    )
    beaching: BeachingConfig = Field(default_factory=BeachingConfig)
    seasonal: SeasonalRampConfig = Field(default_factory=SeasonalRampConfig)
    ekman: EkmanConfig = Field(default_factory=EkmanConfig)


class ParticleClassConfig(BaseModel):
    """Microplastic class metadata for initializing particle ensembles."""

    name: str
    density_kgm3: float = Field(..., gt=0.0)
    diameter_mm: float = Field(..., gt=0.0)
    settling_velocity_mps: float = Field(0.0)
    fraction: float = Field(1.0, ge=0.0, le=1.0)


class SourceConfig(BaseModel):
    """Structure for individual particle sources defined in YAML."""

    type: Literal["rivers", "shipping", "coastal"]
    name: str
    rate_per_day: float = Field(..., ge=0.0)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    composition: List[ParticleClassConfig] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Describe how to persist simulation outputs."""

    directory: Path = Field(Path("results/outputs"), description="Output directory.")
    format: Literal["netcdf", "zarr"] = Field("netcdf")
    chunks: Optional[Dict[str, int]] = Field(
        default=None, description="Chunk sizes for persisted datasets."
    )


class AnimationConfig(BaseModel):
    """Capture visualization parameters for animations."""

    preview_path: Path = Path("results/videos/preview.mp4")
    final_path: Path = Path("results/videos/final_cut.mp4")
    dpi: int = 150
    fade_trail_steps: int = 12
    draw_quivers_every: int = 6


class PerformanceConfig(BaseModel):
    """Select performance tuning presets for save cadence and chunk sizes."""

    profile: Literal["laptop", "workstation", "cluster"] = "laptop"


PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "laptop": {"output_interval_hours": 12.0, "chunks": {"time": 1, "particle": 256}},
    "workstation": {"output_interval_hours": 6.0, "chunks": {"time": 1, "particle": 512}},
    "cluster": {"output_interval_hours": 3.0, "chunks": {"time": 2, "particle": 1024}},
}


class SimulationConfig(BaseModel):
    """Top-level configuration container consumed by the CLI runner."""

    domain: DomainConfig = Field(default_factory=DomainConfig)
    gyre_box: GyreBoxConfig = Field(default_factory=GyreBoxConfig)
    time: TimeConfig
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    sources: List[SourceConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @model_validator(mode="after")
    def _validate_sources(self) -> "SimulationConfig":
        for source in self.sources:
            start = source.start or self.time.start
            end = source.end or self.time.end
            if end < start:
                raise ValueError(f"source {source.name} has end < start")
        return self

    @model_validator(mode="after")
    def _apply_performance_profile(self) -> "SimulationConfig":
        preset = PROFILE_PRESETS[self.performance.profile]
        self.time.output_interval_hours = preset["output_interval_hours"]
        if not self.output.chunks:
            self.output.chunks = dict(preset["chunks"])
        return self


def load_config(path: Path | str) -> SimulationConfig:
    """Load a YAML configuration file into a ``SimulationConfig`` instance.

    Args:
        path: File path to a YAML configuration describing the scenario.

    Returns:
        Parsed ``SimulationConfig`` object.

    Example:
        >>> cfg = load_config("configs/natl_subtropical_gyre.yaml")
        >>> cfg.domain.lon_min
        -100.0
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf8") as fh:
        payload = yaml.safe_load(fh)
    return SimulationConfig(**payload)


def load_many(paths: Iterable[Path | str]) -> List[SimulationConfig]:
    """Load multiple YAML configuration files for batch sweeps."""
    return [load_config(path) for path in paths]
