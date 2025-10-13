# # DriftCast configuration and dataset indexing utilities.
# # Provides pydantic settings, glob-based month discovery, and coordinate helpers.
# # Imported by loaders, simulators, RL modules, and CLI commands to understand local forcing data.

# from __future__ import annotations

# import logging
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple

# import xarray as xr
# from pydantic_settings import BaseSettings
# from pydantic import Field

# LOGGER = logging.getLogger(__name__)
# MONTH_REGEX = re.compile(r"(\d{4})[-_](\d{2})")


# class DataConfig(BaseSettings):
#     """Runtime configuration for DriftCast workflows."""

#     region_north: float = Field(default=60.0, description="Northern latitude bound (deg).")
#     region_south: float = Field(default=0.0, description="Southern latitude bound (deg).")
#     region_west: float = Field(default=-80.0, description="Western longitude bound (degE).")
#     region_east: float = Field(default=10.0, description="Eastern longitude bound (degE).")

#     hycom_root: Path = Field(default=Path("data/raw/hycom"))
#     waves_root: Path = Field(default=Path("data/raw/waves"))
#     era5_root: Path = Field(default=Path("data/raw/era5"))
#     drifters_root: Path = Field(default=Path("data/raw/drifters"))

#     dt_hours: float = Field(default=1.0)
#     diffusion_kappa: float = Field(default=40.0, description="Horizontal diffusivity (m^2/s).")
#     alpha_wind: float = Field(default=0.02, description="Wind drag factor applied to 10 m winds.")
#     stokes_scale: float = Field(default=0.01, description="Fallback Stokes multiplier when wave product missing.")

#     n_particles_default: int = Field(default=2000)
#     random_seed: int = Field(default=1234)
#     max_particle_chunk: int = Field(default=4096)
#     cds_api_key: Optional[str] = Field(default=None, description="CDS API key.")
#     cds_api_url: Optional[str] = Field(default=None, description="CDS API URL.")

#     class Config:
#         env_prefix = "DRIFTCAST_"
#         env_file = ".env"

#     @property
#     def bbox(self) -> Tuple[float, float, float, float]:
#         return (self.region_north, self.region_south, self.region_west, self.region_east)


# @dataclass
# class DatasetIndex:
#     currents: Dict[str, List[Path]]
#     winds: Dict[str, List[Path]]
#     waves: Dict[str, List[Path]]
#     drifters: Dict[str, List[Path]]

#     def months(self) -> Dict[str, List[str]]:
#         return {
#             "currents": sorted(self.currents.keys()),
#             "winds": sorted(self.winds.keys()),
#             "waves": sorted(self.waves.keys()),
#             "drifters": sorted(self.drifters.keys()),
#         }

#     def mandatory_intersection(self) -> List[str]:
#         base = set(self.currents.keys()) & set(self.winds.keys())
#         if not base:
#             return []
#         return sorted(base)

#     def waves_intersection(self) -> List[str]:
#         base = set(self.currents.keys()) & set(self.winds.keys())
#         if not base or not self.waves:
#             return sorted(base)
#         return sorted(base & set(self.waves.keys()))


# def _extract_month(path: Path) -> Optional[str]:
#     match = MONTH_REGEX.search(path.name)
#     if not match:
#         return None
#     year, month = match.groups()
#     return f"{year}-{month}"


# def _index_months(root: Path, pattern: str) -> Dict[str, List[Path]]:
#     if not root.exists():
#         return {}
#     mapping: Dict[str, List[Path]] = {}
#     for file in root.glob(pattern):
#         month = _extract_month(file)
#         if not month:
#             continue
#         mapping.setdefault(month, []).append(file)
#     return {k: sorted(v) for k, v in mapping.items()}


# def index_datasets(cfg: DataConfig) -> DatasetIndex:
#     """Build a per-month index for each forcing product."""
#     index = DatasetIndex(
#         currents=_index_months(cfg.hycom_root, "*.nc"),
#         winds=_index_months(cfg.era5_root, "*.nc"),
#         waves=_index_months(cfg.waves_root, "*.nc"),
#         drifters=_index_months(cfg.drifters_root, "*.csv"),
#     )
#     LOGGER.debug("Indexed datasets: %s", index.months())
#     return index


# def data_availability_report(cfg: DataConfig) -> str:
#     index = index_datasets(cfg)
#     months = index.months()
#     parts = [
#         "Currents months : " + (", ".join(months["currents"]) or "none"),
#         "Winds months    : " + (", ".join(months["winds"]) or "none"),
#         "Waves months    : " + (", ".join(months["waves"]) or "none"),
#         "Drifters months : " + (", ".join(months["drifters"]) or "none"),
#         "Intersection (currents & winds): " + (", ".join(index.mandatory_intersection()) or "none"),
#         "Intersection incl. waves     : " + (", ".join(index.waves_intersection()) or "none"),
#     ]
#     return "\n".join(parts)


# def get_lon_lat_names(ds: xr.Dataset) -> Tuple[str, str]:
#     lon_candidates = ("lon", "longitude", "LONGITUDE", "x", "X")
#     lat_candidates = ("lat", "latitude", "LATITUDE", "y", "Y")
#     lon_name = next((name for name in lon_candidates if name in ds.coords), None)
#     lat_name = next((name for name in lat_candidates if name in ds.coords), None)
#     if lon_name is None or lat_name is None:
#         raise ValueError(f"Dataset missing lon/lat coordinates: {list(ds.coords)}")
#     return lon_name, lat_name


# def get_time_name(ds: xr.Dataset) -> Optional[str]:
#     for candidate in ("time", "Time", "datetime"):
#         if candidate in ds.coords:
#             return candidate
#     return None


# def normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
#     lon_name, _ = get_lon_lat_names(ds)
#     lon = ds[lon_name]
#     if float(lon.min()) >= 0.0 and float(lon.max()) > 180.0:
#         ds = ds.assign_coords({lon_name: ((lon + 180) % 360) - 180}).sortby(lon_name)
#     return ds


# def subset_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
#     north, south, west, east = bbox
#     lon_name, lat_name = get_lon_lat_names(ds)
#     lat_min, lat_max = sorted((south, north))
#     lon_min, lon_max = sorted((west, east))
#     return ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})


# def summarise(cfg: DataConfig) -> None:
#     LOGGER.info("\n%s", data_availability_report(cfg))


# if __name__ == "__main__":  # pragma: no cover
#     logging.basicConfig(level=logging.INFO)
#     summarise(DataConfig())


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # North Atlantic window
    region_north: float = 60.0
    region_south: float = 20.0
    region_west: float = -80.0
    region_east: float = 0.0

    # physics
    dt_hours: float = 1.0
    diffusion_kappa: float = 40.0   # m^2/s
    alpha_wind: float = 0.02        # 2% windage
    stokes_scale: float = 0.01      # fallback stokes = 1% of wind

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.region_north, self.region_south, self.region_west, self.region_east)
