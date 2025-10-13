# Project: plasticparcels-drift-mvp
# ---------------------------------------------------------------
# Minimal working prototype that uses the `plasticparcels` + Parcels
# stack to advect a small ensemble of surface particles for ~1 day,
# optionally adding windage and simple diffusion. If a target location
# is provided, a tiny grid-search "RL-ish" tuner picks a windage
# coefficient that minimises miss distance. Produces a PNG map.
#
# Quick start
# 1) Create venv and install deps:
#    python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
#    pip install -r requirements.txt
# 2) Run a no-wind demo on GlobCurrent example data (auto-download):
#    python main.py --lat 40.0 --lon -40.0 --hours 24 --n 200
# 3) If you have ERA5 10m winds (u10,v10) in a local NetCDF, include them:
#    python main.py --lat 40 --lon -40 --hours 24 --with-wind /path/to/era5_u10_v10.nc
# 4) Optional tuning toward a known end position (toy "RL" search):
#    python main.py --lat 40 --lon -40 --hours 24 --with-wind era5.nc \
#                   --target-lat 41.2 --target-lon -37.9 --tune-windage
#
# Outputs
# - output/forecast.png        : ensemble trajectories and final-position cloud
# - output/ensemble_final.csv  : final particle positions
# - stdout: prints mean prediction and 50%/90% error radii (km)
# ---------------------------------------------------------------

import argparse
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import parcels
from parcels import FieldSet, ParticleSet, JITParticle, Variable
from parcels.application_kernels.advection import AdvectionRK4
from parcels.application_kernels.advectiondiffusion import DiffusionUniformKh

try:
    # plasticparcels is a thin physics-kernel extension on top of Parcels
    from plasticparcels import kernels as pk
    HAVE_PP = True
except Exception:
    HAVE_PP = False

# -------------------------------
# Helpers
# -------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dlat = p2 - p1
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def deg2km_scale(lat_deg):
    # 1 degree latitude ~111.2 km, longitude scales by cos(latitude)
    km_per_deg_lat = 111.2
    km_per_deg_lon = 111.2 * np.cos(np.deg2rad(lat_deg))
    return km_per_deg_lat, km_per_deg_lon


# -------------------------------
# Fieldset builders
# -------------------------------

def build_fieldset_globcurrent():
    """
    Uses the Parcels-provided small GlobCurrent example. This auto-downloads
    a few NetCDF files and is enough to run a realistic ocean-surface demo
    without any accounts or large data pulls.
    """
    example_dir = parcels.download_example_dataset("GlobCurrent_example_data")
    files = {
        "U": f"{example_dir}/20*.nc",
        "V": f"{example_dir}/20*.nc",
    }
    variables = {
        "U": "eastward_eulerian_current_velocity",
        "V": "northward_eulerian_current_velocity",
    }
    dims = {"lon": "lon", "lat": "lat", "time": "time"}
    fs = FieldSet.from_netcdf(files, variables, dims)
    return fs


def maybe_add_era5_winds(fieldset: FieldSet, era5_nc: str):
    """
    Try to add u10,v10 at 10m as Wind_U/Wind_V fields. ERA5 often has
    dims named latitude/longitude. If present, we create Fields and
    attach them to the existing FieldSet.
    """
    if not HAVE_PP:
        print("[warn] plasticparcels not imported, skip WindageDrift kernel. 'pip install plasticparcels'.")
        return False

    if not era5_nc:
        return False
    if not Path(era5_nc).exists():
        print(f"[warn] ERA5 file not found: {era5_nc}. Skipping windage.")
        return False

    try:
        dims = {"lon": "longitude", "lat": "latitude", "time": "time"}
        wind_u = parcels.Field.from_netcdf(era5_nc, variable="u10", dimensions=dims, name="Wind_U")
        wind_v = parcels.Field.from_netcdf(era5_nc, variable="v10", dimensions=dims, name="Wind_V")
        fieldset.add_field(wind_u)
        fieldset.add_field(wind_v)
        return True
    except Exception as e:
        print(f"[warn] Could not add ERA5 winds: {e}")
        return False


# -------------------------------
# Simulation
# -------------------------------

def run_simulation(lat, lon, hours=24, n=200, windage=0.02, kh=25.0, dt_minutes=10,
                   era5_nc=None, seed=0, outdir="output", verbose=True):
    np.random.seed(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    fieldset = build_fieldset_globcurrent()

    # Add constant horizontal diffusivity for a simple random walk
    # Units are m^2/s. Literature often uses O(10-100) m^2/s for surface.
    fieldset.add_constant_field("Kh_zonal", kh, mesh=fieldset.U.grid.mesh)
    fieldset.add_constant_field("Kh_meridional", kh, mesh=fieldset.U.grid.mesh)

    have_wind = maybe_add_era5_winds(fieldset, era5_nc)

    # Particle with a per-particle wind_coefficient, consumed by plasticparcels.WindageDrift
    PlasticParticle = JITParticle.add_variable("wind_coefficient", dtype=np.float32, initial=windage)

    # Small Gaussian cloud around the input point
    # Spread ~2 km std in both directions
    lat_jitter_km = 2.0
    lon_jitter_km = 2.0
    km_lat, km_lon = deg2km_scale(lat)
    lat_sigma_deg = lat_jitter_km / km_lat
    lon_sigma_deg = lon_jitter_km / km_lon

    lats = np.random.normal(lat, lat_sigma_deg, size=n)
    lons = np.random.normal(lon, lon_sigma_deg, size=n)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle, lon=lons, lat=lats)

    # Compose kernels: advection + diffusion, and windage if available
    kernel = AdvectionRK4 + DiffusionUniformKh
    if have_wind:
        kernel = kernel + pk.WindageDrift

    runtime = timedelta(hours=hours)
    dt = timedelta(minutes=dt_minutes)

    # Save hourly to a zarr file we can read back for plotting
    pfile = pset.ParticleFile(name=str(out / "trajectories.zarr"), outputdt=timedelta(hours=1))

    if verbose:
        print(f"[info] Running {n} particles for {hours} h, dt={dt_minutes} min, windage={windage}, Kh={kh} m^2/s")
        if have_wind:
            print("[info] WindageDrift enabled using ERA5 u10/v10")
        else:
            print("[info] No wind fields, currents + diffusion only")

    pset.execute(kernel, runtime=runtime, dt=dt, output_file=pfile)

    # Collect final cloud
    final_lats = np.array([p.lat for p in pset.particles])
    final_lons = np.array([p.lon for p in pset.particles])

    mean_lat = float(np.mean(final_lats))
    mean_lon = float(np.mean(final_lons))

    # Radii: median distance (approx 50%) and 90th percentile
    dists = haversine_km(final_lats, final_lons, mean_lat, mean_lon)
    r50 = float(np.percentile(dists, 50))
    r90 = float(np.percentile(dists, 90))

    # Dump final CSV
    np.savetxt(out / "ensemble_final.csv", np.c_[final_lats, final_lons], delimiter=",", header="lat,lon", comments="")

    # Plot using the saved zarr trajectory file to overlay some tracks
    try:
        ds = xr.open_zarr(out / "trajectories.zarr")
        # ds has dims like time, trajectory; variables lat, lon
        # Draw a thin subset to keep the plot light
        sample = min(n, 100)
        idx = np.linspace(0, n - 1, sample, dtype=int)
        plt.figure(figsize=(8, 6))
        for i in idx:
            plt.plot(ds["lon"].isel(trajectory=i), ds["lat"].isel(trajectory=i), alpha=0.25)
        plt.scatter(final_lons, final_lats, s=6, alpha=0.5, label="final ensemble")
        plt.scatter([mean_lon], [mean_lat], s=60, marker="*", label="mean", zorder=5)
        circle50 = plt.Circle((mean_lon, mean_lat), r50/deg2km_scale(mean_lat)[1], fill=False, linestyle=":", label="50% radius")
        circle90 = plt.Circle((mean_lon, mean_lat), r90/deg2km_scale(mean_lat)[1], fill=False, linestyle="-.", label="90% radius")
        ax = plt.gca()
        ax.add_patch(circle50)
        ax.add_patch(circle90)
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("PlasticParcels drift, final cloud and trajectories")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "forecast.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[warn] Could not render trajectories plot: {e}")

    return {
        "mean_lat": mean_lat,
        "mean_lon": mean_lon,
        "r50_km": r50,
        "r90_km": r90,
        "out_png": str(out / "forecast.png"),
        "out_csv": str(out / "ensemble_final.csv"),
    }


# -------------------------------
# Tiny windage tuner (grid search)
# -------------------------------

def tune_windage_to_target(lat, lon, target_lat, target_lon, era5_nc=None,
                           hours=24, n=80, candidates=(0.01, 0.015, 0.02, 0.03, 0.04),
                           seed=0):
    best_c = None
    best_miss = 1e9
    for c in candidates:
        res = run_simulation(lat, lon, hours=hours, n=n, windage=c, kh=25.0, dt_minutes=20,
                             era5_nc=era5_nc, seed=seed, outdir="_tune", verbose=False)
        miss = float(haversine_km(res["mean_lat"], res["mean_lon"], target_lat, target_lon))
        print(f"  candidate windage={c:.3f} -> miss {miss:.2f} km")
        if miss < best_miss:
            best_miss = miss
            best_c = c
    print(f"[tune] Best windage={best_c:.3f} with miss {best_miss:.2f} km")
    return best_c


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="PlasticParcels drift MVP")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--n", type=int, default=200, help="ensemble size")
    ap.add_argument("--windage", type=float, default=0.02, help="default windage coefficient 0-0.05")
    ap.add_argument("--kh", type=float, default=25.0, help="horizontal diffusivity m^2/s")
    ap.add_argument("--dt-minutes", type=int, default=10)
    ap.add_argument("--with-wind", type=str, default=None, help="path to ERA5 u10/v10 netcdf (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-windage", action="store_true", help="search windage against a target end point")
    ap.add_argument("--target-lat", type=float, default=None)
    ap.add_argument("--target-lon", type=float, default=None)

    args = ap.parse_args()

    windage = args.windage
    if args.tune_windage:
        if args.target_lat is None or args.target_lon is None:
            raise SystemExit("--tune-windage needs --target-lat and --target-lon")
        windage = tune_windage_to_target(args.lat, args.lon, args.target_lat, args.target_lon,
                                         era5_nc=args.with_wind, hours=args.hours, n=max(40, args.n//3), seed=args.seed)

    res = run_simulation(args.lat, args.lon, hours=args.hours, n=args.n, windage=windage,
                         kh=args.kh, dt_minutes=args.dt_minutes, era5_nc=args.with_wind,
                         seed=args.seed)

    print("\n=== Prediction ===")
    print(f"Mean position at T+{args.hours}h: lat={res['mean_lat']:.4f}, lon={res['mean_lon']:.4f}")
    print(f"Error radii: 50% ~ {res['r50_km']:.1f} km, 90% ~ {res['r90_km']:.1f} km")
    print(f"Plot: {res['out_png']}")
    print(f"Final CSV: {res['out_csv']}")


if __name__ == "__main__":
    main()
