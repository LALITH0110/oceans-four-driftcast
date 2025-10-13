# save as nc_to_csv.py
# usage:
#   python nc_to_csv.py ocean_data/cmems_currents.nc
# optional:
#   python nc_to_csv.py ocean_data/cmems_currents.nc --lat-min 20 --lat-max 60 --lon-min -80 --lon-max 0 --stride 1 --vars uo,vo

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def guess_coord_name(names, candidates):
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"Could not find any of {candidates} in {list(names)}")

def main():
    p = argparse.ArgumentParser(description="Convert .nc to tidy CSV")
    p.add_argument("nc_path", type=str, help="Path to NetCDF file")
    p.add_argument("--outdir", type=str, default="ocean_data_csv", help="Output directory")
    p.add_argument("--vars", type=str, default="", help="Comma separated variable names to export (default: all numeric)")
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth grid point to shrink size")
    args = p.parse_args()

    nc_path = Path(args.nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{nc_path.stem}.csv"

    # Open lazily, iterate time by time
    ds = xr.open_dataset(nc_path, chunks={"time": 1})

    # Detect common coordinate names
    lat_name = guess_coord_name(ds.coords, ["latitude", "lat", "y"])
    lon_name = guess_coord_name(ds.coords, ["longitude", "lon", "x"])
    time_name = guess_coord_name(ds.coords, ["time"])
    depth_name = None
    for cand in ["depth", "z", "depthm"]:
        if cand in ds.coords or cand in ds.dims:
            depth_name = cand
            break

    # Normalize longitudes to [-180, 180) for readability
    lon = ds[lon_name]
    if float(lon.max()) > 180.0:
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)}).sortby(lon_name)

    # Optional geographic subset
    if args.lat_min is not None and args.lat_max is not None:
        lat_min, lat_max = sorted([args.lat_min, args.lat_max])
        ds = ds.sel({lat_name: slice(lat_min, lat_max)})
    if args.lon_min is not None and args.lon_max is not None:
        lon_min, lon_max = args.lon_min, args.lon_max
        if lon_min > lon_max:
            # wrap-around is not supported here
            raise ValueError("lon-min must be <= lon-max for this simple slicer")
        ds = ds.sel({lon_name: slice(lon_min, lon_max)})

    # Optional downsampling
    if args.stride > 1:
        slicers = {lat_name: slice(None, None, args.stride), lon_name: slice(None, None, args.stride)}
        ds = ds.isel(**slicers)

    # Choose variables
    if args.vars.strip():
        wanted = [v.strip() for v in args.vars.split(",") if v.strip()]
        missing = [v for v in wanted if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Variables not found in file: {missing}")
        var_list = wanted
    else:
        var_list = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        if not var_list:
            raise ValueError("No numeric variables found to export")

    # Prepare CSV (overwrite if exists)
    if out_csv.exists():
        out_csv.unlink()

    times = ds[time_name].values
    n_times = len(times)

    for i in range(n_times):
        step = ds.isel({time_name: i})[var_list]

        # Convert to tidy DataFrame with coords
        df = step.to_dataframe().reset_index()

        # If dataset lacks an explicit depth coordinate, create one
        if depth_name is None and "depth" not in df.columns:
            df.insert(1, "depth", np.nan)

        # Enforce column order if possible
        cols = ["time"]
        if "depth" in df.columns:
            cols.append("depth")
        cols += [lat_name, lon_name]
        # add variables at the end
        cols += [c for c in var_list if c in df.columns]
        # keep any extra columns too
        for c in df.columns:
            if c not in cols:
                cols.append(c)
        df = df[cols]

        # Append or write header
        df.to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())

        # Light progress print
        if (i + 1) % 5 == 0 or (i + 1) == n_times:
            print(f"Wrote time slice {i+1}/{n_times}")

    print(f"Done. CSV at: {out_csv}")

if __name__ == "__main__":
    main()
