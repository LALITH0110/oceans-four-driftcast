# DriftCast helper script for downloading HYCOM current subsets.
# Pulls an OPeNDAP window over the North Atlantic and writes currents to NetCDF.
# Adjust dataset URLs, dates, and spatial extents to suit your experiment.

import os
import xarray as xr

# Choose the dataset that actually covers your dates
# For Jan 1–5, 2018 use expt_91.2:
url = "https://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/uv3z"
# For Sept 19–Dec 8, 2018 use:
# url = "https://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_93.0/uv3z"

# Open with OPeNDAP via pydap
ds = xr.open_dataset(url, engine="pydap", chunks={"time": 1})

# Pick lon/lat coord names
lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
if lon_name is None or lat_name is None:
    raise ValueError(f"Could not find lon/lat coords. Coords: {list(ds.coords)}")

# Normalize longitudes to [-180, 180] if needed
lon = ds[lon_name]
if float(lon.min()) >= 0.0 and float(lon.max()) > 180.0:
    ds = ds.assign_coords({lon_name: ((lon + 180) % 360) - 180}).sortby(lon_name)

# Your box and time
lon_min, lon_max = -80, 10
lat_min, lat_max = 30, 70
t0, t1 = "2018-01-01", "2018-01-05"  # change to your window if you switch experiments

subset = ds.sel({lon_name: slice(lon_min, lon_max),
                 lat_name: slice(lat_min, lat_max),
                 "time": slice(t0, t1)})

# HYCOM currents are typically u/v (eastward/northward). Keep a few common aliases.
cand_u = ["u", "water_u", "uo", "eastward_sea_water_velocity"]
cand_v = ["v", "water_v", "vo", "northward_sea_water_velocity"]
vars_keep = [v for v in cand_u + cand_v if v in subset.data_vars]
if not vars_keep:
    raise ValueError("No current vars found. Available: " + ", ".join(subset.data_vars))

os.makedirs("data/hycom", exist_ok=True)
out_path = "data/hycom/natl_hycom_currents_20180101_05.nc"
subset[vars_keep].to_netcdf(out_path, engine="netcdf4")  # writing locally uses netCDF4
print(f"Wrote {out_path}")
