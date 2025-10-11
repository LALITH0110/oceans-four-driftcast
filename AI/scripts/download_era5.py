# DriftCast helper script for ERA5 wind downloads.
# Pulls a small North Atlantic window to NetCDF for loader smoke tests.
# Adjust the date range or bounding box before running larger jobs.

import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['10m_u_component_of_wind','10m_v_component_of_wind'],
        'year': ['2024'],
        'month': ['07'],
        'day': ['01','02'],
        'time': [f"{h:02d}:00" for h in range(24)],
        # area format: North, West, South, East
        'area': [60, -80, 0, 10],
        'format': 'netcdf'
    },
    'data/raw/era5/era5_u10_v10_20240701_02_natl.nc'
)
print("Saved ERA5 to data/raw/era5/era5_u10_v10_20240701_02_natl.nc")
