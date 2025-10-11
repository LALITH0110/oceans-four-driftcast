# DriftCast helper script for small ERA5 wind pulls.
# Requests a limited window over the North Atlantic into NetCDF for loader smoke tests.
# Edit the date range or area before running for larger studies.

import cdsapi
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind','10m_v_component_of_wind'
        ],
        'year': '2020',
        'month': '01',
        'day': ['01','02','03','04','05','06','07'],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [70, -80, 30, 10],  # N, W, S, E
        'format': 'netcdf'
    },
    'data/era5/natl_wind_10m_hourly_20200101_07.nc')
