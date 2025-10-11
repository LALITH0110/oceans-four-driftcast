import sys, cdsapi
years  = sys.argv[1].split(",")     # "2019,2020,2021"
months = sys.argv[2].split(",")     # "1,4,7,10"
north, west, south, east = map(float, sys.argv[3:7])
outdir = sys.argv[7]

c = cdsapi.Client()
for y in years:
    for m in months:
        m = f"{int(m):02d}"
        out = f"{outdir}/era5_u10_v10_{y}-{m}_natl.nc"
        print("ERA5:", y, m, "->", out, flush=True)
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["10m_u_component_of_wind","10m_v_component_of_wind"],
                    "year": y,
                    "month": m,
                    "day": [f"{d:02d}" for d in range(1,32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [north, west, south, east],
                    "format": "netcdf",
                },
                out
            )
        except Exception as e:
            print("ERA5 failed for", y, m, "->", e)
