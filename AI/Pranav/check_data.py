import xarray as xr, os, glob, json
for f in glob.glob("ocean_data/*.nc"):
    ds = xr.open_dataset(f)
    print("\nFILE:", f, "| size =", round(os.path.getsize(f)/1e6, 1), "MB")
    print("dims:", dict(ds.dims))
    print("vars:", list(ds.data_vars))
    print("title:", ds.attrs.get("title"))
    print("institution:", ds.attrs.get("institution"))
    print("source:", ds.attrs.get("source"))
    print("history:", ds.attrs.get("history"))
    for v in ("uo","vo","longitude","latitude","time","depth"):
        if v in ds:
            print(v, "units:", ds[v].attrs.get("units"))
