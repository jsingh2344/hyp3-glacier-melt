import xarray as xr
import numpy as np
from osgeo import ogr

ds = xr.open_dataset(r"C:\Users\jaden\Downloads\Research\hyp3_repo\hyp3-glacier-melt\output_nc\Gulkana_32606_S1_cube_VH.nc")

print(ds)
print("\nDATA VARS:")
for name, da in ds.data_vars.items():
    print(name, da.dims, da.shape, da.dtype)

print("\nCOORDS:")
for name, coord in ds.coords.items():
    print(name, coord.dims, coord.shape, coord.dtype)

print("\nATTRS:")
for k, v in ds.attrs.items():
    print(f"{k}: {v}")

def validate_datacube_schema(ds):
    # --- Expected ---
    required_vars = {
        "images": ("time", "y", "x"),
        "dem": ("y", "x"),
        "rgi_ind_glacier_mask": ("y", "x"),
        "rgi_aspect_arr": ("y", "x"),
    }

    required_coords = {
        "time",
        "x",
        "y",
    }

    required_attrs = {
        "description",
        "projection",
        "epsg_str",
        "geotransform",
    }

    # --- Check variables exist ---
    missing_vars = set(required_vars) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")

    # --- Check variable dimensions ---
    for var, expected_dims in required_vars.items():
        actual_dims = ds[var].dims
        if actual_dims != expected_dims:
            raise ValueError(
                f"{var} dims mismatch: expected {expected_dims}, got {actual_dims}"
            )

    # --- Check coordinates ---
    missing_coords = required_coords - set(ds.coords)
    if missing_coords:
        raise ValueError(f"Missing coords: {missing_coords}")

    # --- Check attributes ---
    missing_attrs = required_attrs - set(ds.attrs)
    if missing_attrs:
        raise ValueError(f"Missing attrs: {missing_attrs}") 

    print("Mask unique values sample:", np.unique(ds["rgi_ind_glacier_mask"].values)[:20])
    print("Mask max:", np.nanmax(ds["rgi_ind_glacier_mask"].values))
    print("Aspect min/max:", np.nanmin(ds["rgi_aspect_arr"].values), np.nanmax(ds["rgi_aspect_arr"].values))

    print("✅ Datacube schema is valid")

validate_datacube_schema(ds)



shp_path = r"C:/Users/jaden/Downloads/Research/Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp"

ogr_ds = ogr.Open(shp_path)
layer = ogr_ds.GetLayer()

layer_defn = layer.GetLayerDefn()
for i in range(layer_defn.GetFieldCount()):
    field_defn = layer_defn.GetFieldDefn(i)
    print(field_defn.GetName(), field_defn.GetTypeName())
