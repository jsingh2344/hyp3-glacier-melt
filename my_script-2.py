import xarray as xr
import numpy as np

new_path = r"output_nc\Gulkana_32607_S1_cube_VH.nc"
old_path = r"c:\Users\jaden\Downloads\Research\onset_maps\Kennicott_32607_S1_cube_VH_014_387.nc"

ds_new = xr.open_dataset(new_path)
ds_old = xr.open_dataset(old_path)

def summarize_cube(name, ds):
    ids = np.unique(ds["rgi_ind_glacier_mask"].values)
    ids = ids[~np.isnan(ids)]
    ids = ids[ids != 0]

    print(f"\n=== {name} ===")
    print("path:", name)
    print("projection:", ds.attrs.get("projection"))
    print("epsg_str:", ds.attrs.get("epsg_str"))
    print("geotransform:", ds.attrs.get("geotransform"))
    print("shape:", ds["dem"].shape)
    print("x range:", float(ds["x"].values[0]), "to", float(ds["x"].values[-1]))
    print("y range:", float(ds["y"].values[0]), "to", float(ds["y"].values[-1]))
    print("glacier count:", len(ids))
    print("first 20 ids:", ids[:20])
    print("last 20 ids:", ids[-20:])

summarize_cube(new_path, ds_new)
summarize_cube(old_path, ds_old)

# direct comparison
print("\n=== COMPARISON ===")
print("same projection:", ds_new.attrs.get("projection") == ds_old.attrs.get("projection"))
print("same geotransform:", np.array_equal(
    np.array(ds_new.attrs.get("geotransform")),
    np.array(ds_old.attrs.get("geotransform"))
))
print("same shape:", ds_new["dem"].shape == ds_old["dem"].shape)

new_ids = np.unique(ds_new["rgi_ind_glacier_mask"].values)
new_ids = new_ids[~np.isnan(new_ids)]
new_ids = new_ids[new_ids != 0]

old_ids = np.unique(ds_old["rgi_ind_glacier_mask"].values)
old_ids = old_ids[~np.isnan(old_ids)]
old_ids = old_ids[old_ids != 0]

print("ids only in new:", new_ids[~np.isin(new_ids, old_ids)][:50])
print("ids only in old:", old_ids[~np.isin(old_ids, new_ids)][:50])

ds_new.close()
ds_old.close()