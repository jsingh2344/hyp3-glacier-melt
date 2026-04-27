import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

nc_path = r"output_nc\Gulkana_32607_OPERA_cube_VH.nc"
ds = xr.open_dataset(nc_path)

img = ds.images.isel(time=0).values
dem = ds.dem.values
mask = ds.rgi_ind_glacier_mask.values

print(ds)
print("image shape:", img.shape)
print("dem shape:", dem.shape)
print("mask shape:", mask.shape)
print("x range:", float(ds.x.min()), "to", float(ds.x.max()))
print("y range:", float(ds.y.min()), "to", float(ds.y.max()))
print("DEM min/max:", np.nanmin(dem), np.nanmax(dem))
print("Mask unique sample:", np.unique(mask)[:20])
print("Valid image pixels:", np.sum(np.isfinite(img)), "of", img.size)

glacier = mask > 0
valid_img = np.isfinite(img)
valid_dem = dem > 0

print("glacier pixels:", glacier.sum())
print("valid image on glacier:", np.sum(glacier & valid_img))
print("fraction glacier with valid image:", np.sum(glacier & valid_img) / glacier.sum())
print("valid DEM on glacier:", np.sum(glacier & valid_dem))
print("fraction glacier with valid DEM:", np.sum(glacier & valid_dem) / glacier.sum())

# Image plotting stretch
img_plot = img.astype(float).copy()
img_plot[~np.isfinite(img_plot)] = np.nan

if np.sum(np.isfinite(img_plot)) == 0:
    raise RuntimeError("No valid image pixels found")

vmin = np.nanpercentile(img_plot, 2)
vmax = np.nanpercentile(img_plot, 98)

# DEM plotting: mask zeros so fill areas do not look like real low terrain
dem_plot = dem.astype(float).copy()
dem_plot[dem_plot == 0] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(img_plot, cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title("OPERA VH image")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(dem_plot, cmap="terrain")
axes[1].contour(mask > 0, levels=[0.5], colors="red", linewidths=0.6)
axes[1].set_title("DEM with RGI outline")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(img_plot, cmap="gray", vmin=vmin, vmax=vmax)
axes[2].contour(mask > 0, levels=[0.5], colors="cyan", linewidths=0.6)
axes[2].set_title("VH with RGI outline")
axes[2].axis("off")

plt.tight_layout()
plt.show()