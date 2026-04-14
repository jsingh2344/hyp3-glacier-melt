import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# --- set your datacube path here ---
cube_path = r"C:\Users\jaden\Downloads\Research\hyp3_repo\hyp3-glacier-melt\output_nc\Gulkana_32607_S1_cube_VH.nc"

# --- open dataset ---
ds = xr.open_dataset(cube_path)

# --- grab image data ---
arr = ds["images"].values

# flatten to finite values only
finite = arr[np.isfinite(arr)]

if finite.size == 0:
    raise RuntimeError("No finite values found in ds['images'].")

print("Datacube:", cube_path)
print("Variable:", "images")
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Finite count:", finite.size)
print("NaN count:", np.isnan(arr).sum())
print("Min:", np.nanmin(finite))
print("Max:", np.nanmax(finite))
print("Mean:", np.nanmean(finite))
print("Median:", np.nanmedian(finite))

# simple heuristic
vmin = np.nanmin(finite)
vmax = np.nanmax(finite)
vmean = np.nanmean(finite)

print("\n--- INTERPRETATION ---")
if vmax <= 5 and vmean >= 0:
    print("This looks like POWER / linear-scale SAR.")
elif vmin < -3 and vmax <= 5:
    print("This looks like DECIBEL (dB) SAR.")
else:
    print("Ambiguous range; inspect histogram and sample values below.")

# sample positive values and log-transform them for comparison
positive = finite[finite > 0]
if positive.size > 0:
    test_db = 10 * np.log10(positive)
    print("\nIf treated as power and converted to dB:")
    print("Converted min:", np.min(test_db))
    print("Converted max:", np.max(test_db))
    print("Converted mean:", np.mean(test_db))

# sample raw values
print("\nSample raw values:")
print(finite[:20])

# histogram
plt.figure(figsize=(8, 5))
plt.hist(finite, bins=100)
plt.title("Histogram of datacube image values")
plt.xlabel("Value")
plt.ylabel("Count")
plt.tight_layout()
plt.show()