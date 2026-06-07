# HyP3 glacier-melt

Melt extents and snowlines from RTC products: methodology overview

At each Sentinel-1 acquisition date, Wells et al. (2026) computes melt extents and snowlines by processing full glacier backscatter images. To parallelize on a per-pixel basis, while still producing similar results, we introduce an ‘onset map’ processing framework. Onset maps enable per-pixel parallelization by reducing the problem to independent time-series evaluations at each pixel, with each onset map storing the earliest day of year a condition is met. Subsequent post-processing evaluates these maps at each Sentinel-1 scene date to derive melt extents and snowlines, achieving R2>0.936 relative to Wells et al. (2026) in each month from May–August.

## 1. Melt Extents

### Onset map and mask generation

To calculate melt extents, we produce three spatially-distributed onset maps for each summer: a melt onset map, an ice onset map, and a second melt onset map (Fig. 1). Each map stores, for every pixel on a glacier, the earliest day of year (DOY) at which a given condition (melt, second melt, or ice) is met. 
- The melt condition is triggered when a pixel drops more than 3 dB below its winter (January and February) mean (Figs. 2 and 3).
- The ice condition has two preconditions. The melt condition must already have been met earlier in the year, and the day of minimum backscatter also must have occurred earlier in the year. If both of those hold, the ice condition triggers when either:
    - a pixel is either 4 dB above its 5th percentile summer (May through September) backscatter level, or 
    - when a pixel is less than 3 dB below its winter mean. (Figs. 2 and 3).
- The second melt condition uses the same threshold as the melt condition, but additionally requires that the ice condition has already been met earlier in the year (Fig. 3).


To accommodate the ice condition check, we additionally generate yearly maps of the DOY of minimum backscatter. Note also that for quality control, for each of these checks, DOYs in January, February, and October-December are excluded from consideration.

<p align="center">
  <img width="512" height="499" alt="image" src="https://github.com/user-attachments/assets/25c96fbe-7e04-4475-9736-e29cee08d98f" />
</p>

#### Fig. 1: Example onset maps for glacier 5589 (RGI v7) in 2020

<p align="center">
  <img width="512" height="408" alt="image" src="https://github.com/user-attachments/assets/e3d7041a-22a6-4af9-8dea-1f90ee46f7aa" />
  <img width="512" height="282" alt="image" src="https://github.com/user-attachments/assets/c5cf0a5b-70a2-4d74-94d9-83fcee4b3fd2" />
</p>

#### Fig. 2: Backscatter in 2020 for a pixel in the upper ablation zone on glacier 5589 (RGI v7). Data points are labeled by their scene number in the overall dataset. This pixel meets the melt check at scene 90, the ice check at scene 100, and the second melt check at scene 101. 

<p align="center">
  <img width="512" height="408" alt="image" src="https://github.com/user-attachments/assets/6bcfb701-41e0-43f0-8860-e8e65f041210" />
  <img width="512" height="282" alt="image" src="https://github.com/user-attachments/assets/dd827796-344b-47ca-8d24-9b97303d02ef" />
</p>

#### Fig. 3: Backscatter in 2019 for a pixel in the lower ablation zone of glacier 5589 (RGI v7). Note that for this pixel, the melt onset DOY is at scene 60, and the ice onset DOY is at scene 72 (due to the minimum backscatter DOY requirement, which enforces that a positive ice check may occur only after scene 71).

### Extracting elevation from spatially-distributed maps

For each date with a Sentinel-1 scene, we derive melt, ice, and second melt masks (Fig. 4) from the corresponding spatially-distributed maps (Fig. 1). Binary maps are classified as true for pixels where the date’s DOY is greater than or equal to the onset DOY, and false otherwise.

To subsequently calculate the melt extent altitude, we use the melt mask with an additional adjustment to account for pixels misclassified as false (not melting) in the ablation zone. We first construct an ‘allmelt’ mask by subtracting the ice mask from the melt mask, and then adding the second melt mask (Fig. 4). We then iterate through elevation bins of the ‘allmelt’ mask and identify the highest elevation bin in which more than 90% of pixels are classified as melt (i.e., marked ‘true’ in the ‘allmelt’ mask). If a bin is composed of 100% melt pixels in the ‘allmelt’ mask, we select that bin and stop iteration there. All pixels below the elevation of the selected bin are set as ‘true’ in the original melt mask (Fig. 5). Finally, the melt extent elevation is determined based on the glacier’s cumulative area altitude distribution – for example, if 50 percent of the glacier is now classified as melting, we determine the melt extent elevation to be the glacier’s median elevation (Fig. 5). We further compute uncertainty in the melt extent using estimates of misclassified pixels. The minimum melt extent is computed by counting the number of pixels below the melt extent that remain unclassified in the melt mask (i.e., equal to false), and vice versa for the maximum melt extent. 

<p align="center">
  <img width="512" height="118" alt="image" src="https://github.com/user-attachments/assets/2fc75a0e-b97c-4c8d-8dae-0cd3ebb4c970" />
</p>

#### Fig. 4: Example elevation calculation for glacier 5589 (RGI v7) on June 24th, 2020. To generate the ‘allmelt’ mask, the ice mask is subtracted from the melt mask, and then the second melt mask is added to the result. The red band in the ‘allmelt’ mask highlights the pixels at the elevation of the selected elevation bin. Setting pixels below that elevation to be melting in the original melt mask then produces the final melt mask (Fig. 5).

<p align="center">
  <img width="566" height="573" alt="image" src="https://github.com/user-attachments/assets/57e81ca3-9b2f-499b-90a5-a5d7fe34596e" />
</p>

#### Fig. 5: The final melt mask after adjustments using the ‘allmelt’ correction. Counting pixels in the adjusted mask yields 26,279 melt-classified pixels out of 28,287 total (~93 percent). Accordingly, we take the 93rd percentile of the glacier’s elevation distribution, producing a melt extent elevation of 2,763 m. 

## 2. Snowlines

To calculate snowline elevation, we introduce an additional onset map: the spring baseline map, which uses the same condition as the ice onset map but does not require that the DOY of a given pixel must be greater than that pixel’s melt onset DOY. Note that in most cases, the majority of glacier pixels are immediately assigned the earliest possible DOY, as spring backscatter values are typically more than 4 dB higher than the summer minimum.

As with the melt extent maps, for a given Sentinel-1 scene date, we define a spring baseline mask as ‘true’ for pixels where the ice condition check has already occurred (Fig. 6), per the spring baseline DOY map (e.g., Fig. 1).

We then generate a cumulative ice mask by taking this spring baseline mask, excluding melting pixels, and adding the ice mask previously used in the melt extent calculation (Fig. 6). Next, we mask out pixels with elevations above the minimum melt extent, as these pixels have not yet shown a melt signal. Finally, the snowline elevation is determined based on the cumulative area altitude distribution (Fig. 7).

<p align="center">
  <img width="1558" height="317" alt="image" src="https://github.com/user-attachments/assets/4f2e4e77-d359-4eb7-9446-e1136a0252e8" />
  <img width="596" height="439" alt="image" src="https://github.com/user-attachments/assets/41daff80-5401-4560-b3c0-80e074113f5b" />
</p>

#### Fig. 6: Example elevation calculation for glacier 5999 (RGI v7) on September 28th, 2020. Subtracting the melt mask from the spring baseline mask, and then adding the second ice mask, produces the cumulative ice mask. A DEM is shown for reference.  

<p align="center">
  <img width="952" height="434" alt="image" src="https://github.com/user-attachments/assets/fa8c5e87-94c9-4640-bbb0-7a954aecb2b3" />
</p>

#### Fig. 7: After excluding pixels above the minimum melt extent, 1,305 out of 1,522 pixels (~86%) are classified as ice. The snowline elevation is therefore determined to be the 86th percentile of the glacier elevation distribution, which is 3,506 m.




