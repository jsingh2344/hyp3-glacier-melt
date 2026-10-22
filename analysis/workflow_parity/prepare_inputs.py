"""Export original observations and DEM provenance for end-to-end parity tests."""
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np,xarray as xr,rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window,transform as window_transform
from hyp3_glacier_melt.doy_product import create_and_write_doy_products
R=Path(__file__).resolve().parent
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc')
dem=np.load('analysis/residual_investigation/reconstructed_dem.npy');good=np.load('analysis/residual_investigation/sar_validity.npy')
z=ZipFile(next(Path('doy-inputs').glob('*028627*.zip')))
dc=SimpleNamespace(ds=cube,dem=dem,mask_good_pixels=np.where(good,1.,np.nan),xres=100,yres=100)
for name,attribute in [('melt_onset','melt_onset_doy_maps'),('ice_onset','snowline_post_onset_doy_maps'),('spring_baseline','snowline_onset_doy_maps'),('second_melt_onset','annual_second_onset_map_doy_maps'),('minimum_backscatter','min_dB_doys')]:
 maps={}
 for year in range(2017,2025):
  path=next(n for n in z.namelist() if n.endswith(f'{name}_doy_{year}.tif'))
  with MemoryFile(z.read(path)) as memory:
   with memory.open() as ds:maps[year]=ds.read(1)
 setattr(dc,attribute,maps)
full=R/'inputs'/'single';full.mkdir(parents=True,exist_ok=True)
for year in range(2017,2025):create_and_write_doy_products(dc,year,full)
w=cube.sizes['x'];mid=w//2
for label,start,stop in [('left',0,mid+80),('right',mid-80,w)]:
 out=R/'inputs'/label;out.mkdir(exist_ok=True)
 for path in full.glob('*.tif'):
  with rasterio.open(path) as source:
   window=Window(start,0,stop-start,source.height);profile=source.profile.copy();profile.update(width=stop-start,transform=window_transform(window,source.transform))
   with rasterio.open(out/path.name,'w',**profile) as target:target.write(source.read(window=window));target.update_tags(**source.tags())
print('Prepared single-source and overlapping tiles with exact DEM, validity and acquisition dates.')
