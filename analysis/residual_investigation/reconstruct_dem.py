from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import json,io,re
import numpy as np,pandas as pd,xarray as xr
from osgeo import gdal
from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import source_copernicus_dem,open_and_resample_to_cube_grid,output_bounds_to_projwin
R=Path(__file__).resolve().parent
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc');tf=cube.attrs['geotransform'];h,w=cube.dem.shape;bounds=(tf[0],tf[3]-h*100,tf[0]+w*100,tf[3]);tiles=sorted(Path('DEM/copernicus_glo30_tiles').glob('*.tif'))
print('bounds',bounds,'tiles',len(tiles),flush=True)
with patch('hyp3lib.dem._get_dem_file_paths',return_value=[str(p.resolve()) for p in tiles]), patch('hyp3lib.dem._intersects_dem',return_value=True):
 source_copernicus_dem(R/'reconstructed_30m.tif',bounds,'EPSG:32607')
ds=open_and_resample_to_cube_grid(R/'reconstructed_30m.tif','EPSG:32607',output_bounds_to_projwin(bounds),bounds,100,100,'average');arr=np.rint(ds.ReadAsArray()).astype('float32');print('DEM grid',ds.GetGeoTransform(),arr.shape,flush=True)
np.save(R/'reconstructed_dem.npy',arr)
images=cube.images.values;valid=np.all(np.isfinite(images)&(images>=-1e10),axis=0);del images;np.save(R/'sar_validity.npy',valid)
mask=cube.rgi_ind_glacier_mask.values;z=ZipFile(next(Path('doy-inputs').glob('*028627*.zip')));rows=[]
for n in z.namelist():
 m=re.search(r'time_series_(01\.(\d+))_all_years.csv$',n)
 if not m:continue
 f=pd.read_csv(io.BytesIO(z.read(n)));sel=(mask==int(m[2]))&valid;orig=float(f.glacier_max_elev_m.iloc[0]);rows.append(dict(glacier=m[1],zip_max=orig,stored_max=float(cube.dem.values[sel].max()),reconstructed_max=float(arr[sel].max()),reconstructed_minus_zip=float(arr[sel].max()-orig),dem_mae_vs_stored=float(np.mean(abs(arr[sel]-cube.dem.values[sel])))))
out=pd.DataFrame(rows);out.to_csv(R/'dem_reconstruction_audit.csv',index=False);print(out.to_string(index=False),flush=True)
