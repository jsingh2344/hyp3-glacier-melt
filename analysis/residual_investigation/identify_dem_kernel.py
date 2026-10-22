from pathlib import Path
import pandas as pd,numpy as np
from osgeo import gdal
R=Path(__file__).resolve().parent
ref=gdal.Open('DEM/dem.tif');a=ref.ReadAsArray();gt=ref.GetGeoTransform();bounds=(gt[0],gt[3]+gt[5]*ref.RasterYSize,gt[0]+gt[1]*ref.RasterXSize,gt[3]);tiles=[str(p) for p in sorted(Path('DEM/copernicus_glo30_tiles').glob('*.tif'))];rows=[]
for mode in ['Point','Area']:
 vrt=gdal.BuildVRT(f'/vsimem/residual_{mode}.vrt',tiles);vrt.SetMetadataItem('AREA_OR_POINT',mode);vrt.FlushCache()
 for kernel in ['near','bilinear','cubic','cubicspline','average','lanczos']:
  ds=gdal.Warp('',vrt,format='MEM',dstSRS='EPSG:32607',outputBounds=bounds,xRes=30,yRes=30,targetAlignedPixels=True,resampleAlg=kernel,multithread=True);b=ds.ReadAsArray();d=abs(b-a);rows.append(dict(metadata=mode,kernel=kernel,mae=float(np.nanmean(d)),max_abs=float(np.nanmax(d)),exact_percent=float(100*np.mean(a==b))));print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(R/'local_dem_kernel_audit.csv',index=False)
