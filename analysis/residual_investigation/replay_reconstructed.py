from pathlib import Path
from zipfile import ZipFile
import io,re,json
import numpy as np,pandas as pd,xarray as xr,geopandas as gpd,rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window,transform as window_transform
from hyp3_glacier_melt.doy_workflow.pipeline import AnnualDoyMaps,RasterGrid,onset_datacube,prepare_glacier_spatial_state
from hyp3_glacier_melt.config import MeltConfig
R=Path(__file__).resolve().parent
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc');dem=np.load(R/'reconstructed_dem.npy');good=np.load(R/'sar_validity.npy');mask=cube.rgi_ind_glacier_mask.values
z=ZipFile(next(Path('doy-inputs').glob('*028627*.zip')));keys=['melt_onset','ice_onset','spring_baseline','second_melt_onset'];maps={};source={}
for n in z.namelist():
 m=re.search(r'time_series_(01\.(\d+))_all_years.csv$',n)
 if m:source[int(m[2])]=pd.read_csv(io.BytesIO(z.read(n)))
for yr in range(2017,2025):
 maps[yr]={}
 for k in keys:
  n=next(n for n in z.namelist() if n.endswith(f'{k}_doy_{yr}.tif'))
  with MemoryFile(z.read(n)) as mem:
   with mem.open() as ds:maps[yr][k]=ds.read(1);tf=ds.transform;crs=ds.crs
rgi=gpd.read_file('Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp');rgi['num']=rgi.rgi_id.str[-5:].astype(int)
metrics=['melt_area_m2','ice_area_m2','melt_extent_elev_m','snowline_elev_m','glacier_area_m2','glacier_max_elev_m','melt_fraction','ice_fraction'];rows=[]
for num,orig in source.items():
 m=mask==num;rr,cc=np.where(m);r0=rr.min()-3;r1=rr.max()+4;c0=cc.min()-3;c1=cc.max()+4;sl=np.s_[r0:r1,c0:c1];grid=RasterGrid(crs,window_transform(Window(c0,r0,c1-c0,r1-r0),tf),r1-r0,c1-c0);area=float(rgi.loc[rgi.num==num].area_km2.iloc[0]);rec=(f'01.{num:05}',num,area,m[sl]);aps=[AnnualDoyMaps(yr,grid,*(maps[yr][k][sl] for k in keys),np.ones(m[sl].shape,bool),good[sl],dem[sl]) for yr in maps]
 dc=onset_datacube(aps,dem[sl],[rec],MeltConfig(),1,None,1);dates=pd.to_datetime(orig.date);dc.dates=dates.to_numpy();dc.doys=dates.dt.dayofyear.tolist();dc.years=dates.dt.year.tolist();prepare_glacier_spatial_state(dc,num,area);out=R/'reconstructed_replay';out.mkdir(exist_ok=True);fn=dc.generate_elevs_from_onsets(num,out,rec[0]);f=pd.read_csv(fn)
 for metric in metrics:
  a=orig[metric].to_numpy();b=f[metric].to_numpy();eq=np.isclose(a,b,rtol=0,atol=1e-9,equal_nan=True);d=np.abs(b-a);valid=np.isfinite(d)
  rows.append(dict(glacier=rec[0],metric=metric,rows=len(a),equal=int(eq.sum()),sum_abs=float(d[valid].sum()),finite=int(valid.sum()),max_abs=float(d[valid].max()) if valid.any() else 0,nan_mismatch=int((np.isnan(a)^np.isnan(b)).sum())))
 print(num,'exact metrics',sum(row['equal']==row['rows'] for row in rows[-8:]),'/8',flush=True)
f=pd.DataFrame(rows);f.to_csv(R/'reconstruction_per_glacier.csv',index=False);g=f.groupby('metric').sum(numeric_only=True);g['max_abs']=f.groupby('metric').max_abs.max();g['mae']=g.sum_abs/g.finite;g.to_csv(R/'reconstruction_summary.csv');print(g.to_string(),flush=True)
