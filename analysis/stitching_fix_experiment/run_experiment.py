"""Exercise the corrected export/discovery/stitching/metric path on real inputs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import io,json,re,os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import numpy as np,pandas as pd,xarray as xr,rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window,transform as window_transform
from rasterio.warp import reproject,Resampling
from hyp3_glacier_melt.doy_product import create_and_write_doy_products
from hyp3_glacier_melt.doy_workflow.pipeline import run_doy_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc')
images=cube.images.values;good=np.all(np.isfinite(images)&(images>=-1e10),axis=0);del images
z=ZipFile(next(Path('doy-inputs').glob('*028627*.zip')))
def read(name,year):
 n=next(n for n in z.namelist() if n.endswith(f'/{name}_doy_{year}.tif'))
 with MemoryFile(z.read(n)) as mem:
  with mem.open() as ds:return ds.read(1)
source={};dates=set()
for n in z.namelist():
 m=re.search(r'time_series_(01\.\d+)_all_years.csv$',n)
 if m:
  source[m[1]]=pd.read_csv(io.BytesIO(z.read(n)));dates.update(source[m[1]].date)
cal=pd.to_datetime(sorted(dates));years=range(2017,2025)
def exact_doys(year,start_doy,end_doy,doy_step):return cal[cal.year==year].dayofyear.tolist()
names=['melt_onset','ice_onset','spring_baseline','second_melt_onset','minimum_backscatter']
attrs=['melt_onset_doy_maps','snowline_post_onset_doy_maps','snowline_onset_doy_maps','annual_second_onset_map_doy_maps','min_dB_doys']
dc=SimpleNamespace(ds=cube,dem=cube.dem.values,mask_good_pixels=np.where(good,1.,np.nan),xres=100,yres=100)
for name,attr in zip(names,attrs):setattr(dc,attr,{year:read(name,year) for year in years})
inputdir=R/'inputs'/'full';inputdir.mkdir(parents=True,exist_ok=True)
for year in years:create_and_write_doy_products(dc,year,inputdir)
print('Exported real maps with original SAR validity and native metric DEM',flush=True)
# Split one real product into two overlapping rasters: no new observations or event fusion.
# This tests actual discovery/mosaicking, rather than bypassing the stitcher with premerged arrays.
w=cube.sizes['x'];mid=w//2
for label,start,stop in [('left',0,mid+80),('right',mid-80,w)]:
 out=R/'inputs'/label;out.mkdir(exist_ok=True)
 for f in inputdir.glob('*.tif'):
  with rasterio.open(f) as ds:
   win=Window(start,0,stop-start,ds.height);profile=ds.profile.copy();profile.update(width=stop-start,transform=window_transform(win,ds.transform))
   with rasterio.open(out/f.name,'w',**profile) as target:target.write(ds.read(window=win))
# Legacy inputs lack the new companions. Run their original bilinear/DEM-only behavior as control.
legacy=R/'inputs'/'legacy';legacy.mkdir(exist_ok=True)
for f in inputdir.glob('*.tif'):
 if not f.name.startswith(('validity_mask','metric_dem')):
  target=legacy/f.name
  if not target.exists():os.link(f,target)
def legacy_dem(path,grid):
 arr=np.full((grid.height,grid.width),np.nan,dtype='float32')
 with rasterio.open(path) as ds:
  reproject(rasterio.band(ds,1),arr,src_transform=ds.transform,src_crs=ds.crs,src_nodata=ds.nodata,dst_transform=grid.transform,dst_crs=grid.crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
 return arr
rgi='Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp'
runs={}
for name,inputs in [('old_behavior',[legacy]),('corrected_single',[inputdir]),('corrected_stitched',[R/'inputs'/'left',R/'inputs'/'right']),('corrected_reverse',[R/'inputs'/'right',R/'inputs'/'left'])]:
 out=R/name;out.mkdir(exist_ok=True)
 kwargs=dict(doy_inputs=inputs,output_dir=out,rgi_shapefile=rgi)
 with patch('hyp3_glacier_melt.doy_workflow.pipeline._evaluation_doys',exact_doys):
  if name=='old_behavior':
   with patch('hyp3_glacier_melt.doy_workflow.pipeline.load_dem_on_grid',legacy_dem):result=run_doy_pipeline(**kwargs,dem_path='DEM/dem.tif')
  else:result=run_doy_pipeline(**kwargs)
 produced=set(result.csv_files)
 for stale in out.glob('*_all_years.csv'):
  if stale not in produced:stale.unlink()
 runs[name]={re.search(r'time_series_(.+)_all_years',p.name)[1]:pd.read_csv(p) for p in result.csv_files}
 print(name,'glaciers',len(result.csv_files),'failed',result.failed_glacnos,flush=True)
 if result.failed_glacnos:raise RuntimeError(result.failed_glacnos)
metrics=['melt_area_m2','ice_area_m2','melt_extent_elev_m','snowline_elev_m','glacier_area_m2','glacier_max_elev_m']
records=[];perglacier=[]
for run,frames in runs.items():
 assert set(source).issubset(frames)
 for rid in source:
  df=frames[rid]
  for reference,other in [('original_zip',source[rid]),('corrected_single',runs['corrected_single'][rid])]:
   merged=other.merge(df,on=['year','date'],suffixes=('_reference','_test'),validate='one_to_one');assert len(merged)==len(other)==226
   for metric in metrics:
    a=merged[metric+'_reference'].to_numpy();b=merged[metric+'_test'].to_numpy();eq=np.isclose(a,b,rtol=0,atol=1e-6,equal_nan=True);finite=np.isfinite(a)&np.isfinite(b);d=np.abs(b[finite]-a[finite]);
    records.append(dict(run=run,reference=reference,glacier=rid,metric=metric,rows=len(a),equal=int(eq.sum()),finite=int(finite.sum()),sum_abs=float(d.sum()),max_abs=float(d.max()) if len(d) else np.nan,nan_mismatch=int((np.isnan(a)^np.isnan(b)).sum())))
d=pd.DataFrame(records);d.to_csv(R/'per_glacier_comparison.csv',index=False)
g=d.groupby(['run','reference','metric'],as_index=False).agg(rows=('rows','sum'),equal=('equal','sum'),finite=('finite','sum'),sum_abs=('sum_abs','sum'),max_abs=('max_abs','max'),nan_mismatch=('nan_mismatch','sum'));g['exact_percent']=100*g.equal/g.rows;g['mae']=g.sum_abs/g.finite;g.to_csv(R/'summary.csv',index=False)
for run in ['corrected_stitched','corrected_reverse']:
 check=g[(g.run==run)&(g.reference=='corrected_single')];assert (check.equal==check.rows).all()
case={}
for run in runs:
 row=runs[run]['01.05878'].query("date == '2020-05-19'").iloc[0];case[run]=float(row.melt_area_m2)
summary={'source_glaciers':31,'rows_per_run':7006,'threshold_case_melt_area_m2':case,'split_overlap_columns':160,'scope':'028627 real product, single-source replay and two overlapping spatial tiles of that same product; not two independent bursts','all_stitched_and_reverse_metrics_equal_to_corrected_single':True}
(R/'result.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(12,4.5));sel=g[g.reference=='original_zip'];labels=['Melt area','Ice area','Melt elevation','Snowline'];keys=metrics[:4]
for i,run in enumerate(['old_behavior','corrected_single']):
 s=sel[sel.run==run].set_index('metric').reindex(keys);axs[0].bar(np.arange(4)+(i-.5)*.32,s.exact_percent,.32,label='Before' if i==0 else 'Corrected')
axs[0].set_xticks(range(4),labels,rotation=20);axs[0].set_ylabel('Rows matching original ZIP (%)');axs[0].set_ylim(0,105);axs[0].legend();axs[0].set_title('31 glaciers, 7,006 glacier/date rows')
for run,label,color,ls in [('old_behavior','Before','#b6537e','-'),('corrected_single','Corrected single','#222222','-'),('corrected_stitched','Corrected stitched tiles','#238777','--')]:
 f=runs[run]['01.05878'];f=f[(f.date>='2020-04-01')&(f.date<='2020-08-01')];axs[1].plot(pd.to_datetime(f.date),f.melt_area_m2/1e6,label=label,color=color,ls=ls,lw=2)
axs[1].scatter(pd.Timestamp('2020-05-19'),30.27,color='black',zorder=5);axs[1].set_title('01.05878: threshold case restored');axs[1].set_ylabel('Melt area (km²)');axs[1].tick_params(axis='x',rotation=25);axs[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'fix_results.png',dpi=160)
print(g[g.reference=='original_zip'][['run','metric','exact_percent','mae','max_abs']].to_string(index=False));print(json.dumps(summary,indent=2))
