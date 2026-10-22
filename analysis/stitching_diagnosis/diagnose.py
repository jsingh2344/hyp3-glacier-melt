from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import io,json,re,warnings
import numpy as np,pandas as pd,xarray as xr,rasterio,geopandas as gpd
from rasterio.warp import reproject,Resampling
from rasterio.features import rasterize
from rasterio.windows import Window,transform as window_transform
from rasterio.io import MemoryFile
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hyp3_glacier_melt.doy_workflow.pipeline import AnnualDoyMaps,RasterGrid,onset_datacube,prepare_glacier_spatial_state,load_dem_on_grid
from hyp3_glacier_melt.config import MeltConfig
warnings.filterwarnings('ignore',category=RuntimeWarning)
ROOT=Path(__file__).resolve().parent; ROOT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
zs=[ZipFile(p) for p in sorted(Path('doy-inputs').glob('*.zip'))]
keys=['melt_onset','ice_onset','spring_baseline','second_melt_onset']
def readmap(z,key,year):
 n=next(n for n in z.namelist() if n.endswith(f'/{key}_doy_{year}.tif'))
 with MemoryFile(z.read(n)) as mf:
  with mf.open() as ds:return ds.read(1),ds.transform,ds.crs
ref,tf,crs=readmap(zs[0],keys[0],2024);shape=ref.shape
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc')
assert shape==cube.dem.shape
assert np.allclose(tuple(tf.to_gdal()),cube.attrs['geotransform'])
dem=cube.dem.values.astype('float32'); maskids=cube.rgi_ind_glacier_mask.values
print('loading valid time-series pixels',flush=True)
images=cube.images.values; good=np.all(np.isfinite(images)&(images>=-1e10),axis=0);del images
rgi=gpd.read_file('Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp').to_crs(crs)
rgi['num']=rgi.rgi_id.str[-5:].astype(int)
frames={}
for n in zs[0].namelist():
 m=re.search(r'time_series_(01\.\d+)_all_years.csv$',n)
 if m:frames[int(m[1].split('.')[1])]=pd.read_csv(io.BytesIO(zs[0].read(n)))
# Reproject second burst onto the first burst grid: same CRS, pixel alignment and resolution.
allmaps={};byyear={};stats=[];spatial=[]
for year in range(2017,2025):
 a={};b={};merged={}
 for k in keys:
  a[k],_,_=readmap(zs[0],k,year); bb,bt,bc=readmap(zs[1],k,year)
  b[k]=np.full(shape,np.nan,dtype='float32')
  reproject(bb,b[k],src_transform=bt,src_crs=bc,dst_transform=tf,dst_crs=crs,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.nearest)
  merged[k]=np.fmin(a[k],b[k])
  for num in frames:
   m=maskids==num; both=m&np.isfinite(a[k])&np.isfinite(b[k]);changed=m&(~np.isclose(a[k],merged[k],equal_nan=True))
   stats.append(dict(year=year,glacier=f'01.{num:05}',map=k,pixels=int(m.sum()),both_finite=int(both.sum()),conflicting=int((both&(a[k]!=b[k])).sum()),earlier=int((both&(b[k]<a[k])).sum()),filled=int((m&~np.isfinite(a[k])&np.isfinite(b[k])).sum()),changed=int(changed.sum()),median_delta=float(np.median((b[k]-a[k])[both])) if both.any() else np.nan))
 byyear[year]={'source':a,'other':b,'merged':merged}
 if year==2024: allmaps={'source':a,'other':b,'merged':merged}
pd.DataFrame(stats).to_csv(ROOT/'pixel_changes_2017_2024.csv',index=False)
print('maps and overlap statistics ready',flush=True)
grid=RasterGrid(crs,tf,*shape)
bilinear=load_dem_on_grid(Path('DEM/dem.tif'),grid)
avg=np.full(shape,np.nan,dtype='float32')
with rasterio.open('DEM/dem.tif') as ds:
 reproject(rasterio.band(ds,1),avg,src_transform=ds.transform,src_crs=ds.crs,src_nodata=ds.nodata,dst_transform=tf,dst_crs=crs,dst_nodata=np.nan,resampling=Resampling.average)
avg=np.rint(avg)
rows=[]
for num,fr in frames.items():
 m=maskids==num;geom=rgi.loc[rgi.num==num].iloc[0]
 rm=rasterize([(geom.geometry,1)],out_shape=shape,transform=tf).astype(bool)
 vals=dem[m&good]; valsall=dem[m&np.isfinite(dem)]
 rows.append(dict(glacier=f'01.{num:05}',pixels=int(m.sum()),rgi_mask_changed=int((m^rm).sum()),invalid_sar_pixels=int((m&~good).sum()),dem_finite_pixels=int((m&np.isfinite(dem)).sum()),original_max=float(fr.glacier_max_elev_m.iloc[0]),cube_good_max=float(vals.max()),cube_all_max=float(valsall.max()),bilinear_mae=float(np.nanmean(abs(bilinear[m]-dem[m]))),average_mae=float(np.nanmean(abs(avg[m]-dem[m])))))
pd.DataFrame(rows).to_csv(ROOT/'spatial_audit.csv',index=False)
# Controlled metric runs: identical dates, RGI pixels, source DEM; vary one input at a time.
selected=list(frames)
comparisons=[];results={};errors=[]; stages=['source_good','source_demfinite','merged_good','merged_demfinite','merged_bilinear']
for num in selected:
 m=maskids==num;rr,cc=np.where(m);r0=max(0,rr.min()-3);r1=min(shape[0],rr.max()+4);c0=max(0,cc.min()-3);c1=min(shape[1],cc.max()+4);sl=np.s_[r0:r1,c0:c1]
 localgrid=RasterGrid(crs,window_transform(Window(c0,r0,c1-c0,r1-r0),tf),r1-r0,c1-c0)
 record=(f'01.{num:05}',num,float(rgi.loc[rgi.num==num].area_km2.iloc[0]),m[sl])
 original=frames[num].copy();dates=pd.to_datetime(original.date);results[num]={'original_csv':original}
 for stage in stages:
  maps=allmaps['source' if stage.startswith('source') else 'merged']
  dsdem=(bilinear if stage=='merged_bilinear' else dem)[sl].copy()
  aps=[AnnualDoyMaps(yr,localgrid,*(byyear[yr]['source' if stage.startswith('source') else 'merged'][k][sl] for k in keys),np.ones(dsdem.shape,bool)) for yr in byyear]
  dc=onset_datacube(aps,dsdem,[record],MeltConfig(),1,None,1)
  dc.dates=dates.to_numpy();dc.doys=dates.dayofyear.tolist() if isinstance(dates,pd.DatetimeIndex) else dates.dt.dayofyear.tolist();dc.years=dates.dt.year.tolist()
  if stage.endswith('_good'):dc.mask_good_pixels=np.where(good[sl],1.,np.nan)
  prepare_glacier_spatial_state(dc,num,record[2]);out=ROOT/'runs_all_years'/stage;out.mkdir(parents=True,exist_ok=True)
  
  try:
   f=dc.generate_elevs_from_onsets(num,out,record[0]);df=pd.read_csv(f);results[num][stage]=df
  except Exception as exc:
   errors.append(dict(glacier=record[0],stage=stage,error=repr(exc)));print('ERROR',num,stage,repr(exc),flush=True);continue
  for metric in ['melt_area_m2','ice_area_m2','melt_extent_elev_m','snowline_elev_m','glacier_max_elev_m']:
   x=original[metric].to_numpy();y=df[metric].to_numpy();diff=y-x;finite=np.isfinite(diff)
   comparisons.append(dict(glacier=record[0],stage=stage,metric=metric,rows=len(x),equal=int(np.isclose(x,y,equal_nan=True,rtol=0,atol=1e-6).sum()),mae=float(np.mean(abs(diff[finite]))) if finite.any() else None,max_abs=float(np.max(abs(diff[finite]))) if finite.any() else None,nan_mismatch=int((np.isnan(x)^np.isnan(y)).sum())))
  print(num,stage,'done',flush=True)
(ROOT/'run_errors.json').write_text(json.dumps(errors,indent=2))
pd.DataFrame(comparisons).to_csv(ROOT/'controlled_comparisons_all_years.csv',index=False)
# Plot 1: all-year map edits by glacier and event type.
st=pd.DataFrame(stats);p=st.groupby(['glacier','map'])[['changed','pixels']].sum();p['percent']=100*p.changed/p.pixels
h=p.percent.unstack().reindex(columns=keys);h=h.loc[h.max(axis=1).sort_values(ascending=False).index]
fig,ax=plt.subplots(figsize=(9,10));im=ax.imshow(h.values,aspect='auto',cmap='magma',vmin=0,vmax=max(1,h.values.max()));ax.set_yticks(range(len(h)),h.index);ax.set_xticks(range(4),['Melt onset','Ice onset','Spring baseline','Second melt']);ax.set_title('Pixels changed by stitching, relative to burst 028627\n2017–2024; earlier values plus newly filled pixels');fig.colorbar(im,ax=ax,label='% of glacier pixel-years');fig.tight_layout();fig.savefig(ROOT/'01_pixel_changes.png');plt.close(fig)
# Plot 2: mask and DEM causal audit.
s=pd.DataFrame(rows);fig,axs=plt.subplots(1,3,figsize=(14,5));axs[0].scatter(s.bilinear_mae,s.average_mae,color='#226b91');axs[0].set(xlabel='Bilinear DEM mean absolute difference (m)',ylabel='Average + rounded DEM difference (m)',title='DEM preparation vs stored cube')
axs[1].bar(range(len(s)),100*s.invalid_sar_pixels/s.pixels,color='#b66b39');axs[1].set(xlabel='31 source glaciers',ylabel='Pixels excluded by original SAR mask (%)',title='DOY adapter drops the SAR validity mask');axs[1].set_xticks([])
axs[2].scatter(s.cube_good_max,s.cube_all_max-s.cube_good_max,color='#226b91');axs[2].set(xlabel='Original valid-pixel maximum elevation (m)',ylabel='Increase when SAR mask is removed (m)',title='A mask change adds 78 m to one maximum');
for _,r in s.iterrows():
 if r.cube_all_max-r.cube_good_max>5:axs[2].annotate(r.glacier,(r.cube_good_max,r.cube_all_max-r.cube_good_max),xytext=(5,5),textcoords='offset points')
fig.tight_layout();fig.savefig(ROOT/'02_spatial_audit.png');plt.close(fig)
# Plot 3: selected controlled time series.
fig,axs=plt.subplots(3,2,figsize=(13,11));colors={'original_csv':'black','source_good':'#999999','source_demfinite':'#e29e21','merged_good':'#248a78','merged_demfinite':'#3376b6','merged_bilinear':'#b34d72'}
for row,num in enumerate([5582,5836,5804]):
 for col,metric in enumerate(['melt_area_m2','snowline_elev_m']):
  ax=axs[row,col]
  for stage in ['original_csv','source_good','source_demfinite','merged_good','merged_bilinear']:
   
   if stage not in results[num]:continue
   df=results[num][stage].query('year == 2024');ax.plot(pd.to_datetime(df.date),df[metric]/(1e6 if col==0 else 1),label=stage,color=colors[stage],lw=1.5,alpha=.85,linestyle='--' if stage=='source_good' else '-')
  ax.set_title(f'01.{num:05} — '+('melt area' if col==0 else 'snowline'));ax.set_ylabel('km²' if col==0 else 'm');ax.tick_params(axis='x',rotation=30)
axs[0,0].legend(fontsize=8);fig.suptitle('2024 exact acquisition dates: isolating maps, validity mask, and DEM',y=1.0);fig.tight_layout();fig.savefig(ROOT/'03_controlled_timeseries.png');plt.close(fig)
# Plot 4: a spatial example separating earlier overlap dates and gap filling.
num=5804;m=maskids==num;rr,cc=np.where(m);sl=np.s_[max(0,rr.min()-5):rr.max()+6,max(0,cc.min()-5):cc.max()+6]
a=allmaps['source']['melt_onset'][sl];b=allmaps['other']['melt_onset'][sl];mm=m[sl];change=np.where(mm,np.fmin(a,b)-a,np.nan)
fig,axs=plt.subplots(1,3,figsize=(13,4));
for ax,arr,title in zip(axs,[np.where(mm,a,np.nan),np.where(mm,b,np.nan),change],['028627 melt onset','028628 melt onset','Stitched minus 028627']):
 im=ax.imshow(arr,cmap='coolwarm' if ax==axs[2] else 'viridis');fig.colorbar(im,ax=ax,label='days (DOY)' if ax!=axs[2] else 'days');ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Glacier 01.05804 — 2024, same 100 m grid');fig.tight_layout();fig.savefig(ROOT/'04_overlap_example.png');plt.close(fig)
print('COMPLETE',flush=True)
