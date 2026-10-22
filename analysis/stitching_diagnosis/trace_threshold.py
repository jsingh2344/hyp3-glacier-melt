from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import numpy as np,pandas as pd,xarray as xr,rasterio,json,geopandas as gpd
from rasterio.io import MemoryFile
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
num=5878;name=f'melt_snowline_time_series_01.{num:05}_all_years.csv'
a=pd.read_csv(R/'runs_all_years/source_good'/name);b=pd.read_csv(R/'runs_all_years/source_demfinite'/name)
diff=b.melt_area_m2-a.melt_area_m2;i=abs(diff).idxmax();date=pd.Timestamp(a.date[i]);year=date.year;doy=date.dayofyear
cube=xr.open_dataset('datacubes/32607_S1_cube_VH_T014-028627-IW2.nc');m=cube.rgi_ind_glacier_mask.values==num;rr,cc=np.where(m);sl=np.s_[rr.min()-1:rr.max()+2,cc.min()-1:cc.max()+2];m=m[sl];dem=cube.dem.values[sl];im=cube.images.isel(y=slice(rr.min()-1,rr.max()+2),x=slice(cc.min()-1,cc.max()+2)).values;good=np.all(np.isfinite(im)&(im>=-1e10),axis=0)
z=ZipFile(next(Path('doy-inputs').glob('*028627*.zip')));maps={}
for k in ['melt_onset','ice_onset','spring_baseline','second_melt_onset']:
 n=next(n for n in z.namelist() if n.endswith(f'{k}_doy_{year}.tif'))
 with MemoryFile(z.read(n)) as mf:
  with mf.open() as ds:maps[k]=np.where(m,ds.read(1)[sl],np.nan)
raw=maps['melt_onset']<=doy;mask=(raw&~(maps['ice_onset']<=doy))|(maps['second_melt_onset']<=doy)
rgi=gpd.read_file('Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp');area=float(rgi.loc[rgi.rgi_id.str.endswith('05878'),'area_km2'].iloc[0]);pixels_per_bin=int(min(max(np.ceil(area*1e6/100/10000)*10000,100000),2000000)/10000)
summary={'glacier':'01.05878','date':str(date.date()),'doy':doy,'raw_melt_pixels':int(raw.sum()),'original_area_m2':float(a.melt_area_m2[i]),'unmasked_area_m2':float(b.melt_area_m2[i]),'difference_m2':float(diff[i]),'excluded_pixels':int((m&~good).sum()),'stages':{}}
fig,axs=plt.subplots(1,3,figsize=(14,4.5));details=[]
for stage in ['source_good','source_demfinite']:
 gd=np.where(m&(good if stage=='source_good' else True),dem,np.nan);bins=np.arange(np.floor(np.nanmin(gd)/20)*20,np.ceil(np.nanmax(gd)/20)*20+20,20);chosen=0;locked=False;records=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  bm=(gd>lo)&(gd<=hi);count=int(bm.sum());melt=int((mask&bm).sum());frac=melt/count if count else np.nan;reason=''
  if not locked:
   if chosen==0 and frac>.9 and count>10:chosen=lo;reason='>90% and >10 pixels'
   if frac==1:chosen=lo;locked=True;reason='100% (no minimum pixel count)'
  records.append(dict(stage=stage,lower_m=lo,pixels=count,allmelt_pixels=melt,fraction=frac,selected=reason))
 corrected=raw.copy()
 if chosen>0:corrected[gd<chosen]=1
 summary['stages'][stage]={'first_correction_elevation':float(chosen),'after_first_correction_pixels':int(corrected.sum()),'selected_bins':[r for r in records if r['selected']]}
 
 sorted_dem=np.sort(gd[np.isfinite(gd)]);ab=sorted_dem[::pixels_per_bin]
 if ab[-1]!=sorted_dem[-1]:ab=np.append(ab,sorted_dem[-1])
 chosen2=0;locked=False;records2=[]
 for lo,hi in zip(ab[:-1],ab[1:]):
  bm=(gd>lo)&(gd<=hi);count=int(bm.sum());melt=int((corrected&bm).sum());frac=melt/count if count else np.nan;reason=''
  if not locked:
   if chosen2==0 and frac>.9 and count>10:chosen2=lo;reason='>90% and >10 pixels'
   if frac==1:chosen2=lo;locked=True;reason='100%'
  records2.append(dict(stage=stage,lower_m=float(lo),upper_m=float(hi),pixels=count,allmelt_pixels=melt,fraction=frac,selected=reason))
 final=corrected.copy()
 if chosen2>0:final[gd<chosen2]=1
 summary['stages'][stage].update(second_correction_elevation=float(chosen2),after_second_correction_pixels=int(final.sum()),second_selected_bins=[r for r in records2 if r['selected']])
 df=pd.DataFrame(records2);details.extend(records2)
 axs[0].plot(df.lower_m,df.fraction,label=('Original SAR validity mask' if stage=='source_good' else 'Every finite DEM pixel'));axs[1].plot(df.lower_m,df.pixels,label=('Original SAR validity mask' if stage=='source_good' else 'Every finite DEM pixel'))
axs[0].set_xlim(1850,2150)
axs[0].axhline(.9,color='gray',ls=':');axs[0].set(xlabel='Elevation bin lower edge (m)',ylabel='Fraction in all-melt mask',title='Equal-area correction threshold');axs[0].legend(fontsize=8)
axs[1].set_xlim(1850,2150)
axs[1].set(xlabel='Elevation bin lower edge (m)',ylabel='Pixels in equal-area elevation bin',title='Same DEM; different valid-pixel population')
view=np.where(m,0.,np.nan);view[m&~good]=1.;axs[2].imshow(view,cmap='Oranges',vmin=0,vmax=1);axs[2].set_title('34 SAR-excluded pixels (orange)');axs[2].set_xticks([]);axs[2].set_yticks([])
fig.suptitle(f'01.05878 on {date.date()}: unchanged onset maps, 14.85 km² area difference');fig.tight_layout();fig.savefig(R/'06_threshold_amplification.png',dpi=160)
(R/'threshold_trace.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(details).to_csv(R/'threshold_bins.csv',index=False);print(json.dumps(summary,indent=2))
