"""Observe production calculation locals without changing its source or behavior."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parent
namespace={'__file__':str(R/'replay_reconstructed.py')}
s=(R/'replay_reconstructed.py').read_text();exec(s.split('for num,orig in source.items():')[0],namespace);globals().update({k:v for k,v in namespace.items() if not k.startswith('__')})
records=[]
for num,day in [(5584,'2019-08-29'),(5813,'2017-08-27'),(5904,'2020-05-19'),(5904,'2017-12-25')]:
 m=mask==num;rr,cc=np.where(m);r0=rr.min()-3;r1=rr.max()+4;c0=cc.min()-3;c1=cc.max()+4;sl=np.s_[r0:r1,c0:c1];grid=RasterGrid(crs,window_transform(Window(c0,r0,c1-c0,r1-r0),tf),r1-r0,c1-c0);area=float(rgi.loc[rgi.num==num].area_km2.iloc[0]);rec=(f'01.{num:05}',num,area,m[sl]);date=pd.Timestamp(day);yr=date.year
 for label,dsdem in [('bilinear_30m',cube.dem.values),('cubic_30m',dem)]:
  aps=[AnnualDoyMaps(yr,grid,*(maps[yr][k][sl] for k in keys),np.ones(m[sl].shape,bool),good[sl],dsdem[sl])]
  dc=onset_datacube(aps,dsdem[sl],[rec],MeltConfig(),1,None,1);dc.dates=np.array([date.to_datetime64()]);dc.doys=[date.dayofyear];dc.years=[yr];prepare_glacier_spatial_state(dc,num,area);events=[]
  def trace(frame,event,arg):
   if frame.f_code.co_name!='generate_elevs_from_onsets':return trace
   if event!='line':return trace
   line=frame.f_lineno;l=frame.f_locals
   if line in [656,660,704,707]:
    events.append(dict(step='selected_20m_bin' if line<680 else 'selected_equal_area_bin',lower=float(l['bin_elev_lower']),upper=float(l['bin_elev_upper']),pixels=int(l['bin_count']),melted=int(l['data_cp_bin_count']),fraction=float(l['frac_melt']),trigger='100%' if line in [660,707] else '>90%'))
   if line in [664,675,711,724,736]:
    fields=['allmelt_elev','melt_extent_elev','melt_extent_elev_min','nomelt_pixels_below','min_idx','melt_pixels','melt_area_m2','max_snowline','snowline_elev','ice_area_m2']
    row={'step':{664:'before_first_correction',675:'after_first_correction',711:'before_second_correction',724:'after_second_correction',736:'final'}[line]}
    for field in fields:
     if field in l:row[field]=float(l[field])
    if 'melted_2d' in l:row['melted_pixels']=int(l['melted_2d'].sum())
    if 'final_snowline' in l:row['snowline_pixels']=int(l['final_snowline'].sum())
    events.append(row)
   return trace
  out=R/'traced_runs'/label;out.mkdir(parents=True,exist_ok=True)
  try:
   sys.settrace(trace);dc.generate_elevs_from_onsets(num,out,rec[0])
  finally:sys.settrace(None)
  records.append(dict(glacier=rec[0],date=day,dem=label,events=events))
(R/'outlier_calculation_traces.json').write_text(json.dumps(records,indent=2))
for item in records:print(json.dumps(item),flush=True)
