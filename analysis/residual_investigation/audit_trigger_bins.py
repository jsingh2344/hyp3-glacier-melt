from pathlib import Path
import json
R=Path(__file__).resolve().parent
namespace={'__file__':str(R/'trace_residuals.py')}
s=(R/'trace_residuals.py').read_text();exec(s.split('records=[]')[0],namespace);globals().update({k:v for k,v in namespace.items() if not k.startswith('__')})
rows=[];pixels=[]
for num,day,lo in [(5584,'2019-08-29',3140),(5813,'2017-08-27',2680),(5904,'2017-12-25',2880),(5904,'2020-05-19',2240)]:
 date=pd.Timestamp(day);yr=date.year;doy=date.dayofyear;m=(mask==num)&good;a=maps[yr];melt=(a['melt_onset']<=doy);allmelt=(melt&~(a['ice_onset']<=doy))|(a['second_melt_onset']<=doy);masks={}
 for label,arr in [('bilinear',cube.dem.values),('cubic',dem)]:
  bm=m&(arr>lo)&(arr<=lo+20);masks[label]=bm;count=int(bm.sum());n=int((bm&allmelt).sum());rows.append(dict(glacier=f'01.{num:05}',date=day,bin_lower_m=lo,bin_upper_m=lo+20,dem=label,pixels=count,allmelt_pixels=n,fraction=n/count if count else None))
 for r,c in zip(*np.where(masks['bilinear']^masks['cubic'])):pixels.append(dict(glacier=f'01.{num:05}',date=day,row=int(r),column=int(c),bilinear_elev=float(cube.dem.values[r,c]),cubic_elev=float(dem[r,c]),in_allmelt=bool(allmelt[r,c])))
pd.DataFrame(rows).to_csv(R/'critical_bin_membership.csv',index=False);pd.DataFrame(pixels).to_csv(R/'critical_bin_pixels.csv',index=False);print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(pixels).to_string(index=False))
