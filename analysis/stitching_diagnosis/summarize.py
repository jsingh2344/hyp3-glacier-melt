from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
rows=[]
for fn in (R/'runs_all_years/source_good').glob('*all_years.csv'):
 rid=fn.name.split('time_series_')[1].split('_all_years')[0]
 dfs={s:pd.read_csv(R/'runs_all_years'/s/fn.name) for s in ['source_good','source_demfinite','merged_demfinite','merged_bilinear'] if (R/'runs_all_years'/s/fn.name).exists()}
 for name,a,b in [('mask only','source_good','source_demfinite'),('maps only, DEM-only mask','source_demfinite','merged_demfinite'),('DEM only, merged maps','merged_demfinite','merged_bilinear')]:
  if a not in dfs or b not in dfs:continue
  for metric in ['melt_area_m2','ice_area_m2','melt_extent_elev_m','snowline_elev_m','glacier_max_elev_m']:
   join=dfs[a].merge(dfs[b],on=['year','date'],suffixes=('_a','_b'));x=join[metric+'_a'].to_numpy();y=join[metric+'_b'].to_numpy();delta=y-x;finite=np.isfinite(delta)
   rows.append(dict(glacier=rid,cause=name,metric=metric,rows=len(x),changed=int((~np.isclose(x,y,equal_nan=True,rtol=0,atol=1e-6)).sum()),mae=float(np.mean(abs(delta[finite]))) if finite.any() else np.nan,max_abs=float(np.max(abs(delta[finite]))) if finite.any() else np.nan))
d=pd.DataFrame(rows);d.to_csv(R/'causal_effects.csv',index=False)
# Directly compare interventions so small historical DEM differences cancel.
sel=['01.05550','01.05582','01.05583','01.05836','01.05839','01.05878','01.05804']
fig,axs=plt.subplots(1,2,figsize=(13,5.5));colors=['#e3a327','#328678','#b25c7e']
for j,metric in enumerate(['melt_area_m2','snowline_elev_m']):
 for i,cause in enumerate(d.cause.unique()):
  ss=d[(d.cause==cause)&(d.metric==metric)].set_index('glacier').reindex(sel)
  axs[j].bar(np.arange(len(sel))+(i-1)*.25,ss.mae/(1e6 if j==0 else 1),width=.24,label=cause,color=colors[i])
 axs[j].set_xticks(np.arange(len(sel)),sel,rotation=40,ha='right');axs[j].set_ylabel('Mean absolute change (km²)' if j==0 else 'Mean absolute change (m)');axs[j].set_title('Melt area' if j==0 else 'Snowline elevation')
axs[0].legend(fontsize=8);fig.suptitle('Which intervention changes the result? 2017–2024, exact source dates\nEffects depend on the stated mask/map context; bars are not additive');fig.tight_layout();fig.savefig(R/'05_causal_effects.png',dpi=160)
print(d[d.glacier.isin(sel)&d.metric.isin(['melt_area_m2','snowline_elev_m'])].to_string(index=False))
