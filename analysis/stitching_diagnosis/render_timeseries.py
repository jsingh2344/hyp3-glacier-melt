from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
fig,axs=plt.subplots(2,2,figsize=(12,7.5))
styles=[('source_good','Source maps + original SAR mask','#222222','-'),('source_demfinite','Same maps + DEM-only mask','#d99b20','-'),('merged_good','Merged maps + original SAR mask','#268878','--'),('merged_bilinear','Merged maps + DEM-only mask + bilinear DEM','#b6537e',':')]
for row,num in enumerate([5582,5836]):
 for col,metric in enumerate(['melt_area_m2','snowline_elev_m']):
  ax=axs[row,col]
  for stage,label,color,ls in styles:
   f=R/'runs_all_years'/stage/f'melt_snowline_time_series_01.{num:05}_all_years.csv';df=pd.read_csv(f).query('year == 2024');ax.plot(pd.to_datetime(df.date),df[metric]/(1e6 if col==0 else 1),label=label,color=color,ls=ls,lw=2)
  ax.set_title(f'01.{num:05} — '+('melt area' if col==0 else 'snowline elevation'));ax.set_ylabel('km²' if col==0 else 'm');ax.tick_params(axis='x',rotation=25);ax.spines[['right','top']].set_visible(False)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9,bbox_to_anchor=(.5,-.01));fig.suptitle('Same onset maps, different answers: the validity mask drives these examples\nMerged and source maps are identical on both glaciers across all eight years',fontsize=12);fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(R/'03_controlled_timeseries.png',dpi=160)
