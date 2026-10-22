from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/hyp3-mpl-cache'
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
previous=pd.read_csv('analysis/stitching_fix_experiment/summary.csv');previous=previous[(previous.run=='corrected_single')&(previous.reference=='original_zip')].set_index('metric')
now=pd.read_csv(R/'reconstruction_summary.csv').set_index('metric');keys=['melt_area_m2','ice_area_m2','melt_extent_elev_m','snowline_elev_m','glacier_max_elev_m'];labels=['Melt area','Ice area','Melt elevation','Snowline','Maximum elevation']
fig,axs=plt.subplots(1,2,figsize=(13,5));x=np.arange(5)
axs[0].bar(x-.18,previous.loc[keys].exact_percent,.36,label='Previous replay: local bilinear DEM',color='#b66d4b');axs[0].bar(x+.18,100*now.loc[keys].equal/now.loc[keys].rows,.36,label='Historical cubic DEM reconstructed',color='#26877e');axs[0].set_xticks(x,labels,rotation=25,ha='right');axs[0].set_ylim(0,111);axs[0].set_ylabel('Rows matching archived ZIP (%)');axs[0].set_title('31 glaciers × 226 dates: all residuals disappear');axs[0].legend(fontsize=8,loc='lower left')
bins=pd.read_csv(R/'critical_bin_membership.csv',dtype={'glacier':str});groups=bins[['glacier','date']].drop_duplicates();x=np.arange(len(groups))
for i,dem in enumerate(['bilinear','cubic']):
 ss=bins[bins.dem==dem];values=ss.fraction.fillna(0).to_numpy()*100;bars=axs[1].bar(x+(i-.5)*.35,values,.35,label=dem.capitalize()+' 30 m DEM',color=['#b66d4b','#26877e'][i])
 for bar,(_,row) in zip(bars,ss.iterrows()):axs[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{int(row.allmelt_pixels)}/{int(row.pixels)}' if row.pixels else 'empty',ha='center',fontsize=9)
axs[1].axhline(100,color='gray',ls=':',lw=1);axs[1].set_ylim(0,116);axs[1].set_xticks(x,[f'{r.glacier}\n{r.date}' for _,r in groups.iterrows()],rotation=15);axs[1].set_ylabel('Pixels satisfying all-melt condition (%)');axs[1].set_title('Tiny elevation bins switch the 100% correction');axs[1].legend(fontsize=8,loc='lower left');fig.tight_layout();fig.savefig(R/'residual_source.png',dpi=160)
