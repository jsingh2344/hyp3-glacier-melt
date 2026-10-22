"""Compare packaged CLI outputs with all original CSV metric values."""
from pathlib import Path
from zipfile import ZipFile
import re,io,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
source=next(Path('doy-inputs').glob('*028627*.zip'))
def load(path):
 out={}
 with ZipFile(path) as z:
  for name in z.namelist():
   match=re.search(r'time_series_(01\.\d+)_all_years.csv$',name)
   if match:out[match[1]]=pd.read_csv(io.BytesIO(z.read(name)))
 return out
reference=load(source);rows=[]
for run in ['local_single','local_stitched','container_single','container_stitched','container_reversed','container_external']:
 products=list((R/run).glob('*.zip'))
 if not products:continue
 assert len(products)==1
 with ZipFile(products[0]) as archive:
  metadata=json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('product_metadata.json'))))
  assert metadata['scene_count']==226
  assert metadata['evaluation_calendar']=='source_acquisitions'
  assert metadata['failed_glacier_count']==0
 frames=load(products[0]);assert set(frames)==set(reference),(run,set(frames)^set(reference))
 for rid,a in reference.items():
  b=frames[rid];assert len(a)==len(b)==226
  merged=a.merge(b,on=['year','date'],suffixes=('_source','_test'),validate='one_to_one');assert len(merged)==226
  for metric in a.columns.drop(['year','date']):
   x=merged[metric+'_source'].to_numpy();y=merged[metric+'_test'].to_numpy();assert np.array_equal(x,y,equal_nan=True),(run,rid,metric,np.nanmax(abs(x-y)))
 rows.append(dict(run=run,glaciers=31,dates_per_glacier=226,rows=7006,metrics=8,metric_values=56048,exact=True));print(rows[-1])
(R/'verification.json').write_text(json.dumps(rows,indent=2))
