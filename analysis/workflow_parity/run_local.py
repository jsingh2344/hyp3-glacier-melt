"""Run packaged workflow locally with the repository's copy of the bundled RGI."""
from pathlib import Path
from functools import partial
from unittest.mock import patch
from hyp3_glacier_melt.doy_workflow.pipeline import run_doy_pipeline
from hyp3_glacier_melt.doy_workflow.process import process_doy_maps
R=Path(__file__).resolve().parent
local_pipeline=partial(run_doy_pipeline,rgi_shapefile='Glaciers/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp')
with patch('hyp3_glacier_melt.doy_workflow.process.run_doy_pipeline',local_pipeline):
 for name,inputs in [('local_single',[R/'inputs/single']),('local_stitched',[R/'inputs/left',R/'inputs/right'])]:
  output=process_doy_maps([str(p) for p in inputs],str(R/name));print(name,output,flush=True)
