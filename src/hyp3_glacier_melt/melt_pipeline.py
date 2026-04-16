import logging
from pathlib import Path

from tqdm import tqdm
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import rasterio
from datetime import date, timedelta
from scipy.ndimage import convolve, convolve1d
from rasterio.transform import Affine
from numpy.lib.stride_tricks import sliding_window_view
from netCDF4 import Dataset
import os

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.paths import MeltPaths
from hyp3_glacier_melt.datacube import sar_datacube
from hyp3_glacier_melt.rgi import selectglaciersrgitable
from hyp3_glacier_melt.utils import nan_percentile, DescStr, _zvalue_from_index

log = logging.getLogger(__name__)

def generate_spatial_subsets(ny, nx, tile_y=500, tile_x=500):
    """
    Yields (subset_y, subset_x, y0, y1, x0, x1) covering the spatial domain.
    """
    for y0 in range(0, ny, tile_y):
        y1 = min(y0 + tile_y, ny)
        for x0 in range(0, nx, tile_x):
            x1 = min(x0 + tile_x, nx)
            yield slice(y0, y1), slice(x0, x1), y0, y1, x0, x1

class DescStr:
    def __init__(self):
        self._desc = ''
    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)
    def read(self):
        ret = self._desc
        self._desc = ''
        return ret
    def flush(self):
        pass

def process_datacube_to_melt_extent(datacube_path, config, paths, verbose=False):
    failed_glacnos = []
    ds_fn = datacube_path
    pathrow_str = str(ds_fn).split('.nc')[0].split(config.pol_str)[1][1:]

    print("PIPELINE OPENED:", ds_fn)

    dc = sar_datacube(
        ds_fn,
        scene_name=config.scene_name,
        rgi_reg=config.rgi_reg,
        xres=config.xres,
        yres=config.yres,
        min_glac_area_km2=config.min_glac_area_km2,
        db_threshold=config.db_threshold,
        db_threshold_sl=config.db_threshold_sl,
        zscore_threshold=config.zscore_threshold,
        winter_months=config.winter_months,
        snowmelt_months=config.snowmelt_months,
        months2exclude_cp=config.months2exclude_cp,
        winter_std_threshold=config.winter_std_threshold,
        bin_size=config.bin_size,
        area_bin_size=config.area_bin_size,
        allmelt_threshold=config.allmelt_threshold,
        allmelt_pixels=config.allmelt_pixels,
        subset_y=None,
        subset_x=None,
        rgi_cols_drop=config.rgi_cols_drop,
        paths=paths
    )

    try:
        main_glac_rgi = dc.glacnos_to_process()
    except Exception as e:
        print(f"glacnos_to_process failed: {e}")
        return len(Dataset(datacube_path, mode="r").variables["time"]), failed_glacnos

    print("Glaciers in cube:", main_glac_rgi.rgino_str.values)

    dc.mask_nonglacier_pixels(main_glac_rgi)

    dc.pixel_analysis()
    dc.annual_melt_onset_map()
    dc.annual_snowline_post_onset_map()
    dc.annual_snowline_onset_map()
    dc.annual_second_onset_map()

    for nglac, glacno in enumerate(
        tqdm(
            main_glac_rgi.glacno.values,
            desc=f"Processing {len(main_glac_rgi)} glaciers in path_row {pathrow_str}",
        )
    ):
        if verbose:
            print(f"\nProcessing glacier {glacno}")

        nidx = list(main_glac_rgi.glacno.values).index(glacno)
        rgino_str = main_glac_rgi.loc[nidx, 'rgino_str']
        area_km2 = main_glac_rgi.loc[nidx, 'area_km2']

        dc.area_bin_size = config.area_bin_size
        dc.single_glacier_preprocess(glacno=glacno, area_km2=area_km2)

        dc.generate_elevs_from_onsets(
            glacno,
            paths.csv_dir,
            doy_step=10,
            percentile=1.0,
            min_valid_frac=0.01,
            plot_year=2024
        )

        failed_glacnos.append(glacno)

    return len(Dataset(datacube_path, mode="r").variables["time"]), failed_glacnos


def run_melt_pipeline(datacube_path, config, paths):

    #Paths is an object!
    os.makedirs(paths.output_root, exist_ok=True)
    os.makedirs(paths.csv_dir, exist_ok=True)
    os.makedirs(paths.onset_dir, exist_ok=True)

    scenes_no, failed_glacnos = process_datacube_to_melt_extent(datacube_path=datacube_path, config=config, paths=paths)

    output_file = Path(paths.output_root) / "datacube_summary.txt"
    output_file.write_text(
        f"scenes={scenes_no}\nfailed_glacnos={failed_glacnos}\n"
    )
    return output_file