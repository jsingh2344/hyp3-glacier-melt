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
import os

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.paths import MeltPaths
from hyp3_glacier_melt.datacube import sar_datacube
from hyp3_glacier_melt.rgi import selectglaciersrgitable
from hyp3_glacier_melt.utils import nan_percentile, DescStr, _zvalue_from_index
from hyp3_glacier_melt.product import MeltRunResult

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

def process_datacube_to_melt_extent(datacube_path, config, paths, verbose=False) -> MeltRunResult:
    # Process Datasets
    failed_glacnos: dict[int, str] = {}
    generated_csv_files: list[Path] = []
    ds_fn = datacube_path
    # Path/Row string used for filenames
    pathrow_str = str(ds_fn).split('.nc')[0].split(config.pol_str)[1][1:]

    with xr.open_dataset(ds_fn) as tmp_ds:
        nt = tmp_ds.sizes['time']
        ny = tmp_ds.sizes['y']
        nx = tmp_ds.sizes['x']
        image_itemsize = tmp_ds['images'].dtype.itemsize

    print("PIPELINE OPENED:", ds_fn)
    print("PIPELINE SHAPE:", ny, nx)

    if config.use_spatial_tiling:
        spatial_subsets = generate_spatial_subsets(
            ny,
            nx,
            tile_y=config.tile_y,
            tile_x=config.tile_x,
        )
    else:
        raw_images_gib = nt * ny * nx * image_itemsize / 1024**3
        log.warning(
            'Spatial tiling is disabled. Processing the full %sx%s cube at once; '
            'the raw images array alone is approximately %.2f GiB and working arrays require additional memory.',
            ny,
            nx,
            raw_images_gib,
        )
        spatial_subsets = [(slice(0, ny), slice(0, nx), 0, ny, 0, nx)]

    for subset_y, subset_x, y0, y1, x0, x1 in tqdm(spatial_subsets):

    #for subset_y, subset_x, y0, y1, x0, x1 in [(slice(500, 1000), slice(1500, 2000), -1, -1, -1, -1)]: #5589!

        print(f"\n=== ds_fn={ds_fn}, tile y[{y0}:{y1}], x[{x0}:{x1}] ===")
        
        # Initialize the datacube (this is slow because we're loading/opening large datafiles)
        dc = sar_datacube(ds_fn, 
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
                        subset_y=subset_y,
                        subset_x=subset_x,
                        rgi_cols_drop=config.rgi_cols_drop,
                        paths=paths)

        # Load glaciers to process
        try:
            main_glac_rgi = dc.glacnos_to_process()
        except Exception:
            log.exception(
                "glacnos_to_process failed for tile y[%s:%s], x[%s:%s]",
                y0,
                y1,
                x0,
                x1,
            )
            continue

        if main_glac_rgi.empty:
            log.info(
                "No suitable glaciers for tile y[%s:%s], x[%s:%s]; skipping",
                y0,
                y1,
                x0,
                x1,
            )
            continue
        
        #if verbose: 
        print("Glaciers in tile:", main_glac_rgi.rgino_str.values)
        # if '01.05589' not in main_glac_rgi.rgino_str.values:
        #     continue
    
        # Remove pixels from non-glaciated areas
        dc.mask_nonglacier_pixels(main_glac_rgi)
    
        # PIXEL-BY-PIXEL ANALYSIS
        dc.pixel_analysis()
        dc.annual_melt_onset_map()
        dc.annual_snowline_post_onset_map()
        dc.annual_snowline_onset_map() 
        dc.annual_second_onset_map()

        # SINGLE GLACIER ANALYSIS
        for nglac, glacno in enumerate(
            tqdm(
                main_glac_rgi.glacno.values,
                desc=f"Processing {len(main_glac_rgi)} glaciers in path_row {pathrow_str}",
            )
        ):
        #for nglac, glacno in enumerate([5589]): # or specify a particular glacier
            if verbose: 
                print(f"\nProcessing glacier {glacno}")
            
            # RGI_str for filenames
            nidx = list(main_glac_rgi.glacno.values).index(glacno)
            rgino_str = main_glac_rgi.loc[nidx,'rgino_str']
            area_km2 = main_glac_rgi.loc[nidx,'area_km2']
    
            try:
                # Process individual glacier

                dc.area_bin_size = config.area_bin_size
                dc.single_glacier_preprocess(glacno=glacno, area_km2=area_km2)

                #dc.plot_pixel_timeseries(glacno, 195, 75)
                csv_file = dc.generate_elevs_from_onsets(glacno, paths.csv_dir,
                                                doy_step=10,
                                                percentile=1.0,
                                                min_valid_frac=0.01,
                                                plot_year=2024
                                                )
                generated_csv_files.append(csv_file)
                #dc.plot_melt_onset_maps(glacno, out_dir=paths.onset_dir)
            except Exception as exc:
                log.exception("Processing failed for glacier %s", glacno)
                failed_glacnos[int(glacno)] = f"{type(exc).__name__}: {exc}"
        
    return MeltRunResult(
        scenes=nt,
        csv_files=generated_csv_files,
        failed_glacnos=failed_glacnos,
    )


def run_melt_pipeline(datacube_path, config, paths):

    #Paths is an object!
    os.makedirs(paths.output_root, exist_ok=True)
    os.makedirs(paths.csv_dir, exist_ok=True)
    os.makedirs(paths.onset_dir, exist_ok=True)

    return process_datacube_to_melt_extent(
        datacube_path=datacube_path,
        config=config,
        paths=paths,
    )
