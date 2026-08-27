from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MeltConfig:
    rgi_reg: int = 1
    min_glac_area_km2=2
    db_threshold = -3
    db_threshold_sl = 4
    zscore_threshold = -2 #same as prev
    winter_months = [1, 2]
    snowmelt_months = [4, 5, 6, 7] #same as prev
    months2exclude_cp = [10, 11, 12, 1, 2]
    winter_std_threshold = 3
    xres = 100.0
    yres = 100.0
    bin_size= 20
    area_bin_size = 'variable'
    allmelt_threshold = 0.9
    allmelt_pixels = 10
    nan_filter = -1e10
    min_area_frac = 0.9
    subset_y = (500, 1000)
    subset_x = (500, 1000)
    use_spatial_tiling = True #Setting to false could load many many GBs into RAM!
    tile_y = 500
    tile_x = 500
    rgi_cols_drop = ['glims_id', 'anlys_id', 'subm_id']
    pol_str = 'VH'
    scene_name = 'Kennicott-Cordoba'
    epsg_no = 32607
    path_frame_dict = {'14': ['387']}
    path_direction = 'Ascending'
    frame_cut = 0

    # Runtime mode
    build_datacube = False
    build_opera_datacube = True
    download_opera_files = True
    build_only = False
    dry_run: bool = False

    # OPERA acquisition settings
    opera_burst_id: str = "T014_028627_IW2"
    opera_start: str = "2017-01-01"
    opera_end: str = "2024-12-31"
    opera_download_processes: int = 1
    opera_overwrite_downloads: bool = False

    # OPERA cube build settings
    opera_resample_alg: str = "average"
    opera_write_db: bool = True
    opera_overwrite_cube: bool = True
