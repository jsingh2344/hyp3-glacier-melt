from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.paths import MeltPaths
from hyp3_glacier_melt.rgi import selectglaciersrgitable
from hyp3_glacier_melt.utils import nan_percentile, DescStr, _zvalue_from_index


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


class sar_datacube():
    """
    SAR Datacube for melt extent analyses
    
    Attributes
    ----------
    ds_fn : str
        filename of the datacube
    scene_name : str
        name of the scene for easier referencing and naming files
    """
    def __init__(self, 
                 ds_fn=str(),
                 scene_name=str(),
                 rgi_reg=1,
                 xres=None,
                 yres=None,
                 min_glac_area_km2=0,
                 db_threshold=-3,
                 db_threshold_sl=3,
                 zscore_threshold=-2,
                 winter_months=[1,2],
                 snowmelt_months=[4,5,6,7],
                 months2exclude_cp=[10, 11, 12, 1, 2],
                 winter_std_threshold=3, # maximum winter standard deviation [dB]
                 bin_size=20,
                 area_bin_size=100000,
                 allmelt_threshold=0.9,
                 allmelt_pixels=10,
                 nan_filter=-1e10, # value below which you can threshold for nan data
                 min_area_frac=0.9, # minimum fraction of the total area that has data to be included
                 subset_y = slice(500, 1000),
                 subset_x = slice(500, 1000),
                 rgi_cols_drop = None,
                 paths=None
                 ):
        """
        Add attributes
        """
        self.ds_fn = ds_fn
        self.scene_name = scene_name
        self.rgi_reg = rgi_reg
        self.paths = paths
        self.rgi_cols_drop = rgi_cols_drop

        # Load xarray dataset
        ds = xr.open_dataset(
            self.ds_fn,
            chunks={'x': 500, 'y': 500}   # all 226 timesteps kept in each chunk
        )
        self.ds = ds.isel(y=subset_y, x=subset_x)
        
        
        self.data = (ds.images.isel(y=subset_y, x=subset_x)).values
        if not nan_filter is None:
            self.data[self.data < nan_filter] = np.nan
        mask_good_pixels = np.sum(self.data, axis=0)
        mask_good_pixels[~np.isnan(mask_good_pixels)] = 1
        self.mask_good_pixels = mask_good_pixels
        self.data_good = self.data * self.mask_good_pixels[np.newaxis,:,:]
        
        self.dates = self.ds.time.values

        #self.mask_values = self.ds.rgi_ind_glacier_mask.values
        self.mask_values = (ds.rgi_ind_glacier_mask.isel(y=subset_y, x=subset_x)).values
        self.dem = self.ds.dem.values
        self.dem = (ds.dem.isel(y=subset_y, x=subset_x)).values
        self.xres = xres
        self.yres = yres

        # Single Glacier Dictionary initialization
        self.glac_bounds = {}
        self.glac_mask = {}
        self.glac_mask_good_pixels = {}
        self.glac_data = {}
        self.glac_data_cp = {}
        self.glac_data_sl_cp = {}
        self.min_dB_elevs = {}
        self.glac_dem = {}
        self.glac_bins = {}
        self.glac_bins_center = {}
        self.glac_area_bins = {}
        self.glac_area_bins_center = {}

        self.glac_melt_extent_elevs_percentiles = {}
        self.glac_melt_extent_elevs_percentile_mins = {}
        self.glac_melt_extent_elevs_percentile_maxs = {}
        self.glac_melt_extent_areas_percentiles = {}
        self.glac_melt_extent_areas_percentile_mins = {}
        self.glac_melt_extent_areas_percentile_maxs = {}

        self.glac_snowline_elevs_percentiles = {}
        self.glac_snowline_elevs_percentile_mins = {}
        self.glac_snowline_elevs_percentile_maxs = {}
        self.glac_snowline_areas_percentiles = {}
        self.glac_snowline_areas_percentile_mins = {}
        self.glac_snowline_areas_percentile_maxs = {}
        

        # Attributes
        self.min_glac_area_km2 = min_glac_area_km2
        self.db_threshold=db_threshold
        self.db_threshold_sl=db_threshold_sl
        self.zscore_threshold=zscore_threshold
        self.winter_months=winter_months
        self.snowmelt_months=snowmelt_months
        self.months2exclude_cp=months2exclude_cp
        self.winter_std_threshold=winter_std_threshold

        self.bin_size = bin_size
        self.area_bin_size = area_bin_size
        self.allmelt_threshold = allmelt_threshold,
        self.allmelt_pixels = allmelt_pixels
        self.min_area_frac = min_area_frac
    
               
    def glacnos_to_process(self):
        """
        Identify glacier numbers to process
        
        Parameters
        ----------
        None
        
        Returns
        -------
        main_glac_rgi_sar : pd.DataFrame
            dataframe of relevant RGI attributes and any added ones
        """
        glacnos = sorted(list(np.unique(self.mask_values)))[1:]
        glacnos_str = [str(self.rgi_reg) + '.' + str(x).zfill(5) for x in glacnos]
        
        assert len(glacnos_str) > 0, 'No glaciers to process'
        main_glac_rgi_raw = selectglaciersrgitable(rgi_fp=self.paths.rgi_root, rgi_cols_drop = self.rgi_cols_drop,
                                                   glac_no=glacnos_str, min_glac_area_km2=self.min_glac_area_km2)
        glacnos_raw = list(main_glac_rgi_raw.glacno.values)
        
        # glacnos = glacnos_raw
        
        # Remove glaciers that are on the edge (and thus cut off and incomplete coverage)
        self.mask_values = self.ds.rgi_ind_glacier_mask.values
        glacno_edges = (list(np.unique(self.mask_values[0,:])) + 
                        list(np.unique(self.mask_values[-1,:])) + 
                        list(np.unique(self.mask_values[:,0])) + 
                        list(np.unique(self.mask_values[:,-1])))
        glacno_edges = list(np.unique(np.array(glacno_edges)))
        glacno_edges.remove(0)
        glacnos = [x for x in glacnos_raw if x not in glacno_edges]
        glac_idxs = [glacnos_raw.index(x) for x in glacnos]
        main_glac_rgi = main_glac_rgi_raw.loc[glac_idxs]
        main_glac_rgi.reset_index(inplace=True, drop=True)
        
        glacnos_2process = []
        glacnos_dsfrac = []
        glacnos_sarfrac = []
        for nglac, glacno in enumerate(glacnos):
            area_km2 = main_glac_rgi.loc[nglac,'area_km2']
            area_ds = np.where(self.mask_values == glacno)[0].shape[0] * self.xres * self.yres / 1e6
            area_ds_frac = area_ds / area_km2
            
            area_sar = np.where(~np.isnan(self.data_good[0,:,:][np.where(self.mask_values == glacno)]))[0].shape[0] * self.xres * self.yres / 1e6
            area_sar_frac = area_sar / area_km2
            if area_ds_frac > self.min_area_frac and area_sar_frac > self.min_area_frac:
                glacnos_2process.append(glacno)
                glacnos_dsfrac.append(area_ds_frac)
                glacnos_sarfrac.append(area_sar_frac)
        
        assert len(glacnos_2process) > 0, 'No glaciers suitable for processing'
        glacnos_2process_str = [str(self.rgi_reg) + '.' + str(x).zfill(5) for x in glacnos_2process]
        main_glac_rgi_sar = selectglaciersrgitable(rgi_fp=self.paths.rgi_root, rgi_cols_drop=self.rgi_cols_drop, glac_no=glacnos_2process_str)
        main_glac_rgi_sar['ds_area_frac'] = glacnos_dsfrac
        # main_glac_rgi_sar['sar_area_frac'] = glacnos_sarfrac
        main_glac_rgi_sar['rgino_str'] = [str(main_glac_rgi_sar.loc[x,'o1region']).zfill(2) + '.' + 
                                          str(main_glac_rgi_sar.loc[x,'glacno']).zfill(5) for x in np.arange(main_glac_rgi_sar.shape[0])]
        
        return main_glac_rgi_sar
        

    def mask_nonglacier_pixels(self, main_glac_rgi):
        """
        Mask the non-glaciated pixels
        """
        self.mask_values = self.ds.rgi_ind_glacier_mask.values

        mask_values_minsize = np.zeros(self.mask_values.shape)
        for glacno in main_glac_rgi.glacno.values:
            mask_values_minsize[self.mask_values == glacno] = glacno
        mask_values_minsize_binary = np.zeros(self.mask_values.shape)
        mask_values_minsize_binary[mask_values_minsize>0] = 1
        
        data_masked = np.copy(self.data_good)
        data_masked[self.data_good < -1e10] = np.nan # filtering out very large negative values
        for nscene in np.arange(self.data_good.shape[0]):
            data_scene = data_masked[nscene,:,:]
            data_scene[mask_values_minsize==0] = np.nan
            data_masked[nscene,:,:] = data_scene
        
        self.data_masked = data_masked

    def pixel_analysis(self):

        self.min_dB_doys = {}
        self.min_dB_idx = {}
        dates_pd = pd.DatetimeIndex(self.dates)
        self.years = [x.year for x in dates_pd]
        self.months = [x.month for x in dates_pd]
        self.days = [x.day for x in dates_pd]
        self.doys = [int(x.to_julian_date() - pd.Timestamp(x.year,1,1).to_julian_date()) for x in dates_pd]
        
        winter_idx = [idx for idx, element in enumerate(self.months) if element in self.winter_months]
        
        data_winter_mean = np.nanmean(self.data_masked[winter_idx,:,:], axis=0)
        data_winter_std = np.nanstd(self.data_masked[winter_idx,:,:], axis=0)
        data_winter_res = self.data_masked - data_winter_mean[np.newaxis,:,:]
        self.data_zscore = data_winter_res / data_winter_std[np.newaxis,:,:]

        data_cp = np.zeros(self.data_masked.shape)
        data_cp[np.isnan(self.data_masked)] = np.nan
        data_cp[(data_winter_res < self.db_threshold) & (self.data_zscore < self.zscore_threshold)] = 1
        self.data_cp = data_cp

        # snowline change pixels
        data_sl_cp = np.zeros(self.data_masked.shape)
        data_sl_cp[np.isnan(self.data_masked)] = np.nan

        # snowline change pixels -- based on minimum backscatter from each year
        summer_idx = [idx for idx, element in enumerate(self.months) if element in self.snowmelt_months]
        for yr in set(self.years):
            year_idx = [idx for idx, y in enumerate(self.years) if y == yr]
            comb_idx = list(set(year_idx).intersection(summer_idx))
            if comb_idx: # ensure that we have any data for the year
                # data_summer_min_yr = np.nanpercentile(self.data_masked[comb_idx,:,:], 5, axis=0) # get dB for the 5% of melt pixels
                data_summer_min_yr = nan_percentile(self.data_masked[comb_idx,:,:], q=5, axis=0) # faster nanpercentile alternative
                data_summer_res_yr = self.data_masked[year_idx,:,:] - data_summer_min_yr[np.newaxis,:,:]

                # mask for the indices of the given year
                for i, idx in enumerate(year_idx):
                    mask = (data_winter_res[idx] > self.db_threshold) | (data_summer_res_yr[i] > self.db_threshold_sl)
                    data_sl_cp[idx][mask] = 1
            

                year_data = self.data_masked[year_idx, :, :]
                all_nan_mask = np.all(np.isnan(year_data), axis=0)
                safe_data = np.where(np.isnan(year_data), np.inf, year_data)
                min_idx = np.argmin(safe_data, axis=0).astype(float)
                min_idx[all_nan_mask] = np.nan

                doys_year = np.array(self.doys)[year_idx]
                valid = ~np.isnan(min_idx)

                min_doy = np.full(min_idx.shape, np.nan)
                min_doy[valid] = doys_year[min_idx[valid].astype(int)]

                self.min_dB_idx[yr] = min_idx   # keep as float with NaNs
                self.min_dB_doys[yr] = min_doy
                    
        # data_summer_min = np.nanpercentile(self.data_masked[summer_idx,:,:], 5, axis=0) # get dB for the 5% of melt pixels
        # data_summer_res = self.data_masked[summer_idx,:,:] - data_summer_min[np.newaxis,:,:]
        # data_sl_cp[(data_winter_res > self.db_threshold) | (data_summer_res > self.db_threshold_sl)] = 1
        self.data_sl_cp = data_sl_cp
            

    def annual_melt_onset_map(self):
        
        self.melt_onset_doy_maps = {}
        years_unique = np.unique(self.years)
        for nyear, year in enumerate(years_unique):


            # Subset dates for the given year
            year_idx = list(np.where(np.array(self.years) == year)[0])
            months_subset = [self.months[x] for x in year_idx]
            doys_subset = [self.doys[x] for x in year_idx]

        
            # Prevent melt/SL onset in winter months
            data_cp_year = self.data_cp[year_idx,:,:]
            
            for nmonth, month in enumerate(months_subset):
                if month in self.months2exclude_cp:
                    data_cp_year[nmonth,:,:] = 0

            # Get the first value of 1 (i.e., first change pixel)
            data_cp_year_onset_idx = (data_cp_year != 0).argmax(axis=0)
            data_cp_year_onset_idx = data_cp_year_onset_idx * self.mask_good_pixels

            # Only index months where there's a value of 1
            data_cp_year_sum = data_cp_year.sum(0)
            data_cp_year_sum[data_cp_year_sum>0] = 1
        
            # Remove pixels where there is no melt
            data_cp_year_sum[np.isnan(data_cp_year_sum)] = 0
            data_cp_year_onset_idx[data_cp_year_sum == 0] = np.nan
        
            # Plot the julian day of melt onset
            onset_idx_unique = np.unique(data_cp_year_onset_idx)
            data_cp_year_onset_doy = np.zeros(data_cp_year_onset_idx.shape)

            for onset_idx in onset_idx_unique:
                if not np.isnan(onset_idx):
                    onset_idx = int(onset_idx)
                    doy = doys_subset[onset_idx]
                    data_cp_year_onset_doy[data_cp_year_onset_idx == onset_idx] = doy            
        
            data_cp_year_onset_doy[data_cp_year_onset_doy==0] = np.nan
            self.melt_onset_doy_maps[year] = data_cp_year_onset_doy

    def generate_snowline_post_onset_mask(self):
        """
        Generate a mask of shape (Time, H, W) where each pixel is 1 if:
        - The current DOY is >= the melt onset DOY for that pixel (per year)
        - AND the summer residual (backscatter - summer_min) exceeds the 
            snowline threshold self.db_threshold_sl.

        Output stored as: self.data_sl_post_onset
        """

        # Allocate output mask
        T, H, W = self.data_masked.shape
        data_sl_post_onset = np.zeros((T, H, W), dtype=float)
        data_sl_post_onset[:] = 0   # default zero

        # Precompute residuals for entire cube ------------------------------
        summer_idx = [i for i, m in enumerate(self.months) if m in self.snowmelt_months]

        # For each year, compute summer-min and residuals
        years_unique = np.unique(self.years)

        for year in years_unique:

            # Indices for this year
            year_idx = np.where(np.array(self.years) == year)[0]
            months_subset = [self.months[i] for i in year_idx]

            # Get melt onset DOY map (2D)
            onset_map = self.melt_onset_doy_maps[year]   # shape (H, W)

            # Compute summer-min for this year (same logic as pixel_analysis)
            summer_idx_year = list(set(year_idx).intersection(summer_idx))
            if len(summer_idx_year) == 0:
                continue

            data_summer_min_yr = nan_percentile(
                self.data_masked[summer_idx_year, :, :],
                q=5, axis=0
            )

            # Compute summer residual for this year
            data_summer_res_yr = (
                self.data_masked[year_idx, :, :] - 
                data_summer_min_yr[np.newaxis, :, :]
            )  # shape: (#year_frames, H, W)


            # Loop through each timestep of the year -------------------------
            for k, t in enumerate(year_idx):
                
                month_t = self.months[t]
                # DOY of this timestep
                doy_t = self.doys[t]

                if month_t in self.months2exclude_cp:
                    # Set everything to 0
                    tmp = np.zeros((H, W), dtype=float)
                    tmp[self.mask_good_pixels == 1] = 0
                    data_sl_post_onset[t] = tmp
                    continue

                # Condition 1: DOY must be > local melt-onset DOY
                cond_doy = doy_t > onset_map

                # Condition 2: residual > SL threshold
                cond_res = data_summer_res_yr[k] > self.db_threshold_sl

                # Combine conditions
                mask = cond_doy & cond_res & (self.mask_good_pixels == 1)

                
                data_sl_post_onset[t][:] = 0
                data_sl_post_onset[t][mask] = 1

        # Save result
        self.data_sl_post_onset = data_sl_post_onset

    def annual_second_onset_map(self):

        self.annual_second_onset_map_doy_maps = {}
        years_unique = np.unique(self.years)

        for nyear, year in enumerate(years_unique):
            # Subset dates for the given year
            melt_onset_map = self.melt_onset_doy_maps[year] #to implement post onset logic
            sl_onset_map = self.snowline_post_onset_doy_maps[year] #to implement post onset logic
            year_idx = list(np.where(np.array(self.years) == year)[0])
            months_subset = [self.months[x] for x in year_idx]
            doys_subset = [self.doys[x] for x in year_idx]

            # Prevent melt/SL onset in winter months
            data_cp_year = self.data_cp[year_idx,:,:]
            
            for nmonth, month in enumerate(months_subset):
                if month in self.months2exclude_cp:
                    data_cp_year[nmonth,:,:] = 0

            post_melt_cp_mask = np.zeros_like(data_cp_year, dtype=bool)
            for i, doy in enumerate(doys_subset):
                # For timestep i, mark pixels as True only if:
                # 1. cp is non-zero AND
                # 2. current DOY > that pixel's melt onset DOY and the DOY > the snowline (so the pixel dipped, and then rose again)
                post_melt_cp_mask[i, :, :] = (data_cp_year[i, :, :] != 0) & (doy > melt_onset_map) & (doy > sl_onset_map)

            # Now get the first True value in this combined mask
            data_cp_year_onset_idx = post_melt_cp_mask.argmax(axis=0)
            data_cp_year_onset_idx = data_cp_year_onset_idx * self.mask_good_pixels

            # Only keep pixels where there was at least one True (same as normal melt logic from here)
            data_cp_year_sum = post_melt_cp_mask.sum(0)
            data_cp_year_sum[data_cp_year_sum > 0] = 1
            data_cp_year_sum[np.isnan(data_cp_year_sum)] = 0
            data_cp_year_onset_idx[data_cp_year_sum == 0] = np.nan
        
            # Plot the julian day of ice onset
            onset_idx_unique = np.unique(data_cp_year_onset_idx)
            data_cp_year_onset_doy = np.zeros(data_cp_year_onset_idx.shape)
            for onset_idx in onset_idx_unique:
                if not np.isnan(onset_idx):
                    onset_idx = int(onset_idx)
                    doy = doys_subset[onset_idx]
                    # convert onset_idx to doy 
                    data_cp_year_onset_doy[data_cp_year_onset_idx == onset_idx] = doy            
        
            data_cp_year_onset_doy[data_cp_year_onset_doy==0] = np.nan

            self.annual_second_onset_map_doy_maps[year] = data_cp_year_onset_doy

    def annual_snowline_post_onset_map(self):
        """
        A snowline onset map without post onset map, derived directly from the data_sl_cp masks generated in
        pixel_analysis()
        """
        
        self.snowline_post_onset_doy_maps = {}
        
        years_unique = np.unique(self.years)
        for nyear, year in enumerate(years_unique):
            # Subset dates for the given year
            melt_onset_map = self.melt_onset_doy_maps[year] #to implement post onset logic
            min_dB_map = self.min_dB_doys[year]
            year_idx = list(np.where(np.array(self.years) == year)[0])
            months_subset = [self.months[x] for x in year_idx]
            doys_subset = [self.doys[x] for x in year_idx]
        
            # Prevent melt/SL onset in winter months
            data_sl_cp_year = self.data_sl_cp[year_idx,:,:]
            
            for nmonth, month in enumerate(months_subset):
                if month in self.months2exclude_cp:
                    data_sl_cp_year[nmonth,:,:] = 0

            # Create a combined boolean mask:
            # True only if (sl_cp == 1) AND (current_doy > melt_onset_doy)
            post_melt_sl_mask = np.zeros_like(data_sl_cp_year, dtype=bool)
            for i, doy in enumerate(doys_subset):
                # For timestep i, mark pixels as True only if:
                # 1. sl_cp is non-zero AND
                # 2. current DOY > that pixel's melt onset DOY
                post_melt_sl_mask[i, :, :] = (data_sl_cp_year[i, :, :] != 0) & (doy > melt_onset_map) & (doy > min_dB_map) # adding the condition that it also has to be past the minimum dB day, to avoid false positives where the snowline is detected before the melt onset (which can happen when the minimum dB is in winter and then rises above the threshold in spring, without a clear melt onset signal)

            # Now get the first True value in this combined mask
            data_sl_cp_year_onset_idx = post_melt_sl_mask.argmax(axis=0)
            data_sl_cp_year_onset_idx = data_sl_cp_year_onset_idx * self.mask_good_pixels

            # Only keep pixels where there was at least one True (same as normal melt logic from here)
            data_sl_cp_year_sum = post_melt_sl_mask.sum(0)
            data_sl_cp_year_sum[data_sl_cp_year_sum > 0] = 1
            data_sl_cp_year_sum[np.isnan(data_sl_cp_year_sum)] = 0
            data_sl_cp_year_onset_idx[data_sl_cp_year_sum == 0] = np.nan
        
            # Plot the julian day of ice onset
            onset_idx_unique = np.unique(data_sl_cp_year_onset_idx)
            data_sl_cp_year_onset_doy = np.zeros(data_sl_cp_year_onset_idx.shape)
            for onset_idx in onset_idx_unique:
                if not np.isnan(onset_idx):
                    onset_idx = int(onset_idx)
                    doy = doys_subset[onset_idx]
                    # convert onset_idx to doy 
                    data_sl_cp_year_onset_doy[data_sl_cp_year_onset_idx == onset_idx] = doy            
        
            data_sl_cp_year_onset_doy[data_sl_cp_year_onset_doy==0] = np.nan

            self.snowline_post_onset_doy_maps[year] = data_sl_cp_year_onset_doy

    def annual_snowline_onset_map(self):
        """
        A snowline onset map without post onset map, derived directly from the data_sl_cp masks generated in
        pixel_analysis()
        """
        
        self.snowline_onset_doy_maps = {}
        
        years_unique = np.unique(self.years)
        for nyear, year in enumerate(years_unique):
            # Subset dates for the given year
            year_idx = list(np.where(np.array(self.years) == year)[0])
            months_subset = [self.months[x] for x in year_idx]
            doys_subset = [self.doys[x] for x in year_idx]
        
            # Prevent melt/SL onset in winter months
            data_sl_cp_year = self.data_sl_cp[year_idx,:,:]
            
            for nmonth, month in enumerate(months_subset):
                if month in self.months2exclude_cp:
                    data_sl_cp_year[nmonth,:,:] = 0

            # Create a combined boolean mask:
            # True only if (sl_cp == 1) 
            post_melt_sl_mask = np.zeros_like(data_sl_cp_year, dtype=bool)
            for i, doy in enumerate(doys_subset):
                # For timestep i, mark pixels as True only if:
                # 1. sl_cp is non-zero AND
                # 2. current DOY > that pixel's melt onset DOY
                post_melt_sl_mask[i, :, :] = (data_sl_cp_year[i, :, :] != 0) #no check for melt

            # Now get the first True value in this combined mask
            data_sl_cp_year_onset_idx = post_melt_sl_mask.argmax(axis=0)
            data_sl_cp_year_onset_idx = data_sl_cp_year_onset_idx * self.mask_good_pixels

            # Only keep pixels where there was at least one True (same as normal melt logic from here)
            data_sl_cp_year_sum = post_melt_sl_mask.sum(0)
            data_sl_cp_year_sum[data_sl_cp_year_sum > 0] = 1
            data_sl_cp_year_sum[np.isnan(data_sl_cp_year_sum)] = 0
            data_sl_cp_year_onset_idx[data_sl_cp_year_sum == 0] = np.nan
        
            # Plot the julian day of ice onset
            onset_idx_unique = np.unique(data_sl_cp_year_onset_idx)
            data_sl_cp_year_onset_doy = np.zeros(data_sl_cp_year_onset_idx.shape)
            for onset_idx in onset_idx_unique:
                if not np.isnan(onset_idx):
                    onset_idx = int(onset_idx)
                    doy = doys_subset[onset_idx]
                    # convert onset_idx to doy 
                    data_sl_cp_year_onset_doy[data_sl_cp_year_onset_idx == onset_idx] = doy            
        
            data_sl_cp_year_onset_doy[data_sl_cp_year_onset_doy==0] = np.nan

            self.snowline_onset_doy_maps[year] = data_sl_cp_year_onset_doy

    def annual_snowline_second_onset_map(self):
        
        self.snowline_second_onset_doy_maps = {}
        
        years_unique = np.unique(self.years)
        for nyear, year in enumerate(years_unique):
            # Subset dates for the given year
            second_melt_onset_map = self.annual_second_onset_map_doy_maps[year] #to implement post onset logic
            
            year_idx = list(np.where(np.array(self.years) == year)[0])
            months_subset = [self.months[x] for x in year_idx]
            doys_subset = [self.doys[x] for x in year_idx]
        
            # Prevent melt/SL onset in winter months
            data_sl_cp_year = self.data_sl_cp[year_idx,:,:]
            
            for nmonth, month in enumerate(months_subset):
                if month in self.months2exclude_cp:
                    data_sl_cp_year[nmonth,:,:] = 0

            # Create a combined boolean mask:
            # True only if (sl_cp == 1) AND (current_doy > melt_onset_doy)
            post_melt_sl_mask = np.zeros_like(data_sl_cp_year, dtype=bool)
            for i, doy in enumerate(doys_subset):
                # For timestep i, mark pixels as True only if:
                # 1. sl_cp is non-zero AND
                # 2. current DOY > that pixel's melt onset DOY
                post_melt_sl_mask[i, :, :] = (data_sl_cp_year[i, :, :] != 0) & (doy > second_melt_onset_map)

            # Now get the first True value in this combined mask
            data_sl_cp_year_onset_idx = post_melt_sl_mask.argmax(axis=0)
            data_sl_cp_year_onset_idx = data_sl_cp_year_onset_idx * self.mask_good_pixels

            # Only keep pixels where there was at least one True (same as normal melt logic from here)
            data_sl_cp_year_sum = post_melt_sl_mask.sum(0)
            data_sl_cp_year_sum[data_sl_cp_year_sum > 0] = 1
            data_sl_cp_year_sum[np.isnan(data_sl_cp_year_sum)] = 0
            data_sl_cp_year_onset_idx[data_sl_cp_year_sum == 0] = np.nan
        
            # Plot the julian day of ice onset
            onset_idx_unique = np.unique(data_sl_cp_year_onset_idx)
            data_sl_cp_year_onset_doy = np.zeros(data_sl_cp_year_onset_idx.shape)
            for onset_idx in onset_idx_unique:
                if not np.isnan(onset_idx):
                    onset_idx = int(onset_idx)
                    doy = doys_subset[onset_idx]
                    # convert onset_idx to doy 
                    data_sl_cp_year_onset_doy[data_sl_cp_year_onset_idx == onset_idx] = doy            
        
            data_sl_cp_year_onset_doy[data_sl_cp_year_onset_doy==0] = np.nan

            self.snowline_second_onset_doy_maps[year] = data_sl_cp_year_onset_doy
    


    def save_melt_onset_tiffs(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        # get geoinfo from the xarray dataset
        gt = self.ds.attrs.get('geotransform', None)
        crs = self.ds.attrs.get('projection', None)

        if gt is None:
            raise ValueError("Dataset is missing 'geotransform' in attrs.")
        if crs is None:
            raise ValueError("Dataset is missing 'projection' in attrs.")

        # turn GDAL-style geotransform into rasterio Affine
        # geotransform = [x_min, x_res, 0, y_max, 0, y_res_neg]
        transform = Affine.from_gdal(*gt)

        for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            onset_arr = self.melt_onset_doy_maps[year]
            # make sure it's float32 so NaNs work
            arr = onset_arr.astype("float32")

            height, width = arr.shape

            out_fn = os.path.join(out_dir, f"melt_onset_{year}.tif")

            with rasterio.open(
                out_fn,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(arr, 1)


    def generate_elevs_from_onsets(self, glacno, out_dir, 
                                                doy_step=10,
                                                percentile=1.0,
                                                min_valid_frac=0.01,
                                                plot_year=2024,
                                                plot_dir=None):
        
        #Setting up directories, glacier mask
        plot_dir = os.path.join(os.getcwd(), 'testing')
        mask_glac = self.glac_mask[glacno] 
        total_pixels = np.nansum(mask_glac)
        glac_dem = self.glac_dem[glacno]
        min_dB_elevs_glac = self.min_dB_elevs[glacno]
        

        dem_values_sorted = np.sort(self.glac_dem[glacno].reshape(1,-1))[0,:]
        dem_values_sorted = dem_values_sorted[dem_values_sorted > -9999]

        yearly_csv_paths = []

        for year in self.melt_onset_doy_maps.keys():

            melt_onset_map = self.melt_onset_doy_maps[year]
            original_snowline_onset_map = self.snowline_onset_doy_maps[year]
            snowline_onset_map = self.snowline_post_onset_doy_maps[year]
            second_melt_onset_map = self.annual_second_onset_map_doy_maps[year]
            second_snow_onset_map = self.snowline_second_onset_doy_maps[year]
            min_dB_map = self.min_dB_doys[year]

            # Mask valid glacier pixels
            h, w = mask_glac.shape
            #print("h, w:", h, w)

            xb0 = self.glac_bounds[glacno]['xmin']
            yb0 = self.glac_bounds[glacno]['ymin']

            # Slice onset_map using height/width from mask
            melt_onset_glac = melt_onset_map[xb0 : xb0 + h,      # h rows
                                             yb0 : yb0 + w]       # w cols
            
            snowline_onset_glac = snowline_onset_map[xb0 : xb0 + h,      # h rows
                                                     yb0 : yb0 + w]       # w cols

            second_melt_onset_glac = second_melt_onset_map[xb0 : xb0 + h,      # h rows
                                                     yb0 : yb0 + w]       # w cols
            second_snow_onset_glac = second_snow_onset_map[xb0 : xb0 + h,      # h rows
                                                     yb0 : yb0 + w]       # w cols
            original_snowline_onset_map = original_snowline_onset_map[xb0 : xb0 + h,      # h rows
                                                     yb0 : yb0 + w]       # w cols
            
            min_dB_map_glac = min_dB_map[xb0 : xb0 + h,      # h rows
                                         yb0 : yb0 + w]       # w cols
         
            
            # Generate onset maps clipped to glacier bounds
            melt_onset_map = np.where(mask_glac == 1, melt_onset_glac, np.nan) #primary mask for extent math
            original_snowline_onset_map = np.where(mask_glac == 1, original_snowline_onset_map, np.nan)
            snowline_onset_map = np.where(mask_glac == 1, snowline_onset_glac, np.nan)
            second_melt_onset_map = np.where(mask_glac == 1, second_melt_onset_glac, np.nan)
            second_snow_onset_map = np.where(mask_glac == 1, second_snow_onset_glac, np.nan)
            min_dB_map = np.where(mask_glac == 1, min_dB_map_glac, np.nan)
            

            #pure onset mask
            valid_mask = (~np.isnan(melt_onset_map))
            melt_onset_flat = melt_onset_map[valid_mask]
            snowline_onset_flat = snowline_onset_map[valid_mask] #these are 1d due to numpy indexing behavior!


            if (melt_onset_flat.size / total_pixels < min_valid_frac or
               snowline_onset_flat.size / total_pixels < min_valid_frac):
                
                print(f"  Skipping {year}: insufficient valid pixels.")
                continue

            year_idx = list(np.where(np.array(self.years) == year)[0])
            doys = [self.doys[x] for x in year_idx]
            results = []

            # Derive pixel area from DEM 
            pixel_area_m2 = self.xres * self.yres
            total_glacier_area_m2 = np.sum(mask_glac == 1) * pixel_area_m2
            
            for i, doy in enumerate(doys):
                nscene = year_idx[i]
                min_dB_elevs_glac_single = min_dB_elevs_glac[nscene]
                # Convert DOY → datetime
                glac_times = np.array(self.dates).astype("datetime64[D]")
                date = glac_times[nscene]
                # build 2D melted mask for this DOY
                melted_2d = (melt_onset_map <= doy)
                melted_2d_mask = melted_2d.copy()

                original_snowline_2d = (original_snowline_onset_map <= doy)
                snowline_2d = (snowline_onset_map <= doy) 
                second_melt_2d = (second_melt_onset_map <= doy)
                second_snow_2d = (second_snow_onset_map <= doy)
                min_dB_2d = (min_dB_map <= doy)

                # now flatten using valid_mask
                melt_elevs = np.sort((self.glac_dem[glacno])[valid_mask & melted_2d])

                ## POST - PROCESSING ##

                #apply snowline correction to remove lower pixels for allmelt logic
                allmelt_mask = melted_2d & (~snowline_2d) 
                allmelt_mask = allmelt_mask | second_melt_2d
                
                # ---------- ---------- ALL MELT CORRECTION: repeat for equal area elevation bin ---------- ----------
                allmelt_elev = 0
                allmelt_100 = False

                for nbin, bin_elev_lower in enumerate(self.glac_bins[glacno][:-1]):
                    bin_elev_upper = self.glac_bins[glacno][nbin+1]
                    if not allmelt_100:
                        # Create mask based on elevations
                        mask_bin = np.zeros(self.glac_dem[glacno].shape)
                        mask_bin[(self.glac_dem[glacno] > bin_elev_lower) & (self.glac_dem[glacno] <= bin_elev_upper)] = 1
                        bin_count = mask_bin.sum()
                
                        data_cp_bin = allmelt_mask * mask_bin  # changed from data_cp_single to melted
                        data_cp_bin_count = np.nansum(data_cp_bin)
                
                        frac_melt = data_cp_bin_count / bin_count
                
                        # Record "all melt" elevation 
                        if allmelt_elev == 0 and frac_melt > self.allmelt_threshold and bin_count >  self.allmelt_pixels:
                            allmelt_elev = bin_elev_lower
                                
                        # Record "all melt" elevation in the case that 100% hasn't been found yet
                        if not allmelt_100 and frac_melt == 1:
                            allmelt_elev = bin_elev_lower
                            allmelt_100 = True

                # Apply "all melt" correction
                snow_allmelt_elev = allmelt_elev
                if allmelt_elev > 0:
                    melted_2d[glac_dem < allmelt_elev] = 1

                if melt_elevs.size == 0:
                    melt_extent_elev = np.nan
                    n_melt = 0
                else:
                    n_melt = np.sum(melted_2d) - 1  
                    melt_extent_elev = dem_values_sorted[n_melt]

                #print("Onset:", year, nscene, n_melt, allmelt_elev, melt_extent_elev)

                ### Add nomelt pixels to help snowline post-processing
                nomelt_pixels_below = np.nansum((glac_dem < melt_extent_elev) & (melted_2d == 0))
                melt_extent_elev_min = dem_values_sorted[n_melt - nomelt_pixels_below]
                
                first_melt = original_snowline_2d & (~melted_2d_mask) #use mask to avoid adjusting w allmelt included melt pixels
                final_snowline = first_melt | snowline_2d
                #final_snowline = final_snowline & min_dB_2d #add min dB constraint to snowline consideration
                #final_snowline = snowline_2d.copy()
                # prev_pixels = np.nansum(snowline_2d)
                # snowline_2d = snowline_2d & (~second_melt_2d) #remove second melt pixels from snowline consideration
                # pixels_after_removing_second_melt = np.nansum(snowline_2d)
                # snowline_2d = snowline_2d | second_snow_2d #add second snowline pixels to snowline consideration
                # pixels_after_adding_second_snow = np.nansum(snowline_2d)

                # print(f'''Onset method at year {year}, scene {nscene}, date {date},
                #       snowline after adding snow pixels 1st time: {np.nansum(final_snowline)},
                #       after elevation adjustment: {pixels_after_elev_adjustment},
                #       ''')
                # print(f'''Onset method at year {year}, scene {nscene}, date {date}: 
                #       After subtracting melt: {first_melt_pixels}
                #       Adding original snow onset: {prev_pixels}, 
                # after removing 2nd melt: {pixels_after_removing_second_melt}, 
                # after adding 2nd snow: {pixels_after_adding_second_snow},
                # allmelt elevation (currently out of use): {allmelt_elev}
                # after allmelt correction: {pixels_after_allmelt},
                # after elev adjustment: {pixels_after_elev_adjustment}''')

                
                ### allmelt area binning:

                # ---------- ---------- ALL MELT CORRECTION: repeat for equal area elevation bin ---------- ----------
                allmelt_elev = 0
                allmelt_100 = False
                for nbin, bin_elev_lower in enumerate(self.glac_area_bins[glacno][:-1]):
                    bin_elev_upper = self.glac_area_bins[glacno][nbin+1]
                    if not allmelt_100:
                        # Create mask based on elevations
                        mask_bin = np.zeros(self.glac_dem[glacno].shape)
                        mask_bin[(self.glac_dem[glacno] > bin_elev_lower) & (self.glac_dem[glacno] <= bin_elev_upper)] = 1
                        bin_count = mask_bin.sum()
                
                        data_cp_bin = melted_2d * mask_bin
                        data_cp_bin_count = np.nansum(data_cp_bin)
                
                        frac_melt = data_cp_bin_count / bin_count
                
                        # Record "all melt" elevation 
                        if allmelt_elev == 0 and frac_melt > self.allmelt_threshold and bin_count > self.allmelt_pixels:
                            allmelt_elev = bin_elev_lower
                        # Record "all melt" elevation in the case that 100% hasn't been found yet
                        if not allmelt_100 and frac_melt == 1:
                            allmelt_elev = bin_elev_lower
                            allmelt_100 = True

                # Apply "all melt" correction
                if allmelt_elev > 0:
                    melted_2d[self.glac_dem[glacno] < allmelt_elev] = 1

                # ----- PERCENTILE METHOD -----
                melt_pixels = np.nansum(melted_2d)
        
                # The index is associated with one less than the sum of the pixels to account for indexing starting with 0 not 1
                if melt_pixels == 0:
                    melt_idx = int(0)
                else:
                    melt_idx = int(melt_pixels - 1)               
                melt_area_m2 = melt_idx*self.xres*self.yres


                ### NEW SNOWLINE METHOD: allmelt-based

                # allmelt_elev = 0
                # allmelt_100 = False
                # rev_allmelt_threshold = 0.9
                # rev_allmelt_pixels = 10

                # for nbin, bin_elev_lower in reversed(list(enumerate(self.glac_bins[glacno][:-1]))):
                    
                #     bin_elev_upper = self.glac_bins[glacno][nbin+1]
                #     if not allmelt_100:
                #         # Create mask based on elevations
                #         mask_bin = np.zeros(self.glac_dem[glacno].shape)
                #         mask_bin[(self.glac_dem[glacno] > bin_elev_lower) & (self.glac_dem[glacno] <= bin_elev_upper)] = 1
                #         bin_count = mask_bin.sum()
                
                #         data_cp_bin = allmelt_mask * mask_bin  # changed from data_cp_single to melted
                #         data_cp_bin_count = np.nansum(data_cp_bin)
                
                #         frac_melt = data_cp_bin_count / bin_count
                
                #         # Record "all melt" elevation 
                #         if allmelt_elev == 0 and frac_melt > rev_allmelt_threshold and bin_count > rev_allmelt_pixels:
                #             allmelt_elev = bin_elev_lower
                                
                #         # Record "all melt" elevation in the case that 100% hasn't been found yet
                #         if not allmelt_100 and frac_melt == 1:
                #             allmelt_elev = bin_elev_lower
                #             allmelt_100 = True


                # allmelt_elev = 0
                # allmelt_100 = False
                # snow_allmelt_threshold = 0.9
                # snow_allmelt_pixels = 10

                # for nbin, bin_elev_lower in reversed(list(enumerate(self.glac_bins[glacno][:-1]))):

                #     bin_elev_upper = self.glac_bins[glacno][nbin+1]
                #     if not allmelt_100:
                #         # Create mask based on elevations
                #         mask_bin = np.zeros(self.glac_dem[glacno].shape)
                #         mask_bin[(self.glac_dem[glacno] > bin_elev_lower) & (self.glac_dem[glacno] <= bin_elev_upper)] = 1
                #         bin_count = mask_bin.sum()
                
                #         data_cp_bin = final_snowline * mask_bin  # changed from data_cp_single to melted
                #         data_cp_bin_count = np.nansum(data_cp_bin)
                
                #         frac_melt = data_cp_bin_count / bin_count
                #         # if year == 2019 and nscene in [58, 59, 60, 61, 62, 63]:
                #         #     print(f"  Bin {nbin}: elev {bin_elev_lower}-{bin_elev_upper}, count: {bin_count}, melt count: {data_cp_bin_count}, frac melt: {frac_melt}")
                
                #         # Record "all melt" elevation 
                #         if allmelt_elev == 0 and frac_melt > snow_allmelt_threshold and bin_count > snow_allmelt_pixels and bin_elev_lower < melt_extent_elev:
                #             allmelt_elev = bin_elev_lower
                                
                #         # Record "all melt" elevation in the case that 100% hasn't been found yet
                #         if not allmelt_100 and frac_melt == 1 and bin_elev_lower < melt_extent_elev:
                #             allmelt_elev = bin_elev_lower
                #             allmelt_100 = True
                
                #max_snowline = allmelt_elev #to use new allmelt
                #using allmelt_mask, which currently has second melt pixels included:
                #snow_pixels = (~allmelt_mask) & valid_mask

                #pre_elev_snow_pixels = np.nansum(final_snowline)
                
                #derive similar elev cap:
                
                max_snowline = melt_extent_elev_min
                #if snow_allmelt_elev > 0:

                final_snowline_for_mask = final_snowline.copy()
                final_snowline = final_snowline & (glac_dem < max_snowline) #so, using pixels after first melt + first ice + elev filter
                #note this is with ground-up allmelt! aka the same used for the melt extent

                n_melt = np.nansum(final_snowline) 
                if n_melt == 0:
                    snowline_elev = np.nan
                    ice_area_m2 = 0.0
                else:
                    snowline_elev = dem_values_sorted[n_melt - 1]
                    ice_area_m2 = n_melt * pixel_area_m2

                # print(f'''year: {year}, scene: {nscene}, allmelt elevation: {snow_allmelt_elev}, 
                #       pre_adj_pixels: {pre_elev_snow_pixels}, n_melt : {n_melt}, snowline elev: {snowline_elev}''')

                # print(f'''Onset method at year {year}, scene {nscene}, date {date}, 
                #       allmelt elevation (reverse): {allmelt_elev}, pre-elev adj snow pixels: {pre_elev_snow_pixels},
                #       post-elev adj snow pixels: {n_melt},
                #       melt elevation: {melt_extent_elev}, snowline elevation: {snowline_elev},
                #       original_snow_pixel_count: {np.nansum(original_snowline_2d)},'''
                #       )


                results.append({
                    "year": year,
                    "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                    "melt_extent_elev_m": melt_extent_elev,
                    "snowline_elev_m" : snowline_elev,
                    "ice_area_m2" : ice_area_m2,
                    "melt_area_m2": melt_area_m2,
                    "melt_fraction": melt_area_m2 / total_glacier_area_m2,  # normalized fraction of glacier melted
                    "ice_fraction" : ice_area_m2 / total_glacier_area_m2,
                    "glacier_area_m2" : total_glacier_area_m2,
                })

                # --- Plot melt mask for specified year ---
                plot_dir = None #not plotting rn
                #if plot_dir and year == plot_year:
                if glacno == 5999 and plot_dir and year == 2019:

                    # Extract the SAR dB slice
                    glac_cube = self.data_masked
                    sar_dB = glac_cube[nscene, :, :]
                    sar_dB_glac = sar_dB[xb0 : xb0 + h,      # h rows
                                        yb0 : yb0 + w]       # w cols

                    # ---------- NEW: cp mask at nearest scene ----------
                    # data_cp_glac has shape (time, y, x) for this glacier
                    data_cp_glac = self.glac_data_cp[glacno]
                    cp_scene = data_cp_glac[nscene, :, :]  

                    data_sl_cp_glac = self.glac_data_sl_cp[glacno]
                    data_sl_cp_glac_single = data_sl_cp_glac[nscene, :, :]  # This is already glacier-sized
                    sl_cp_overlay = np.where(data_sl_cp_glac_single == 1, 1.0, np.nan)

                    #print(nscene, np.sum(snowline_2d), np.sum(data_sl_cp_glac_single == 1))

                    # build an overlay: 1 where cp==1, NaN elsewhere
                    cp_overlay = np.where(cp_scene == 1, 1.0, np.nan)
                    melt_overlay = np.where(melted_2d_mask == 1, 1.0, np.nan)
                    ice_overlay = np.where(snowline_2d == 1, 1.0, np.nan)
                    allmelt_overlay = np.where(allmelt_mask == 1, 1.0, np.nan)

                    # ---------- NEW: Calculate difference between ice_mask_full and sl_cp_overlay ----------
                    # Create binary versions for comparison (1 where valid, 0 elsewhere)
                    sl_cp_binary = np.where(sl_cp_overlay == 1, 1, 0)

                    final_snowline_overlay = np.where(final_snowline_for_mask == 1, 1.0, np.nan)

                    ice_binary = np.where(final_snowline_overlay == 1, 1, 0)

                    # Calculate difference: positive where ice but not sl_cp, negative where sl_cp but not ice
                    snow_difference_mask = ice_binary - sl_cp_binary
                    # Convert 0s to NaN for better visualization
                    snow_difference_overlay = np.where(snow_difference_mask != 0, snow_difference_mask, np.nan)

                    second_melt_overlay = np.where(second_melt_2d, 1.0, np.nan)


                    # Calculated difference between first and second melt
                    first_melt_binary = np.where(melt_overlay == 1, 1, 0)
                    second_melt_binary = np.where(second_melt_overlay == 1, 1, 0)

                    melt_difference_mask = first_melt_binary - second_melt_binary
                    melt_difference_overlay = np.where(melt_difference_mask != 0, melt_difference_mask, np.nan)

                    # DEM for this glacier
                    dem_glac = self.glac_dem[glacno]

                    grey = np.where(np.isnan(dem_glac), np.nan, 0.5)

                    sar_dB_masked = np.where(np.isnan(dem_glac), np.nan, sar_dB_glac)

                    melt_elev_mask = np.abs(glac_dem - melt_extent_elev) <= 50
                    melt_elev_mask_overlay = np.where((melt_elev_mask & allmelt_mask), 1.0, np.nan)

                    max_snow_elev_mask = np.abs(glac_dem - max_snowline) <= 50
                    max_snow_elev_mask_overlay = np.where((max_snow_elev_mask & final_snowline_for_mask), 1.0, np.nan)
                    #print("nscene:", nscene, "snow_allmelt_elev:", snow_allmelt_elev, "elev_mask pixels:", np.nansum(elev_mask))

                    plot_coords = (195, 75) #(x, y) 

                    # Plot
                    backscatter_cmap = plt.cm.coolwarm_r  # reversed so low values = red, high = blue
                                            
                    fig, axes = plt.subplots(1, 5, figsize=(24, 6), constrained_layout=True)

                    # ---------- LEFT: Backscatter + onset overlays ----------
                    ax = axes[3]
                    im2 = ax.imshow(grey, cmap="gray", vmin=0, vmax=1)  
                    ax.imshow(
                        allmelt_overlay,
                        cmap = mcolors.ListedColormap(["red"]),
                        alpha = 1.0
                    )
                    ax.imshow(
                        melt_elev_mask_overlay,
                        cmap=mcolors.ListedColormap(["black"]),  
                        alpha=0.8
                    )
                    ax.set_title(
                        f"{year} DOY {doy}\n({date}) — Melted pixels, max snow elev: {max_snowline}",
                        fontsize=18
                    )
                    ax.axis("off")

                    ax = axes[0]
                    im0 = ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0)
                    ax.set_title(
                        f"Backscatter (dB)",
                        fontsize=18
                    )
                    ax.axis("off")
                    # ax.plot(plot_coords[0], plot_coords[1], marker='*', color='lime', markersize=12, 
                    #     markeredgecolor='black', markeredgewidth=0.5)


                    ax = axes[1]
                    im1 = ax.imshow(grey, cmap="gray", vmin=0, vmax=1)  
                    # overlay cp mask (e.g. magenta where cp==1)
                    ax.imshow(
                        sl_cp_overlay,
                        cmap=mcolors.ListedColormap(["blue"]),
                        alpha=0.6
                    )
                    ax.set_title(f"sl_cp mask, scene {nscene}", fontsize=18)
                    ax.axis("off")

                    ax = axes[2]
                    im1 = ax.imshow(grey, cmap="gray", vmin=0, vmax=1)  
                    # overlay cp mask (e.g. magenta where cp==1)
                    ax.imshow(
                        cp_overlay,
                        cmap=mcolors.ListedColormap(["black"]),
                        alpha=0.6
                    )
                    ax.set_title(f"cp mask", fontsize=18)
                    ax.axis("off")

                    ax = axes[4]
                    im3 = ax.imshow(grey, cmap="gray", vmin=0, vmax=1) 
                    # overlay cp mask (e.g. magenta where cp==1)
                    ax.imshow(
                        final_snowline_overlay,
                        cmap=mcolors.ListedColormap(["cyan"]),
                        alpha=0.6
                    )
                    ax.imshow(
                        max_snow_elev_mask_overlay,
                        cmap=mcolors.ListedColormap(["black"]),  
                        alpha=0.8
                    )
                    ax.set_title(f"Final snowline mask", fontsize=18)
                    ax.axis("off")

                    # ax = axes[0]
                    # im0 = ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0)
                    # ax.imshow(
                    #     melt_overlay,
                    #     cmap=mcolors.ListedColormap(["black"]),
                    #     alpha=0.6
                    # )
                    # # ice 
                    # ax.imshow(
                    #     ice_overlay,
                    #     cmap=mcolors.ListedColormap(["cyan"]),  
                    #     alpha=0.8
                    # )
                    # ax.set_title(
                    #     f"{year} DOY {doy}\n({date}) — Melted pixels (black)",
                    #     fontsize=11
                    # )
                    # ax.axis("off")

                    # # ---------- SECOND: Backscatter + cp mask ----------
                    # ax = axes[1]
                    # im1 = ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0)
                    # # overlay cp mask (e.g. magenta where cp==1)
                    # ax.imshow(
                    #     sl_cp_overlay,
                    #     cmap=mcolors.ListedColormap(["magenta"]),
                    #     alpha=0.6
                    # )
                    # ax.set_title(f"Backscatter (dB) + sl_cp mask, scene {nscene}", fontsize=11)
                    # ax.axis("off")

                    # # ---------- THIRD: DEM ----------
                    # ax = axes[2]
                    # im2 = ax.imshow(dem_glac, cmap="terrain")
                    # ax.set_title("DEM (m)", fontsize=11)
                    # ax.axis("off")

                    # # ---------- FOURTH: Difference (ice_mask_full - sl_cp_overlay) ----------
                    # ax = axes[3]
                    # # Show backscatter as background
                    # ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0, alpha=0.3)
                    # # Show difference: 
                    # # +1 (ice but not sl_cp) in one color, -1 (sl_cp but not ice) in another
                    # im3 = ax.imshow(
                    #     snow_difference_overlay,
                    #     cmap=mcolors.ListedColormap(["red", "blue"]),  # red for -1, blue for +1
                    #     #cmap=mcolors.ListedColormap(["magenta"]),
                    #     vmin=-1,
                    #     vmax=1,
                    #     alpha=0.8
                    # )
                    # ax.set_title("Difference mask: onset - sl_cp", fontsize=11)
                    # ax.axis("off")

                    # # Fifth : second melt mask
                    # ax = axes[4]
                    # im1 = ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0)
                    # # overlay cp mask (e.g. magenta where cp==1)
                    # ax.imshow(
                    #     final_snowline_overlay,
                    #     cmap=mcolors.ListedColormap(["magenta"]),
                    #     alpha=0.6
                    # )
                    # ax.set_title(f"Final snowline mask", fontsize=11)
                    # ax.axis("off")

                    # ax = axes[5]
                    # im1 = ax.imshow(sar_dB_masked, cmap=backscatter_cmap, vmin=-25, vmax=0)
                    # # overlay cp mask (e.g. magenta where cp==1)
                    # ax.imshow(
                    #     cp_overlay,
                    #     cmap=mcolors.ListedColormap(["black"]),
                    #     alpha=0.6
                    # )
                    # ax.set_title(f"Backscatter (dB) + cp mask, scene {nscene}", fontsize=11)
                    # ax.axis("off")


                    # Add one shared colorbar for backscatter (left + middle)
                    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # (left, bottom, width, height)
                    # cbar = fig.colorbar(im1, cax=cbar_ax)
                    # cbar.set_label("σ⁰ (dB)", rotation=270, labelpad=15)

                    # Save plot
                    out_path = os.path.join(plot_dir, f"{glacno}_onset_map_{year}_doy{doy:03d}.png")
                    # fig.text(0.5, -0.02, f'''cp: {cp_scene[plot_coords[1],  plot_coords[0]]}, sl_cp: {data_sl_cp_glac_single[plot_coords[1],  plot_coords[0]]}
                    #                         melt mask: {melt_overlay[plot_coords[1],  plot_coords[0]]}, snowline mask: {final_snowline_overlay[plot_coords[1],  plot_coords[0]]}''', 
                    #     ha='center', fontsize=14)
                    plt.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close()

            # Save CSV
            # df = pd.DataFrame(results)
            # min_elev = np.nanmin(self.glac_dem[glacno][self.glac_mask[glacno] == 1])
            # max_elev = np.nanmax(self.glac_dem[glacno][self.glac_mask[glacno] == 1])
            # df["melt_extent_elev_m"].fillna(0.0, inplace=True)
            # df["melt_extent_elev_m"].replace(0.0, min_elev, inplace=True)
            # df["glacier_max_elev_m"] = max_elev
            # df["snowline_elev_m"].fillna(0.0, inplace=True)
            # df["snowline_elev_m"].replace(0.0, min_elev, inplace=True)

            df = pd.DataFrame(results)
            min_elev = np.nanmin(self.glac_dem[glacno][self.glac_mask[glacno] == 1])
            max_elev = np.nanmax(self.glac_dem[glacno][self.glac_mask[glacno] == 1])

            for col in ["melt_extent_elev_m", "snowline_elev_m"]:
                df[col] = df[col].fillna(0.0)
                df[col] = df[col].replace(0.0, min_elev)

            df["glacier_max_elev_m"] = max_elev

            csv_path = os.path.join(out_dir, f"melt_snowline_time_series_{glacno}_{year}.csv")
            df.to_csv(csv_path, index=False)
            yearly_csv_paths.append(csv_path)

        #Merge csvs, delete yearly files
        if yearly_csv_paths:
            merged_dfs = []

            for csv_path in yearly_csv_paths:
                if os.path.exists(csv_path):
                    merged_dfs.append(pd.read_csv(csv_path))

            if merged_dfs:
                merged_df = pd.concat(merged_dfs, ignore_index=True)

                # Optional but usually helpful
                if "date" in merged_df.columns:
                    merged_df = merged_df.sort_values("date").reset_index(drop=True)

                merged_csv_path = os.path.join(
                    out_dir,
                    f"melt_snowline_time_series_{glacno}_all_years.csv"
                )
                merged_df.to_csv(merged_csv_path, index=False)

                for csv_path in yearly_csv_paths:
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
            
    

    def plot_melt_onset_maps(self, glacno, out_dir, verbose=False):
        """
        Plot and save melt onset DOY maps for a single glacier for each year.

        Output:
            <out_dir>/<glacno>_melt_onset_<year>.png
            <out_dir>/<glacno>_melt_onset_cons_<year>.png
            <out_dir>/<glacno>_melt_onset_diff_<year>.png
        """

        os.makedirs(out_dir, exist_ok=True)

        mask_glac = self.glac_mask[glacno]
        h, w = mask_glac.shape

        xb0 = self.glac_bounds[glacno]['xmin']
        yb0 = self.glac_bounds[glacno]['ymin']

        total_pixels = np.nansum(mask_glac)

        vmin = 100
        vmax = 300

        # reversed plasma colormap for DOY maps
        cmap = plt.cm.plasma_r.copy()
        cmap.set_bad(alpha=0)   # fully transparent for NaN

        # --- original onset maps ---
        for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:

            onset_map_full = self.melt_onset_doy_maps[year]
            onset_map_full = self.min_dB_doys[year]

            onset_glac = onset_map_full[xb0 : xb0 + h,
                                        yb0 : yb0 + w]

            # apply glacier mask
            onset_map_full_range = np.where(mask_glac == 1, onset_glac, np.nan)
            onset_map_full_range = onset_map_full_range.astype(float)  # converts None -> nan

            # apply DOY subset: keep ONLY 100–300 (clip outside range)
            onset_map = onset_map_full_range.copy()
            onset_map[onset_map_full_range < vmin] = vmin
            onset_map[onset_map_full_range > vmax] = vmax

            valid_frac = np.sum(~np.isnan(onset_map)) / total_pixels
            if valid_frac < 0.01:
                print(f"Skipping {year}: insufficient valid melt onset pixels")
                continue

            # mask off-glacier and no-onset pixels
            display_mask = np.isnan(onset_map) | (mask_glac == 0)
            display_ma = np.ma.array(onset_map, mask=display_mask)

            plt.figure(figsize=(8, 6))
            im = plt.imshow(display_ma, cmap=cmap, vmin=vmin, vmax=vmax)

            # glacier outline in green
            plt.contour(mask_glac, levels=[0.5], colors='green', linewidths=1.5)

            plt.title(f"Melt Onset DOY (100–300) — Glacier {glacno}, {year} (original)")
            plt.colorbar(im, label="Day of Year")
            plt.axis("off")

            out_path = os.path.join(out_dir, f"{glacno}_melt_onset_{year}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

        



    def single_glacier_preprocess(self, glacno=None, area_km2=10, verbose=False):
        """
        Glacier melt extent elevations for individual glaciers

        Parameters
        ----------
        glacno : int
            glacier number within the region (SAR datacube) of interest

        Returns
        -------
        self.data_cp_glac : dictionary of np.arrays
            change potential data cubes for each glacier
        """
        self.glac_bounds[glacno] = {}
        xmin = np.where(self.mask_values == glacno)[0].min() - 1
        xmax = np.where(self.mask_values == glacno)[0].max() + 1
        ymin = np.where(self.mask_values == glacno)[1].min() - 1
        ymax = np.where(self.mask_values == glacno)[1].max() + 1
        self.glac_bounds[glacno]['xmin'] = xmin
        self.glac_bounds[glacno]['xmax'] = xmax
        self.glac_bounds[glacno]['ymin'] = ymin
        self.glac_bounds[glacno]['ymax'] = ymax    
        
        # Single Glacier Mask
        mask_values_glac = self.mask_values[xmin:xmax+1,ymin:ymax+1]
        mask_values_binary_nan_glac = np.copy(mask_values_glac).astype(np.float64)
        mask_values_binary_nan_glac[mask_values_binary_nan_glac!=glacno] = np.nan
        mask_values_binary_nan_glac[mask_values_binary_nan_glac>0] = 1
        self.glac_mask[glacno] = mask_values_binary_nan_glac
        
        # Good Pixel Mask
        self.glac_mask_good_pixels[glacno] = self.mask_good_pixels[xmin:xmax+1,ymin:ymax+1]
        
        # Subset data
        self.glac_data[glacno] = self.data_masked[:,xmin:xmax+1,ymin:ymax+1] * self.glac_mask[glacno][np.newaxis,:,:]
        self.glac_data_cp[glacno] = self.data_cp[:,xmin:xmax+1,ymin:ymax+1] * self.glac_mask[glacno][np.newaxis,:,:]
        self.glac_data_sl_cp[glacno] = self.data_sl_cp[:,xmin:xmax+1,ymin:ymax+1] * self.glac_mask[glacno][np.newaxis,:,:]
        
        glac_dem = self.dem[xmin:xmax+1,ymin:ymax+1].astype(np.float64)
        glac_dem = glac_dem * self.glac_mask[glacno] * self.glac_mask_good_pixels[glacno]
        # glac_dem[np.isnan(self.glac_data_cp[glacno][0,:,:])] = np.nan
        self.glac_dem[glacno] = glac_dem

        # get the elevation of minimum backscatter (for defining maximum snowline cutoff)
        glac_data = self.glac_data[glacno] * self.glac_mask[glacno] * self.glac_mask_good_pixels[glacno]
        flat_idx = np.nanargmin(glac_data.reshape(glac_data.shape[0], -1), axis=1)
        rows, cols = np.unravel_index(flat_idx, glac_data.shape[1:])
        min_dB_elevs = glac_dem[rows, cols]
        self.min_dB_elevs[glacno] = min_dB_elevs
        
        # equal elevation bins
        bin_min = int(np.floor(np.nanmin(glac_dem) / self.bin_size) * self.bin_size)
        bin_max = int(np.ceil(np.nanmax(glac_dem) / self.bin_size) * self.bin_size)
        if verbose:
            print('bin_min:', bin_min, '\nbin_max:', bin_max)
        
        bins = np.arange(bin_min, bin_max+self.bin_size, self.bin_size)
        bins_center = np.arange(bin_min + self.bin_size/2, bin_max, self.bin_size).astype(int)
        bins_count, bins = np.histogram(glac_dem, bins=bins)
        nbins = bins_center.shape[0]
        
        # equal area bins
        if self.area_bin_size == 'variable': # if area bin size is variable
            min_bins = 100 # get area based on 50 bins
            min_bin_size = (area_km2 * 1e6) / min_bins 
            area_bin_size = np.ceil(min_bin_size / (self.xres * self.yres)) * (self.xres * self.yres)
            area_bin_size = max(area_bin_size, 100000) # minimum of 0.1 km2 bins
            area_bin_size = min(area_bin_size, 2e6) # maximum of 2 km2 bins
            self.area_bin_size = area_bin_size

            print(area_bin_size, self.xres, self.yres, type(area_bin_size))
    
        assert self.area_bin_size % (self.xres * self.yres) == 0, f'`area_bin_size` is an area not compatible with DEM resolution ({self.xres} m)'
        
        pixels_per_area_bin = int(self.area_bin_size/(self.xres*self.yres))  # pixels per elevation bin
        dem_sort = np.sort(glac_dem[~np.isnan(glac_dem)].flatten()) # remove NaN and sort elevation
        
        area_bins = dem_sort[::pixels_per_area_bin] # find elevation of bin edges
        if area_bins[-1] != dem_sort[-1]:  # include last bin edge
            area_bins = np.append(area_bins, dem_sort[-1])
        area_bins_center = 0.5 * (area_bins[:-1] + area_bins[1:]) # find bin centers
        if verbose:
            print('area_bin_min:', area_bins[0], '\narea_bin_max:', area_bins[-1])

        self.glac_bins[glacno] = bins
        self.glac_bins_center[glacno] = bins_center
        self.glac_area_bins[glacno] = area_bins
        self.glac_area_bins_center[glacno] = area_bins_center
            
    
    def melt_elev_percentile_method(self, glacno, csv_fn=None, csv_sl_fn=None, verbose=True):
        """
        Compute Melt Elevations using the Percentile Method

        Note: the problem with the percentile method is that ice pixels that are still melting, 
        but undetected due to the lack of snow prevent the method from using a simple pixel count.  
        Hence, this method identifies those pixels based on an "all melt" threshold that identfies
        pixels melting above them. These pixels are then assumed to be melting. 
        
        "All-Melt Threshold"
        This threshold is used to identify the elevation at which the bin is melting. 
        To avoid issues with this being applied too early (e.g., around the ELA where you may have 
        a mix of ice and firn pixels) this uses a fraction and 100%.

        Parameters
        ----------
        glacno : int
            glacier number within the region (SAR datacube) of interest
        verbose : Boolean
            print some debugging information or not

        Returns
        -------
        glac_melt_extent_elevs_percentiles : dictionary of np.arrays
            time series of the melt extent elevations for each glacier number
        """
        # Sorted DEM values to use with the percentile method for easy indexing
        dem_values_sorted = np.sort(self.glac_dem[glacno].reshape(1,-1))[0,:]
        dem_values_sorted = dem_values_sorted[dem_values_sorted > -9999]
        
        # Process scenes
        glac_dem = self.glac_dem[glacno]
        data_cp_glac = self.glac_data_cp[glacno]
        min_dB_elevs_glac = self.min_dB_elevs[glacno]
        melt_extent_elevs, melt_extent_elev_mins, melt_extent_elev_maxs = [], [], []
        melt_extent_areas, melt_extent_area_mins, melt_extent_area_maxs = [], [], []
        snowline_elevs, snowline_elev_mins, snowline_elev_maxs = [], [], []
        snowline_areas, snowline_area_mins, snowline_area_maxs = [], [], []

        scene_dates = np.array(self.dates)   # shape: (n_scenes,)
        scene_years = scene_dates.astype("datetime64[Y]").astype(int) + 1970

        prev_cp = None
        prev_year = None  

        for nscene in np.arange(data_cp_glac.shape[0]):
            data_cp_glac_single = data_cp_glac[nscene,:,:]
            min_dB_elevs_glac_single = min_dB_elevs_glac[nscene]
            
            year = scene_years[nscene]  

            if prev_year is not None and year != prev_year:
                prev_cp = None
        
            # Check if all nan values
            if len(np.where(~np.isnan(data_cp_glac_single))[0]) == 0:
                melt_extent_elevs.append(np.nan)
                melt_extent_areas.append(np.nan)
            # Otherwise, calculate extent
            else:
                # ---------- ---------- ALL MELT CORRECTION: equal elevation bins ---------- ----------
                allmelt_elev = 0
                allmelt_100 = False
                for nbin, bin_elev_lower in enumerate(self.glac_bins[glacno][:-1]):
                    bin_elev_upper = self.glac_bins[glacno][nbin+1]
                    if not allmelt_100:
                        # Create mask based on elevations
                        mask_bin = np.zeros(glac_dem.shape)
                        mask_bin[(glac_dem > bin_elev_lower) & (glac_dem <= bin_elev_upper)] = 1
                        bin_count = mask_bin.sum()
                
                        data_cp_bin = data_cp_glac_single * mask_bin
                        data_cp_bin_count = np.nansum(data_cp_bin)
                
                        frac_melt = data_cp_bin_count / bin_count
                        # Record "all melt" elevation 
                        if allmelt_elev == 0 and frac_melt > allmelt_threshold and bin_count > allmelt_pixels:
                            allmelt_elev = bin_elev_lower
                        # Record "all melt" elevation in the case that 100% hasn't been found yet
                        if not allmelt_100 and frac_melt == 1:
                            allmelt_elev = bin_elev_lower
                            allmelt_100 = True

                # if prev_cp is not None:
                # # any pixel that was melting (==1) in previous scene
                # # is forced to still be melting in current scene
                #     data_cp_glac_single[prev_cp == 1] = 1
                
                # store processed cp (after all corrections) for next iteration
                prev_cp = data_cp_glac_single.copy()
                prev_year = year
                
                # Apply "all melt" correction
                if allmelt_elev > 0:
                    data_cp_glac_single[glac_dem < allmelt_elev] = 1
            
                # ----- PERCENTILE METHOD -----
                melt_pixels = np.nansum(data_cp_glac_single)
        
                # The index is associated with one less than the sum of the pixels to account for indexing starting with 0 not 1
                if melt_pixels == 0:
                    melt_idx = int(0)
                else:
                    melt_idx = int(melt_pixels - 1)

                melt_extent_elev = dem_values_sorted[melt_idx]
                #print("Prev:", year, nscene, allmelt_elev, melt_extent_elev)
                melt_extent_elevs.append(melt_extent_elev)

                
                # percentile method uncertainty
                nomelt_pixels_below = np.nansum((glac_dem < melt_extent_elev) & (data_cp_glac_single == 0))
                melt_pixels_above = np.nansum((glac_dem > melt_extent_elev) & (data_cp_glac_single == 1))
                melt_extent_elev_min = dem_values_sorted[melt_idx - nomelt_pixels_below]
                melt_extent_elev_max = dem_values_sorted[melt_idx + melt_pixels_above]
                melt_extent_elev_mins.append(melt_extent_elev_min)
                melt_extent_elev_maxs.append(melt_extent_elev_max)

                # ---------- ---------- ALL MELT CORRECTION: repeat for equal area elevation bin ---------- ----------
                allmelt_elev = 0
                allmelt_100 = False
                for nbin, bin_elev_lower in enumerate(self.glac_area_bins[glacno][:-1]):
                    bin_elev_upper = self.glac_area_bins[glacno][nbin+1]
                    if not allmelt_100:
                        # Create mask based on elevations
                        mask_bin = np.zeros(glac_dem.shape)
                        mask_bin[(glac_dem > bin_elev_lower) & (glac_dem <= bin_elev_upper)] = 1
                        bin_count = mask_bin.sum()
                
                        data_cp_bin = data_cp_glac_single * mask_bin
                        data_cp_bin_count = np.nansum(data_cp_bin)
                
                        frac_melt = data_cp_bin_count / bin_count
                
                        # Record "all melt" elevation 
                        if allmelt_elev == 0 and frac_melt > allmelt_threshold and bin_count > allmelt_pixels:
                            allmelt_elev = bin_elev_lower
                        # Record "all melt" elevation in the case that 100% hasn't been found yet
                        if not allmelt_100 and frac_melt == 1:
                            allmelt_elev = bin_elev_lower
                            allmelt_100 = True

                # Apply "all melt" correction
                if allmelt_elev > 0:
                    data_cp_glac_single[glac_dem < allmelt_elev] = 1
            
                # ----- PERCENTILE METHOD -----
                melt_pixels = np.nansum(data_cp_glac_single)
        
                # The index is associated with one less than the sum of the pixels to account for indexing starting with 0 not 1
                if melt_pixels == 0:
                    melt_idx = int(0)
                else:
                    melt_idx = int(melt_pixels - 1)               
                melt_extent_area = melt_idx*self.xres*self.yres
                melt_extent_areas.append(melt_extent_area)

                # percentile method uncertainty
                nomelt_pixels_below = np.nansum((glac_dem < dem_values_sorted[melt_idx]) & (data_cp_glac_single == 0))
                melt_pixels_above = np.nansum((glac_dem > dem_values_sorted[melt_idx]) & (data_cp_glac_single == 1))
                melt_extent_area_min = (melt_idx - nomelt_pixels_below)*self.xres*self.yres
                melt_extent_area_max = (melt_idx + melt_pixels_above)*self.xres*self.yres
                melt_extent_area_mins.append(melt_extent_area_min)
                melt_extent_area_maxs.append(melt_extent_area_max)

                # ------------------------- SNOWLINES ------------------------------
                # SNOWLINES: cp values that are back to 0 below the melt extent (minimum to be conservative) or elevation of min backscatter
                data_sl_cp_glac = self.glac_data_sl_cp[glacno]
                data_sl_cp_glac_single = data_sl_cp_glac[nscene,:,:]
                data_sl_elev_max = min(min_dB_elevs_glac_single, melt_extent_elev_min)
                snowline_cp = (data_sl_cp_glac_single == 1) & (glac_dem < data_sl_elev_max)
                #print("Prev method:", year, nscene, (np.nansum(data_sl_cp_glac_single == 1)), np.nansum(snowline_cp))
                
                snowline_idx = max(np.nansum(snowline_cp) - 1, 0)
                snowline_elev = dem_values_sorted[snowline_idx]
                snowline_area = snowline_idx*self.xres*self.yres

                # print(f'''Prev method: year {year}, scene {nscene} total_snow_cp_pixels: {np.nansum(data_sl_cp_glac_single == 1)}, snowline_pixels after 
                #       adding pixels below ME min: {np.nansum(snowline_cp)},
                #       snowline_elev: {snowline_elev}''')

                # percentile method uncertainty for snowlines
                # snow_pixels_below = np.nansum((glac_dem < snowline_elev) & (data_sl_cp_glac_single == 0))
                # nosnow_pixels_above = np.nansum((glac_dem > snowline_elev) & (data_sl_cp_glac_single == 1) & 
                #                                 (glac_dem < melt_extent_elev_min))
                snow_pixels_below = (np.nansum((glac_dem < snowline_elev) & (data_sl_cp_glac_single == 0)) + 
                                     np.nansum((glac_dem < snowline_elev) & self.glac_mask[glacno].astype(bool) & ~self.glac_mask_good_pixels[glacno].astype(bool)))
                nosnow_pixels_above = (np.nansum((glac_dem > snowline_elev) & (data_sl_cp_glac_single == 1) & (glac_dem < melt_extent_elev_min)) +
                                       np.nansum((glac_dem > snowline_elev) & self.glac_mask[glacno].astype(bool) & ~self.glac_mask_good_pixels[glacno].astype(bool)))

                snowline_elev_min = dem_values_sorted[snowline_idx - snow_pixels_below]
                snowline_elev_max = dem_values_sorted[snowline_idx + nosnow_pixels_above]
                snowline_area_min = (snowline_idx - snow_pixels_below)*self.xres*self.yres
                snowline_area_max = (snowline_idx + nosnow_pixels_above)*self.xres*self.yres

                # add to lists
                snowline_elevs.append(snowline_elev)
                snowline_areas.append(snowline_area)
                snowline_elev_mins.append(snowline_elev_min)
                snowline_elev_maxs.append(snowline_elev_max)
                snowline_area_mins.append(snowline_area_min)
                snowline_area_maxs.append(snowline_area_max)

        
        self.glac_melt_extent_elevs_percentiles[glacno] = np.array(melt_extent_elevs)
        self.glac_melt_extent_elevs_percentile_mins[glacno] = np.array(melt_extent_elev_mins)
        self.glac_melt_extent_elevs_percentile_maxs[glacno] = np.array(melt_extent_elev_maxs)
        self.glac_melt_extent_areas_percentiles[glacno] = np.array(melt_extent_areas)
        self.glac_melt_extent_areas_percentile_mins[glacno] = np.array(melt_extent_area_mins)
        self.glac_melt_extent_areas_percentile_maxs[glacno] = np.array(melt_extent_area_maxs)

        self.glac_snowline_elevs_percentiles[glacno] = np.array(snowline_elevs)
        self.glac_snowline_elevs_percentile_mins[glacno] = np.array(snowline_elev_mins)
        self.glac_snowline_elevs_percentile_maxs[glacno] = np.array(snowline_elev_maxs)
        self.glac_snowline_areas_percentiles[glacno] = np.array(snowline_areas)
        self.glac_snowline_areas_percentile_mins[glacno] = np.array(snowline_area_mins)
        self.glac_snowline_areas_percentile_maxs[glacno] = np.array(snowline_area_maxs)
        
        # Export binned data        
        if not csv_fn is None:
            me_df = pd.DataFrame(self.glac_melt_extent_elevs_percentiles[glacno], columns=['melt_extent_elev_m'], index=self.dates)
            me_df['melt_extent_elev_min_m'] = self.glac_melt_extent_elevs_percentile_mins[glacno]
            me_df['melt_extent_elev_max_m'] = self.glac_melt_extent_elevs_percentile_maxs[glacno]
            me_df['melt_extent_elev_diff_mean_m'] = ((me_df['melt_extent_elev_max_m'] - me_df['melt_extent_elev_m']) +
                                                     (me_df['melt_extent_elev_m'] - me_df['melt_extent_elev_min_m']))/2
            me_df.to_csv(csv_fn)
            
            me_df = pd.DataFrame(self.glac_melt_extent_areas_percentiles[glacno], columns=['melt_extent_area_m2'], index=self.dates)
            me_df['melt_extent_area_min_m2'] = self.glac_melt_extent_areas_percentile_mins[glacno]
            me_df['melt_extent_area_max_m2'] = self.glac_melt_extent_areas_percentile_maxs[glacno]
            me_df['melt_extent_area_diff_mean_m2'] = ((me_df['melt_extent_area_max_m2'] - me_df['melt_extent_area_m2']) +
                                                      (me_df['melt_extent_area_m2'] - me_df['melt_extent_area_min_m2']))/2
            me_df.to_csv(csv_fn[:-4]+'_eabin.csv')
        else:
            return (self.dates, self.glac_melt_extent_elevs_percentiles[glacno], self.glac_melt_extent_areas_percentiles[glacno], 
                    self.glac_snowline_elevs_percentiles[glacno], self.glac_snowline_areas_percentiles[glacno])

        if not csv_sl_fn is None:
            sl_df = pd.DataFrame(self.glac_snowline_elevs_percentiles[glacno], columns=['snowline_elev_m'], index=self.dates)
            sl_df['snowline_elev_min_m'] = self.glac_snowline_elevs_percentile_mins[glacno]
            sl_df['snowline_elev_max_m'] = self.glac_snowline_elevs_percentile_maxs[glacno]
            sl_df['snowline_elev_diff_mean_m'] = ((sl_df['snowline_elev_max_m'] - sl_df['snowline_elev_m']) +
                                                  (sl_df['snowline_elev_m'] - sl_df['snowline_elev_min_m']))/2
            sl_df.to_csv(csv_sl_fn)
            
            sl_df = pd.DataFrame(self.glac_snowline_areas_percentiles[glacno], columns=['snowline_area_m2'], index=self.dates)
            sl_df['snowline_area_min_m2'] = self.glac_snowline_areas_percentile_mins[glacno]
            sl_df['snowline_area_max_m2'] = self.glac_snowline_areas_percentile_maxs[glacno]
            sl_df['snowline_area_diff_mean_m2'] = ((sl_df['snowline_area_max_m2'] - sl_df['snowline_area_m2']) +
                                                   (sl_df['snowline_area_m2'] - sl_df['snowline_area_min_m2']))/2
            sl_df.to_csv(csv_sl_fn[:-4]+'_eabin.csv')

    
    def db_heatmap(self, glacno, csv_fn=None, hyps_fn=None):
        """
        Bin backscatter to produce heatmaps

        Parameters
        ----------
        glacno : int
            glacier number within the region (SAR datacube) of interest

        Returns
        -------
        glac_binned_db : dictionary of np.arrays
            binned backscatter for each glacier number
        """
        for i in [0, 1]:
            if i == 0:
                bins = self.glac_bins[glacno]
                bins_center = self.glac_bins_center[glacno]
            else:
                bins = self.glac_area_bins[glacno]
                bins_center = self.glac_area_bins_center[glacno]
            
            glac_dem = self.glac_dem[glacno]
            glac_data = self.glac_data[glacno]
            
            db_bin = np.zeros((len(bins)-1, glac_data.shape[0]))
            db_bin[:,:] = np.nan
            binned_pixels = np.zeros((len(bins)-1))
            for nbin, bin_elev_lower in enumerate(bins[:-1]):
                # Create mask based on elevations
                bin_elev_upper = bins[nbin+1]
                mask_bin = np.zeros(glac_dem.shape)
                mask_bin[(glac_dem > bin_elev_lower) & (glac_dem <= bin_elev_upper)] = 1
                bin_pixels = mask_bin.sum()
            
                # Mask pixels
                data_glac_singlebin = glac_data * mask_bin[np.newaxis,:,:]
            
                # Manually average based on summing and pixel counts
                #  note: this avoids masking the entire data stack which is slow
                bin_sar_series = np.nansum(data_glac_singlebin, axis=(1, 2)) / bin_pixels
                bin_sar_series[bin_sar_series == 0] = np.nan
                db_bin[nbin,:] = bin_sar_series
            
                binned_pixels[nbin] = bin_pixels
    
            # Export binned data
            if not csv_fn is None:
                db_bin_df = pd.DataFrame(db_bin, columns=self.dates, index=bins_center)
                if i == 0:
                    db_bin_df.to_csv(csv_fn)
                else:
                    db_bin_df.to_csv(csv_fn[:-4]+'_eabin.csv')

            # Export hypsometry
            binned_area = binned_pixels * self.xres * self.yres
            if not hyps_fn is None:
                hyps_df = pd.DataFrame(binned_area, columns=['area_m2'], index=bins_center)
                hyps_df.index.name = 'Elev_m'
                if i == 0:
                    hyps_df.to_csv(hyps_fn)
                else:
                    hyps_df.to_csv(hyps_fn[:-4]+'_eabin.csv')
