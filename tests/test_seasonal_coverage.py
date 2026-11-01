"""Regression tests for late-year-only bursts and map-only products."""

import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.datacube import sar_datacube
from hyp3_glacier_melt.doy_product import create_and_write_doy_products
from hyp3_glacier_melt.doy_workflow import pipeline
from hyp3_glacier_melt.product import MeltRunResult, package_product


def seasonal_cube(include_partial_year=True):
    """A late-year-only 2018 followed by an observed 2019 melt season."""
    dc = object.__new__(sar_datacube)
    dates = ['2019-01-02', '2019-02-02', '2019-05-01', '2019-06-01',
             '2019-07-01', '2019-08-01', '2019-09-01']
    backscatter = [-10, -12, -18, -19, -15, -9, -18]
    if include_partial_year:
        # August is allowed by the onset code but cannot supply its April-July
        # reference. It must not produce an inferred melt/spring-baseline event.
        dates = ['2018-08-16', '2018-11-08', '2018-12-14'] + dates
        backscatter = [-19, -10, -10] + backscatter
    config = MeltConfig()
    for name in ('winter_months', 'snowmelt_months', 'months2exclude_cp',
                 'db_threshold', 'db_threshold_sl', 'zscore_threshold',
                 'bin_size', 'area_bin_size', 'allmelt_threshold', 'allmelt_pixels'):
        setattr(dc, name, getattr(config, name))
    dc.dates = np.array(dates, dtype='datetime64[s]')
    dc.data_masked = np.broadcast_to(np.array(backscatter, dtype=float)[:, None, None],
                                    (len(dates), 6, 6)).copy()
    dc.mask_good_pixels = np.ones((6, 6))
    dc.dem = (np.arange(36).reshape(6, 6) * 10 + 100).astype(float)
    dc.xres = dc.yres = 100
    dc.ds = xr.Dataset(coords={'time': dc.dates, 'x': np.arange(6) * 100 + 50,
                              'y': 550 - np.arange(6) * 100}, attrs={'epsg_str': 'EPSG:32607'})
    dc.pixel_analysis()
    dc.annual_melt_onset_map()
    dc.annual_snowline_post_onset_map()
    dc.annual_snowline_onset_map()
    dc.annual_second_onset_map()
    return dc


def glacier_mask():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    return mask


def test_missing_season_has_nodata_maps_without_changing_observed_year():
    dc = seasonal_cube()
    reference = seasonal_cube(include_partial_year=False)
    assert set(dc.skipped_years) == {2018}
    for attribute in ('min_dB_doys', 'melt_onset_doy_maps', 'snowline_post_onset_doy_maps',
                      'snowline_onset_doy_maps', 'annual_second_onset_map_doy_maps'):
        assert np.isnan(getattr(dc, attribute)[2018]).all()
        np.testing.assert_array_equal(getattr(dc, attribute)[2019], getattr(reference, attribute)[2019])
    assert np.isfinite(dc.melt_onset_doy_maps[2019]).any()


def test_primary_csv_omits_unobserved_year(tmp_path):
    dc = seasonal_cube()
    dc.mask_values = glacier_mask().astype('uint32')
    for name in ('glac_bounds', 'glac_mask', 'glac_mask_good_pixels', 'glac_dem',
                 'glac_bins', 'glac_bins_center', 'glac_area_bins', 'glac_area_bins_center'):
        setattr(dc, name, {})
    pipeline.prepare_glacier_spatial_state(dc, 1, 0.16)
    csv = dc.generate_elevs_from_onsets(1, tmp_path, rgi_id='01.00001')
    frame = pd.read_csv(csv)
    assert set(frame.year) == {2019}
    assert frame.date.tolist() == dc.dates[3:].astype('datetime64[D]').astype(str).tolist()


def test_all_years_missing_season_can_still_export_a_map_only_product(tmp_path):
    dc = seasonal_cube()
    dc.dates = dc.dates[:3]
    dc.data_masked = dc.data_masked[:3]
    dc.ds = dc.ds.isel(time=slice(0, 3))
    dc.pixel_analysis()
    dc.annual_melt_onset_map()
    dc.annual_snowline_post_onset_map()
    dc.annual_snowline_onset_map()
    dc.annual_second_onset_map()
    paths = create_and_write_doy_products(dc, 2018, tmp_path / 'maps')
    result = MeltRunResult(3, [], {}, paths, dc.skipped_years)
    product = package_product(result, 'no_season', tmp_path / 'products', {})
    with ZipFile(product) as archive:
        metadata = json.loads(archive.read('no_season/product_metadata.json'))
        assert metadata['skipped_year_count'] == 1
        assert metadata['doy_map_count'] == 7


def test_map_only_zip_records_missing_year_and_remains_readable(tmp_path):
    dc = seasonal_cube()
    paths = []
    for year in (2018, 2019):
        paths.extend(create_and_write_doy_products(dc, year, tmp_path / 'maps'))
    result = MeltRunResult(len(dc.dates), [], {}, paths, dc.skipped_years)
    product = package_product(result, 'map_only', tmp_path / 'products', {})
    with ZipFile(product) as archive:
        metadata = json.loads(archive.read('map_only/product_metadata.json'))
        assert metadata['successful_glacier_count'] == 0
        assert metadata['doy_map_count'] == 14
        assert metadata['status'] == 'partial'
        assert '2018' in metadata['skipped_years']
    with rasterio.open(tmp_path / 'maps/validity_mask_doy_2018.tif') as source:
        assert source.tags()['analysis_status'] == 'insufficient_observations'
        assert json.loads(source.tags()['acquisition_dates']) == ['2018-08-16', '2018-11-08', '2018-12-14']
        # Acquisition validity and native DEM remain stable across years.
        np.testing.assert_array_equal(source.read(1), dc.mask_good_pixels)
    discovered = pipeline.collect_doy_files([product], tmp_path / 'extracted')
    first = pipeline.mosaic_annual_maps(discovered[2018], 2018)
    second = pipeline.mosaic_annual_maps(discovered[2019], 2019)
    assert not first.coverage.any()
    assert second.coverage.all()
    np.testing.assert_array_equal(first.validity, second.validity)
    np.testing.assert_array_equal(first.metric_dem, second.metric_dem)


def test_stitching_retains_observed_year_when_another_year_has_no_season(tmp_path, monkeypatch):
    dc = seasonal_cube()
    for year in (2018, 2019):
        create_and_write_doy_products(dc, year, tmp_path / 'maps')
    monkeypatch.setattr(pipeline, 'glaciers_on_grid', lambda *a: [('01.00001', 1, 0.16, glacier_mask())])
    result = pipeline.run_doy_pipeline([tmp_path / 'maps'], tmp_path / 'output')
    assert not result.failed_glacnos
    assert set(result.skipped_years) == {2018}
    assert len(result.csv_files) == 1
    assert set(pd.read_csv(result.csv_files[0]).year) == {2019}


def test_missing_source_does_not_inflate_seasonal_coverage(tmp_path):
    dc = seasonal_cube()
    unavailable = tmp_path / 'unavailable'
    available = tmp_path / 'available'
    create_and_write_doy_products(dc, 2018, unavailable)
    # A neighboring source has usable observations only on its right half.
    dc.skipped_years = {}
    for attribute in ('min_dB_doys', 'melt_onset_doy_maps', 'snowline_post_onset_doy_maps',
                      'snowline_onset_doy_maps', 'annual_second_onset_map_doy_maps'):
        getattr(dc, attribute)[2018] = np.full((6, 6), 150.0)
    dc.mask_good_pixels[:, :3] = np.nan
    create_and_write_doy_products(dc, 2018, available)
    discovered = pipeline.collect_doy_files([unavailable, available], tmp_path / 'extracted')
    maps = pipeline.mosaic_annual_maps(discovered[2018], 2018)
    assert not maps.coverage[:, :3].any()
    assert maps.coverage[:, 3:].all()
    assert maps.validity.all()  # static acquisition coverage still includes both sources
    assert np.isnan(maps.melt_onset[:, :3]).all()
    assert np.isfinite(maps.melt_onset[:, 3:]).all()
