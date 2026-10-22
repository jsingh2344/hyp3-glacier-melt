"""Tests for stitching DOY maps and adapting them to the existing metrics code."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.doy_workflow.pipeline import (
    DOY_MAP_NAMES,
    AnnualDoyMaps,
    RasterGrid,
    collect_doy_files,
    mosaic_annual_maps,
    onset_datacube,
    prepare_glacier_spatial_state,
    usable_onset_years,
)


def write_doy_map(path: Path, values: np.ndarray, left: float) -> Path:
    """Write a small projected DOY map fixture."""
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype='float32',
        crs='EPSG:32607',
        transform=from_origin(left, 200, 100, 100),
        nodata=np.nan,
    ) as dataset:
        dataset.write(values.astype('float32'), 1)
    return path


def test_mosaic_annual_maps_stitches_bursts_and_uses_earliest_onset(
    tmp_path: Path,
) -> None:
    paths_by_name = {}
    left_values = np.array([[100, 150], [np.nan, 200]], dtype='float32')
    right_values = np.array([[120, 180], [190, np.nan]], dtype='float32')
    for map_name in DOY_MAP_NAMES:
        paths_by_name[map_name] = [
            write_doy_map(tmp_path / f'left_{map_name}.tif', left_values, 0),
            write_doy_map(tmp_path / f'right_{map_name}.tif', right_values, 100),
        ]

    products = mosaic_annual_maps(paths_by_name, 2024)

    expected = np.array([[100, 120, 180], [np.nan, 190, np.nan]], dtype='float32')
    np.testing.assert_allclose(products.melt_onset, expected, equal_nan=True)
    assert products.coverage.all()
    assert products.grid.width == 3
    assert products.grid.height == 2
    assert products.grid.bounds == (0.0, 0.0, 300.0, 200.0)


def test_collect_doy_files_requires_every_metric_map(tmp_path: Path) -> None:
    write_doy_map(tmp_path / 'melt_onset_doy_2024.tif', np.ones((2, 2)), 0)

    with pytest.raises(ValueError, match='Missing DOY maps for 2024'):
        collect_doy_files([tmp_path], tmp_path / 'extract')


def test_collect_doy_files_extracts_all_zips_in_input_directory(tmp_path: Path) -> None:
    product_directory = tmp_path / 'products'
    map_directory = tmp_path / 'maps'
    product_directory.mkdir()
    map_directory.mkdir()

    for burst_name in ('burst-a', 'burst-b'):
        archive_path = product_directory / f'{burst_name}.zip'
        with ZipFile(archive_path, 'w') as archive:
            for map_name in DOY_MAP_NAMES:
                map_path = write_doy_map(
                    map_directory / f'{map_name}_doy_2024.tif',
                    np.ones((2, 2)),
                    0,
                )
                archive.write(map_path, f'{burst_name}/doy_maps/{map_path.name}')

    # An extracted copy alongside ZIP products must not be counted a second time.
    write_doy_map(
        product_directory / 'melt_onset_doy_2024.tif',
        np.ones((2, 2)),
        0,
    )

    discovered = collect_doy_files(
        [product_directory],
        tmp_path / 'extracted',
    )

    assert set(discovered[2024]) == set(DOY_MAP_NAMES)
    assert all(len(paths) == 2 for paths in discovered[2024].values())


def test_collect_doy_files_rejects_product_zip_without_doy_maps(tmp_path: Path) -> None:
    archive_path = tmp_path / 'csv-only.zip'
    with ZipFile(archive_path, 'w') as archive:
        archive.writestr('product/glacier.csv', 'date,melt_area_m2\n')

    with pytest.raises(ValueError, match='contains no recognized annual DOY maps'):
        collect_doy_files([archive_path], tmp_path / 'extracted')


def test_mosaic_tracks_gaps_between_burst_footprints(tmp_path: Path) -> None:
    paths_by_name = {}
    for map_name in DOY_MAP_NAMES:
        paths_by_name[map_name] = [
            write_doy_map(tmp_path / f'left_{map_name}.tif', np.array([[100]]), 0),
            write_doy_map(tmp_path / f'right_{map_name}.tif', np.array([[120]]), 200),
        ]

    products = mosaic_annual_maps(paths_by_name, 2024)

    assert products.coverage.tolist() == [[True, False, True]]
    np.testing.assert_allclose(
        products.melt_onset,
        np.array([[100, np.nan, 120]], dtype='float32'),
        equal_nan=True,
    )


def test_onset_adapter_uses_existing_glacier_metric_implementation(
    tmp_path: Path,
) -> None:
    grid = RasterGrid(
        crs=rasterio.crs.CRS.from_epsg(32607),
        transform=from_origin(0, 600, 100, 100),
        height=6,
        width=6,
    )
    glacier_mask = np.zeros((6, 6), dtype=bool)
    glacier_mask[1:5, 1:5] = True

    def onset(value: float) -> np.ndarray:
        array = np.full((6, 6), np.nan, dtype='float32')
        array[glacier_mask] = value
        return array

    products = AnnualDoyMaps(
        year=2024,
        grid=grid,
        melt_onset=onset(100),
        ice_onset=onset(200),
        spring_baseline=onset(50),
        second_melt_onset=onset(250),
        coverage=np.ones((6, 6), dtype=bool),
    )
    dem = np.arange(36, dtype='float32').reshape(6, 6) * 10 + 100
    config = MeltConfig()
    datacube = onset_datacube(
        [products],
        dem,
        [('01.00001', 1, 0.16, glacier_mask)],
        config,
        start_doy=1,
        end_doy=301,
        doy_step=100,
    )
    prepare_glacier_spatial_state(datacube, glacier_number=1, area_km2=0.16)

    output = datacube.generate_elevs_from_onsets(
        1,
        tmp_path,
        rgi_id='01.00001',
        min_valid_frac=0.01,
    )
    frame = pd.read_csv(output)

    assert output.name == 'melt_snowline_time_series_01.00001_all_years.csv'
    assert frame['date'].tolist() == [
        '2024-01-01',
        '2024-04-10',
        '2024-07-19',
        '2024-10-27',
    ]
    assert frame.columns.tolist() == [
        'year',
        'date',
        'melt_extent_elev_m',
        'snowline_elev_m',
        'ice_area_m2',
        'melt_area_m2',
        'melt_fraction',
        'ice_fraction',
        'glacier_area_m2',
        'glacier_max_elev_m',
    ]


def test_usable_onset_years_matches_existing_validity_check() -> None:
    grid = RasterGrid(
        crs=rasterio.crs.CRS.from_epsg(32607),
        transform=from_origin(0, 200, 100, 100),
        height=2,
        width=2,
    )
    glacier_mask = np.ones((2, 2), dtype=bool)
    valid = np.full((2, 2), 100, dtype='float32')
    missing = np.full((2, 2), np.nan, dtype='float32')
    products = [
        AnnualDoyMaps(2023, grid, missing, valid, valid, valid, glacier_mask),
        AnnualDoyMaps(2024, grid, valid, valid, valid, valid, glacier_mask),
    ]

    assert usable_onset_years(products, glacier_mask) == {2024}


def test_explicit_validity_masks_sources_before_merging_and_keeps_no_event_pixels(tmp_path):
    sources = []
    for name, left, values, valid, dem in [
        ('a', 0, [[100, 10]], [[1, 0]], [[1000, 1100]]),
        ('b', 100, [[180, np.nan]], [[1, 1]], [[1200, 1300]]),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        paths = {}
        for metric in DOY_MAP_NAMES:
            paths[metric] = write_doy_map(directory / f'{metric}.tif', np.array(values), left)
        paths['validity_mask'] = write_doy_map(directory / 'validity.tif', np.array(valid), left)
        paths['metric_dem'] = write_doy_map(directory / 'dem.tif', np.array(dem), left)
        sources.append(paths)
    paths = {key: [source[key] for source in sources] for key in sources[0]}
    result = mosaic_annual_maps(paths, 2024)
    reverse = mosaic_annual_maps({key: list(reversed(value)) for key, value in paths.items()}, 2024)
    np.testing.assert_allclose(result.melt_onset, [[100, 180, np.nan]], equal_nan=True)
    assert result.validity.tolist() == [[True, True, True]]
    np.testing.assert_array_equal(result.metric_dem, [[1000, 1200, 1300]])
    np.testing.assert_array_equal(result.metric_dem, reverse.metric_dem)
    np.testing.assert_array_equal(result.melt_onset, reverse.melt_onset)


def test_explicit_validity_requires_companions_for_every_source(tmp_path):
    paths = {name: [write_doy_map(tmp_path / f'{name}.tif', np.ones((2, 2)), 0)]
             for name in DOY_MAP_NAMES}
    paths['validity_mask'] = [write_doy_map(tmp_path / 'valid.tif', np.ones((2, 2)), 0)]
    with pytest.raises(ValueError, match='Every source needs'):
        mosaic_annual_maps(paths, 2024)


def test_adapter_masks_events_and_dem_with_the_same_validity():
    grid = RasterGrid(rasterio.crs.CRS.from_epsg(32607), from_origin(0, 200, 100, 100), 2, 2)
    values = np.full((2, 2), 100, dtype='float32')
    validity = np.array([[True, False], [True, True]])
    product = AnnualDoyMaps(2024, grid, values, values, values, values, np.ones((2, 2), bool), validity)
    dc = onset_datacube([product], np.ones((2, 2)), [], MeltConfig(), 1, 2, 1)
    assert np.isnan(dc.melt_onset_doy_maps[2024][0, 1])
    assert np.isnan(dc.mask_good_pixels[0, 1])
    assert np.isfinite(dc.mask_good_pixels[1, 1])


def test_adapter_rejects_year_dependent_validity_in_global_metric_state():
    grid = RasterGrid(rasterio.crs.CRS.from_epsg(32607), from_origin(0, 200, 100, 100), 2, 2)
    values = np.ones((2, 2), dtype='float32')
    coverage = np.ones((2, 2), bool)
    first = AnnualDoyMaps(2023, grid, values, values, values, values, coverage, coverage)
    second = AnnualDoyMaps(2024, grid, values, values, values, values, coverage, ~coverage)
    with pytest.raises(ValueError, match='same explicit validity'):
        onset_datacube([first, second], values, [], MeltConfig(), 1, 2, 1)


def test_corrected_pipeline_enforces_original_sar_area_eligibility(tmp_path, monkeypatch):
    from hyp3_glacier_melt.doy_workflow import pipeline

    values = np.full((6, 6), 100, dtype='float32')
    validity = np.ones((6, 6), dtype='float32')
    validity[1, 1:4] = 0  # 13/16 valid glacier pixels: below the original 90% cutoff.
    for name in DOY_MAP_NAMES:
        write_doy_map(tmp_path / f'{name}_doy_2024.tif', values, 0)
    write_doy_map(tmp_path / 'validity_mask_doy_2024.tif', validity, 0)
    write_doy_map(tmp_path / 'metric_dem_doy_2024.tif', values, 0)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    monkeypatch.setattr(pipeline, 'glaciers_on_grid', lambda *args: [('01.00001', 1, .16, mask)])

    def unexpected_dem_download(*args):
        pytest.fail('Embedded metric DEM should avoid downloading or reprojecting a DEM')

    monkeypatch.setattr(pipeline, 'source_copernicus_dem', unexpected_dem_download)
    result = pipeline.run_doy_pipeline([tmp_path], tmp_path / 'output', doy_step=100)
    assert result.csv_files == []
    assert result.failed_glacnos == {}


def test_adapter_defaults_to_original_acquisitions_and_supports_explicit_grid():
    grid = RasterGrid(rasterio.crs.CRS.from_epsg(32607), from_origin(0, 200, 100, 100), 2, 2)
    values = np.full((2, 2), 100, dtype='float32')
    coverage = np.ones((2, 2), bool)
    product = AnnualDoyMaps(
        2024, grid, values, values, values, values, coverage, coverage, values,
        ('2024-01-02', '2024-02-29', '2024-12-24'),
    )
    dc = onset_datacube([product], values, [], MeltConfig(), 1, None, None)
    assert dc.doys == [2, 60, 359]
    dc = onset_datacube([product], values, [], MeltConfig(), 1, 101, 100)
    assert dc.doys == [1, 101]


def test_mosaic_unions_source_acquisition_dates(tmp_path):
    import json

    paths = {name: [] for name in (*DOY_MAP_NAMES, 'validity_mask', 'metric_dem')}
    for label, dates in [('a', ['2024-02-29']), ('b', ['2024-02-29', '2024-03-12'])]:
        directory = tmp_path / label
        directory.mkdir()
        for name in paths:
            path = write_doy_map(directory / f'{name}.tif', np.ones((2, 2)), 0)
            paths[name].append(path)
            if name == 'validity_mask':
                with rasterio.open(path, 'r+') as ds:
                    ds.update_tags(acquisition_dates=json.dumps(dates))
    result = mosaic_annual_maps(paths, 2024)
    assert result.acquisition_dates == ('2024-02-29', '2024-03-12')


def test_primary_parity_default_rejects_legacy_missing_provenance(tmp_path):
    from hyp3_glacier_melt.doy_workflow.pipeline import run_doy_pipeline

    for name in DOY_MAP_NAMES:
        write_doy_map(tmp_path / f'{name}_doy_2024.tif', np.ones((2, 2)), 0)
    with pytest.raises(ValueError, match='Re-export source products'):
        run_doy_pipeline([tmp_path], tmp_path / 'output')


def test_year_eligibility_matches_primary_when_ice_has_no_event():
    grid = RasterGrid(rasterio.crs.CRS.from_epsg(32607), from_origin(0, 200, 100, 100), 2, 2)
    values = np.full((2, 2), 100, dtype='float32')
    no_event = np.full((2, 2), np.nan, dtype='float32')
    mask = np.ones((2, 2), bool)
    product = AnnualDoyMaps(2024, grid, values, no_event, values, values, mask)
    assert usable_onset_years([product], mask) == {2024}


def test_external_dem_matches_primary_gdal_values(tmp_path):
    from hyp3_glacier_melt.doy_workflow.pipeline import load_dem_on_grid
    from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import read_metric_dem

    path = write_doy_map(tmp_path / 'dem.tif', np.array([[100.4, 200.6], [300.5, np.nan]]), 0)
    grid = RasterGrid(rasterio.crs.CRS.from_epsg(32607), from_origin(0, 200, 100, 100), 2, 2)
    _, expected = read_metric_dem(str(path), 'EPSG:32607', grid.bounds, 100, 100)
    values = load_dem_on_grid(path, grid)
    np.testing.assert_array_equal(values[np.isfinite(values)], expected[expected != -32768])
    np.testing.assert_allclose(values, [[100, 201], [300, np.nan]], equal_nan=True)
