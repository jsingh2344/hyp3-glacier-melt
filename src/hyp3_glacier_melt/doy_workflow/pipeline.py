"""Derive glacier melt extents and snowlines from annual DOY maps."""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.datacube import sar_datacube
from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import read_metric_dem, source_copernicus_dem
from hyp3_glacier_melt.paths import BUNDLED_RGI_SHAPEFILE
from hyp3_glacier_melt.product import MeltRunResult


log = logging.getLogger(__name__)

DOY_MAP_NAMES = (
    'melt_onset',
    'ice_onset',
    'spring_baseline',
    'second_melt_onset',
)
DOY_FILENAME_PATTERN = re.compile(
    rf"^({'|'.join((*DOY_MAP_NAMES, 'validity_mask', 'metric_dem'))})_doy_(\d{{4}})\.tiff?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RasterGrid:
    """A common raster grid shared by one mosaicked set of DOY maps."""

    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    height: int
    width: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the grid bounds in left, bottom, right, top order."""
        return array_bounds(self.height, self.width, self.transform)

    @property
    def pixel_area_m2(self) -> float:
        """Return pixel area, requiring a projected grid with metre units."""
        if not self.crs.is_projected:
            raise ValueError('DOY maps must use a projected CRS')
        return abs(self.transform.a * self.transform.e)


@dataclass(frozen=True)
class AnnualDoyMaps:
    """Mosaicked onset maps for one year."""

    year: int
    grid: RasterGrid
    melt_onset: np.ndarray
    ice_onset: np.ndarray
    spring_baseline: np.ndarray
    second_melt_onset: np.ndarray
    coverage: np.ndarray
    validity: np.ndarray | None = None
    metric_dem: np.ndarray | None = None
    acquisition_dates: tuple[str, ...] | None = None


def _masked_source(
    stack: ExitStack, path: Path, validity_path: Path, dem_path: Path,
    seasonal: bool = False,
) -> rasterio.io.DatasetReader:
    """Mask each source before merging so invalid events cannot win overlaps."""
    source = stack.enter_context(rasterio.open(path))
    validity = stack.enter_context(rasterio.open(validity_path))
    dem = stack.enter_context(rasterio.open(dem_path))
    for companion in (validity, dem):
        if (companion.crs, companion.transform, companion.shape) != (
            source.crs, source.transform, source.shape
        ):
            raise ValueError(f'Companion grid does not match {path}')
    valid_values = validity.read(1, masked=True).filled(0)
    valid = np.isfinite(valid_values) & (valid_values > 0)
    valid &= np.isfinite(dem.read(1, masked=True).filled(np.nan))
    if seasonal and validity.tags().get('analysis_status') == 'insufficient_observations':
        valid[:] = False
    values = source.read(1, masked=True).astype('float32').filled(np.nan)
    if path == validity_path:
        values = np.where(valid, 1.0, np.nan).astype('float32')
    else:
        values[~valid] = np.nan
    memory = stack.enter_context(MemoryFile())
    profile = source.profile.copy()
    profile.update(dtype='float32', nodata=np.nan)
    if not profile.get('tiled', False):
        profile.pop('blockxsize', None)
    with memory.open(**profile) as target:
        target.write(values, 1)
    return stack.enter_context(memory.open())


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting members outside the destination."""
    destination = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f'Unsafe ZIP member in {archive_path}: {member.filename}')
        archive.extractall(destination)


def _extract_doy_archive(archive_path: Path, destination: Path) -> list[Path]:
    """Extract a prior product ZIP and return its recognized DOY maps."""
    _safe_extract(archive_path, destination)
    tiff_paths = list(destination.rglob('*.tif')) + list(destination.rglob('*.tiff'))
    doy_paths = [
        path for path in tiff_paths if DOY_FILENAME_PATTERN.fullmatch(path.name) is not None
    ]
    if not doy_paths:
        raise ValueError(
            f'ZIP product contains no recognized annual DOY maps: {archive_path}'
        )
    return doy_paths


def collect_doy_files(
    inputs: Iterable[str | Path], extraction_root: Path
) -> dict[int, dict[str, list[Path]]]:
    """Discover annual DOY maps in TIFFs, directories, or prior product ZIPs."""
    discovered: dict[int, dict[str, list[Path]]] = {}
    input_paths = [Path(item) for item in inputs]
    if not input_paths:
        raise ValueError('At least one DOY input is required')

    candidates: list[Path] = []
    for index, input_path in enumerate(input_paths):
        if not input_path.exists():
            raise FileNotFoundError(f'DOY input does not exist: {input_path}')
        if input_path.is_dir():
            archive_paths = sorted(input_path.rglob('*.zip'))
            if archive_paths:
                for archive_index, archive_path in enumerate(archive_paths):
                    extract_dir = extraction_root / f'input_{index}_archive_{archive_index}'
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    candidates.extend(_extract_doy_archive(archive_path, extract_dir))
            else:
                candidates.extend(input_path.rglob('*.tif'))
                candidates.extend(input_path.rglob('*.tiff'))
        elif input_path.suffix.lower() == '.zip':
            extract_dir = extraction_root / f'input_{index}'
            extract_dir.mkdir(parents=True, exist_ok=True)
            candidates.extend(_extract_doy_archive(input_path, extract_dir))
        elif input_path.suffix.lower() in {'.tif', '.tiff'}:
            candidates.append(input_path)
        else:
            raise ValueError(
                f'DOY inputs must be TIFFs, directories, or ZIP products: {input_path}'
            )

    for candidate in sorted(set(candidates)):
        match = DOY_FILENAME_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        map_name = match.group(1).lower()
        year = int(match.group(2))
        discovered.setdefault(year, {}).setdefault(map_name, []).append(candidate)

    if not discovered:
        raise ValueError('No recognized annual DOY map files were found')

    for year, maps in discovered.items():
        missing = sorted(set(DOY_MAP_NAMES) - set(maps))
        if missing:
            raise ValueError(f'Missing DOY maps for {year}: {", ".join(missing)}')

    return discovered


def _validate_source_grids(paths: Iterable[Path]) -> None:
    """Validate that source maps have compatible projected grids."""
    reference_crs = None
    reference_resolution = None
    reference_transform = None
    for path in paths:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f'DOY map must contain exactly one band: {path}')
            if dataset.crs is None or not dataset.crs.is_projected:
                raise ValueError(f'DOY map must have a projected CRS: {path}')
            if not np.allclose((dataset.transform.b, dataset.transform.d), (0, 0)):
                raise ValueError(f'DOY maps must use north-up grids: {path}')
            resolution = tuple(abs(value) for value in dataset.res)
            if reference_crs is None:
                reference_crs = dataset.crs
                reference_resolution = resolution
                reference_transform = dataset.transform
            elif dataset.crs != reference_crs:
                raise ValueError(f'DOY maps use different CRSs: {path}')
            elif not np.allclose(resolution, reference_resolution):
                raise ValueError(f'DOY maps use different pixel sizes: {path}')
            else:
                assert reference_transform is not None
                x_offset = (dataset.transform.c - reference_transform.c) / resolution[0]
                y_offset = (dataset.transform.f - reference_transform.f) / resolution[1]
                if not np.allclose((x_offset, y_offset), np.round((x_offset, y_offset))):
                    raise ValueError(f'DOY maps are not aligned to the same pixel grid: {path}')


def mosaic_annual_maps(paths_by_name: dict[str, list[Path]], year: int) -> AnnualDoyMaps:
    """Mosaic all maps for a year, keeping the earliest finite onset per pixel."""
    all_paths = [path for paths in paths_by_name.values() for path in paths]
    _validate_source_grids(all_paths)

    arrays: dict[str, np.ndarray] = {}
    grid: RasterGrid | None = None
    common_coverage = None
    companions = ('validity_mask', 'metric_dem')
    explicit_validity = any(name in paths_by_name for name in companions)
    grouped = {
        name: {path.parent: path for path in paths}
        for name, paths in paths_by_name.items()
    }
    if explicit_validity:
        parents = set(grouped['melt_onset'])
        for name in (*DOY_MAP_NAMES, *companions):
            if name not in grouped or set(grouped[name]) != parents:
                raise ValueError(f'Every source needs all event maps, validity_mask and metric_dem for {year}')
            if len(grouped[name]) != len(paths_by_name[name]):
                raise ValueError(f'Duplicate {name} maps in one source for {year}')
    acquisition_dates = set()
    complete_calendar = explicit_validity
    has_skipped_source = False
    for path in paths_by_name.get('validity_mask', []):
        with rasterio.open(path) as dataset:
            encoded_dates = dataset.tags().get('acquisition_dates')
            has_skipped_source |= dataset.tags().get('analysis_status') == 'insufficient_observations'
        if encoded_dates is None:
            complete_calendar = False
            continue
        dates = json.loads(encoded_dates)
        if not isinstance(dates, list) or not dates:
            raise ValueError(f'Invalid acquisition dates in {path}')
        for value in dates:
            parsed = date.fromisoformat(value)
            if parsed.year != year:
                raise ValueError(f'Acquisition date outside {year} in {path}')
            acquisition_dates.add(parsed.isoformat())
    names = (*DOY_MAP_NAMES, *companions) if explicit_validity else DOY_MAP_NAMES
    for map_name in names:
        with ExitStack() as stack:
            if explicit_validity:
                datasets = [
                    _masked_source(
                        stack, path, grouped['validity_mask'][path.parent],
                        grouped['metric_dem'][path.parent],
                    )
                    for path in paths_by_name[map_name]
                ]
                # DEM priority is geographic, then content-based, independent of input order.
                datasets.sort(key=lambda ds: (
                    tuple(ds.bounds), hashlib.sha256(ds.read(1).tobytes()).hexdigest()
                ))
            else:
                datasets = [stack.enter_context(rasterio.open(path)) for path in paths_by_name[map_name]]
            mosaic, transform = merge(
                datasets,
                nodata=np.nan,
                dtype='float32',
                method='first' if map_name == 'metric_dem' else 'min',
            )
            array = mosaic[0]
            current_grid = RasterGrid(
                crs=datasets[0].crs,
                transform=transform,
                height=array.shape[0],
                width=array.shape[1],
            )
            footprints = []
            for dataset in datasets:
                left, bottom, right, top = dataset.bounds
                footprints.append(
                    (
                        {
                            'type': 'Polygon',
                            'coordinates': [[
                                (left, bottom),
                                (left, top),
                                (right, top),
                                (right, bottom),
                                (left, bottom),
                            ]],
                        },
                        1,
                    )
                )
            map_coverage = rasterize(
                footprints,
                out_shape=array.shape,
                transform=transform,
                fill=0,
                dtype='uint8',
            ).astype(bool)

        if grid is None:
            grid = current_grid
        elif (
            current_grid.crs != grid.crs
            or current_grid.transform != grid.transform
            or current_grid.height != grid.height
            or current_grid.width != grid.width
        ):
            raise ValueError(f'DOY map types do not resolve to the same grid for {year}')
        arrays[map_name] = array
        common_coverage = (
            map_coverage
            if common_coverage is None
            else common_coverage & map_coverage
        )

    assert grid is not None
    assert common_coverage is not None
    if has_skipped_source:
        # Preserve the shared SAR validity/DEM for metric parity, but measure
        # this year's coverage using only sources with seasonal observations.
        with ExitStack() as stack:
            sources = [
                _masked_source(stack, path, path, grouped['metric_dem'][path.parent], seasonal=True)
                for path in paths_by_name['validity_mask']
            ]
            seasonal_validity, _ = merge(
                sources, bounds=grid.bounds, nodata=np.nan, dtype='float32', method='min',
            )
            common_coverage &= np.isfinite(seasonal_validity[0])
    validity = arrays.pop('validity_mask', None)
    metric_dem = arrays.pop('metric_dem', None)
    return AnnualDoyMaps(
        year=year, grid=grid, coverage=common_coverage,
        validity=np.isfinite(validity) if validity is not None else None,
        metric_dem=metric_dem, **arrays,
        acquisition_dates=tuple(sorted(acquisition_dates)) if complete_calendar else None,
    )


def load_dem_on_grid(dem_path: Path, grid: RasterGrid) -> np.ndarray:
    """Use the primary container's exact GDAL resampling and integer conversion."""
    with rasterio.open(dem_path) as source:
        if source.crs is None:
            raise ValueError(f'DEM has no CRS: {dem_path}')
    dataset, values = read_metric_dem(
        str(dem_path), grid.crs.to_string(), grid.bounds,
        abs(grid.transform.a), abs(grid.transform.e),
    )
    if values.shape != (grid.height, grid.width) or not np.allclose(
        dataset.GetGeoTransform(), grid.transform.to_gdal(), rtol=0, atol=1e-7
    ):
        raise ValueError('Primary DEM preparation does not match the DOY grid')
    destination = values.astype('float32')
    destination[values == -32768] = np.nan
    return destination


def _evaluation_doys(year: int, start_doy: int, end_doy: int | None, doy_step: int) -> list[int]:
    days_in_year = 366 if calendar.isleap(year) else 365
    final_doy = days_in_year if end_doy is None else min(end_doy, days_in_year)
    if not 1 <= start_doy <= days_in_year:
        raise ValueError(f'start_doy must be between 1 and {days_in_year} for {year}')
    if final_doy < start_doy:
        raise ValueError('end_doy must be greater than or equal to start_doy')
    if doy_step < 1:
        raise ValueError('doy_step must be at least 1')
    return list(range(start_doy, final_doy + 1, doy_step))


def _rgi_id_column(glaciers: gpd.GeoDataFrame) -> str:
    for candidate in ('rgi_id', 'RGIId', 'rgiid'):
        if candidate in glaciers.columns:
            return candidate
    raise ValueError('RGI shapefile does not contain an RGI ID column')


def _area_column(glaciers: gpd.GeoDataFrame) -> str:
    for candidate in ('area_km2', 'Area', 'area'):
        if candidate in glaciers.columns:
            return candidate
    raise ValueError('RGI shapefile does not contain an area column')


def glaciers_on_grid(
    rgi_shapefile: Path, grid: RasterGrid, minimum_area_km2: float
) -> list[tuple[str, int, float, np.ndarray]]:
    """Rasterize qualifying RGI glaciers that intersect a DOY grid."""
    glaciers = gpd.read_file(rgi_shapefile).to_crs(grid.crs)
    id_column = _rgi_id_column(glaciers)
    area_column = _area_column(glaciers)
    left, bottom, right, top = grid.bounds
    glaciers = glaciers.cx[left:right, bottom:top]
    glaciers = glaciers[glaciers[area_column] > minimum_area_km2]

    output = []
    for row in glaciers.itertuples(index=False):
        row_data = row._asdict()
        rgi_id = str(row_data[id_column])
        short_id_match = re.search(r'(\d{2})-(\d{5})$', rgi_id)
        short_id = (
            f'{short_id_match.group(1)}.{short_id_match.group(2)}'
            if short_id_match
            else rgi_id
        )
        glacier_number_match = re.search(r'(\d{5})$', short_id)
        if glacier_number_match is None:
            raise ValueError(f'Cannot derive glacier number from RGI ID: {rgi_id}')
        glacier_number = int(glacier_number_match.group(1))
        mask = rasterize(
            [(row_data['geometry'], 1)],
            out_shape=(grid.height, grid.width),
            transform=grid.transform,
            fill=0,
            dtype='uint8',
        ).astype(bool)
        output.append((short_id, glacier_number, float(row_data[area_column]), mask))
    return output


def onset_datacube(
    annual_maps: list[AnnualDoyMaps],
    dem: np.ndarray,
    glacier_records: list[tuple[str, int, float, np.ndarray]],
    config: MeltConfig,
    start_doy: int,
    end_doy: int | None,
    doy_step: int | None,
) -> sar_datacube:
    """Populate the existing datacube interface with stitched onset products."""
    if not annual_maps:
        raise ValueError('No annual DOY maps were supplied')
    grid = annual_maps[0].grid
    datacube = object.__new__(sar_datacube)

    validity = annual_maps[0].validity
    for item in annual_maps:
        if (item.validity is None) != (validity is None) or (
            validity is not None and not np.array_equal(item.validity, validity)
        ):
            raise ValueError('The metric adapter requires the same explicit validity mask for every year')
    good_pixels = np.isfinite(dem)
    if validity is not None:
        good_pixels &= validity

    def valid_events(values: np.ndarray) -> np.ndarray:
        return np.where(good_pixels, values, np.nan)

    datacube.melt_onset_doy_maps = {item.year: valid_events(item.melt_onset) for item in annual_maps}
    datacube.snowline_post_onset_doy_maps = {item.year: valid_events(item.ice_onset) for item in annual_maps}
    datacube.snowline_onset_doy_maps = {item.year: valid_events(item.spring_baseline) for item in annual_maps}
    datacube.annual_second_onset_map_doy_maps = {
        item.year: valid_events(item.second_melt_onset) for item in annual_maps
    }

    dates = []
    doys = []
    years = []
    for item in annual_maps:
        if doy_step is None and item.acquisition_dates is not None:
            evaluation_doys = [date.fromisoformat(value).timetuple().tm_yday for value in item.acquisition_dates]
            allowed_doys = set(_evaluation_doys(item.year, start_doy, end_doy, 1))
            evaluation_doys = [doy for doy in evaluation_doys if doy in allowed_doys]
        else:
            evaluation_doys = _evaluation_doys(
                item.year, start_doy, end_doy, 1 if doy_step is None else doy_step
            )
        for doy in evaluation_doys:
            dates.append(np.datetime64(date(item.year, 1, 1) + timedelta(days=doy - 1)))
            doys.append(doy)
            years.append(item.year)
    datacube.dates = np.array(dates)
    datacube.doys = doys
    datacube.years = years

    datacube.dem = dem
    datacube.mask_values = np.zeros((grid.height, grid.width), dtype=np.uint32)
    for _, glacier_number, _, mask in glacier_records:
        datacube.mask_values[mask] = glacier_number
    datacube.mask_good_pixels = np.where(good_pixels, 1.0, np.nan)
    datacube.xres = abs(grid.transform.a)
    datacube.yres = abs(grid.transform.e)
    datacube.bin_size = config.bin_size
    datacube.area_bin_size = config.area_bin_size
    datacube.allmelt_threshold = config.allmelt_threshold
    datacube.allmelt_pixels = config.allmelt_pixels

    for attribute in (
        'glac_bounds',
        'glac_mask',
        'glac_mask_good_pixels',
        'glac_dem',
        'glac_bins',
        'glac_bins_center',
        'glac_area_bins',
        'glac_area_bins_center',
    ):
        setattr(datacube, attribute, {})
    return datacube


def prepare_glacier_spatial_state(
    datacube: sar_datacube,
    glacier_number: int,
    area_km2: float,
) -> None:
    """Prepare only the spatial state required by the existing onset metrics."""
    rows, columns = np.where(datacube.mask_values == glacier_number)
    if rows.size == 0:
        raise ValueError(f'Glacier {glacier_number} has no pixels on the stitched grid')

    xmin = rows.min() - 1
    xmax = rows.max() + 1
    ymin = columns.min() - 1
    ymax = columns.max() + 1
    datacube.glac_bounds[glacier_number] = {
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
    }

    mask_values = datacube.mask_values[xmin : xmax + 1, ymin : ymax + 1]
    glacier_mask = mask_values.astype(np.float64, copy=True)
    glacier_mask[glacier_mask != glacier_number] = np.nan
    glacier_mask[glacier_mask > 0] = 1
    datacube.glac_mask[glacier_number] = glacier_mask
    good_pixels = datacube.mask_good_pixels[xmin : xmax + 1, ymin : ymax + 1]
    datacube.glac_mask_good_pixels[glacier_number] = good_pixels

    glacier_dem = datacube.dem[xmin : xmax + 1, ymin : ymax + 1].astype(np.float64)
    glacier_dem = glacier_dem * glacier_mask * good_pixels
    datacube.glac_dem[glacier_number] = glacier_dem

    bin_min = int(np.floor(np.nanmin(glacier_dem) / datacube.bin_size) * datacube.bin_size)
    bin_max = int(np.ceil(np.nanmax(glacier_dem) / datacube.bin_size) * datacube.bin_size)
    bins = np.arange(bin_min, bin_max + datacube.bin_size, datacube.bin_size)
    bins_center = np.arange(
        bin_min + datacube.bin_size / 2,
        bin_max,
        datacube.bin_size,
    ).astype(int)
    _, bins = np.histogram(glacier_dem, bins=bins)

    if datacube.area_bin_size == 'variable':
        minimum_bin_area = area_km2 * 1e6 / 100
        area_bin_size = (
            np.ceil(minimum_bin_area / (datacube.xres * datacube.yres))
            * datacube.xres
            * datacube.yres
        )
        datacube.area_bin_size = min(max(area_bin_size, 100_000), 2_000_000)

    pixel_area = datacube.xres * datacube.yres
    if datacube.area_bin_size % pixel_area != 0:
        raise ValueError(
            f'area_bin_size is incompatible with DEM resolution ({datacube.xres} m)'
        )
    pixels_per_area_bin = int(datacube.area_bin_size / pixel_area)
    sorted_dem = np.sort(glacier_dem[np.isfinite(glacier_dem)])
    area_bins = sorted_dem[::pixels_per_area_bin]
    if area_bins[-1] != sorted_dem[-1]:
        area_bins = np.append(area_bins, sorted_dem[-1])

    datacube.glac_bins[glacier_number] = bins
    datacube.glac_bins_center[glacier_number] = bins_center
    datacube.glac_area_bins[glacier_number] = area_bins
    datacube.glac_area_bins_center[glacier_number] = 0.5 * (
        area_bins[:-1] + area_bins[1:]
    )


def usable_onset_years(
    annual_products: list[AnnualDoyMaps],
    glacier_mask: np.ndarray,
    minimum_valid_fraction: float = 0.01,
) -> set[int]:
    """Return years that satisfy the legacy onset-map validity checks."""
    total_pixels = np.count_nonzero(glacier_mask)
    if total_pixels == 0:
        return set()

    usable = set()
    for products in annual_products:
        melt_fraction = (
            np.count_nonzero(np.isfinite(products.melt_onset[glacier_mask]))
            / total_pixels
        )
        # The primary metric indexes both flattened onset arrays with the melt
        # validity mask; its year eligibility therefore depends on that mask.
        if melt_fraction >= minimum_valid_fraction:
            usable.add(products.year)
    return usable


def run_doy_pipeline(
    doy_inputs: Iterable[str | Path],
    output_dir: str | Path,
    dem_path: str | Path | None = None,
    rgi_shapefile: str | Path = BUNDLED_RGI_SHAPEFILE,
    start_doy: int = 1,
    end_doy: int | None = None,
    doy_step: int | None = None,
    config: MeltConfig | None = None,
) -> MeltRunResult:
    """Run DOY-only melt/snowline processing and write per-glacier CSVs."""
    config = config or MeltConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_glaciers: dict[int, str] = {}
    total_evaluations = 0

    with TemporaryDirectory(prefix='.doy-inputs-', dir=output_dir) as temporary_directory:
        temporary_root = Path(temporary_directory)
        discovered = collect_doy_files(doy_inputs, temporary_root / 'extracted')
        annual_products = [
            mosaic_annual_maps(discovered[year], year) for year in sorted(discovered)
        ]
        reference_grid = annual_products[0].grid
        for annual_maps in annual_products[1:]:
            if annual_maps.grid != reference_grid:
                raise ValueError('All annual DOY maps must resolve to the same grid')

        if doy_step is None and any(
            item.validity is None or item.metric_dem is None or item.acquisition_dates is None
            for item in annual_products
        ):
            raise ValueError(
                'Matching the primary workflow requires source validity, metric DEM and acquisition dates. '
                'Re-export source products with the updated primary container, or explicitly select '
                '--doy-step for legacy approximate processing.'
            )

        embedded_dem = annual_products[0].metric_dem
        if embedded_dem is not None and any(
            not np.array_equal(item.metric_dem, embedded_dem, equal_nan=True)
            for item in annual_products
        ):
            raise ValueError('Embedded metric DEM must be identical for every year')
        if annual_products[0].validity is None:
            log.warning(
                'Legacy DOY inputs have no explicit SAR validity mask; exact metric reproduction is unavailable'
            )
        if dem_path is None and embedded_dem is not None:
            dem = embedded_dem
        elif dem_path is None:
            generated_dem = temporary_root / 'copernicus_dem.tif'
            source_copernicus_dem(
                generated_dem,
                reference_grid.bounds,
                reference_grid.crs.to_string(),
            )
            dem = load_dem_on_grid(generated_dem, reference_grid)
        else:
            selected_dem = Path(dem_path)
            if not selected_dem.is_file():
                raise FileNotFoundError(f'DEM does not exist: {selected_dem}')
            dem = load_dem_on_grid(selected_dem, reference_grid)
        glacier_records = glaciers_on_grid(
            Path(rgi_shapefile), reference_grid, config.min_glac_area_km2
        )
        datacube = onset_datacube(
            annual_products,
            dem,
            glacier_records,
            config,
            start_doy,
            end_doy,
            doy_step,
        )
        total_evaluations = len(datacube.dates)
        if total_evaluations == 0:
            raise ValueError('No acquisition dates fall inside the requested DOY range')

        csv_files = []
        skipped_without_onsets = []
        for rgi_id, glacier_number, area_km2, glacier_mask in glacier_records:
            if (
                np.any(glacier_mask[0, :])
                or np.any(glacier_mask[-1, :])
                or np.any(glacier_mask[:, 0])
                or np.any(glacier_mask[:, -1])
            ):
                continue
            coverage = (
                np.count_nonzero(glacier_mask) * reference_grid.pixel_area_m2 / 1e6
            ) / area_km2
            if coverage <= config.min_area_frac:
                continue
            if annual_products[0].validity is not None:
                valid_area_fraction = (
                    np.count_nonzero(glacier_mask & np.isfinite(datacube.mask_good_pixels))
                    * reference_grid.pixel_area_m2 / 1e6 / area_km2
                )
                if valid_area_fraction <= config.min_area_frac:
                    continue
            usable_years = usable_onset_years(annual_products, glacier_mask)
            usable_years &= {
                products.year for products in annual_products
                if np.count_nonzero(glacier_mask & products.coverage)
                / np.count_nonzero(glacier_mask) >= config.min_area_frac
            }
            usable_years &= set(datacube.years)
            if not usable_years:
                skipped_without_onsets.append(rgi_id)
                continue

            onset_attributes = (
                'melt_onset_doy_maps',
                'snowline_post_onset_doy_maps',
                'snowline_onset_doy_maps',
                'annual_second_onset_map_doy_maps',
            )
            complete_onset_maps = {
                attribute: getattr(datacube, attribute) for attribute in onset_attributes
            }
            try:
                for attribute, maps in complete_onset_maps.items():
                    setattr(
                        datacube,
                        attribute,
                        {year: array for year, array in maps.items() if year in usable_years},
                    )
                datacube.area_bin_size = config.area_bin_size
                prepare_glacier_spatial_state(datacube, glacier_number, area_km2)
                csv_file = datacube.generate_elevs_from_onsets(
                    glacier_number,
                    output_dir,
                    rgi_id=rgi_id,
                    doy_step=doy_step or 1,
                    min_valid_frac=0.01,
                )
                csv_files.append(csv_file)
            except Exception as exc:
                failed_glaciers[glacier_number] = f'{type(exc).__name__}: {exc}'
                log.exception('DOY processing failed for glacier %s', rgi_id)
            finally:
                for attribute, maps in complete_onset_maps.items():
                    setattr(datacube, attribute, maps)

        if skipped_without_onsets:
            log.info(
                'Skipped %s glaciers with no year meeting the existing onset-validity checks',
                len(skipped_without_onsets),
            )

    return MeltRunResult(
        scenes=total_evaluations,
        csv_files=csv_files,
        failed_glacnos=failed_glaciers,
        tiff_files=[],
        skipped_years={
            item.year: 'No source has usable seasonal coverage'
            for item in annual_products if not item.coverage.any()
        },
    )
