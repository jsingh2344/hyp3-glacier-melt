"""Product result and packaging utilities."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


@dataclass
class MeltRunResult:

    scenes: int
    csv_files: list[Path]
    failed_glacnos: dict[int, str]
    tiff_files: list[Path]



def package_product(
    result: MeltRunResult,
    product_name,
    output_root,
    metadata,
) -> Path:
    """Package the explicitly listed outputs from one pipeline run.

    Args:
        result: Structured result returned by the melt pipeline.
        product_name: Name used for both the ZIP and its internal root directory.
        output_root: Directory in which to create the ZIP.
        metadata: Additional product metadata. Derived run fields take precedence.

    Returns:
        Path to the validated product ZIP.

    Raises:
        FileExistsError: If the destination ZIP already exists.
        RuntimeError: If the run did not produce any CSV files or ZIP validation fails.
        ValueError: If an output file list is invalid.
        FileNotFoundError: If a listed output file does not exist.
    """

    csv_files = [Path(csv_file) for csv_file in result.csv_files]
    if not csv_files:
        raise RuntimeError('Cannot package a run that produced no CSV files')

    csv_names = [csv_file.name for csv_file in csv_files]
    duplicate_names = sorted({name for name in csv_names if csv_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f'Duplicate CSV filenames cannot be packaged: {duplicate_names}')

    for csv_file in csv_files:
        if not csv_file.is_file():
            raise FileNotFoundError(f'Generated CSV does not exist: {csv_file}')
        if csv_file.suffix.lower() != '.csv':
            raise ValueError(f'Expected a CSV file, got: {csv_file}')

    tiff_files = [Path(tiff_file) for tiff_file in result.tiff_files]
    tiff_names = [tiff_file.name for tiff_file in tiff_files]
    duplicate_tiff_names = sorted(
        {name for name in tiff_names if tiff_names.count(name) > 1}
    )
    if duplicate_tiff_names:
        raise ValueError(
            f'Duplicate TIFF filenames cannot be packaged: {duplicate_tiff_names}'
        )

    for tiff_file in tiff_files:
        if not tiff_file.is_file():
            raise FileNotFoundError(f'Generated TIFF does not exist: {tiff_file}')
        if tiff_file.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError(f'Expected a TIFF file, got: {tiff_file}')

    packaged_files = list(csv_names)
    for tiff_name in tiff_names:
        packaged_files.append(f'doy_maps/{tiff_name}')

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    product_path = output_root / f'{product_name}.zip'
    if product_path.exists():
        raise FileExistsError(f'Product already exists: {product_path}')

    status = 'complete' if not result.failed_glacnos else 'partial'
    product_metadata = dict(metadata)
    product_metadata.update(
        {
            'product_name': product_name,
            'scene_count': result.scenes,
            'successful_glacier_count': len(csv_files),
            'doy_map_count': len(tiff_files),
            'failed_glacier_count': len(result.failed_glacnos),
            'failed_glacnos': result.failed_glacnos,
            'status': status,
            'files': sorted(packaged_files),
        }
    )
    metadata_text = json.dumps(product_metadata, indent=2, sort_keys=True)
    readme_text = (
        f'{product_name}\n'
        f'Status: {status}\n'
        f'Scenes processed: {result.scenes}\n'
        f'Successful glaciers: {len(csv_files)}\n'
        f'DOY maps: {len(tiff_files)}\n'
        f'Failed glaciers: {len(result.failed_glacnos)}\n\n'
        'Each CSV contains the melt and snowline time series for one glacier.\n'
        'The doy_maps directory contains the annual intermediate GeoTIFFs.\n'
        'See product_metadata.json for run metadata and recorded glacier failures.\n'
    )

    try:
        with ZipFile(product_path, mode='x', compression=ZIP_DEFLATED) as archive:
            for csv_file in sorted(csv_files, key=lambda path: path.name):
                archive.write(csv_file, arcname=f'{product_name}/{csv_file.name}')
            for tiff_file in sorted(tiff_files, key=lambda path: path.name):
                archive.write(
                    tiff_file,
                    arcname=f'{product_name}/doy_maps/{tiff_file.name}',
                )
            archive.writestr(f'{product_name}/product_metadata.json', metadata_text)
            archive.writestr(f'{product_name}/README.txt', readme_text)

        with ZipFile(product_path, mode='r') as archive:
            corrupt_file = archive.testzip()
        if corrupt_file is not None:
            raise RuntimeError(f'Product ZIP validation failed for {corrupt_file}')
    except FileExistsError:
        raise
    except Exception:
        product_path.unlink(missing_ok=True)
        raise

    return product_path
