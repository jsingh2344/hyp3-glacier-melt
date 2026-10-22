"""Top-level product creation for DOY-only glacier processing."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from hyp3_glacier_melt.doy_workflow.pipeline import run_doy_pipeline
from hyp3_glacier_melt.product import package_product


def process_doy_maps(
    doy_inputs: list[str],
    output_root: str,
    dem: str | None = None,
    start_doy: int = 1,
    end_doy: int | None = None,
    doy_step: int | None = None,
) -> Path:
    """Create a packaged melt/snowline product from DOY map inputs."""
    final_output_root = Path(output_root)
    final_output_root.mkdir(parents=True, exist_ok=True)
    source_names = '_'.join(Path(item).stem for item in doy_inputs)
    source_id = re.sub(r'[^A-Za-z0-9._-]+', '_', source_names).strip('._-')
    source_id = source_id or 'DOY_MAPS'
    product_name = f'HYP3_GLACIER_MELT_FROM_DOY_{source_id}'

    with TemporaryDirectory(prefix='.hyp3-glacier-melt-doy-', dir=final_output_root) as staging:
        csv_dir = Path(staging) / 'csv'
        result = run_doy_pipeline(
            doy_inputs=doy_inputs,
            output_dir=csv_dir,
            dem_path=dem,
            start_doy=start_doy,
            end_doy=end_doy,
            doy_step=doy_step,
        )
        return package_product(
            result=result,
            product_name=product_name,
            output_root=final_output_root,
            metadata={
                'source_id': source_id,
                'source_type': 'annual_doy_maps',
                'doy_inputs': [Path(item).name for item in doy_inputs],
                'start_doy': start_doy,
                'end_doy': end_doy,
                'doy_step': doy_step,
                'evaluation_calendar': 'source_acquisitions' if doy_step is None else 'regular_doy_grid',
            },
        )
