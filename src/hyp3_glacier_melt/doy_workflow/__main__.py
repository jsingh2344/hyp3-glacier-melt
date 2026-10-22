"""Command-line entry point for the DOY-only container."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from hyp3lib.aws import upload_file_to_s3

from hyp3_glacier_melt.doy_workflow.process import process_doy_maps


DEFAULT_WORK_ROOT = Path('/tmp/hyp3-glacier-melt-doy')


def main() -> None:
    """Run DOY-only melt extent and snowline processing."""
    parser = ArgumentParser()
    parser.add_argument(
        '--doy-input',
        action='append',
        required=True,
        help='DOY GeoTIFF, directory of maps/ZIP products, or prior product ZIP.',
    )
    parser.add_argument(
        '--dem', help='Override the embedded source DEM; uses the primary container\'s DEM preparation.',
    )
    parser.add_argument('--start-doy', type=int, default=1)
    parser.add_argument('--end-doy', type=int)
    parser.add_argument(
        '--doy-step', type=int,
        help='Override the source acquisition calendar with a regular DOY interval.',
    )
    parser.add_argument('--output-root', default=str(DEFAULT_WORK_ROOT / 'output'))
    parser.add_argument('--bucket', help='AWS S3 bucket for the final product')
    parser.add_argument('--bucket-prefix', default='', help='S3 prefix for the final product')
    args = parser.parse_args()

    if args.start_doy < 1 or args.start_doy > 366:
        parser.error('--start-doy must be between 1 and 366')
    if args.end_doy is not None and (args.end_doy < 1 or args.end_doy > 366):
        parser.error('--end-doy must be between 1 and 366')
    if args.end_doy is not None and args.end_doy < args.start_doy:
        parser.error('--end-doy must be on or after --start-doy')
    if args.doy_step is not None and args.doy_step < 1:
        parser.error('--doy-step must be at least 1')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )
    product = process_doy_maps(
        doy_inputs=args.doy_input,
        output_root=args.output_root,
        dem=args.dem,
        start_doy=args.start_doy,
        end_doy=args.end_doy,
        doy_step=args.doy_step,
    )
    logging.info('Created DOY-derived product: %s', product)
    if args.bucket:
        upload_file_to_s3(product, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
