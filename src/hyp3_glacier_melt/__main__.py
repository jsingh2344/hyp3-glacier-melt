import logging
from argparse import ArgumentParser

from hyp3lib.aws import upload_file_to_s3
from hyp3_glacier_melt.process import process_glacier_melt


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--bucket', help='AWS S3 bucket for HyP3 to upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--username', help='Earthdata Login username')
    parser.add_argument('--password', help='Earthdata Login password')
    parser.add_argument('--datacube', help='Path to input datacube .nc file')
    parser.add_argument('--build-datacube', action='store_true', help='Build datacube from ASF/HyP3 before processing')
    parser.add_argument('--output-root', help='Directory for outputs')
    parser.add_argument('--rgi-root', help='Directory containing RGI data folders')
    parser.add_argument('--rgi-shapefile', help='Path to RGI shapefile')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    datacube_arg = None if args.build_datacube else args.datacube

    product_file = process_glacier_melt(
        datacube=datacube_arg,
        output_root=args.output_root,
        rgi_root=args.rgi_root,
        rgi_shapefile=args.rgi_shapefile,
    )

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()