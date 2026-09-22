import logging
import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import date
from pathlib import Path

from hyp3lib.aws import upload_file_to_s3

from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.paths import BUNDLED_RGI_SHAPEFILE
from hyp3_glacier_melt.process import process_glacier_melt


#Scratch directory for docker image tmp directory
DEFAULT_WORK_ROOT = Path("/tmp/hyp3-glacier-melt")

#For start/end dates
def iso_date(value: str) -> str:
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ArgumentTypeError(f"Invalid date: {value}; expected YYYY-MM-DD") from exc

    if parsed_date.isoformat() != value:
        raise ArgumentTypeError(f"Invalid date: {value}; expected YYYY-MM-DD")

    return value



def main() -> None:
    parser = ArgumentParser()

    # Optional HyP3/S3 output settings
    parser.add_argument("--bucket", help="AWS S3 bucket for HyP3 to upload final product(s)")
    parser.add_argument("--bucket-prefix", default="", help="S3 bucket prefix for final product(s)")

    # Mode selector:
    # If this is provided, process this cube directly.
    # If this is omitted, build an OPERA cube first.
    parser.add_argument("--datacube", help="Path to existing input datacube .nc file")

    # OPERA build/download auxiliary inputs
    parser.add_argument(
        "--opera-burst-id",
        action="append",
        dest="opera_burst_ids",
        help=(
            "OPERA burst ID to process. Repeat this option for multiple bursts."
        ),
    )
    parser.add_argument(
        "--opera-input-dir",
        default=str(DEFAULT_WORK_ROOT / "opera"),
        help="Directory containing local OPERA GeoTIFF files. Also used as download target if --opera-download-dir is omitted.",
    )
    parser.add_argument(
        "--opera-download-dir",
        help="Optional separate directory where downloaded OPERA files should be written.",
    )
    parser.add_argument(
        "--start-date",
        type=iso_date,
        help="First OPERA acquisition date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=iso_date,
        help="Last OPERA acquisition date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--opera-dem",
        help=(
            "Optional path to a DEM GeoTIFF. If omitted, create a Copernicus "
            "GLO-30 DEM covering the OPERA burst."
        ),
    )
    parser.add_argument(
        "--opera-output-dir",
        default=str(DEFAULT_WORK_ROOT / "datacubes"),
        help="Directory where generated OPERA datacube .nc should be written.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_WORK_ROOT / "output"),
        help="Directory for melt pipeline outputs",
    )

    args = parser.parse_args()
    burst_ids = []

    # HyP3 passes array parameters to the container as one space-separated value.
    # Splitting each value also keeps repeated local --opera-burst-id flags working.
    for value in args.opera_burst_ids or []:
        burst_ids.extend(value.split())

    config = MeltConfig()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    datacubes_to_process = []

    # Existing datacube supplied on CLI.
    if args.datacube:
        if len(burst_ids) > 1:
            parser.error("Only one --opera-burst-id can be used with --datacube")

        burst_id = None
        if burst_ids:
            burst_id = burst_ids[0]

        logging.info("Using existing datacube from CLI: %s", args.datacube)
        datacubes_to_process.append((burst_id, Path(args.datacube)))

    # No datacube supplied, so download and build one cube per burst.
    else:
        logging.info("No --datacube provided; building OPERA datacube(s) first.")

        if not args.start_date:
            parser.error("--start-date is required when --datacube is not provided")
        if not args.end_date:
            parser.error("--end-date is required when --datacube is not provided")
        if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
            parser.error("--start-date must be on or before --end-date")
        if not burst_ids:
            parser.error("--opera-burst-id is required when --datacube is not provided")
        if not args.opera_input_dir:
            parser.error("--opera-input-dir is required when --datacube is not provided")
        if not args.opera_output_dir:
            parser.error("--opera-output-dir is required when --datacube is not provided")

        if config.download_opera_files:
            from hyp3_glacier_melt.hyp3_datacube.download_opera_burst_asf import (
                run_download_opera_burst,
            )

        from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import generate_opera_cube

        multiple_bursts = len(burst_ids) > 1

        for burst_id in burst_ids:
            directory_name = burst_id.replace("-", "_")
            opera_input_dir = Path(args.opera_input_dir)
            opera_output_dir = Path(args.opera_output_dir)

            if multiple_bursts:
                opera_input_dir = opera_input_dir / directory_name
                opera_output_dir = opera_output_dir / directory_name

            cube_input_dir = opera_input_dir

            if config.download_opera_files:
                if args.opera_download_dir:
                    opera_download_dir = Path(args.opera_download_dir)
                    if multiple_bursts:
                        opera_download_dir = opera_download_dir / directory_name
                else:
                    opera_download_dir = opera_input_dir

                download_args = Namespace(
                    opera_burst_id=burst_id,
                    start=args.start_date,
                    end=args.end_date,
                    output_dir=opera_download_dir,
                    processing_level=getattr(config, "opera_processing_level", "RTC"),
                    polarization=config.pol_str,
                    asset_mode=getattr(config, "opera_asset_mode", "cube"),
                    asset_regex=getattr(config, "opera_asset_regex", None),
                    max_results=getattr(config, "opera_max_results", 2000),
                    processes=config.opera_download_processes,
                    overwrite=config.opera_overwrite_downloads,
                    username=os.environ.get("EARTHDATA_USERNAME"),
                    password=os.environ.get("EARTHDATA_PASSWORD"),
                    edl_token=os.environ.get("EARTHDATA_TOKEN"),
                )

                logging.info(
                    "Downloading OPERA files for burst %s into: %s",
                    burst_id,
                    opera_download_dir,
                )
                download_status = run_download_opera_burst(download_args)

                if download_status != 0:
                    raise RuntimeError(
                        f"OPERA download failed for {burst_id} with status code {download_status}"
                    )

                cube_input_dir = opera_download_dir

            generated_datacube = generate_opera_cube(
                opera_input_dir=cube_input_dir,
                dem_path=args.opera_dem,
                rgi_shapefile_path=BUNDLED_RGI_SHAPEFILE,
                out_dir=opera_output_dir,
                polarization=config.pol_str,
                xres=config.xres,
                yres=config.yres,
                resample_alg=config.opera_resample_alg,
                write_db=config.opera_write_db,
                overwrite=config.opera_overwrite_cube,
                opera_burst_id=burst_id,
            )

            datacubes_to_process.append((burst_id, Path(generated_datacube)))
            logging.info("Built OPERA datacube for %s: %s", burst_id, generated_datacube)

        if config.build_only:
            return

    for burst_id, datacube_path in datacubes_to_process:
        logging.info("About to run process_glacier_melt on datacube: %s", datacube_path)

        product_file = process_glacier_melt(
            datacube=str(datacube_path),
            output_root=args.output_root,
            opera_burst_id=burst_id,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        logging.info("process_glacier_melt returned: %s", product_file)

        if args.bucket:
            upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == "__main__":
    main()
