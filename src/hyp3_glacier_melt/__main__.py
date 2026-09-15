import logging
import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import date
from pathlib import Path

from hyp3lib.aws import upload_file_to_s3

from hyp3_glacier_melt.config import MeltConfig
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
        help="OPERA burst ID to download/build, e.g. T014_028627_IW2",
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


    # Melt pipeline auxiliary inputs/outputs
    # parser.add_argument("--output-root", help="Directory for melt pipeline outputs") #Switched to using tmp structure
    parser.add_argument(
        "--rgi-root",
        default=os.environ.get("RGI_ROOT"),
        help="Directory containing RGI data folders; defaults to RGI_ROOT if set",
    )
    parser.add_argument(
        "--rgi-shapefile",
        default=os.environ.get("RGI_SHAPEFILE"),
        help="Path to RGI shapefile; defaults to RGI_SHAPEFILE if set",
    )

    args = parser.parse_args()
    config = MeltConfig()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    # -------------------------------------------------------------------------
    # MODE 1: Existing datacube supplied on CLI
    # -------------------------------------------------------------------------
    if args.datacube:
        logging.info("Using existing datacube from CLI: %s", args.datacube)
        datacube_arg = args.datacube

    # -------------------------------------------------------------------------
    # MODE 2: No datacube supplied, so build OPERA cube first
    # -------------------------------------------------------------------------
    else:
        logging.info("No --datacube provided; building OPERA datacube first.")


        #Check dates are valid
        if not args.start_date:
            parser.error("--start-date is required when --datacube is not provided")
        if not args.end_date:
            parser.error("--end-date is required when --datacube is not provided")
        if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
            parser.error("--start-date must be on or before --end-date")
        if not args.opera_burst_id:
            raise ValueError("--opera-burst-id is required when --datacube is not provided")
        if not args.opera_input_dir:
            raise ValueError("--opera-input-dir is required when --datacube is not provided")
        if not args.opera_output_dir:
            raise ValueError("--opera-output-dir is required when --datacube is not provided")
        if not args.rgi_shapefile:
            raise ValueError("--rgi-shapefile is required when --datacube is not provided")
        
        

        # ---------------------------------------------------------------------
        # Optional OPERA download stage, controlled by config.py
        # ---------------------------------------------------------------------
        if config.download_opera_files:
            opera_download_dir = args.opera_download_dir or args.opera_input_dir

            from hyp3_glacier_melt.hyp3_datacube.download_opera_burst_asf import (
                run_download_opera_burst,
            )

            download_args = Namespace(
                opera_burst_id=args.opera_burst_id,
                start=args.start_date, #Switched from config control to command line
                end=args.end_date,
                output_dir=Path(opera_download_dir),
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

            logging.info("Downloading OPERA files for burst %s into: %s", args.opera_burst_id, opera_download_dir)
            download_status = run_download_opera_burst(download_args)

            if download_status != 0:
                raise RuntimeError(f"OPERA download failed with status code {download_status}")

            # Make sure the cube builder scans the actual download location.
            args.opera_input_dir = str(opera_download_dir)

        # ---------------------------------------------------------------------
        # Build OPERA cube
        # ---------------------------------------------------------------------
        from hyp3_glacier_melt.hyp3_datacube.generate_opera_cube import generate_opera_cube

        generated_datacube = generate_opera_cube(
            opera_input_dir=args.opera_input_dir,
            dem_path=args.opera_dem,
            rgi_shapefile_path=args.rgi_shapefile,
            out_dir=args.opera_output_dir,
            polarization=config.pol_str,
            xres=config.xres,
            yres=config.yres,
            resample_alg=config.opera_resample_alg,
            write_db=config.opera_write_db,
            overwrite=config.opera_overwrite_cube,
            opera_burst_id=args.opera_burst_id,
        )

        datacube_arg = str(generated_datacube)

        if config.build_only:
            logging.info("Built OPERA datacube: %s", datacube_arg)
            return

    # -------------------------------------------------------------------------
    # Run melt processing
    # -------------------------------------------------------------------------
    logging.info("About to run process_glacier_melt on datacube: %s", datacube_arg)

    product_file = process_glacier_melt(
        datacube=datacube_arg,
        output_root=args.output_root,
        rgi_root=args.rgi_root,
        rgi_shapefile=args.rgi_shapefile,
        opera_burst_id=args.opera_burst_id, #Passing to name products after burst id
        start_date=args.start_date,
        end_date=args.end_date,
    )

    logging.info("process_glacier_melt returned: %s", product_file)

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == "__main__":
    main()
