"""glacier melt processing."""

import logging
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import re

from hyp3_glacier_melt.product import package_product
from hyp3_glacier_melt.melt_pipeline import run_melt_pipeline
from hyp3_glacier_melt.config import MeltConfig
from hyp3_glacier_melt.paths import MeltPaths



log = logging.getLogger(__name__)


def process_glacier_melt(
    datacube: str | None = None,
    output_root: str | None = None,
    rgi_root: str | None = None,
    rgi_shapefile: str | None = None,
) -> Path:
    """
    Run the glacier melt pipeline.

    If a datacube path is supplied, use it directly.
    Otherwise, build the datacube from ASF/HyP3 first.
    """
    config = MeltConfig()
    cwd = Path.cwd()

    output_root = output_root or str(cwd / "output")
    rgi_root = rgi_root or os.environ.get("RGI_ROOT") or str(cwd / "Glaciers")
    rgi_shapefile = rgi_shapefile or os.environ.get("RGI_SHAPEFILE") or str(
        Path(rgi_root) / "RGI2000-v7.0-G-01_alaska" / "RGI2000-v7.0-G-01_alaska.shp"
    )

    if datacube is not None:
        datacube_path = Path(datacube)
        log.info("Running melt pipeline on existing datacube: %s", datacube_path)
    else:
        log.info("No datacube provided; building one from ASF/HyP3 first")

        from hyp3_glacier_melt.hyp3_datacube.create_datacube import (
            DatacubeBuildConfig,
            build_datacube,
        )

        dc_cfg = DatacubeBuildConfig(
            rgi_shapefile=Path(rgi_shapefile),
            scene_name=config.scene_name,
            epsg_no=config.epsg_no,
            path_frame_dict=config.path_frame_dict,
            direction=None,
            pol=config.pol_str,
            start_date="2017-01-01",
            end_date="2024-12-31",
            out_nc_dir=cwd / "datacubes",
            cache_dir=cwd / "hyp3_cache",
            resample_alg="bilinear",
        )

        datacube_path = build_datacube(dc_cfg)
        log.info("Built datacube: %s", datacube_path)


    final_output_root = Path(output_root or cwd / "output")
    final_output_root.mkdir(parents=True, exist_ok=True)

    source_id = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        datacube_path.stem,
    ).strip("._-")


    product_name = f"HYP3_GLACIER_MELT_{source_id}"

    with TemporaryDirectory(
        prefix=".hyp3-glacier-melt-",
        dir=final_output_root,
    ) as staging_directory:
        staging_paths = MeltPaths(
            rgi_root=rgi_root,
            rgi_shapefile=rgi_shapefile,
            output_root=staging_directory,
        )

        result = run_melt_pipeline(
            datacube_path,
            config,
            staging_paths,
        )

        product_file = package_product(
            result=result,
            product_name=product_name,
            output_root=final_output_root,
            metadata={
                "source_id": source_id,
                "polarization": config.pol_str,
                "pixel_spacing_x_m": config.xres,
                "pixel_spacing_y_m": config.yres,
                "use_spatial_tiling": config.use_spatial_tiling,
            },
        )

    return product_file


# def process_glacier_melt(datacube: str | None = None) -> Path:
#     """
#     Run the glacier melt pipeline.

#     If a datacube path is supplied, use it directly.
#     Otherwise, build the datacube from ASF/HyP3 first.
#     """
#     config = MeltConfig()
#     cwd = os.getcwd()
#     paths = MeltPaths(
#         rgi_root=os.path.join(cwd, "Glaciers"),
#         rgi_shapefile=os.path.join(
#             cwd, "RGI2000-v7.0-G-01_alaska", "RGI2000-v7.0-G-01_alaska.shp"
#         ),
#         output_root=os.path.join(cwd, "output"),
#     )

#     if datacube is not None:
#         datacube_path = Path(datacube)
#         log.info("Running melt pipeline on existing datacube: %s", datacube_path)
#     else:
#         log.info("No datacube provided; building one from ASF/HyP3")

#         dc_cfg = DatacubeBuildConfig(
#             rgi_shapefile=Path(r"C:\Users\jaden\Downloads\Research\RGI2000-v7.0-G-01_alaska\RGI2000-v7.0-G-01_alaska.shp"),
#             scene_name="Kennicott",
#             epsg_no=32607,
#             path_frame_dict={"14": ["387"]},
#             direction=None,
#             pol=None,
#             start_date="2017-01-01",
#             end_date="2024-12-31",
#             out_nc_dir=Path(cwd) / "datacubes",
#             cache_dir=Path(cwd) / "hyp3_cache",
#         )

#         datacube_path = build_datacube(dc_cfg)
#         log.info("Built datacube: %s", datacube_path)

#     output_file = run_melt_pipeline(datacube_path, config, paths)

#     log.info("Pipeline returned: %s", output_file)
#     return Path(output_file)
