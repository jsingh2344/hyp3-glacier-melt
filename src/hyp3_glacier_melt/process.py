"""glacier melt processing."""

import logging
from pathlib import Path
import os

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
    rgi_root = rgi_root or str(cwd / "Glaciers")
    rgi_shapefile = rgi_shapefile or str(
        cwd / "RGI2000-v7.0-G-01_alaska" / "RGI2000-v7.0-G-01_alaska.shp"
    )

    paths = MeltPaths(
        rgi_root=rgi_root,
        rgi_shapefile=rgi_shapefile,
        output_root=output_root,
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

    output_file = run_melt_pipeline(datacube_path, config, paths)

    log.info("Pipeline returned: %s", output_file)
    return Path(output_file)


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