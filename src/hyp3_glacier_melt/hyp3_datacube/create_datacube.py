from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import asf_search as asf
from collections import defaultdict
from hyp3_sdk import HyP3
import time
import requests
import zipfile
from osgeo import ogr, osr, gdal
import numpy as np
import pandas as pd
import xarray as xr

import logging

log = logging.getLogger(__name__)


@dataclass
class DatacubeBuildConfig:
    rgi_shapefile: Path
    scene_name: str
    epsg_no: int
    path_frame_dict: dict[str, list[str]]
    direction: str
    pol: str
    start_date: str
    end_date: str
    resample_alg: str

    out_nc_dir: Path
    cache_dir: Path

    xres: float = 100.0
    yres: float = 100.0
    frame_buffer: int = 1


    include_dem: bool = True
    rtc_resolution: int = 30
    rtc_radiometry: str = "gamma0"
    rtc_scale: str = "decibel"
    speckle_filter: bool = False

    def validate(self) -> None:
        # if self.direction.upper() not in {"ASCENDING", "DESCENDING"}:
        #     raise ValueError(f"Invalid direction: {self.direction}")

        # if self.pol.upper() not in {"VV", "VH", "HH", "HV"}:
        #     raise ValueError(f"Invalid polarization: {self.pol}")

        if not self.path_frame_dict:
            raise ValueError("path_frame_dict cannot be empty")

        self.out_nc_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

def read_band_as_array(ds, label: str):
    """
    Read band 1 from a GDAL dataset and log basic stats.
    """
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    log.info("%s nodata: %r", label, nodata)
    log.info("%s dtype: %s", label, arr.dtype)
    log.info("%s shape: %s", label, arr.shape)
    log.info("%s min: %s", label, np.nanmin(arr))
    log.info("%s max: %s", label, np.nanmax(arr))

    return arr, nodata

def derive_projwin_and_epsg_from_dems(dem_datasets):
    """
    Replicate the notebook's projWin logic from a list of GDAL DEM datasets.

    Returns
    -------
    projWin : tuple[int, int, int, int]
        (ulx, uly, lrx, lry)
    output_bounds : tuple[int, int, int, int]
        (min_x, min_y, max_x, max_y)
    epsg_str_for_cube : str
    """

    min_xs, max_xs, min_ys, max_ys = [], [], [], []

    for ds in dem_datasets:
        gt = ds.GetGeoTransform()
        if gt is None:
            raise RuntimeError("geotransform unknown")

        top_left_x = gt[0]
        pixel_width = gt[1]
        top_left_y = gt[3]
        pixel_height = gt[5]

        cols = ds.RasterXSize
        rows = ds.RasterYSize

        min_xs.append(top_left_x)
        max_xs.append(top_left_x + cols * pixel_width)
        max_ys.append(top_left_y)
        min_ys.append(top_left_y + rows * pixel_height)

    projWin = (
        int(np.min(min_xs)),
        int(np.max(max_ys)),
        int(np.max(max_xs)),
        int(np.min(min_ys)),
    )

    output_bounds = (projWin[0], projWin[3], projWin[2], projWin[1])

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(dem_datasets[0].GetProjection())
    epsg_code = spatial_ref.GetAttrValue("AUTHORITY", 1)
    epsg_str_for_cube = f"EPSG:{epsg_code}"

    return projWin, output_bounds, epsg_str_for_cube

def scene_time_from_granule(granule: str) -> np.datetime64:
    """
    Extract acquisition time from a Sentinel-1 granule name.
    Example:
      S1B_IW_SLC__1SDV_20170111T031940_20170111T032007_...
    """
    parts = granule.split("_")
    start_str = parts[5]  # e.g. 20170111T031940
    ts = pd.to_datetime(start_str, format="%Y%m%dT%H%M%S")
    return np.datetime64(ts)

def validate_datacube_schema(ds):
    required_vars = {"images", "dem", "rgi_ind_glacier_mask", "rgi_aspect_arr"}
    required_coords = {"time", "x", "y"}
    required_attrs = {"description", "projection", "epsg_str", "geotransform"}

    missing_vars = required_vars - set(ds.data_vars)
    missing_coords = required_coords - set(ds.coords)
    missing_attrs = required_attrs - set(ds.attrs)

    if missing_vars or missing_coords or missing_attrs:
        raise ValueError(
            f"Schema mismatch:"
            f"\nmissing vars: {missing_vars}"
            f"\nmissing coords: {missing_coords}"
            f"\nmissing attrs: {missing_attrs}"
        )
    
def validate_datacube_schema(ds):
    # --- Expected ---
    required_vars = {
        "images": ("time", "y", "x"),
        "dem": ("y", "x"),
        "rgi_ind_glacier_mask": ("y", "x"),
        "rgi_aspect_arr": ("y", "x"),
    }

    required_coords = {
        "time",
        "x",
        "y",
    }

    required_attrs = {
        "description",
        "projection",
        "epsg_str",
        "geotransform",
    }

    # --- Check variables exist ---
    missing_vars = set(required_vars) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")

    # --- Check variable dimensions ---
    for var, expected_dims in required_vars.items():
        actual_dims = ds[var].dims
        if actual_dims != expected_dims:
            raise ValueError(
                f"{var} dims mismatch: expected {expected_dims}, got {actual_dims}"
            )

    # --- Check coordinates ---
    missing_coords = required_coords - set(ds.coords)
    if missing_coords:
        raise ValueError(f"Missing coords: {missing_coords}")

    # --- Check attributes ---
    missing_attrs = required_attrs - set(ds.attrs)
    if missing_attrs:
        raise ValueError(f"Missing attrs: {missing_attrs}")

    print("Datacube schema is valid")

def assemble_datacube_dataset(
    images,
    dem,
    rgi_mask,
    aspect,
    time,
    x,
    y,
    description,
    projection,
    epsg_str,
    geotransform,
):
    ds = xr.Dataset(
        data_vars=dict(
            images=(["time", "y", "x"], images),
            dem=(["y", "x"], dem),
            rgi_ind_glacier_mask=(["y", "x"], rgi_mask),
            rgi_aspect_arr=(["y", "x"], aspect),
        ),
        coords=dict(
            time=time,
            x=x,
            y=y,
        ),
        attrs=dict(
            description=description,
            projection=projection,
            epsg_str=epsg_str,
            geotransform=geotransform,
        ),
    )
    validate_datacube_schema(ds)
    return ds

def build_single_scene_dataset(
    rtc_arr: np.ndarray,
    dem_arr: np.ndarray,
    rgi_ind_glacier_mask: np.ndarray,
    rgi_aspect_arr: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    granule: str,
    epsg_str_for_cube: str,
    pol: str,
    geotransform,
) -> xr.Dataset:
    """
    Build a one-time-step xarray dataset for a single RTC scene.
    """
    time_val = scene_time_from_granule(granule)

    ds = xr.Dataset(
        data_vars=dict(
            images=(["time", "y", "x"], rtc_arr[np.newaxis, :, :]),
            dem=(["y", "x"], dem_arr),
            rgi_ind_glacier_mask=(["y", "x"], rgi_ind_glacier_mask),
            rgi_aspect_arr=(["y", "x"], rgi_aspect_arr),
        ),
        coords=dict(
            time=np.array([time_val]),
            x=x_vec,
            y=y_vec,
        ),
        attrs=dict(
            description=f"Sentinel-1 {pol} SAR (HyP3 RTC)",
            projection=epsg_str_for_cube,
            epsg_str=epsg_str_for_cube,
            geotransform=geotransform,
        ),
    )

    return ds

def open_gdal_from_zip(zip_path: Path, member_name: str):
    """
    Open a file inside a ZIP using GDAL /vsizip/.
    """
    vsi_path = f"/vsizip/{zip_path.as_posix()}/{member_name}"
    ds = gdal.Open(vsi_path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open {vsi_path}")
    return ds

def find_rtc_members(members: list[str], pol: str = "VH") -> tuple[str, str]:
    """
    Return the RTC TIFF member and DEM TIFF member from a HyP3 RTC ZIP member list.
    """
    rtc_member = None
    dem_member = None

    for member in members:
        lower = member.lower()

        if lower.endswith(f"_{pol.lower()}.tif"):
            rtc_member = member

        if lower.endswith("_dem.tif"):
            dem_member = member

    if rtc_member is None:
        raise RuntimeError(f"Could not find {pol} RTC TIFF in ZIP")

    if dem_member is None:
        raise RuntimeError("Could not find DEM TIFF in ZIP")

    return rtc_member, dem_member

def log_zip_contents(zip_path: Path, max_items: int = 50) -> list[str]:
    """
    Log the first few members of a HyP3 RTC ZIP and return the full member list.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    log.info("ZIP contains %d files", len(members))
    for member in members[:max_items]:
        log.info("ZIP member: %s", member)

    return members

def _scene_date(scene: Any) -> str:
    """
    Return scene acquisition date as YYYYMMDD.
    """
    props = getattr(scene, "properties", {}) or {}

    start_time = props.get("startTime")
    if start_time:
        return start_time[:10].replace("-", "")

    scene_name = getattr(scene, "sceneName", None)
    if scene_name:
        parts = scene_name.split("_")
        if len(parts) > 4:
            return parts[4][:8]

    raise ValueError(f"Could not determine date for scene: {scene}")

def _scene_name(scene: Any) -> str:
    """
    Return the granule / scene name.
    """
    scene_name = getattr(scene, "sceneName", None)
    if scene_name:
        return scene_name

    props = getattr(scene, "properties", {}) or {}
    if "sceneName" in props:
        return props["sceneName"]

    raise ValueError(f"Could not determine sceneName for scene: {scene}")

def _scene_granule(scene: Any) -> str:
    """
    Return the Sentinel-1 granule name for HyP3 submission.
    """
    props = getattr(scene, "properties", {}) or {}

    scene_name = props.get("sceneName")
    if scene_name:
        return scene_name

    scene_name = getattr(scene, "sceneName", None)
    if scene_name:
        return scene_name

    raise ValueError(f"Could not determine granule name for scene: {scene}")

def make_hyp3() -> HyP3:
    """
    Create an authenticated HyP3 client.
    Uses .netrc if available, otherwise prompts for credentials.
    """
    return HyP3(prompt="password")


def find_existing_rtc_jobs(hyp3: HyP3, granule: str):
    """
    Return existing RTC jobs whose name matches the granule.
    """
    jobs = hyp3.find_jobs(
        name=granule,
        job_type="RTC_GAMMA",
    )
    return list(jobs)

def search_s1_scenes_for_path_frame(
    path: str,
    frame: str,
    start_date: str,
    end_date: str,
    direction: str,
    frame_buffer: int = 1,
) -> list[Any]:
    """
    Search ASF for Sentinel-1 IW SLC scenes for a given relative orbit (path)
    and frame, with optional +/- frame buffer.
    """
    frame_int = int(frame)
    frame_min = max(0, frame_int - frame_buffer)
    frame_max = frame_int + frame_buffer

    log.info(
        "Searching ASF: path=%s frame=%s frame_range=%s-%s direction=%s start=%s end=%s",
        path,
        frame,
        frame_min,
        frame_max,
        direction,
        start_date,
        end_date,
    )

    results = asf.search(
        dataset=asf.DATASET.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        start=start_date,
        end=end_date,
        relativeOrbit=int(path),
        frame = [int(frame)],
        maxResults=5000,
    )

    results_list = list(results)

    # if results_list:
    #     sample = results_list[0]
    #     props = getattr(sample, "properties", {}) or {}

    #     log.info("Sample scene name: %s", getattr(sample, "sceneName", None))
    #     log.info("Sample property keys: %s", sorted(props.keys()))

    #     for key in [
    #         "pathNumber",
    #         "frameNumber",
    #         "frame",
    #         "asfFrame",
    #         "flightDirection",
    #         "beamModeType",
    #         "startTime",
    #         "sceneName",
    #     ]:
    #         log.info("Sample property %s = %r", key, props.get(key))


    # log.info(
    #     "ASF returned %d scenes for path=%s frame=%s",
    #     len(results_list),
    #     path,
    #     frame,
    # )

    return results_list

def build_scene_index(cfg) -> dict[str, dict[str, list[Any]]]:
    """
    Build:
        {
            path: {
                YYYYMMDD: [scene1, scene2, ...]
            }
        }

    For each path, keep dates where at least the requested number of frames
    are represented.
    """
    scene_index: dict[str, dict[str, list[Any]]] = {}

    for path, frames in cfg.path_frame_dict.items():

        grouped_by_date: dict[str, list[Any]] = defaultdict(list)
        log.info("Building scene index for path %s with frames %s", path, frames)

        for frame in frames:
            results = search_s1_scenes_for_path_frame(
                path=path,
                frame=frame,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                direction=cfg.direction,
                frame_buffer=cfg.frame_buffer,
            )

            for scene in results:
                props = getattr(scene, "properties", {}) or {}

                # Filter direction
                scene_dir = props.get("flightDirection", "")

                if cfg.direction is not None and scene_dir.upper() != cfg.direction.upper():
                    continue

                # Filter beam mode
                if props.get("beamModeType", "") != "IW":
                    continue

                grouped_by_date[_scene_date(scene)].append(scene)

        needed_count = len(frames)
        valid_dates: dict[str, list[Any]] = {}

        for date_str, scenes in grouped_by_date.items():
            deduped: dict[str, Any] = {}
            for scene in scenes:
                deduped[_scene_name(scene)] = scene

            deduped_scenes = sorted(deduped.values(), key=_scene_name)

            if len(deduped_scenes) >= needed_count:
                valid_dates[date_str] = deduped_scenes

        scene_index[path] = dict(sorted(valid_dates.items()))

        log.info(
            "Path %s retained %d valid dates after grouping",
            path,
            len(scene_index[path]),
        )

    return scene_index

def get_or_submit_rtc_job(hyp3: HyP3, granule: str):
    batch = hyp3.submit_rtc_job(
        granule=granule,
        name=f"{granule}_decibel_test",
        include_dem=True,
        resolution=30,
        radiometry="gamma0",
        scale="decibel",
    )

    watched = hyp3.watch(batch)
    jobs = list(watched)

    if not jobs:
        raise RuntimeError(f"No HyP3 job returned for granule {granule}")

    return jobs[0]

def submit_rtc_job_no_wait(hyp3: HyP3, granule: str):
    """
    Submit one RTC job and return immediately without waiting.
    Reuse is disabled so parameter changes (for example scale='decibel')
    actually create a fresh job.
    """
    log.info("Submitting new RTC job (reuse disabled) for granule %s", granule)

    batch = hyp3.submit_rtc_job(
        granule=granule,
        name=f"{granule}_decibel_test",
        include_dem=True,
        resolution=30,
        radiometry="gamma0",
        scale="decibel",
        speckle_filter=False,
    )

    return _first_job_from_batch(batch)

def _first_job_from_batch(batch):
    """
    Extract the first Job from a HyP3 Batch.
    """
    jobs = list(batch)
    if not jobs:
        raise RuntimeError("HyP3 batch contained no jobs")
    return jobs[0]

def refresh_job(hyp3: HyP3, job_id: str):
    """
    Re-fetch a HyP3 job by ID.
    """
    jobs = list(hyp3.find_jobs())
    for job in jobs:
        if getattr(job, "job_id", None) == job_id:
            return job

    raise RuntimeError(f"Could not find HyP3 job with id {job_id}")

def wait_for_job_completion(
    hyp3: HyP3,
    job_id: str,
    poll_seconds: int = 20,
    max_polls: int = 30,
):
    """
    Poll a HyP3 job until it reaches a terminal state.
    """
    for poll_idx in range(max_polls):
        job = refresh_job(hyp3, job_id)
        status = getattr(job, "status_code", None)

        log.info(
            "Poll %d/%d for job %s: status=%s (elapsed ~%d min)",
            poll_idx + 1,
            max_polls,
            job_id,
            status,
            ((poll_idx + 1) * poll_seconds) // 60,
        )

        if status == "SUCCEEDED":
            return job

        if status in {"FAILED", "CANCELED", "EXPIRED"}:
            raise RuntimeError(f"HyP3 job {job_id} ended with status {status}")

        time.sleep(poll_seconds)

    raise TimeoutError(
        f"HyP3 job {job_id} did not complete after {max_polls} polls"
    )

def log_job_outputs(job) -> None:
    """
    Log the output files/URLs attached to a finished HyP3 job.
    """
    log.info("Finished job output inspection:")
    log.info("  job_id=%s", getattr(job, "job_id", None))
    log.info("  name=%s", getattr(job, "name", None))
    log.info("  status=%s", getattr(job, "status_code", None))
    log.info("  files=%r", getattr(job, "files", None))
    log.info("  browse_images=%r", getattr(job, "browse_images", None))
    log.info("  logs=%r", getattr(job, "logs", None))


def download_job_zip(job, out_dir: Path) -> Path:
    """
    Download the RTC ZIP file for a finished HyP3 job.
    """
    files = getattr(job, "files", None)
    if not files:
        raise RuntimeError("No files found on HyP3 job")

    file_info = files[0]
    url = file_info.get("url")
    filename = file_info.get("filename")

    if not url or not filename:
        raise RuntimeError("Missing URL or filename in job files")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists():
        log.info("ZIP already exists, skipping download: %s", out_path)
        return out_path

    log.info("Downloading RTC ZIP: %s", filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    log.info("Downloaded to: %s", out_path)

    return out_path

def build_rgi_mask_from_shapefile(
    rgi_shapefile: Path,
    xsize: int,
    ysize: int,
    geotransform,
    epsg_no: int,
) -> np.ndarray:
    """
    Rasterize the RGI shapefile onto the current raster grid using an integer
    glacier ID field derived from rgi_id, producing a uint32 glacier mask.
    """

    shp = ogr.Open(str(rgi_shapefile), update=1)
    if shp is None:
        raise RuntimeError(f"Could not open shapefile: {rgi_shapefile}")

    layer = shp.GetLayer()

    # Create rgi_int field if it does not already exist
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

    if "rgi_int" not in field_names:
        fld_def = ogr.FieldDefn("rgi_int", ogr.OFTInteger)
        layer.CreateField(fld_def)

        for feat in layer:
            rgi_id_str = feat.GetField("rgi_id")
            rgi_int = int(rgi_id_str.split("-")[-1])
            feat.SetField("rgi_int", rgi_int)
            layer.SetFeature(feat)

        layer.ResetReading()

    mem_driver = gdal.GetDriverByName("MEM")
    out_ds = mem_driver.Create("", xsize, ysize, 1, gdal.GDT_UInt32)
    out_ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_no)
    out_ds.SetProjection(srs.ExportToWkt())

    band = out_ds.GetRasterBand(1)
    band.Fill(0)

    err = gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=["ATTRIBUTE=rgi_int"],
    )
    if err != 0:
        raise RuntimeError("RasterizeLayer failed for glacier mask")

    arr = band.ReadAsArray().astype(np.uint32)
    return arr

def build_xy_vectors(ds):
    """
    Build x and y coordinate vectors from a GDAL dataset geotransform.
    """
    gt = ds.GetGeoTransform()
    x0, dx, _, y0, _, dy = gt

    nx = ds.RasterXSize
    ny = ds.RasterYSize

    x = x0 + dx * (0.5 + np.arange(nx))
    y = y0 + dy * (0.5 + np.arange(ny))

    log.info("x vector shape: %s", x.shape)
    log.info("y vector shape: %s", y.shape)
    log.info("x range: %s to %s", x[0], x[-1])
    log.info("y range: %s to %s", y[0], y[-1])

    return x, y

def write_dataset_to_netcdf(ds: xr.Dataset, out_path: Path) -> Path:
    """
    Write an xarray dataset to NetCDF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Writing NetCDF to: %s", out_path)

    ds.to_netcdf(out_path)

    log.info("Finished writing NetCDF")

    return out_path

def log_dataset_geo(ds, label: str) -> None:
    """
    Log projection and geotransform for a GDAL dataset.
    """
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()

    log.info("%s projection: %s", label, projection)
    log.info("%s geotransform: %s", label, geotransform)

def build_rgi_aspect_from_shapefile(
    rgi_shapefile: Path,
    xsize: int,
    ysize: int,
    geotransform,
    epsg_no: int,
) -> np.ndarray:
    """
    Rasterize the shapefile's aspect_deg field onto the current raster grid.
    """
    

    shp = ogr.Open(str(rgi_shapefile), update=0)
    if shp is None:
        raise RuntimeError(f"Could not open shapefile: {rgi_shapefile}")

    layer = shp.GetLayer()

    mem_driver = gdal.GetDriverByName("MEM")
    out_ds = mem_driver.Create("", xsize, ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_no)
    out_ds.SetProjection(srs.ExportToWkt())

    band = out_ds.GetRasterBand(1)
    band.Fill(0)

    err = gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=["ATTRIBUTE=aspect_deg"],
    )
    if err != 0:
        raise RuntimeError("RasterizeLayer failed for aspect")

    arr = band.ReadAsArray().astype(np.float32)
    return arr

def get_output_bounds_from_gdal_dataset(src_ds) -> tuple[float, float, float, float]:
    """
    Return output_bounds in GDAL Warp order:
    (min_x, min_y, max_x, max_y)
    """
    gt = src_ds.GetGeoTransform()
    if gt is None:
        raise RuntimeError("Dataset geotransform is missing")

    min_x = gt[0]
    max_y = gt[3]
    max_x = gt[0] + src_ds.RasterXSize * gt[1]
    min_y = gt[3] + src_ds.RasterYSize * gt[5]

    return (min_x, min_y, max_x, max_y)


def get_output_bounds_in_target_crs(src_ds, epsg_str: str) -> tuple[float, float, float, float]:
    """
    Compute dataset bounds in the target CRS without hardcoding them.
    Returns bounds as (min_x, min_y, max_x, max_y).
    """

    vrt = gdal.Warp(
        "",
        src_ds,
        format="VRT",
        dstSRS=epsg_str,
    )
    if vrt is None:
        raise RuntimeError("Failed to warp dataset for bounds calculation")

    gt = vrt.GetGeoTransform()
    min_x = gt[0]
    max_y = gt[3]
    max_x = gt[0] + vrt.RasterXSize * gt[1]
    min_y = gt[3] + vrt.RasterYSize * gt[5]

    return (min_x, min_y, max_x, max_y)

def build_datacube(cfg: DatacubeBuildConfig) -> Path:
    """
    Main entrypoint for recreating the original 01_create_datacube workflow,
    but sourcing imagery through ASF search + HyP3 instead of local files.

    This function preserves the notebook's original sequence:
      1. discover scenes by path/frame/date
      2. obtain RTC products + DEM
      3. mosaic/reproject/subset
      4. assemble xarray datacube
      5. write NetCDF
    """
    cfg.validate()

    log.info("Starting datacube build for scene: %s", cfg.scene_name)

    # Step placeholders; these will be filled in one by one
    scene_index = build_scene_index(cfg)

    granules = []

    for path, date_dict in scene_index.items():
        for date_str, scenes in date_dict.items():
            if scenes:
                granules.append(_scene_granule(scenes[0]))
            if len(granules) >= 35:
                break
        if len(granules) >= 35:
            break

    # print("cfg.path_frame_dict =", cfg.path_frame_dict)
    # print("cfg.direction =", cfg.direction)
    # print("cfg.start_date =", cfg.start_date)
    # print("cfg.end_date =", cfg.end_date)
    # print("scene_index =", {k: list(v.keys())[:5] for k, v in scene_index.items()})
    # print("scene_index counts =", {k: len(v) for k, v in scene_index.items()})

    if not granules:
        raise RuntimeError("No scenes found")

    log.info("Selected %d granules: %s", len(granules), granules)

    hyp3 = make_hyp3()

    # Derive cube grid once from the first granule's DEM
    first_granule = granules[0]
    first_job = submit_rtc_job_no_wait(hyp3, first_granule)
    first_finished_job = wait_for_job_completion(
        hyp3,
        first_job.job_id,
        poll_seconds=30,
        max_polls=80,
    )

    first_zip_path = download_job_zip(first_finished_job, cfg.cache_dir)
    first_members = log_zip_contents(first_zip_path)
    _, first_dem_member = find_rtc_members(first_members, pol=cfg.pol)
    first_dem_ds = open_gdal_from_zip(first_zip_path, first_dem_member)

    projWin, output_bounds, epsg_str_for_cube = derive_projwin_and_epsg_from_dems([first_dem_ds])

    print("GLOBAL projWin:", projWin)
    print("GLOBAL output_bounds:", output_bounds)
    print("GLOBAL epsg_str_for_cube:", epsg_str_for_cube)

    epsg_no_for_cube = int(epsg_str_for_cube.split(":")[-1])

    datasets = []

    for granule in granules:
        log.info("Processing granule: %s", granule)

        rtc_job = submit_rtc_job_no_wait(hyp3, granule)
        finished_job = wait_for_job_completion(
            hyp3,
            rtc_job.job_id,
            poll_seconds=30,
            max_polls=80,
        )

        zip_path = download_job_zip(finished_job, cfg.cache_dir)
        members = log_zip_contents(zip_path)

        rtc_member, dem_member = find_rtc_members(members, pol=cfg.pol)

        rtc_ds = open_gdal_from_zip(zip_path, rtc_member)
        dem_ds = open_gdal_from_zip(zip_path, dem_member)
        

        dem_arr, _, _, _ = warp_to_target_grid(
            dem_ds,
            output_bounds=output_bounds,
            epsg_str=epsg_str_for_cube,
            xres=cfg.xres,
            yres=cfg.yres,
            resample_alg=cfg.resample_alg,
        )

        dem_arr = dem_arr.astype(np.int16)

        rtc_arr, geotransform, x_vec, y_vec = warp_to_target_grid(
            rtc_ds,
            output_bounds=output_bounds,
            epsg_str=epsg_str_for_cube,
            xres=cfg.xres,
            yres=cfg.yres,
            resample_alg=cfg.resample_alg,
        )

        print("rtc geotransform:", rtc_ds.GetGeoTransform())
        print("rtc size:", rtc_ds.RasterYSize, rtc_ds.RasterXSize)

        epsg_no_for_cube = int(epsg_str_for_cube.split(":")[-1])

        rgi_ind_glacier_mask = build_rgi_mask_from_shapefile(
            rgi_shapefile=cfg.rgi_shapefile,
            xsize=rtc_arr.shape[1],
            ysize=rtc_arr.shape[0],
            geotransform=geotransform,
            epsg_no=epsg_no_for_cube,
        )

        rgi_aspect_arr = build_rgi_aspect_from_shapefile(
            rgi_shapefile=cfg.rgi_shapefile,
            xsize=rtc_arr.shape[1],
            ysize=rtc_arr.shape[0],
            geotransform=geotransform,
            epsg_no=epsg_no_for_cube,
        )

        ds = build_single_scene_dataset(
            rtc_arr=rtc_arr,
            dem_arr=dem_arr,
            rgi_ind_glacier_mask=rgi_ind_glacier_mask,
            rgi_aspect_arr=rgi_aspect_arr,
            x_vec=x_vec,
            y_vec=y_vec,
            granule=granule,
            epsg_str_for_cube=epsg_str_for_cube,
            pol=cfg.pol,
            geotransform=geotransform,
        )

        print("\nSINGLE-SCENE DATASET:")
        print(ds)
        print("dem dims before concat:", ds["dem"].dims)

        datasets.append(ds)
    
    ds_combined = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    log.info("Combined dataset dims: %s", ds_combined.dims)

    epsg_no_for_cube = int(epsg_str_for_cube.split(":")[-1])
    out_path = cfg.out_nc_dir / f"{cfg.scene_name}_{epsg_no_for_cube}_S1_cube_{cfg.pol}.nc"
    write_dataset_to_netcdf(ds_combined, out_path)

    product_index: dict[str, Any] = {}
    total_dates = sum(len(date_dict) for date_dict in scene_index.values())
    log.info("Built scene index with %d total retained dates", total_dates)

    for path, date_dict in scene_index.items():
        preview_dates = list(date_dict.keys())[:5]
        log.info("Path %s first retained dates: %s", path, preview_dates)
    output_nc = cfg.out_nc_dir / f"{cfg.scene_name}_{epsg_no_for_cube}_S1_cube_{cfg.pol}.nc"

    log.info("Scene index placeholder created")
    log.info("Product index placeholder created")
    log.info("Output will be written to: %s", output_nc)

    return output_nc

def warp_to_target_grid(
    src_ds,
    output_bounds,
    epsg_str,
    xres,
    yres,
    resample_alg,
):
    """
    Warp a GDAL dataset to the target grid using the same logic as the original workflow.
    """

    dst = gdal.Warp(
        "",
        src_ds,
        format="VRT",
        outputBounds=output_bounds,
        dstSRS=epsg_str,
        xRes=xres,
        yRes=yres,
        resampleAlg=resample_alg,
    )

    arr = dst.ReadAsArray()

    gt = dst.GetGeoTransform()

    x_vec = (gt[0] + gt[1]/2.0) + (gt[1] * np.arange(dst.RasterXSize))
    y_vec = (gt[3] + gt[5]/2.0) + (gt[5] * np.arange(dst.RasterYSize))

    return arr, gt, x_vec, y_vec

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    
    cfg = DatacubeBuildConfig(
        scene_name="Gulkana",
        epsg_no=32607,
        path_frame_dict={"14": ["387"]},
        direction=None,
        pol='VH',
        start_date="2017-01-01",
        end_date="2024-12-31",
        out_nc_dir=Path("output_nc"),
        cache_dir=Path("hyp3_cache"),
        rgi_shapefile=Path(r"C:\Users\jaden\Downloads\Research\Glaciers\RGI2000-v7.0-G-01_alaska\RGI2000-v7.0-G-01_alaska.shp"),
        resample_alg="bilinear",
    )

    print("cfg.xres =", cfg.xres)
    print("cfg.yres =", cfg.yres)
    print("cfg.rtc_resolution =", cfg.rtc_resolution)

    out_path = build_datacube(cfg)
    print(f"Returned output path: {out_path}")


if __name__ == "__main__":
    main()