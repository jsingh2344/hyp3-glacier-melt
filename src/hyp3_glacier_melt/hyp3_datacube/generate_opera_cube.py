import math
import re
import warnings
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, ogr, osr
from tqdm import tqdm


gdal.UseExceptions()
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

# ============================================================
# USER INPUTS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

OPERA_BURST_DIR = SCRIPT_DIR / "opera_burst_files"

DEM_PATH = None
OUT_DIR = SCRIPT_DIR / "output_nc"

WRITE_DB = True

# You still need this, because your shown folder structure does not include the RGI shapefile.
RGI_SHAPEFILE_PATH = Path(
    r"C:\Users\jaden\Downloads\Research\Glaciers\RGI2000-v7.0-G-01_alaska\RGI2000-v7.0-G-01_alaska.shp"
)

POLARIZATION = "VH"

XRES = 100.0
YRES = 100.0
RESAMPLE_ALG = "average"
OVERWRITE = True

# ============================================================
# HELPERS
# ============================================================

def convert_backscatter(arr, nodata_value=None, write_db=True):
    arr = arr.astype(np.float32, copy=True)

    if nodata_value is None:
        invalid = ~np.isfinite(arr)
    else:
        invalid = (~np.isfinite(arr)) | (arr == nodata_value)

    if write_db:
        # dB = 10 * log10(linear power)
        positive = (~invalid) & (arr > 0)
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        out[positive] = 10.0 * np.log10(arr[positive])
        return out

    # keep linear scale, but normalize invalid values to NaN
    arr[invalid] = np.nan
    return arr

def return_epsg_str_from_gdal_image(in_gd_img):
    img_srs = in_gd_img.GetSpatialRef()
    if img_srs is None:
        raise RuntimeError("Dataset has no spatial reference.")

    img_srs_dict = in_gd_img.GetSpatialRef().ExportToPROJJSON()
    import json

    img_srs_dict = json.loads(img_srs_dict)
    return f"{img_srs_dict['id']['authority']}:{img_srs_dict['id']['code']}"


def parse_input_paths(opera_burst_dir: Path):
    """
    Scan opera_burst_dir for local OPERA GeoTIFFs.

    Expected:
      opera_burst_dir/
        OPERA_L2_RTC-S1_..._VH.tif
        OPERA_L2_RTC-S1_..._VH.tif
        ...

    DEM is handled separately through DEM_PATH, not from this folder.
    """
    if not opera_burst_dir.exists():
        raise FileNotFoundError(f"Could not find OPERA burst directory: {opera_burst_dir}")

    tif_paths = list(opera_burst_dir.rglob("*.tif")) + list(opera_burst_dir.rglob("*.tiff"))

    sar_paths = [
        p for p in tif_paths
        if re.search(rf"_{POLARIZATION}\.tiff?$", p.name, flags=re.IGNORECASE)
    ]

    return [], sort_paths_stably(sar_paths)


def extract_acquisition_timestamp_from_filename(path: Path):
    m = re.search(r"_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_", path.name)
    if m is None:
        raise ValueError(f"Could not extract acquisition timestamp from filename: {path.name}")
    return m.group(1)


def extract_production_timestamp_from_filename(path: Path):
    m = re.search(r"_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_", path.name)
    if m is None:
        raise ValueError(f"Could not extract production timestamp from filename: {path.name}")
    return m.group(2)


def select_latest_opera_files(sar_paths: list[Path]) -> list[Path]:
    """Select the latest-produced OPERA file for each acquisition timestamp."""
    paths_by_acquisition = {}
    for path in sar_paths:
        path = Path(path)
        acquisition_timestamp = extract_acquisition_timestamp_from_filename(path)
        paths_by_acquisition.setdefault(acquisition_timestamp, []).append(path)

    selected_paths = []
    for acquisition_timestamp, candidates in paths_by_acquisition.items():
        latest_production_timestamp = max(
            extract_production_timestamp_from_filename(path)
            for path in candidates
        )
        latest_candidates = [
            path
            for path in candidates
            if extract_production_timestamp_from_filename(path) == latest_production_timestamp
        ]
        if len(latest_candidates) != 1:
            filenames = ", ".join(sorted(path.name for path in latest_candidates))
            raise ValueError(
                f"Multiple OPERA files for acquisition {acquisition_timestamp} have "
                f"the same latest production timestamp {latest_production_timestamp}: {filenames}"
            )
        selected_paths.append(latest_candidates[0])

    return sorted(selected_paths, key=extract_acquisition_timestamp_from_filename)


def opera_timestamp_to_datetime64(timestamp: str):
    """Convert an OPERA compact UTC timestamp to a NumPy datetime."""
    formatted = (
        f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        f"T{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
    )
    return np.datetime64(formatted, "s")


def extract_opera_burst_id_from_filename(path: Path):
    m = re.search(r"_(T\d+[-_]\d+[-_]IW\d)_", path.name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not extract OPERA burst ID from filename: {path.name}")

    return m.group(1).upper().replace("_", "-")


def normalize_opera_burst_id(value: str) -> str:
    """Return an OPERA burst ID in canonical hyphen-separated form."""
    normalized = str(value).strip().upper().replace("_", "-")
    if re.fullmatch(r"T\d+-\d+-IW\d", normalized) is None:
        raise ValueError(f"Invalid OPERA burst ID: {value}")
    return normalized


def verify_opera_burst_paths(
    sar_paths: list[Path],
    expected_burst_id: str | None = None,
) -> str:
    """Verify that every SAR input belongs to one expected OPERA burst."""
    if not sar_paths:
        raise ValueError("No OPERA SAR paths were provided for burst verification.")

    paths_by_burst = {}
    for path in sar_paths:
        path = Path(path)
        burst_id = normalize_opera_burst_id(extract_opera_burst_id_from_filename(path))
        paths_by_burst.setdefault(burst_id, []).append(path)

    discovered_bursts = sorted(paths_by_burst)
    if len(discovered_bursts) != 1:
        counts = ", ".join(
            f"{burst_id}: {len(paths_by_burst[burst_id])} file(s)"
            for burst_id in discovered_bursts
        )
        raise ValueError(
            "OPERA input directory contains files from multiple bursts: "
            f"{counts}. Use a directory containing only the requested burst."
        )

    discovered_burst_id = discovered_bursts[0]
    if expected_burst_id is not None:
        expected_burst_id = normalize_opera_burst_id(expected_burst_id)
        if discovered_burst_id != expected_burst_id:
            raise ValueError(
                f"Requested OPERA burst {expected_burst_id}, but all "
                f"{len(sar_paths)} discovered SAR file(s) belong to {discovered_burst_id}."
            )

    return discovered_burst_id


def extract_pol_from_filename(path: Path):
    m = re.search(r"_([Vv][Vv]|[Vv][Hh]|[Hh][Hh]|[Hh][Vv])\.tiff?$", path.name)
    if m is None:
        raise ValueError(f"Could not extract polarization from filename: {path.name}")
    return m.group(1).upper()


def sort_paths_stably(paths):
    return sorted(paths, key=lambda p: p.name)


def get_dataset_bounds(ds):
    gt = ds.GetGeoTransform()
    if gt is None:
        raise RuntimeError("Dataset geotransform unknown.")

    cols = ds.RasterXSize
    rows = ds.RasterYSize

    corners_px = [(0, 0), (cols, 0), (0, rows), (cols, rows)]
    xs = []
    ys = []

    for px, py in corners_px:
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        xs.append(x)
        ys.append(y)

    return (min(xs), min(ys), max(xs), max(ys))


def make_srs(epsg_str):
    srs = osr.SpatialReference()
    srs.SetFromUserInput(epsg_str)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def transform_bounds(bounds, src_epsg_str, dst_epsg_str):
    if src_epsg_str == dst_epsg_str:
        return bounds

    src_srs = make_srs(src_epsg_str)
    dst_srs = make_srs(dst_epsg_str)
    transformer = osr.CoordinateTransformation(src_srs, dst_srs)

    min_x, min_y, max_x, max_y = bounds
    corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ]

    out_x = []
    out_y = []

    for x, y in corners:
        tx, ty, _ = transformer.TransformPoint(x, y)
        out_x.append(tx)
        out_y.append(ty)

    return (min(out_x), min(out_y), max(out_x), max(out_y))


def polygon_from_bounds(bounds):
    """Create an OGR polygon from min-x, min-y, max-x, max-y bounds."""
    min_x, min_y, max_x, max_y = bounds
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(min_x, min_y)
    ring.AddPoint(min_x, max_y)
    ring.AddPoint(max_x, max_y)
    ring.AddPoint(max_x, min_y)
    ring.AddPoint(min_x, min_y)

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def source_copernicus_dem(output_path, output_bounds, output_epsg_str):
    """Create a Copernicus GLO-30 DEM covering the cube bounds."""
    from hyp3lib.dem import prepare_dem_geotiff

    geographic_bounds = transform_bounds(
        output_bounds,
        output_epsg_str,
        "EPSG:4326",
    )
    geographic_polygon = polygon_from_bounds(geographic_bounds)
    epsg_code = int(output_epsg_str.split(":")[-1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return prepare_dem_geotiff(
        output_name=output_path,
        geometry=geographic_polygon,
        epsg_code=epsg_code,
        pixel_size=30.0,
        buffer_size_in_degrees=0.01,
        height_above_ellipsoid=False,
    )


def snap_bounds_to_grid(bounds, xres, yres, mode="outer"):
    min_x, min_y, max_x, max_y = bounds

    if mode == "outer":
        # Expands bounds outward: good for union
        min_x = math.floor(min_x / xres) * xres
        min_y = math.floor(min_y / yres) * yres
        max_x = math.ceil(max_x / xres) * xres
        max_y = math.ceil(max_y / yres) * yres

    elif mode == "inner":
        # Shrinks bounds inward: good for intersection
        min_x = math.ceil(min_x / xres) * xres
        min_y = math.ceil(min_y / yres) * yres
        max_x = math.floor(max_x / xres) * xres
        max_y = math.floor(max_y / yres) * yres

    else:
        raise ValueError(f"Unknown snap mode: {mode}")

    if min_x >= max_x or min_y >= max_y:
        raise ValueError(
            f"Snapped bounds are empty: {[min_x, min_y, max_x, max_y]}. "
            "Your rasters may not share a common overlap."
        )

    return [min_x, min_y, max_x, max_y]


def output_bounds_to_projwin(output_bounds):
    min_x, min_y, max_x, max_y = output_bounds
    return (min_x, max_y, max_x, min_y)


def get_output_bounds_from_sar_scenes(sar_paths, epsg_str_for_cube, xres, yres):
    scene_bounds = []

    for p in sar_paths:
        ds = gdal.Open(str(p))
        if ds is None:
            raise FileNotFoundError(f"Could not open SAR raster: {p}")

        src_epsg = return_epsg_str_from_gdal_image(ds)
        bounds = get_dataset_bounds(ds)
        bounds = transform_bounds(bounds, src_epsg, epsg_str_for_cube)
        scene_bounds.append(bounds)
        ds = None

    # INTERSECTION, not union
    min_x = max(b[0] for b in scene_bounds)
    min_y = max(b[1] for b in scene_bounds)
    max_x = min(b[2] for b in scene_bounds)
    max_y = min(b[3] for b in scene_bounds)

    if min_x >= max_x or min_y >= max_y:
        raise ValueError(
            "SAR scenes do not have a common intersection. "
            f"Intersection bounds before snapping: {(min_x, min_y, max_x, max_y)}"
        )

    return snap_bounds_to_grid((min_x, min_y, max_x, max_y), xres, yres, mode="inner")


def add_rgi_int_field(mem_vector_ds):
    layer = mem_vector_ds.GetLayer()
    layer_defn = layer.GetLayerDefn()
    existing = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

    if "rgi_int" not in existing:
        fld_def = ogr.FieldDefn("rgi_int", ogr.OFTInteger)
        layer.CreateField(fld_def)

    layer.ResetReading()
    for feat in layer:
        rgi_id_str = feat.GetField("rgi_id")
        if rgi_id_str is None:
            continue

        m = re.search(r"(\d+)$", str(rgi_id_str))
        if m is None:
            continue

        rgi_int = int(m.group(1))
        feat.SetField("rgi_int", rgi_int)
        layer.SetFeature(feat)


def rasterize_rgi_layers(rgi_shapefile_path, output_bounds, epsg_str_for_cube, xres, yres):
    src_ds = gdal.OpenEx(str(rgi_shapefile_path))
    if src_ds is None:
        raise FileNotFoundError(f"Could not open shapefile: {rgi_shapefile_path}")

    mem_vec = gdal.VectorTranslate(
        "",
        src_ds,
        format="MEM",
        spatFilter=output_bounds,
        spatSRS=epsg_str_for_cube,
        reproject=True,
        dstSRS=epsg_str_for_cube,
    )

    add_rgi_int_field(mem_vec)
    layer = mem_vec.GetLayer()
    layer_name = layer.GetName()

    out_mem_rgi = "/vsimem/raster_rgi_ids.tif"
    gdal.Rasterize(
        out_mem_rgi,
        mem_vec,
        outputType=gdal.GDT_UInt32,
        outputSRS=epsg_str_for_cube,
        xRes=xres,
        yRes=yres,
        outputBounds=output_bounds,
        layers=[layer_name],
        attribute="rgi_int",
    )
    rgi_ds = gdal.Open(out_mem_rgi)
    rgi_arr = rgi_ds.ReadAsArray()
    rgi_ds = None
    gdal.Unlink(out_mem_rgi)

    out_mem_aspect = "/vsimem/raster_aspect.tif"
    gdal.Rasterize(
        out_mem_aspect,
        mem_vec,
        outputType=gdal.GDT_Float32,
        outputSRS=epsg_str_for_cube,
        xRes=xres,
        yRes=yres,
        outputBounds=output_bounds,
        layers=[layer_name],
        attribute="aspect_deg",
    )
    aspect_ds = gdal.Open(out_mem_aspect)
    aspect_arr = aspect_ds.ReadAsArray()
    aspect_ds = None
    gdal.Unlink(out_mem_aspect)

    mem_vec = None
    src_ds = None

    return rgi_arr, aspect_arr


def open_and_resample_to_cube_grid(src_path, epsg_str_for_cube, projwin, output_bounds, xres, yres, resample_alg):
    ds = gdal.Open(str(src_path))
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {src_path}")

    ds_epsg = return_epsg_str_from_gdal_image(ds)

    if ds_epsg == epsg_str_for_cube:
        out = gdal.Translate(
            "",
            ds,
            format="VRT",
            projWin=projwin,
            xRes=xres,
            yRes=yres,
            resampleAlg=resample_alg,
        )
    else:
        out = gdal.Warp(
            "",
            ds,
            format="VRT",
            outputBounds=output_bounds,
            dstSRS=epsg_str_for_cube,
            xRes=xres,
            yRes=yres,
            resampleAlg=resample_alg,
        )

    ds = None
    return out


# ============================================================
# MAIN
# ============================================================

def generate_opera_cube(
    opera_input_dir,
    dem_path,
    rgi_shapefile_path,
    out_dir,
    polarization="VH",
    xres=100.0,
    yres=100.0,
    resample_alg="average",
    write_db=True,
    overwrite=True,
    opera_burst_id=None,
):
    global OPERA_BURST_DIR
    global DEM_PATH
    global RGI_SHAPEFILE_PATH
    global OUT_DIR
    global POLARIZATION
    global XRES
    global YRES
    global RESAMPLE_ALG
    global WRITE_DB
    global OVERWRITE

    OPERA_BURST_DIR = Path(opera_input_dir)
    DEM_PATH = Path(dem_path) if dem_path else None
    RGI_SHAPEFILE_PATH = Path(rgi_shapefile_path)
    OUT_DIR = Path(out_dir)

    POLARIZATION = polarization
    XRES = float(xres)
    YRES = float(yres)
    RESAMPLE_ALG = resample_alg
    WRITE_DB = bool(write_db)
    OVERWRITE = bool(overwrite)

    return run_generate_opera_cube(expected_opera_burst_id=opera_burst_id)


def run_generate_opera_cube(expected_opera_burst_id=None):
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _, sar_paths = parse_input_paths(OPERA_BURST_DIR)

    dem_path = None
    if DEM_PATH is not None:
        if not DEM_PATH.exists():
            raise FileNotFoundError(f"Could not find the requested DEM: {DEM_PATH}")
        dem_path = DEM_PATH

    sar_paths = [p for p in sar_paths if p.exists()]
    if len(sar_paths) == 0:
        raise RuntimeError(
            f"No existing OPERA {POLARIZATION} SAR .tif files found in {OPERA_BURST_DIR} "
        )

    sar_paths = sort_paths_stably(sar_paths)

    opera_burst_id = verify_opera_burst_paths(
        sar_paths,
        expected_burst_id=expected_opera_burst_id,
    )
    discovered_file_count = len(sar_paths)
    sar_paths = select_latest_opera_files(sar_paths)
    superseded_file_count = discovered_file_count - len(sar_paths)
    pol_str = extract_pol_from_filename(sar_paths[0])

    sar_ref_ds = gdal.Open(str(sar_paths[0]))
    if sar_ref_ds is None:
        raise FileNotFoundError(f"Could not open first SAR raster: {sar_paths[0]}")

    epsg_str_for_cube = return_epsg_str_from_gdal_image(sar_ref_ds)
    sar_ref_ds = None

    output_bounds = get_output_bounds_from_sar_scenes(
        sar_paths=sar_paths,
        epsg_str_for_cube=epsg_str_for_cube,
        xres=XRES,
        yres=YRES,
    )
    projwin = output_bounds_to_projwin(output_bounds)

    if dem_path is None:
        dem_path = out_dir / f"copernicus_glo30_{opera_burst_id}_30m.tif"
        if dem_path.exists():
            print(f"Reusing generated Copernicus GLO-30 DEM: {dem_path}")
        else:
            print(f"Creating Copernicus GLO-30 DEM: {dem_path}")
            source_copernicus_dem(
                output_path=dem_path,
                output_bounds=output_bounds,
                output_epsg_str=epsg_str_for_cube,
            )

    out_nc_filename = out_dir / f"{epsg_str_for_cube.split(':')[-1]}_S1_cube_{pol_str}_{opera_burst_id}.nc"
    tmp_nc = out_nc_filename.with_name(out_nc_filename.stem + "_partial.nc")

    if out_nc_filename.exists():
        if OVERWRITE:
            out_nc_filename.unlink()
        else:
            print(f"Found existing file: {out_nc_filename}")
            return

    if tmp_nc.exists():
        tmp_nc.unlink()

    print(f"DEM: {dem_path}")
    print(f"SAR files found: {discovered_file_count}")
    print(f"Latest OPERA files selected: {len(sar_paths)}")
    print(f"Superseded OPERA files excluded: {superseded_file_count}")
    print(f"Polarization: {pol_str}")
    print(f"EPSG for cube: {epsg_str_for_cube}")
    print(f"OPERA burst ID: {opera_burst_id}")
    print(f"projWin (from scenes): {projwin}")
    print(f"outputBounds (from scenes): {output_bounds}")
    print(f"Output netCDF: {out_nc_filename}")

    acquisition_and_tif = {
        extract_acquisition_timestamp_from_filename(path): path
        for path in sar_paths
    }
    acquisition_timestamps = sorted(acquisition_and_tif)
    num_dates = len(acquisition_timestamps)

    print(f"Unique acquisitions: {num_dates}")

    rgi_ind_shape_arr, rgi_aspect_arr = rasterize_rgi_layers(
        RGI_SHAPEFILE_PATH,
        output_bounds,
        epsg_str_for_cube,
        XRES,
        YRES,
    )

    sub_gd_dem = open_and_resample_to_cube_grid(
        dem_path,
        epsg_str_for_cube,
        projwin,
        output_bounds,
        XRES,
        YRES,
        RESAMPLE_ALG,
    )
    DEM_NODATA = np.int16(-32768)

    dem_arr = sub_gd_dem.ReadAsArray().astype(np.float32)

    # Preserve invalid pixels before converting to integer
    dem_invalid = ~np.isfinite(dem_arr)

    # Round to nearest meter and convert to integer
    full_dem_sm = np.rint(dem_arr).astype(np.int16)

    # Set invalid pixels to integer nodata
    full_dem_sm[dem_invalid] = DEM_NODATA

    gt = sub_gd_dem.GetGeoTransform()
    min_x, pix_x_m, _, max_y, _, pix_y_m = gt

    x_vec_sm = (min_x + pix_x_m / 2.0) + (pix_x_m * np.arange(sub_gd_dem.RasterXSize))
    y_vec_sm = (max_y + pix_y_m / 2.0) + (pix_y_m * np.arange(sub_gd_dem.RasterYSize))

    ny, nx = full_dem_sm.shape

    time_vec = np.array(
        [
            opera_timestamp_to_datetime64(timestamp).astype(np.int64)
            for timestamp in acquisition_timestamps
        ],
        dtype=np.int64,
    )

    root = None
    try:
        root = Dataset(tmp_nc, "w", format="NETCDF4")

        root.createDimension("time", num_dates)
        root.createDimension("y", ny)
        root.createDimension("x", nx)

        v_time = root.createVariable("time", "i8", ("time",))
        v_y = root.createVariable("y", "f8", ("y",))
        v_x = root.createVariable("x", "f8", ("x",))

        v_images = root.createVariable(
            "images",
            "f4",
            ("time", "y", "x"),
            zlib=True,
            complevel=2,
            chunksizes=(1, min(512, ny), min(512, nx)),
            fill_value=np.nan,
        )
        v_dem = root.createVariable(
            "dem",
            "i2",
            ("y", "x"),
            zlib=True,
            complevel=2,
            fill_value=DEM_NODATA,
        )
        v_rgi = root.createVariable(
            "rgi_ind_glacier_mask",
            "u4",
            ("y", "x"),
            zlib=True,
            complevel=2,
            fill_value=np.uint32(0),
        )
        v_aspect = root.createVariable(
            "rgi_aspect_arr",
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=2,
            fill_value=np.nan,
        )

        v_time.units = "seconds since 1970-01-01 00:00:00"
        v_time.calendar = "proleptic_gregorian"
        v_y.units = "m"
        v_x.units = "m"

        root.description = f"Sentinel 1 {pol_str} SAR (RTC Gamma processed)"
        root.projection = str(epsg_str_for_cube)
        root.epsg_str = str(epsg_str_for_cube)
        root.setncattr("geotransform", tuple(float(v) for v in gt))

        v_time[:] = time_vec
        v_y[:] = y_vec_sm.astype(np.float64)
        v_x[:] = x_vec_sm.astype(np.float64)

        v_dem[:, :] = full_dem_sm
        v_rgi[:, :] = np.asarray(rgi_ind_shape_arr, dtype=np.uint32)
        v_aspect[:, :] = np.asarray(rgi_aspect_arr, dtype=np.float32)

        root.sync()

        for framecount, acquisition_timestamp in enumerate(
            tqdm(acquisition_timestamps, desc="reading/writing images")
        ):
            tif_path = acquisition_and_tif[acquisition_timestamp]
            sub_gd_img = open_and_resample_to_cube_grid(
                tif_path,
                epsg_str_for_cube,
                projwin,
                output_bounds,
                XRES,
                YRES,
                RESAMPLE_ALG,
            )
            full_img_sm = sub_gd_img.ReadAsArray().astype(np.float32)

            band1 = sub_gd_img.GetRasterBand(1)
            img_nodata_val = band1.GetNoDataValue()
            band1 = None
            sub_gd_img = None

            full_img_sm = convert_backscatter(
                full_img_sm,
                nodata_value=img_nodata_val,
                write_db=WRITE_DB,
            )
            v_images[framecount, :, :] = full_img_sm

            if framecount % 10 == 0:
                root.sync()

        root.sync()
        root.close()
        root = None

        tmp_nc.replace(out_nc_filename)

        print("\nDone.")
        print(f"Saved: {out_nc_filename}")
        print(f"Cube shape: time={num_dates}, y={ny}, x={nx}")

        return out_nc_filename

    except Exception:
        if root is not None:
            try:
                root.close()
            except Exception:
                pass
        try:
            if tmp_nc.exists():
                tmp_nc.unlink()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    run_generate_opera_cube()
