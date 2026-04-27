import math
import re
import os
from pathlib import Path
import warnings

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

DEM_PATH = SCRIPT_DIR / "dem.tif"
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


def extract_date_from_filename(path: Path):
    m = re.search(r"_(\d{8})T\d{6}Z_", path.name)
    if m is None:
        raise ValueError(f"Could not extract acquisition date from filename: {path.name}")
    return m.group(1)


def extract_opera_burst_id_from_filename(path: Path):
    m = re.search(r"_(T\d+[-_]\d+[-_]IW\d)_", path.name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not extract OPERA burst ID from filename: {path.name}")

    return m.group(1).upper().replace("_", "-")

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


def merge_same_date_arrays(arrays, nodata_value):
    merged = arrays[0].copy()

    if nodata_value is None:
        for arr in arrays[1:]:
            merged = np.where(np.isnan(merged), arr, merged)
        return merged

    for arr in arrays[1:]:
        merged[merged == nodata_value] = arr[merged == nodata_value]

    return merged


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
    DEM_PATH = Path(dem_path)
    RGI_SHAPEFILE_PATH = Path(rgi_shapefile_path)
    OUT_DIR = Path(out_dir)

    POLARIZATION = polarization
    XRES = float(xres)
    YRES = float(yres)
    RESAMPLE_ALG = resample_alg
    WRITE_DB = bool(write_db)
    OVERWRITE = bool(overwrite)

    return run_generate_opera_cube()


def run_generate_opera_cube():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    listed_dem_candidates, sar_paths = parse_input_paths(OPERA_BURST_DIR)

    if DEM_PATH.exists():
        dem_path = DEM_PATH
    elif listed_dem_candidates:
        dem_path = listed_dem_candidates[0]
    else:
        raise FileNotFoundError(
            f"Could not find DEM. Expected {DEM_PATH}, or dem.tif inside {OPERA_BURST_DIR}."
        )

    sar_paths = [p for p in sar_paths if p.exists()]
    if len(sar_paths) == 0:
        raise RuntimeError(
            f"No existing OPERA {POLARIZATION} SAR .tif files found in {OPERA_BURST_DIR} "
        )

    sar_paths = sort_paths_stably(sar_paths)

    opera_burst_id = extract_opera_burst_id_from_filename(sar_paths[0])
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
    print(f"SAR files found: {len(sar_paths)}")
    print(f"Polarization: {pol_str}")
    print(f"EPSG for cube: {epsg_str_for_cube}")
    print(f"OPERA burst ID: {opera_burst_id}")
    print(f"projWin (from scenes): {projwin}")
    print(f"outputBounds (from scenes): {output_bounds}")
    print(f"Output netCDF: {out_nc_filename}")

    indate_and_tifs = {}
    for p in sar_paths:
        acq_date = extract_date_from_filename(p)
        indate_and_tifs.setdefault(acq_date, []).append(p)

    unique_dates = sorted(indate_and_tifs.keys())
    num_dates = len(unique_dates)

    print(f"Unique acquisition dates: {num_dates}")

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
            np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}", "s").astype("datetime64[s]").astype(np.int64)
            for d in unique_dates
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

        for framecount, framedate in enumerate(tqdm(unique_dates, desc="reading/writing images")):
            same_date_files = sort_paths_stably(indate_and_tifs[framedate])
            sub_gd_imgs = []

            for tif_path in same_date_files:
                sub_gd_img = open_and_resample_to_cube_grid(
                    tif_path,
                    epsg_str_for_cube,
                    projwin,
                    output_bounds,
                    XRES,
                    YRES,
                    RESAMPLE_ALG,
                )
                sub_gd_imgs.append(sub_gd_img)

            img_sms = [ds.ReadAsArray() for ds in sub_gd_imgs]

            band1 = sub_gd_imgs[0].GetRasterBand(1)
            img_nodata_val = band1.GetNoDataValue()
            band1 = None
            sub_gd_imgs = None

            full_img_sm = merge_same_date_arrays(img_sms, img_nodata_val).astype(np.float32)
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