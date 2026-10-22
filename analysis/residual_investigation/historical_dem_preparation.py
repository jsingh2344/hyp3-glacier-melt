def prepare_dem_geotiff(
    output_name: Path,
    geometry: ogr.Geometry,
    epsg_code: int,
    pixel_size: float,
    buffer_size_in_degrees: float = 0.0,
    height_above_ellipsoid: bool = False,
) -> Path:
    """Create a DEM mosaic GeoTIFF covering a given geometry.

    The DEM mosaic is assembled from the Copernicus GLO-30 Public DEM and reprojected into the given EPSG code and pixel
    size. The extent of the output GeoTIFF is the extent of the buffered input geometry.

    Args:
        output_name: Path for the output GeoTIFF
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a DEM mosaic. Must be a POLYGON with
          longitude coordinates between -200 and +200 degrees.
        epsg_code: EPSG code for the output GeoTIFF projection.
        pixel_size: Pixel size for the DEM in units of the DEM's projection
        buffer_size_in_degrees: Extent of the output geotiff will be the extent of the buffered input geometry.
        height_above_ellipsoid:
          If False, output pixel values will be meters above mean sea level.
          If True, the output pixel values will be meters above the ellipsoid.
          Only supported for geometries between -180 and +180 degrees longitude (i.e. geometries not crossing the antimeridian).
    """
    if geometry.GetGeometryName() != 'POLYGON':
        raise DemError(f'{geometry.GetGeometryName()} geometry is invalid; only POLYGON is supported.')

    buffered_geometry = geometry.Buffer(buffer_size_in_degrees)
    minx, maxx, miny, maxy = buffered_geometry.GetEnvelope()
    if not (-200 <= minx <= maxx <= 200):
        raise DemError(f'Extent of buffered geometry ({minx} - {maxx}) is not between -200 and +200 degrees longitude.')

    if not (-180 <= minx <= maxx <= 180) and height_above_ellipsoid:
        raise DemError(
            'height_above_ellipsoid is not supported for buffered geometries with coordinates outside -180 to +180 degrees longitude.'
        )

    if not _intersects_dem(geometry):
        raise DemError(f'Copernicus GLO-30 Public DEM does not intersect this geometry: {geometry}')

    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dem_file_paths = _get_dem_file_paths(buffered_geometry)

            dem_vrt = temp_path / 'dem.vrt'
            gdal.BuildVRT(str(dem_vrt), dem_file_paths)
            # This is required to ensure the VRT is treated as a point dataset
            vrt_ds = gdal.Open(str(dem_vrt), gdal.GA_Update)
            vrt_ds.SetMetadataItem('AREA_OR_POINT', 'Point')
            vrt_ds = None
            gdal.Warp(
                str(output_name),
                str(dem_vrt),
                dstSRS=f'EPSG:{epsg_code}',
                outputBoundsSRS='EPSG:4326',
                outputBounds=[minx, miny, maxx, maxy],
                xRes=pixel_size,
                yRes=pixel_size,
                targetAlignedPixels=True,
                resampleAlg='cubic',
                multithread=True,
                format='GTiff',
            )

    if height_above_ellipsoid:
        _convert_to_height_above_ellipsoid(output_name)

    return output_name

