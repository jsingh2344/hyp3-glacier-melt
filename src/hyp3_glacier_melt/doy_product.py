"""Objects used to hold annual day-of-year maps."""

import os

import rasterio
from rasterio.transform import from_origin


class AnnualDoyProducts:
    """Hold the intermediate day-of-year maps for one year."""

    def __init__(
        self,
        year,
        melt_onset,
        ice_onset,
        spring_baseline,
        second_melt_onset,
        minimum_backscatter,
    ):
        self.year = year
        self.melt_onset = melt_onset
        self.ice_onset = ice_onset
        self.spring_baseline = spring_baseline
        self.second_melt_onset = second_melt_onset
        self.minimum_backscatter = minimum_backscatter

    def as_dict(self):
        """Return the map names and arrays as a dictionary."""
        return {
            "melt_onset": self.melt_onset,
            "ice_onset": self.ice_onset,
            "spring_baseline": self.spring_baseline,
            "second_melt_onset": self.second_melt_onset,
            "minimum_backscatter": self.minimum_backscatter,
        }


def create_and_write_doy_products(datacube, year, output_dir):
    """Create the annual DOY object and write each map to a GeoTIFF."""
    products = AnnualDoyProducts(
        year=year,
        melt_onset=datacube.melt_onset_doy_maps[year],
        ice_onset=datacube.snowline_post_onset_doy_maps[year],
        spring_baseline=datacube.snowline_onset_doy_maps[year],
        second_melt_onset=datacube.annual_second_onset_map_doy_maps[year],
        minimum_backscatter=datacube.min_dB_doys[year],
    )

    os.makedirs(output_dir, exist_ok=True)

    x_values = datacube.ds.x.values
    y_values = datacube.ds.y.values

    if len(x_values) > 1:
        x_resolution = abs(x_values[1] - x_values[0])
    else:
        x_resolution = datacube.xres

    if len(y_values) > 1:
        y_resolution = abs(y_values[1] - y_values[0])
    else:
        y_resolution = datacube.yres

    left = min(x_values) - x_resolution / 2
    top = max(y_values) + y_resolution / 2
    transform = from_origin(left, top, x_resolution, y_resolution)

    crs = datacube.ds.attrs.get("epsg_str")
    if crs is None:
        crs = datacube.ds.attrs.get("projection")

    output_files = []

    for map_name, array in products.as_dict().items():
        output_path = os.path.join(
            output_dir,
            f"{map_name}_doy_{year}.tif",
        )

        output_array = array.astype("float32")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=output_array.shape[0],
            width=output_array.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=float("nan"),
            compress="deflate",
        ) as output_tiff:
            output_tiff.write(output_array, 1)

        output_files.append(output_path)

    return output_files
