import os
from dataclasses import dataclass


BUNDLED_RGI_ROOT = "/opt/rgi"
BUNDLED_RGI_SHAPEFILE = (
    "/opt/rgi/RGI2000-v7.0-G-01_alaska/"
    "RGI2000-v7.0-G-01_alaska.shp"
)


@dataclass
class MeltPaths:
    rgi_root: str
    output_root: str

    @property
    def csv_dir(self):
        return os.path.join(self.output_root, "csv_hyp3")

    @property
    def onset_dir(self):
        return os.path.join(self.output_root, "onset_maps_hyp3")
