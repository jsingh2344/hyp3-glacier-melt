from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class MeltPaths:
    rgi_root: str
    rgi_shapefile: str
    output_root: str

    @property
    def csv_dir(self):
        return os.path.join(self.output_root, "csv_hyp3")

    @property
    def onset_dir(self):
        return os.path.join(self.output_root, "onset_maps_hyp3")