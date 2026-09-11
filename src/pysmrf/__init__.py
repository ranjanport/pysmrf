"""PySMRF: High-Performance Parallel Simple Morphological Filter for LiDAR Point Clouds.

A clean, modular, and optimized Python library implementing the Simple Morphological Filter (SMRF)
algorithm for binary ground/object classification and Digital Elevation Model (DEM) generation.

Academic Citation:
==================
If you use PySMRF in research or industry, please cite the original foundational paper:

    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
    An improved simple morphological filter for producing ground surfaces from LiDAR point clouds.
    ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002

And the software implementation:
    Pingel, T. (2021). SMRF: Simple Morphological Filter for LiDAR ground classification.
    https://github.com/thomaspingel/smrf
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Aman Ranjan; Thomas Pingel (original algorithm)"
__email__ = "er.amanranjan@gmail.com"
__license__ = "MIT"
__doi__ = "10.1016/j.isprsjprs.2012.12.002"

__citation_text__ = (
    "Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). "
    "An improved simple morphological filter for producing ground surfaces from LiDAR point clouds. "
    "ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30. "
    "https://doi.org/10.1016/j.isprsjprs.2012.12.002"
)

__citation_bibtex__ = """@article{pingel2013smrf,
  title={An improved simple morphological filter for producing ground surfaces from LiDAR point clouds},
  author={Pingel, Thomas J. and Clarke, Keith C. and McBride, William A.},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={77},
  pages={21--30},
  year={2013},
  publisher={Elsevier},
  doi={10.1016/j.isprsjprs.2012.12.002},
  url={https://doi.org/10.1016/j.isprsjprs.2012.12.002}
}"""

from .core import SMRF, classify, smrf
from .grid import coords_to_pixel, create_dem, edges_from_IT
from .inpaint import inpaint_dem, inpaint_nans_by_fda, inpaint_nans_by_springs
from .interp import (
    compute_surface_slope,
    interpolate_elevation_and_slope,
    interpolate_surface,
)
from .io import (
    read_las,
    read_point_cloud,
    write_classified_las,
    write_geotiff,
)
from .morphology import (
    disk_footprint,
    progressive_filter,
    tiled_morphological_opening,
)
from .types import FilterConfig, FilterResult
from .visualization import hillshade, pssm

__all__ = [
    "SMRF",
    "FilterConfig",
    "FilterResult",
    "classify",
    "compute_surface_slope",
    "coords_to_pixel",
    "create_dem",
    "disk_footprint",
    "edges_from_IT",
    "hillshade",
    "inpaint_dem",
    "inpaint_nans_by_fda",
    "inpaint_nans_by_springs",
    "interpolate_elevation_and_slope",
    "interpolate_surface",
    "progressive_filter",
    "pssm",
    "read_las",
    "read_point_cloud",
    "smrf",
    "tiled_morphological_opening",
    "write_classified_las",
    "write_geotiff",
    "__citation_bibtex__",
    "__citation_text__",
    "__version__",
]
