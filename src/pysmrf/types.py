"""Data types, configuration structures, and result containers for PySMRF.

Cites:
    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
    An improved simple morphological filter for producing ground surfaces from
    LiDAR point clouds. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class FilterConfig:
    """Configuration parameters for the Simple Morphological Filter (SMRF).

    Parameters
    ----------
    cellsize : float
        Grid resolution in real-world units (meters or feet). Default is 1.0.
    windows : Union[int, list[int], np.ndarray]
        Filter window radii sequence in pixels. If an int is provided (e.g. 5),
        radii 1..5 are tested sequentially. Default is 5.
    slope_threshold : float
        dz/dx slope threshold for ground/object differentiation (e.g., 0.15 = 15%).
        Default is 0.15.
    elevation_threshold : float
        Base elevation distance tolerance in map units from the provisional DTM.
        Points within this distance are classified as ground. Default is 0.5.
    elevation_scaler : float
        Multiplier scaling elevation threshold on steeper slopes:
        required_diff = elevation_threshold + elevation_scaler * slope.
        Default is 1.25. Set to 0 to disable slope scaling.
    low_filter_slope : float
        dz/dx threshold for identifying low outlier points. Default is 5.0 (500%).
    low_outlier_fill : bool
        If True, re-interpolates cells identified as low outliers prior to morphological
        filtering. Default is False.
    workers : int
        Number of parallel worker threads/processes. -1 uses all available CPU cores.
        Default is 1.
    inpaint_method : str
        Algorithm for void filling ('spring', 'laplace', 'nearest', or 'idw').
        Default is 'spring'.
    interp_method : str
        Method for back-interpolating DTM elevation to point coordinates:
        'spline' (bivariate spline) or 'linear' (fast bilinear). Default is 'spline'.
    """

    cellsize: float = 1.0
    windows: Union[int, list[int], np.ndarray] = 5
    slope_threshold: float = 0.15
    elevation_threshold: float = 0.5
    elevation_scaler: float = 1.25
    low_filter_slope: float = 5.0
    low_outlier_fill: bool = False
    workers: int = 1
    inpaint_method: str = "spring"
    interp_method: str = "spline"

    def get_window_array(self) -> np.ndarray:
        """Returns the window sequence as a 1D numpy array of integers."""
        if np.isscalar(self.windows):
            return np.arange(1, int(self.windows) + 1, dtype=np.int32)
        return np.asarray(self.windows, dtype=np.int32)


@dataclass
class FilterResult:
    """Comprehensive result of running PySMRF ground classification.

    Attributes
    ----------
    dem : np.ndarray
        Provisional digital elevation model (DTM / bare-earth surface).
    transform : Affine
        Affine geotransform matrix for the raster grid.
    is_ground : np.ndarray
        Boolean 1D array, True for ground points, False for object points.
    is_object : np.ndarray
        Boolean 1D array, True for object (non-ground) points, False for ground points.
    classification : np.ndarray
        Standard ASPRS LAS classification codes: 2 = Ground, 1 = Unassigned / Object.
    object_cells : np.ndarray
        Boolean 2D mask of the same shape as `dem` marking raster cells identified as objects.
    elevation_values : Optional[np.ndarray]
        Interpolated bare-earth elevation at each (x, y) point coordinate.
    above_ground_height : Optional[np.ndarray]
        Height of each point above the provisional DTM (z - elevation_values).
    slope_values : Optional[np.ndarray]
        Interpolated terrain slope magnitude (dz/dx) at each point coordinate.
    slope_grid : Optional[np.ndarray]
        2D raster grid of terrain slope magnitude.
    drop_raster : Optional[np.ndarray]
        2D raster where pixel values record the window iteration at which the cell was filtered.
    when_dropped : Optional[np.ndarray]
        1D array indicating the window iteration at which each point was dropped.
    extra : Dict[str, Any]
        Additional auxiliary metadata or custom algorithm outputs.
    """

    dem: np.ndarray
    transform: Affine
    is_ground: np.ndarray
    is_object: np.ndarray
    classification: np.ndarray
    object_cells: np.ndarray
    elevation_values: Optional[np.ndarray] = None
    above_ground_height: Optional[np.ndarray] = None
    slope_values: Optional[np.ndarray] = None
    slope_grid: Optional[np.ndarray] = None
    drop_raster: Optional[np.ndarray] = None
    when_dropped: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        """Total number of classified points."""
        return len(self.is_ground)

    @property
    def ground_count(self) -> int:
        """Total count of classified ground points."""
        return int(np.sum(self.is_ground))

    @property
    def object_count(self) -> int:
        """Total count of classified object points."""
        return int(np.sum(self.is_object))

    @property
    def ground_percentage(self) -> float:
        """Percentage of points classified as ground."""
        return (self.ground_count / self.num_points) * 100.0 if self.num_points > 0 else 0.0

    def to_tuple(self, return_extras: bool = False) -> Tuple[Any, ...]:
        """Convert result to the legacy smrf tuple format.

        Returns
        -------
        Tuple of (dem, transform, object_cells, is_object, [extras])
        """
        if not return_extras:
            return (self.dem, self.transform, self.object_cells, self.is_object)
        extras = {
            "above_ground_height": self.above_ground_height,
            "drop_raster": self.drop_raster,
            "when_dropped": self.when_dropped,
        }
        extras.update(self.extra)
        return (self.dem, self.transform, self.object_cells, self.is_object, extras)

    def save_dem(
        self,
        filepath: Union[str, Path],
        crs: Optional[Any] = None,
        nodata: float = -9999.0,
    ) -> None:
        """Export the provisional DEM to a GeoTIFF raster file."""
        from .io import write_geotiff

        write_geotiff(filepath, self.dem, self.transform, crs=crs, nodata=nodata)

    def save_las(
        self,
        filepath: Union[str, Path],
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        source_las_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Export the point cloud with updated classification to a LAS/LAZ file."""
        from .io import write_classified_las

        write_classified_las(
            filepath=filepath,
            x=x,
            y=y,
            z=z,
            classification=self.classification,
            source_las_path=source_las_path,
        )
