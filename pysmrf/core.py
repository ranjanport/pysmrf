"""Core Simple Morphological Filter (SMRF) implementation.

Cites:
    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
    An improved simple morphological filter for producing ground surfaces from
    LiDAR point clouds. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from rasterio.transform import Affine

from .grid import coords_to_pixel, create_dem
from .inpaint import inpaint_dem
from .interp import interpolate_elevation_and_slope
from .io import read_point_cloud, write_classified_las
from .morphology import progressive_filter
from .parallel import get_worker_count
from .types import FilterConfig, FilterResult


def classify(
    x: Union[np.ndarray, str, Path, Any],
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    cellsize: float = 1.0,
    windows: Union[int, Sequence[int], np.ndarray] = 5,
    slope_threshold: float = 0.15,
    elevation_threshold: float = 0.5,
    elevation_scaler: float = 1.25,
    low_filter_slope: float = 5.0,
    low_outlier_fill: bool = False,
    workers: int = 1,
    inpaint_method: str = "spring",
    interp_method: str = "spline",
) -> FilterResult:
    """Classify LiDAR 3D point cloud into Ground and Object (non-ground) points using SMRF.

    This modern function returns a structured `FilterResult` object containing boolean masks,
    ASPRS classification codes, the provisional bare-earth DTM raster, and elevation metrics.

    Parameters
    ----------
    x : np.ndarray, str, or Path
        1D x-coordinate array, or path to a LAS/LAZ/text point cloud file.
    y, z : np.ndarray, optional
        1D coordinate arrays (required if x is an array).
    cellsize : float
        Grid resolution in real-world units (meters or feet). Default is 1.0.
    windows : int or sequence of int
        Filter window radii sequence in pixels. If an int is provided (e.g. 5),
        radii 1..5 are tested sequentially. Default is 5.
    slope_threshold : float
        dz/dx slope tolerance for ground/object differentiation (0.15 = 15%).
        Default is 0.15.
    elevation_threshold : float
        Base elevation distance tolerance from the provisional DTM. Default is 0.5.
    elevation_scaler : float
        Multiplier scaling elevation threshold on steeper slopes. Default is 1.25.
    low_filter_slope : float
        Slope threshold for filtering low outlier points. Default is 5.0 (500%).
    low_outlier_fill : bool
        If True, re-interpolates cells identified as low outliers prior to morphological
        filtering. Default is False.
    workers : int
        Number of parallel worker threads. -1 uses all available CPU cores. Default is 1.
    inpaint_method : str
        Algorithm for void filling ('spring', 'fda', or 'nearest'). Default is 'spring'.
    interp_method : str
        Method for back-interpolating DTM elevation to point coordinates:
        'spline' (bivariate spline) or 'linear' (fast bilinear). Default is 'spline'.

    Returns
    -------
    result : FilterResult
        Complete classification result and provisional bare-earth model.
    """
    metadata: Optional[Dict[str, Any]] = None
    if y is None or z is None:
        # Load from file path or object
        x_arr, y_arr, z_arr, metadata = read_point_cloud(x)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.float64)

    workers = get_worker_count(workers)

    if np.isscalar(windows):
        windows_arr = np.arange(1, int(windows) + 1, dtype=np.int32)
    else:
        windows_arr = np.asarray(windows, dtype=np.int32)

    # 1. Create initial minimum elevation surface (Zmin)
    Zmin, transform = create_dem(
        x_arr, y_arr, z_arr, cellsize=cellsize, bin_type="min"
    )
    is_empty_cell = np.isnan(Zmin)

    # 2. Inpaint empty cells
    Zmin = inpaint_dem(Zmin, method=inpaint_method)

    # 3. Identify low outliers by filtering inverted surface
    low_outliers = progressive_filter(
        -Zmin,
        windows=np.array([1], dtype=np.int32),
        cellsize=cellsize,
        slope_threshold=low_filter_slope,
        workers=workers,
    )

    if low_outlier_fill and np.any(low_outliers):
        Zmin[low_outliers] = np.nan
        Zmin = inpaint_dem(Zmin, method=inpaint_method)

    # 4. Progressive morphological filtering
    object_cells, drop_raster = progressive_filter(
        Zmin,
        windows=windows_arr,
        cellsize=cellsize,
        slope_threshold=slope_threshold,
        workers=workers,
        return_when_dropped=True,
    )

    # 5. Build provisional bare-earth surface (Zpro)
    # An object cell is any of: empty cell, low outlier, or morphological object cell
    all_object_cells = is_empty_cell | low_outliers | object_cells

    Zpro = Zmin.copy()
    Zpro[all_object_cells] = np.nan
    Zpro = inpaint_dem(Zpro, method=inpaint_method)

    # 6. Interpolate provisional surface elevation and slope to original points
    elevation_values, slope_values, slope_grid = interpolate_elevation_and_slope(
        dem=Zpro,
        transform=transform,
        cellsize=cellsize,
        x=x_arr,
        y=y_arr,
        method=interp_method,
        workers=workers,
    )

    # 7. Classify points based on elevation and slope tolerance
    required_diff = elevation_threshold + (elevation_scaler * slope_values)
    is_object_point = np.abs(elevation_values - z_arr) > required_diff
    is_ground_point = ~is_object_point

    # ASPRS LAS standard: 2 = Ground, 1 = Unassigned / Object
    classification = np.where(is_ground_point, np.uint8(2), np.uint8(1))

    # 8. Extract auxiliary metrics
    above_ground_height = z_arr - elevation_values

    # Point-level drop scale extraction
    c, r = coords_to_pixel(x_arr, y_arr, transform)
    r_int = np.clip(np.round(r).astype(np.int64), 0, Zpro.shape[0] - 1)
    c_int = np.clip(np.round(c).astype(np.int64), 0, Zpro.shape[1] - 1)
    when_dropped = drop_raster[r_int, c_int]

    extra_dict: Dict[str, Any] = {}
    if metadata:
        extra_dict["source_metadata"] = metadata

    return FilterResult(
        dem=Zpro,
        transform=transform,
        is_ground=is_ground_point,
        is_object=is_object_point,
        classification=classification,
        object_cells=all_object_cells,
        elevation_values=elevation_values,
        above_ground_height=above_ground_height,
        slope_values=slope_values,
        slope_grid=slope_grid,
        drop_raster=drop_raster,
        when_dropped=when_dropped,
        extra=extra_dict,
    )


def smrf(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cellsize: float = 1.0,
    windows: Union[int, Sequence[int], np.ndarray] = 5,
    slope_threshold: float = 0.15,
    elevation_threshold: float = 0.5,
    elevation_scaler: float = 1.25,
    low_filter_slope: float = 5.0,
    low_outlier_fill: bool = False,
    return_extras: bool = False,
    workers: int = 1,
    inpaint_method: str = "spring",
    interp_method: str = "spline",
) -> Union[
    Tuple[np.ndarray, Affine, np.ndarray, np.ndarray],
    Tuple[np.ndarray, Affine, np.ndarray, np.ndarray, Dict[str, Any]],
]:
    """Execute the Simple Morphological Filter (SMRF) with legacy API compatibility.

    Matches the exact signature, behavior, and return types of Thomas Pingel's original
    `smrf.smrf` function, while leveraging 10-50x faster vectorized routines and optional
    built-in multi-threaded parallel spatial tiling.

    Parameters
    ----------
    x, y, z : np.ndarray
        Points in space (e.g., LiDAR point cloud coordinates).
    cellsize : float
        Resolution of provisional surface. Default is 1.0.
    windows : int or sequence of int
        Maximum radius in pixels or sequence of radii to test. Default is 5.
    slope_threshold : float
        dz/dx slope tolerance for ground classification. Default is 0.15.
    elevation_threshold : float
        Base elevation distance tolerance in map units. Default is 0.5.
    elevation_scaler : float
        Slope scaling factor for elevation tolerance. Default is 1.25.
    low_filter_slope : float
        Low outlier identification slope tolerance. Default is 5.0.
    low_outlier_fill : bool
        If True, removes and re-interpolates low outlier cells. Default is False.
    return_extras : bool
        If True, returns a 5th element: dictionary containing 'above_ground_height',
        'drop_raster', and 'when_dropped'.
    workers : int
        Parallel worker count. -1 uses all CPU cores. Default is 1.
    inpaint_method : str
        Void inpainting strategy ('spring', 'fda', or 'nearest'). Default is 'spring'.
    interp_method : str
        Interpolation method ('spline' or 'linear'). Default is 'spline'.

    Returns
    -------
    dtm : np.ndarray
        Provisional bare-earth digital elevation model.
    transform : Affine
        Rasterio affine transform matrix.
    object_cells : np.ndarray
        Boolean 2D raster where True marks object cells, False marks ground cells.
    is_object_point : np.ndarray
        Boolean 1D vector where True marks object points, False marks ground points.
    extras : dict (optional)
        Dictionary with above_ground_height, drop_raster, and when_dropped.
    """
    res = classify(
        x=x,
        y=y,
        z=z,
        cellsize=cellsize,
        windows=windows,
        slope_threshold=slope_threshold,
        elevation_threshold=elevation_threshold,
        elevation_scaler=elevation_scaler,
        low_filter_slope=low_filter_slope,
        low_outlier_fill=low_outlier_fill,
        workers=workers,
        inpaint_method=inpaint_method,
        interp_method=interp_method,
    )
    return res.to_tuple(return_extras=return_extras)


class SMRF:
    """Object-oriented Simple Morphological Filter pipeline.

    Provides a reusable, configurable filter instance for processing multiple point clouds
    or LiDAR files with consistent parameters and parallel execution.
    """

    def __init__(
        self,
        cellsize: float = 1.0,
        windows: Union[int, Sequence[int], np.ndarray] = 5,
        slope_threshold: float = 0.15,
        elevation_threshold: float = 0.5,
        elevation_scaler: float = 1.25,
        low_filter_slope: float = 5.0,
        low_outlier_fill: bool = False,
        workers: int = 1,
        inpaint_method: str = "spring",
        interp_method: str = "spline",
    ) -> None:
        self.config = FilterConfig(
            cellsize=cellsize,
            windows=windows,
            slope_threshold=slope_threshold,
            elevation_threshold=elevation_threshold,
            elevation_scaler=elevation_scaler,
            low_filter_slope=low_filter_slope,
            low_outlier_fill=low_outlier_fill,
            workers=workers,
            inpaint_method=inpaint_method,
            interp_method=interp_method,
        )

    def classify(
        self,
        source: Union[str, Path, np.ndarray, Any],
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> FilterResult:
        """Classify points from arrays, LAS file path, or LasData object."""
        return classify(
            x=source,
            y=y,
            z=z,
            cellsize=self.config.cellsize,
            windows=self.config.windows,
            slope_threshold=self.config.slope_threshold,
            elevation_threshold=self.config.elevation_threshold,
            elevation_scaler=self.config.elevation_scaler,
            low_filter_slope=self.config.low_filter_slope,
            low_outlier_fill=self.config.low_outlier_fill,
            workers=self.config.workers,
            inpaint_method=self.config.inpaint_method,
            interp_method=self.config.interp_method,
        )

    def classify_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_dem_path: Optional[Union[str, Path]] = None,
    ) -> FilterResult:
        """Classify a LAS/LAZ file, optionally saving the classified LAS and bare-earth DEM."""
        input_path = Path(input_path)
        x, y, z, hdr = read_point_cloud(input_path)
        result = self.classify(x, y, z)

        if output_path is not None:
            write_classified_las(
                filepath=output_path,
                x=x,
                y=y,
                z=z,
                classification=result.classification,
                source_las_path=input_path,
            )

        if save_dem_path is not None:
            crs = hdr.get("crs") if hdr else None
            result.save_dem(save_dem_path, crs=crs)

        return result
