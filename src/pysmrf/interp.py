"""High-performance interpolation and terrain slope estimation for point clouds."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from rasterio.transform import Affine
from scipy import interpolate
from scipy.ndimage import map_coordinates

from .grid import coords_to_pixel
from .parallel import chunk_indices, get_worker_count


def compute_surface_slope(
    grid: np.ndarray,
    cellsize: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local terrain gradient (dz/dy, dz/dx) and total slope magnitude (dz/dx).

    Parameters
    ----------
    grid : np.ndarray
        2D elevation raster (provisional DTM).
    cellsize : float
        Grid cell size in map units.

    Returns
    -------
    slope : np.ndarray
        2D raster of terrain slope magnitude sqrt((dz/dy)^2 + (dz/dx)^2).
    gy : np.ndarray
        Vertical gradient (dz/dy).
    gx : np.ndarray
        Horizontal gradient (dz/dx).
    """
    gy, gx = np.gradient(grid, cellsize)
    slope = np.sqrt(gy**2 + gx**2)
    return slope, gy, gx


def interpolate_surface(
    grid: np.ndarray,
    transform: Affine,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spline",
    workers: int = 1,
) -> np.ndarray:
    """Interpolate continuous 2D surface elevation at arbitrary (x, y) point coordinates.

    Parameters
    ----------
    grid : np.ndarray
        2D raster surface of shape (rows, cols).
    transform : Affine
        Rasterio affine transform.
    x, y : np.ndarray
        1D arrays of real-world point coordinates.
    method : str
        Interpolation method: 'spline' (Dierckx bivariate cubic spline, matching Pingel 2013)
        or 'linear' (ultra-fast C-level bilinear interpolation via ndimage.map_coordinates).
    workers : int
        Number of parallel worker threads.

    Returns
    -------
    values : np.ndarray
        Interpolated elevation values for each point.
    """
    workers = get_worker_count(workers)
    c, r = coords_to_pixel(x, y, transform)
    num_points = len(x)

    method_lower = method.lower()
    if method_lower == "linear":
        # Pixel coordinates in ndimage convention (origin at center of first pixel, i.e. r - 0.5, c - 0.5)
        coords = np.vstack([r - 0.5, c - 0.5])
        return map_coordinates(grid, coords, order=1, mode="nearest")

    # Bivariate cubic spline matching original SMRF
    H, W = grid.shape
    row_centers = np.arange(0.5, H + 0.5, dtype=np.float64)
    col_centers = np.arange(0.5, W + 0.5, dtype=np.float64)

    spline = interpolate.RectBivariateSpline(
        row_centers, col_centers, grid, kx=3, ky=3, s=0
    )

    if workers <= 1 or num_points < 100_000:
        return spline.ev(r, c)

    # Parallel chunked evaluation for large point clouds
    from concurrent.futures import ThreadPoolExecutor

    chunks = chunk_indices(num_points, workers)
    out = np.empty(num_points, dtype=np.float64)

    def eval_chunk(start: int, end: int) -> None:
        out[start:end] = spline.ev(r[start:end], c[start:end])

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(eval_chunk, s, e) for s, e in chunks]
        for fut in futures:
            fut.result()

    return out


def interpolate_elevation_and_slope(
    dem: np.ndarray,
    transform: Affine,
    cellsize: float,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spline",
    workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneously interpolate bare-earth elevation and terrain slope at point coordinates.

    Parameters
    ----------
    dem : np.ndarray
        Provisional DTM grid.
    transform : Affine
        Affine geotransform.
    cellsize : float
        Cell resolution.
    x, y : np.ndarray
        Point coordinates.
    method : str
        'spline' or 'linear'.
    workers : int
        Parallel worker count.

    Returns
    -------
    elevation_values : np.ndarray
        Interpolated surface height for each point.
    slope_values : np.ndarray
        Interpolated slope magnitude for each point.
    slope_grid : np.ndarray
        2D raster grid of slope magnitudes.
    """
    workers = get_worker_count(workers)
    slope_grid, _, _ = compute_surface_slope(dem, cellsize)

    c, r = coords_to_pixel(x, y, transform)
    num_points = len(x)

    method_lower = method.lower()
    if method_lower == "linear":
        coords = np.vstack([r - 0.5, c - 0.5])
        elevations = map_coordinates(dem, coords, order=1, mode="nearest")
        slopes = map_coordinates(slope_grid, coords, order=1, mode="nearest")
        return elevations, slopes, slope_grid

    # Spline interpolation
    H, W = dem.shape
    row_centers = np.arange(0.5, H + 0.5, dtype=np.float64)
    col_centers = np.arange(0.5, W + 0.5, dtype=np.float64)

    f_elev = interpolate.RectBivariateSpline(row_centers, col_centers, dem, kx=3, ky=3, s=0)
    f_slope = interpolate.RectBivariateSpline(row_centers, col_centers, slope_grid, kx=3, ky=3, s=0)

    if workers <= 1 or num_points < 100_000:
        elevations = f_elev.ev(r, c)
        slopes = f_slope.ev(r, c)
        return elevations, slopes, slope_grid

    from concurrent.futures import ThreadPoolExecutor

    chunks = chunk_indices(num_points, workers)
    elevations = np.empty(num_points, dtype=np.float64)
    slopes = np.empty(num_points, dtype=np.float64)

    def eval_chunk(start: int, end: int) -> None:
        r_sub = r[start:end]
        c_sub = c[start:end]
        elevations[start:end] = f_elev.ev(r_sub, c_sub)
        slopes[start:end] = f_slope.ev(r_sub, c_sub)

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(eval_chunk, s, e) for s, e in chunks]
        for fut in futures:
            fut.result()

    return elevations, slopes, slope_grid
