"""Optimized progressive morphological filtering with built-in multi-threaded parallel spatial tiling."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple, Union
import numpy as np
from scipy.ndimage import grey_opening

from .parallel import get_worker_count


@lru_cache(maxsize=128)
def disk_footprint(radius: int) -> np.ndarray:
    """Generate a 2D boolean structuring element (disk or 3x3 square for r=1).

    Parameters
    ----------
    radius : int
        Filter radius in pixel units.

    Returns
    -------
    footprint : np.ndarray
        Boolean 2D array representing the morphological kernel.
    """
    radius = int(radius)
    if radius <= 1:
        return np.ones((3, 3), dtype=bool)

    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X**2 + Y**2) <= (radius**2)


def tiled_morphological_opening(
    image: np.ndarray,
    radius: int,
    workers: int = 1,
) -> np.ndarray:
    """Perform morphological grey opening on a 2D raster with optional parallel spatial tiling.

    Because grayscale opening is an erosion followed by dilation, a margin of 2 * radius
    ensures that boundary tiles do not incur any edge artifacts. Parallel tiling yields
    near-linear speedup on multi-core CPUs for large rasters.

    Parameters
    ----------
    image : np.ndarray
        2D float or integer array.
    radius : int
        Structuring element radius in pixels.
    workers : int
        Number of parallel worker threads. -1 uses all CPU cores.

    Returns
    -------
    opened : np.ndarray
        Opened 2D raster of identical shape and dtype.
    """
    workers = get_worker_count(workers)
    footprint = disk_footprint(radius)
    H, W = image.shape

    # Minimum size threshold for multi-threading overhead to be worthwhile
    pad = 2 * radius
    min_dimension_for_split = pad * 3 + 20

    if workers <= 1 or H < min_dimension_for_split:
        return grey_opening(image, footprint=footprint)

    n_tiles = min(workers, H // (pad * 2 + 10))
    if n_tiles <= 1:
        return grey_opening(image, footprint=footprint)

    from concurrent.futures import ThreadPoolExecutor

    tile_height = int(np.ceil(H / n_tiles))
    opened = np.empty_like(image)

    def process_tile(tile_idx: int) -> Tuple[int, int, np.ndarray]:
        r_start = tile_idx * tile_height
        r_end = min(H, (tile_idx + 1) * tile_height)

        r_start_pad = max(0, r_start - pad)
        r_end_pad = min(H, r_end + pad)

        tile_input = image[r_start_pad:r_end_pad, :]
        tile_opened = grey_opening(tile_input, footprint=footprint)

        valid_top = r_start - r_start_pad
        valid_bottom = valid_top + (r_end - r_start)
        return r_start, r_end, tile_opened[valid_top:valid_bottom, :]

    with ThreadPoolExecutor(max_workers=n_tiles) as pool:
        futures = [pool.submit(process_tile, i) for i in range(n_tiles)]
        for fut in futures:
            r_s, r_e, tile_result = fut.result()
            opened[r_s:r_e, :] = tile_result

    return opened


def progressive_filter(
    Z: np.ndarray,
    windows: Union[int, Sequence[int], np.ndarray],
    cellsize: float = 1.0,
    slope_threshold: float = 0.15,
    workers: int = 1,
    return_when_dropped: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Execute the Progressive Morphological Filter across increasing window radii.

    At each scale k with radius w_k:
    1. Grayscale opening of the surface: Z_opened = open(Z_{k-1}, w_k)
    2. Difference calculation: diff = Z_{k-1} - Z_opened
    3. Cells exceeding elevation threshold (slope_threshold * w_k * cellsize)
       are identified as object (non-ground) cells.
    4. Surface is progressively updated for subsequent iterations.

    Parameters
    ----------
    Z : np.ndarray
        2D elevation grid (inpainted bare minimum surface).
    windows : int or sequence of int
        Filter window radii sequence. If scalar, range 1..windows is used.
    cellsize : float
        Grid cell size in map units. Default is 1.0.
    slope_threshold : float
        dz/dx slope tolerance. Default is 0.15 (15%).
    workers : int
        Parallel worker count for tiled spatial opening.
    return_when_dropped : bool
        If True, also returns a 2D uint8 grid recording the window scale
        at which each cell was flagged.

    Returns
    -------
    is_object_cell : np.ndarray
        2D boolean array (True for object cells).
    when_dropped : np.ndarray (optional)
        2D uint8 array of drop scales (returned if return_when_dropped is True).
    """
    if np.isscalar(windows):
        windows = np.arange(1, int(windows) + 1, dtype=np.int32)
    else:
        windows = np.asarray(windows, dtype=np.int32)

    last_surface = Z.copy()
    elevation_thresholds = slope_threshold * (windows * cellsize)
    is_object_cell = np.zeros(Z.shape, dtype=bool)

    when_dropped = (
        np.zeros(Z.shape, dtype=np.uint8) if return_when_dropped else None
    )

    num_windows = len(windows)
    for i, window in enumerate(windows):
        elevation_threshold = float(elevation_thresholds[i])
        current_surface = tiled_morphological_opening(
            last_surface, radius=int(window), workers=workers
        )

        diff = last_surface - current_surface
        new_objects = diff > elevation_threshold

        is_object_cell |= new_objects
        if return_when_dropped and when_dropped is not None:
            when_dropped[new_objects] = np.uint8(i)

        if i < num_windows - 1 and num_windows > 1:
            last_surface = current_surface

    if return_when_dropped:
        return is_object_cell, when_dropped
    return is_object_cell
