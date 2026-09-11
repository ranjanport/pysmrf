"""Tests for point cloud gridding and DEM creation."""

import numpy as np
import pytest
from rasterio.transform import Affine

from pysmrf.grid import coords_to_pixel, create_dem, edges_from_IT


def test_create_dem_min():
    x = np.array([0.2, 0.8, 1.2, 1.8, 0.5])
    y = np.array([0.2, 0.8, 1.2, 1.8, 0.5])
    z = np.array([10.0, 5.0, 20.0, 15.0, 2.0])

    grid, transform = create_dem(x, y, z, cellsize=1.0, bin_type="min")
    assert isinstance(grid, np.ndarray)
    assert isinstance(transform, Affine)
    assert grid.ndim == 2
    # Check that minimum in cell containing (0.2, 0.2) and (0.5, 0.5) is 2.0
    assert np.nanmin(grid) == 2.0


def test_create_dem_statistics():
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.1, 0.2, 0.3])
    z = np.array([10.0, 20.0, 30.0])

    g_min, _ = create_dem(x, y, z, cellsize=1.0, bin_type="min")
    g_max, _ = create_dem(x, y, z, cellsize=1.0, bin_type="max")
    g_mean, _ = create_dem(x, y, z, cellsize=1.0, bin_type="mean")
    g_cnt, _ = create_dem(x, y, z, cellsize=1.0, bin_type="count")

    assert np.nanmin(g_min) == 10.0
    assert np.nanmax(g_max) == 30.0
    assert np.isclose(np.nanmax(g_mean), 20.0)
    assert np.nanmax(g_cnt) == 3.0


def test_edges_and_pixel_transforms():
    grid = np.zeros((10, 20), dtype=np.float32)
    from rasterio.transform import from_origin
    t = from_origin(100.0, 200.0, 2.0, 2.0)

    x_edges, y_edges = edges_from_IT(grid, t)
    assert len(x_edges) == 21
    assert len(y_edges) == 11
    assert x_edges[0] == 100.0
    assert y_edges[0] == 200.0

    # Test coordinate to pixel mapping
    px, py = coords_to_pixel(np.array([101.0, 105.0]), np.array([199.0, 195.0]), t)
    assert np.allclose(px, [0.5, 2.5])
    assert np.allclose(py, [0.5, 2.5])
