"""Tests for void inpainting algorithms."""

import numpy as np
import pytest

from pysmrf.inpaint import inpaint_dem, inpaint_nans_by_fda, inpaint_nans_by_springs


def test_inpaint_springs():
    # 20x20 planar surface z = 2x + 3y
    x, y = np.meshgrid(np.arange(20), np.arange(20))
    plane = (2.0 * x + 3.0 * y).astype(np.float64)

    # Insert a 4x4 hole of NaNs in the center
    corrupted = plane.copy()
    corrupted[8:12, 8:12] = np.nan

    filled = inpaint_nans_by_springs(corrupted)
    assert not np.any(np.isnan(filled))
    # Interpolated values on a plane should closely match original plane
    assert np.allclose(filled[8:12, 8:12], plane[8:12, 8:12], atol=0.5)


def test_inpaint_no_nans():
    grid = np.ones((10, 10), dtype=np.float64)
    filled = inpaint_nans_by_springs(grid)
    assert np.array_equal(grid, filled)


def test_inpaint_all_nans():
    grid = np.full((10, 10), np.nan)
    filled = inpaint_nans_by_springs(grid)
    assert not np.any(np.isnan(filled))


def test_inpaint_fda():
    x, y = np.meshgrid(np.arange(15), np.arange(15))
    grid = (x + y).astype(np.float64)
    grid[6:9, 6:9] = np.nan

    filled = inpaint_nans_by_fda(grid)
    assert not np.any(np.isnan(filled))


def test_inpaint_dem_unified():
    grid = np.zeros((10, 10), dtype=np.float64)
    grid[4:6, 4:6] = np.nan

    res_spring = inpaint_dem(grid, method="spring")
    assert not np.any(np.isnan(res_spring))

    res_nearest = inpaint_dem(grid, method="nearest")
    assert not np.any(np.isnan(res_nearest))
