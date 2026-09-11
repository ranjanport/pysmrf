"""Tests for interpolation and slope estimation."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from pysmrf.interp import (
    compute_surface_slope,
    interpolate_elevation_and_slope,
    interpolate_surface,
)


def test_compute_surface_slope():
    # Inclined plane with 10% slope in x and 0% in y
    # z = 0.1 * x
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    grid = 0.1 * x.astype(np.float64)
    slope, gy, gx = compute_surface_slope(grid, cellsize=1.0)

    # Interior slope should be ~0.1
    assert np.allclose(slope[2:8, 2:8], 0.1, atol=1e-3)
    assert np.allclose(gy[2:8, 2:8], 0.0, atol=1e-3)
    assert np.allclose(gx[2:8, 2:8], 0.1, atol=1e-3)


def test_interpolate_surface():
    grid = np.arange(100, dtype=np.float64).reshape((10, 10))
    t = from_origin(0.0, 10.0, 1.0, 1.0)

    px = np.array([2.5, 5.0])
    py = np.array([7.5, 5.0])

    ev_spline = interpolate_surface(grid, t, px, py, method="spline")
    ev_linear = interpolate_surface(grid, t, px, py, method="linear")

    assert len(ev_spline) == 2
    assert len(ev_linear) == 2
    # Both methods should give close values in the interior
    assert np.allclose(ev_spline, ev_linear, atol=1.0)


def test_interpolate_elevation_and_slope():
    grid = np.full((20, 20), 50.0, dtype=np.float64)
    t = from_origin(0.0, 20.0, 1.0, 1.0)

    px = np.array([5.0, 10.0])
    py = np.array([15.0, 10.0])

    elev, slopes, slope_grid = interpolate_elevation_and_slope(
        dem=grid, transform=t, cellsize=1.0, x=px, y=py
    )

    assert np.allclose(elev, 50.0)
    assert np.allclose(slopes, 0.0)
    assert slope_grid.shape == (20, 20)
