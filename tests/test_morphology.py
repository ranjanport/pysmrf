"""Tests for morphology filters and parallel spatial tiling."""

import numpy as np
import pytest

from pysmrf.morphology import (
    disk_footprint,
    progressive_filter,
    tiled_morphological_opening,
)


def test_disk_footprint():
    footprint_1 = disk_footprint(1)
    assert footprint_1.shape == (3, 3)
    assert np.all(footprint_1)  # 3x3 all ones

    footprint_2 = disk_footprint(2)
    assert footprint_2.shape == (5, 5)
    # Center should be true
    assert footprint_2[2, 2]
    # Corners at (0,0) distance sqrt(4+4) = sqrt(8) > 2 should be false
    assert not footprint_2[0, 0]


def test_tiled_morphological_opening_serial_vs_parallel():
    np.random.seed(42)
    # Generate random terrain surface with elevated features
    grid = np.random.uniform(10.0, 50.0, (200, 200)).astype(np.float32)

    # Serial opening
    opened_serial = tiled_morphological_opening(grid, radius=3, workers=1)

    # Parallel opening (2 workers)
    opened_par2 = tiled_morphological_opening(grid, radius=3, workers=2)

    # Parallel opening (4 workers)
    opened_par4 = tiled_morphological_opening(grid, radius=3, workers=4)

    assert np.allclose(opened_serial, opened_par2, atol=1e-5)
    assert np.allclose(opened_serial, opened_par4, atol=1e-5)


def test_progressive_filter():
    # Grid of flat ground with one elevated 10x10 building
    grid = np.full((100, 100), 10.0, dtype=np.float64)
    grid[45:55, 45:55] = 30.0  # 20m high building

    obj_cells, dropped = progressive_filter(
        grid, windows=[1, 2, 5, 8], cellsize=1.0, slope_threshold=0.15, return_when_dropped=True
    )

    assert obj_cells.shape == grid.shape
    assert dropped.shape == grid.shape

    # Center of building should be identified as object
    assert np.all(obj_cells[47:53, 47:53])
    # Distant ground should remain false
    assert not np.any(obj_cells[0:10, 0:10])
