"""Tests verifying backward compatibility with original smrf package."""

import numpy as np
import pytest

from pysmrf import (
    create_dem,
    edges_from_IT,
    inpaint_nans_by_fda,
    inpaint_nans_by_springs,
    progressive_filter,
    pssm,
    read_las,
    smrf,
)


def test_smrf_api_compatibility():
    # Verify exact argument order and types
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    z = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    Zpro, t, object_cells, is_object_point = smrf(
        x, y, z,
        cellsize=1.0,
        windows=2,
        slope_threshold=0.15,
        elevation_threshold=0.5,
        elevation_scaler=1.25,
        low_filter_slope=5.0,
        low_outlier_fill=False,
        return_extras=False,
    )

    assert isinstance(Zpro, np.ndarray)
    assert hasattr(t, "a")  # rasterio.transform.Affine
    assert isinstance(object_cells, np.ndarray)
    assert object_cells.dtype == bool
    assert isinstance(is_object_point, np.ndarray)
    assert is_object_point.dtype == bool


def test_functions_exposed():
    assert callable(create_dem)
    assert callable(edges_from_IT)
    assert callable(inpaint_nans_by_springs)
    assert callable(inpaint_nans_by_fda)
    assert callable(progressive_filter)
    assert callable(pssm)
    assert callable(read_las)
