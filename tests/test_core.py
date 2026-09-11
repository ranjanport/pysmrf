"""End-to-end tests for SMRF classification on synthetic point clouds."""

import numpy as np
import pytest

from pysmrf import SMRF, FilterResult, classify, smrf


@pytest.fixture
def synthetic_scene():
    """Create a synthetic scene: a 50x50m flat ground with a 10x10m building (height=10m)

    and some scattered vegetation points.
    """
    np.random.seed(123)
    n_ground = 2500
    gx = np.random.uniform(0, 50, n_ground)
    gy = np.random.uniform(0, 50, n_ground)
    gz = np.random.normal(10.0, 0.05, n_ground)

    # Building at [20, 30] x [20, 30], height = 20m (10m above ground)
    n_bldg = 500
    bx = np.random.uniform(20, 30, n_bldg)
    by = np.random.uniform(20, 30, n_bldg)
    bz = np.random.normal(20.0, 0.05, n_bldg)

    # Vegetation points scattered
    n_veg = 200
    vx = np.random.uniform(5, 15, n_veg)
    vy = np.random.uniform(5, 15, n_veg)
    vz = np.random.uniform(12.0, 18.0, n_veg)

    x = np.concatenate([gx, bx, vx])
    y = np.concatenate([gy, by, vy])
    z = np.concatenate([gz, bz, vz])
    is_true_ground = np.concatenate([
        np.ones(n_ground, dtype=bool),
        np.zeros(n_bldg + n_veg, dtype=bool),
    ])

    return x, y, z, is_true_ground


def test_classify_functional(synthetic_scene):
    x, y, z, true_ground = synthetic_scene

    result = classify(
        x, y, z,
        cellsize=1.0,
        windows=8,
        slope_threshold=0.15,
        elevation_threshold=0.5,
        elevation_scaler=1.25,
        workers=1,
    )

    assert isinstance(result, FilterResult)
    assert len(result.is_ground) == len(x)
    assert len(result.is_object) == len(x)
    assert np.all(result.is_ground == ~result.is_object)

    # Check accuracy on synthetic data
    accuracy = np.mean(result.is_ground == true_ground)
    assert accuracy > 0.95, f"Classification accuracy {accuracy:.3f} was below 95%"


def test_smrf_legacy_interface(synthetic_scene):
    x, y, z, _ = synthetic_scene

    dem, t, obj_cells, is_obj = smrf(
        x, y, z,
        cellsize=1.0,
        windows=5,
        return_extras=False,
    )

    assert dem.ndim == 2
    assert obj_cells.shape == dem.shape
    assert len(is_obj) == len(x)

    # Test with return_extras=True
    dem, t, obj_cells, is_obj, extras = smrf(
        x, y, z,
        cellsize=1.0,
        windows=5,
        return_extras=True,
    )
    assert "above_ground_height" in extras
    assert "drop_raster" in extras
    assert "when_dropped" in extras


def test_smrf_oop_class(synthetic_scene):
    x, y, z, true_ground = synthetic_scene

    clf = SMRF(cellsize=1.0, windows=8, slope_threshold=0.15, workers=2)
    res = clf.classify(x, y, z)

    assert isinstance(res, FilterResult)
    acc = np.mean(res.is_ground == true_ground)
    assert acc > 0.95
