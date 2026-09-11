"""Tests for point cloud and raster I/O."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from rasterio.transform import from_origin

from pysmrf.io import read_point_cloud, write_classified_las, write_geotiff


def test_write_and_read_geotiff():
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = Path(tmpdir) / "test.tif"
        data = np.random.uniform(10, 50, (30, 40)).astype(np.float32)
        transform = from_origin(500000.0, 4000000.0, 1.0, 1.0)

        write_geotiff(tif_path, data, transform, nodata=-9999.0)
        assert tif_path.exists()

        import rasterio
        with rasterio.open(str(tif_path)) as src:
            read_arr = src.read(1)
            assert read_arr.shape == (30, 40)
            assert np.allclose(read_arr, data, atol=1e-4)


def test_write_and_read_las():
    with tempfile.TemporaryDirectory() as tmpdir:
        las_path = Path(tmpdir) / "test.las"
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([100.0, 200.0, 300.0])
        z = np.array([5.0, 6.0, 7.0])
        cls_arr = np.array([2, 1, 2], dtype=np.uint8)

        write_classified_las(las_path, x, y, z, cls_arr)
        assert las_path.exists()

        rx, ry, rz, hdr = read_point_cloud(las_path)
        assert len(rx) == 3
        assert np.allclose(rx, x)
        assert np.allclose(ry, y)
        assert np.allclose(rz, z)
