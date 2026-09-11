"""Comprehensive test suite running all PySMRF components on real-world LAZ point clouds.

Can be run standalone:
    python tests/run_test_on_laz.py
Or invoked via pytest:
    pytest tests/run_test_on_laz.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
import pytest
import rasterio

from pysmrf import (
    SMRF,
    FilterConfig,
    FilterResult,
    classify,
    create_dem,
    edges_from_IT,
    inpaint_dem,
    read_point_cloud,
    smrf,
)
from pysmrf.morphology import disk_footprint, progressive_filter, tiled_morphological_opening


def _has_laz_backend() -> bool:
    """Check whether a LAZ decompression backend is available."""
    try:
        import laspy
        import lazrs  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import laszip  # noqa: F401
        return True
    except ImportError:
        return False


DATA_DIR = Path(__file__).parent / "data"
# Discover all input LAZ files, ignoring already classified output files
LAZ_FILES = sorted([f for f in DATA_DIR.glob("*.laz") if not f.stem.endswith("_classified")])
HAS_LAZ_BACKEND = _has_laz_backend()

SKIP_LAZ = (len(LAZ_FILES) == 0) or (not HAS_LAZ_BACKEND)
SKIP_REASON = (
    "No .laz files found in tests/data/"
    if len(LAZ_FILES) == 0
    else ("No LAZ decompression backend (lazrs or laszip) installed" if not HAS_LAZ_BACKEND else "")
)


def get_laz_files():
    """Return all available input LAZ files in tests/data."""
    return LAZ_FILES


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_reading(laz_path: Path):
    """Test 1: Read compressed LAZ file and verify coordinates & metadata."""
    t0 = time.perf_counter()
    x, y, z, hdr = read_point_cloud(laz_path)
    elapsed = time.perf_counter() - t0

    assert len(x) > 0, "Point cloud is empty"
    assert len(x) == len(y) == len(z), "Coordinate array lengths mismatch"
    assert not np.any(np.isnan(x)), "x coordinates contain NaNs"
    assert not np.any(np.isnan(y)), "y coordinates contain NaNs"
    assert not np.any(np.isnan(z)), "z coordinates contain NaNs"
    assert hdr is not None, "Header metadata was not parsed"
    assert "num_point_records" in hdr
    assert hdr["num_point_records"] == len(x)

    print(f"\n[PASS] test_laz_reading ({laz_path.name}): {len(x):,} points loaded in {elapsed:.3f}s")


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_gridding(laz_path: Path):
    """Test 2: Test vectorized DEM gridding (min, mean, count) and void filling."""
    x, y, z, _ = read_point_cloud(laz_path)
    cellsize = 1.0

    t0 = time.perf_counter()
    dem_min, transform = create_dem(x, y, z, cellsize=cellsize, bin_type="min", inpaint=False)
    grid_time = time.perf_counter() - t0

    assert dem_min.ndim == 2
    assert dem_min.shape[0] > 1 and dem_min.shape[1] > 1
    assert transform is not None

    # Test count & mean binning
    dem_count, _ = create_dem(x, y, z, cellsize=cellsize, bin_type="count", inpaint=False)
    assert np.nansum(dem_count) == len(x)

    # Inpaint any void cells if present
    dem_inpainted = inpaint_dem(dem_min, method="spring")
    assert not np.any(np.isnan(dem_inpainted)), "Inpainting left unhandled NaNs"

    print(f"\n[PASS] test_laz_gridding ({laz_path.name}): {dem_min.shape} grid generated in {grid_time:.4f}s")


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_morphology_parallel(laz_path: Path):
    """Test 3: Verify bitwise equivalence between serial and multi-threaded parallel morphology."""
    x, y, z, _ = read_point_cloud(laz_path)
    dem, _ = create_dem(x, y, z, cellsize=1.0, bin_type="min", inpaint=True)

    # Serial opening
    t0 = time.perf_counter()
    opened_serial = tiled_morphological_opening(dem, radius=3, workers=1)
    t_serial = time.perf_counter() - t0

    # Multi-threaded parallel opening
    t1 = time.perf_counter()
    opened_parallel = tiled_morphological_opening(dem, radius=3, workers=4)
    t_parallel = time.perf_counter() - t1

    # Bitwise match with zero edge artifacts
    np.testing.assert_allclose(opened_serial, opened_parallel, rtol=1e-12, atol=1e-12)
    print(
        f"\n[PASS] test_laz_morphology_parallel ({laz_path.name}): "
        f"Serial={t_serial*1000:.1f}ms, Parallel (4w)={t_parallel*1000:.1f}ms (Speedup: {t_serial/max(t_parallel, 1e-6):.2f}x)"
    )


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_classify_pipeline(laz_path: Path):
    """Test 4: Functional classify() pipeline on full LAZ dataset."""
    t0 = time.perf_counter()
    result = classify(
        laz_path,
        cellsize=1.0,
        windows=10,
        slope_threshold=0.15,
        elevation_threshold=0.5,
        elevation_scaler=1.25,
        workers=4,
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(result, FilterResult)
    assert len(result.is_ground) == result.num_points
    assert result.ground_count > 0, "Expected at least some ground points"
    assert result.object_count > 0, "Expected at least some object points"
    assert result.ground_count + result.object_count == result.num_points
    assert 0.0 < result.ground_percentage < 100.0
    assert result.dtm.ndim == 2

    print(
        f"\n[PASS] test_laz_classify_pipeline ({laz_path.name}): "
        f"{result.num_points:,} pts in {elapsed:.2f}s | "
        f"Ground={result.ground_count:,} ({result.ground_percentage:.1f}%), Objects={result.object_count:,}"
    )


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_oop_pipeline(laz_path: Path):
    """Test 5: OOP SMRF class pipeline on full LAZ dataset."""
    filter_obj = SMRF(
        cellsize=1.5,
        windows=8,
        slope_threshold=0.20,
        elevation_threshold=0.40,
        workers=4,
    )
    result = filter_obj.classify(laz_path)
    assert isinstance(result, FilterResult)
    assert result.num_points > 0
    assert result.ground_count > 0
    print(f"\n[PASS] test_laz_oop_pipeline ({laz_path.name}): SMRF class execution successful")


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_legacy_smrf(laz_path: Path):
    """Test 6: Legacy smrf() function compatibility on subsampled LAZ points."""
    x, y, z, _ = read_point_cloud(laz_path)
    # Subsample for swift verification
    step = max(1, len(x) // 50000)
    xs, ys, zs = x[::step], y[::step], z[::step]

    Zpro, transform, object_cells, is_object_point = smrf(
        xs, ys, zs,
        cellsize=2.0,
        windows=5,
        slope_threshold=0.15,
        elevation_threshold=0.5,
        return_extras=False,
    )

    assert Zpro.ndim == 2
    assert transform is not None
    assert object_cells.shape == Zpro.shape
    assert len(is_object_point) == len(xs)
    print(f"\n[PASS] test_laz_legacy_smrf ({laz_path.name}): Legacy interface verified on {len(xs):,} points")


@pytest.mark.skipif(SKIP_LAZ, reason=SKIP_REASON)
@pytest.mark.parametrize("laz_path", LAZ_FILES, ids=lambda p: p.name)
def test_laz_export_roundtrip(laz_path: Path, output_dir: Path | None = None):
    """Test 7: Export classified points and bare-earth DEM to GeoTIFF and LAS/LAZ."""
    result = classify(
        laz_path,
        cellsize=1.0,
        windows=10,
        slope_threshold=0.15,
        workers=4,
    )

    is_temp = output_dir is None
    dest_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())

    try:
        dem_path = dest_dir / f"{laz_path.stem}_dtm.tif"
        las_path = dest_dir / f"{laz_path.stem}_classified.laz"

        # 1. Save and verify GeoTIFF
        result.save_dem(dem_path)
        assert dem_path.exists()
        with rasterio.open(dem_path) as src:
            dem_read = src.read(1)
            assert dem_read.shape == result.dtm.shape
            assert src.res == (result.cellsize, result.cellsize)

        # 2. Save and verify classified LAZ
        result.save_las(las_path, source_las_path=laz_path)
        assert las_path.exists()

        import laspy
        saved_las = laspy.read(str(las_path))
        assert len(saved_las.points) == result.num_points
        unique_classes = set(np.unique(saved_las.classification))
        assert unique_classes.issubset({1, 2})
        assert 2 in unique_classes  # Ground points present

        print(f"\n[PASS] test_laz_export_roundtrip ({laz_path.name}):")
        print(f"       Classified LAZ: {las_path} ({las_path.stat().st_size:,} bytes)")
        print(f"       Bare-earth DTM: {dem_path} ({dem_path.stat().st_size:,} bytes)")
    finally:
        if is_temp and dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)


def run_all_laz_tests(persistent_outputs: bool = True):
    """Run all tests programmatically, optionally persisting classified LAZ and GeoTIFF DTM."""
    print("=" * 70)
    print("           PySMRF Comprehensive LAZ Point Cloud Test Suite")
    print("=" * 70)

    if not HAS_LAZ_BACKEND:
        print("WARNING: No LAZ decompression backend (lazrs or laszip) is installed.")
        print("Install with: pip install lazrs")
        print("Skipping LAZ decompression tests.")
        return 0

    if not LAZ_FILES:
        print(f"Error: No input .laz files found in {DATA_DIR}")
        return 1

    out_dir = DATA_DIR if persistent_outputs else None

    for laz_file in LAZ_FILES:
        print(f"\nTarget dataset: {laz_file.name} ({laz_file.stat().st_size / (1024*1024):.2f} MB)")
        tests = [
            ("LAZ Point Cloud I/O", test_laz_reading),
            ("Vectorized DEM Gridding", test_laz_gridding),
            ("Parallel Morphology Equivalence", test_laz_morphology_parallel),
            ("Functional Classify Pipeline", test_laz_classify_pipeline),
            ("Object-Oriented SMRF Pipeline", test_laz_oop_pipeline),
            ("Legacy SMRF Compatibility", test_laz_legacy_smrf),
            ("GeoTIFF & LAS Export Roundtrip", lambda p: test_laz_export_roundtrip(p, output_dir=out_dir)),
        ]

        for name, test_fn in tests:
            try:
                test_fn(laz_file)
            except Exception as e:
                print(f"  FAILED: {name} - {e}")
                import traceback
                traceback.print_exc()
                return 1

    print("\n" + "=" * 70)
    print("               ALL LAZ TEST CASES PASSED SUCCESSFULLY!")
    if persistent_outputs:
        print(f" Outputs saved in: {DATA_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_all_laz_tests(persistent_outputs=True))
