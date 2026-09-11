"""Command-line interface (CLI) for PySMRF LiDAR processing."""

from __future__ import annotations

import glob
from pathlib import Path
import sys
import time
from typing import List, Optional
import click
import numpy as np

from . import __citation_bibtex__, __citation_text__, __version__
from .core import classify
from .grid import create_dem
from .io import read_point_cloud, write_classified_las, write_geotiff
from .parallel import get_worker_count, parallel_map


def _parse_windows(ctx, param, value):
    if value is None:
        return 5
    val_str = str(value).strip()
    if "," in val_str:
        return [int(w.strip()) for w in val_str.split(",") if w.strip()]
    try:
        return int(val_str)
    except ValueError:
        raise click.BadParameter(f"Windows must be an integer (e.g. 10) or comma-separated list (e.g. 1,2,3,5,10), got '{value}'")


@click.group()
@click.version_option(version=__version__, message="PySMRF %(version)s")
def cli():
    """PySMRF: High-performance parallel Simple Morphological Filter for LiDAR ground classification."""
    pass


@cli.command("cite")
def cite_command():
    """Print official academic citation for SMRF."""
    click.echo("\n--- Plain Text Citation (APA) ---")
    click.echo(__citation_text__)
    click.echo("\n--- BibTeX Entry ---")
    click.echo(__citation_bibtex__)
    click.echo("")


@cli.command("info")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
def info_command(input_path):
    """Display summary information for a LiDAR point cloud file."""
    click.echo(f"Reading: {input_path}")
    t0 = time.perf_counter()
    x, y, z, hdr = read_point_cloud(input_path)
    t1 = time.perf_counter()

    click.echo(f"  Points:       {len(x):,}")
    click.echo(f"  X bounds:     [{np.min(x):.3f}, {np.max(x):.3f}] (span: {np.ptp(x):.3f})")
    click.echo(f"  Y bounds:     [{np.min(y):.3f}, {np.max(y):.3f}] (span: {np.ptp(y):.3f})")
    click.echo(f"  Z bounds:     [{np.min(z):.3f}, {np.max(z):.3f}] (span: {np.ptp(z):.3f})")
    click.echo(f"  Load time:    {t1 - t0:.3f} s")
    if hdr:
        click.echo(f"  Metadata:     {hdr}")


@cli.command("classify")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False), help="Output classified LAS/LAZ file path.")
@click.option("--dem", "dem_output_path", type=click.Path(dir_okay=False), help="Optional output bare-earth DTM GeoTIFF path.")
@click.option("-s", "--cellsize", default=1.0, type=float, show_default=True, help="Cell resolution in spatial units.")
@click.option("-w", "--windows", default="5", callback=_parse_windows, help="Window radii (e.g. 5 or 1,2,3,5,10,15).")
@click.option("--slope", "slope_threshold", default=0.15, type=float, show_default=True, help="Slope threshold (dz/dx).")
@click.option("--elevation-threshold", default=0.5, type=float, show_default=True, help="Elevation tolerance threshold.")
@click.option("--elevation-scaler", default=1.25, type=float, show_default=True, help="Slope elevation scaling multiplier.")
@click.option("--low-slope", "low_filter_slope", default=5.0, type=float, show_default=True, help="Low outlier slope threshold.")
@click.option("--fill-low-outliers", is_flag=True, default=False, help="Inpaint low outlier cells prior to filtering.")
@click.option("-j", "--workers", default=-1, type=int, show_default=True, help="Parallel worker threads (-1 for all cores).")
@click.option("--inpaint", "inpaint_method", default="spring", type=click.Choice(["spring", "fda", "nearest"]), show_default=True, help="Void inpainting method.")
@click.option("--interp", "interp_method", default="spline", type=click.Choice(["spline", "linear"]), show_default=True, help="Interpolation method.")
def classify_command(
    input_path,
    output_path,
    dem_output_path,
    cellsize,
    windows,
    slope_threshold,
    elevation_threshold,
    elevation_scaler,
    low_filter_slope,
    fill_low_outliers,
    workers,
    inpaint_method,
    interp_method,
):
    """Classify LiDAR point cloud into ground and non-ground points."""
    click.echo(f"Processing: {input_path}")
    num_cores = get_worker_count(workers)
    click.echo(f"Configuration: cellsize={cellsize}, windows={windows}, slope={slope_threshold}, workers={num_cores}")

    t0 = time.perf_counter()
    x, y, z, hdr = read_point_cloud(input_path)
    t_load = time.perf_counter() - t0
    click.echo(f"Loaded {len(x):,} points in {t_load:.2f}s")

    t_start = time.perf_counter()
    result = classify(
        x=x,
        y=y,
        z=z,
        cellsize=cellsize,
        windows=windows,
        slope_threshold=slope_threshold,
        elevation_threshold=elevation_threshold,
        elevation_scaler=elevation_scaler,
        low_filter_slope=low_filter_slope,
        low_outlier_fill=fill_low_outliers,
        workers=workers,
        inpaint_method=inpaint_method,
        interp_method=interp_method,
    )
    t_filter = time.perf_counter() - t_start

    click.echo(
        f"Classification completed in {t_filter:.2f}s:\n"
        f"  Ground points: {result.ground_count:,} ({result.ground_percentage:.1f}%)\n"
        f"  Object points: {result.object_count:,} ({100 - result.ground_percentage:.1f}%)"
    )

    if output_path:
        click.echo(f"Saving classified LAS to: {output_path}")
        write_classified_las(
            filepath=output_path,
            x=x,
            y=y,
            z=z,
            classification=result.classification,
            source_las_path=input_path,
        )

    if dem_output_path:
        click.echo(f"Saving bare-earth DTM to: {dem_output_path}")
        crs = hdr.get("crs") if hdr else None
        result.save_dem(dem_output_path, crs=crs)

    click.echo("Done.")


@cli.command("batch")
@click.argument("pattern", type=str)
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False), help="Directory to save classified files.")
@click.option("-s", "--cellsize", default=1.0, type=float, show_default=True, help="Cell resolution.")
@click.option("-w", "--windows", default="5", callback=_parse_windows, help="Window radii.")
@click.option("--slope", "slope_threshold", default=0.15, type=float, show_default=True, help="Slope threshold.")
@click.option("-j", "--workers", default=-1, type=int, show_default=True, help="Parallel worker threads (-1 for all cores).")
def batch_command(pattern, output_dir, cellsize, windows, slope_threshold, workers):
    """Batch classify multiple LiDAR files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        click.echo(f"No files matched pattern: '{pattern}'", err=True)
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Found {len(files)} files to process. Output directory: {out_dir}")

    def process_file(fpath):
        f = Path(fpath)
        out_file = out_dir / f"{f.stem}_classified{f.suffix}"
        t0 = time.perf_counter()
        x, y, z, _ = read_point_cloud(f)
        res = classify(x, y, z, cellsize=cellsize, windows=windows, slope_threshold=slope_threshold, workers=1)
        write_classified_las(out_file, x, y, z, res.classification, source_las_path=f)
        elapsed = time.perf_counter() - t0
        return f.name, len(x), elapsed

    workers_count = get_worker_count(workers)
    t_start = time.perf_counter()
    results = parallel_map(process_file, files, workers=workers_count, use_processes=False)
    total_time = time.perf_counter() - t_start

    click.echo(f"\nSuccessfully processed {len(results)} files in {total_time:.2f}s:")
    for fname, pts, el in results:
        click.echo(f"  {fname}: {pts:,} points in {el:.2f}s")


@cli.command("dem")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(dir_okay=False), help="Output GeoTIFF path.")
@click.option("-s", "--cellsize", default=1.0, type=float, show_default=True, help="Grid resolution.")
@click.option("--bin-type", default="min", type=click.Choice(["min", "max", "mean", "count"]), show_default=True)
@click.option("--inpaint", is_flag=True, default=False, help="Inpaint void cells.")
def dem_command(input_path, output_path, cellsize, bin_type, inpaint):
    """Generate a raster DEM from a point cloud."""
    click.echo(f"Gridding {input_path} with resolution={cellsize}...")
    x, y, z, hdr = read_point_cloud(input_path)
    grid, transform = create_dem(x, y, z, cellsize=cellsize, bin_type=bin_type, inpaint=inpaint)
    crs = hdr.get("crs") if hdr else None
    write_geotiff(output_path, grid, transform, crs=crs)
    click.echo(f"Wrote DEM grid {grid.shape} to {output_path}")


if __name__ == "__main__":
    cli()
