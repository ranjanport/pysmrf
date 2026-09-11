# PySMRF: High-Performance Parallel Simple Morphological Filter for LiDAR

[![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.isprsjprs.2012.12.002-blue.svg)](https://doi.org/10.1016/j.isprsjprs.2012.12.002)

**PySMRF** is a clean, modern, and highly optimized Python library implementing the **Simple Morphological Filter (SMRF)** algorithm for binary ground/object classification of airborne LiDAR point clouds and bare-earth Digital Elevation Model (DEM / DTM) generation.

This project is a complete rewrite of the original [`smrf`](https://github.com/thomaspingel/smrf) codebase by Thomas Pingel, re-engineered for **maximum performance**, **built-in multi-threaded/multi-process parallelism**, **native LAS/LAZ I/O**, and **production GIS pipelines**.

---

## 📚 Academic Citation

If you use PySMRF in your academic research, geospatial workflows, or publications, **please cite the foundational paper**:

> **Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).**
> *An improved simple morphological filter for producing ground surfaces from LiDAR point clouds.*
> **ISPRS Journal of Photogrammetry and Remote Sensing**, 77, 21–30.
> [https://doi.org/10.1016/j.isprsjprs.2012.12.002](https://doi.org/10.1016/j.isprsjprs.2012.12.002)

### BibTeX
```bibtex
@article{pingel2013smrf,
  title = {An improved simple morphological filter for producing ground surfaces from LiDAR point clouds},
  author = {Pingel, Thomas J. and Clarke, Keith C. and McBride, William A.},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {77},
  pages = {21--30},
  year = {2013},
  publisher = {Elsevier},
  doi = {10.1016/j.isprsjprs.2012.12.002},
  url = {https://doi.org/10.1016/j.isprsjprs.2012.12.002}
}
```

---

## 🚀 Key Improvements in PySMRF

| Feature | Original `smrf` | **PySMRF (Rewrite)** |
| :--- | :--- | :--- |
| **Grid DEM Creation** | Pandas `groupby.min()` (slow, heavy RAM) | **Vectorized C-level `np.minimum.at` (10–50x faster)** |
| **Parallelism** | None (single-threaded CPU only) | **Built-in multi-threading (`workers=-1`) with tiled spatial domain decomposition** |
| **Morphology Engine** | Sequential `skimage.morphology.opening` | **High-speed `scipy.ndimage.grey_opening` with cached disk footprints & parallel tiling** |
| **Point Interpolation** | `RectBivariateSpline` on full arrays | **Parallel chunked evaluation & optional bilinear `order=1`** |
| **LAS / LAZ Support** | Pure-Python custom reader (no LAZ) | **Unified `laspy` + `lazrs` integration with metadata/attribute preservation** |
| **Void Inpainting** | Uncached spring LSQR matrix | **Optimized sparse Laplacian network, FDA, and fast nearest/IDW solvers** |
| **API Design** | Monolithic script | **Modern functional API, OOP pipeline (`SMRF`), dataclass returns, & legacy compatibility** |
| **CLI Tool** | None | **Full-featured CLI: `pysmrf classify`, `pysmrf dem`, `pysmrf batch`, `pysmrf cite`** |

---

## 📦 Installation

### From Source
```bash
git clone https://github.com/thomaspingel/smrf.git
cd smrf
pip install -e .
```

### With Optional Dependencies
```bash
# With LAZ compression support (lazrs)
pip install -e ".[laz]"

# With visualization support (matplotlib)
pip install -e ".[viz]"

# Complete installation with all optional features
pip install -e ".[all]"
```

---

## ⚡ Quickstart

### 1. Modern Functional API (`classify`)
```python
import pysmrf

# Classify points directly from arrays or file path
result = pysmrf.classify(
    "survey.las",
    cellsize=1.0,
    windows=[1, 2, 3, 5, 10],
    slope_threshold=0.15,
    elevation_threshold=0.5,
    elevation_scaler=1.25,
    workers=-1,  # Use all available CPU cores!
)

print(f"Total points: {result.num_points:,}")
print(f"Ground points: {result.ground_count:,} ({result.ground_percentage:.1f}%)")
print(f"Non-ground objects: {result.object_count:,}")

# Export bare-earth DTM raster and classified LAS
result.save_dem("bare_earth_dtm.tif")
```

### 2. Object-Oriented Pipeline (`SMRF`)
```python
from pysmrf import SMRF

# Configure reusable filter
filter = SMRF(
    cellsize=1.0,
    windows=10,
    slope_threshold=0.15,
    elevation_threshold=0.4,
    workers=8,
)

# Process file directly, saving classified output and DTM GeoTIFF
result = filter.classify_file(
    input_path="input.las",
    output_path="classified.las",
    save_dem_path="dtm.tif",
)
```

### 3. 100% Backward-Compatible Legacy Interface (`smrf.smrf`)
Existing code written for `smrf` works out-of-the-box:

```python
import pysmrf as smrf

# Exact legacy signature and return values
dtm, transform, object_cells, is_object_point = smrf.smrf(
    x, y, z,
    cellsize=1.0,
    windows=5,
    slope_threshold=0.15,
    elevation_threshold=0.5,
    elevation_scaler=1.25,
)
```

---

## 💻 Command-Line Interface (CLI)

PySMRF provides an intuitive CLI for processing files directly from the terminal:

### Classify a LAS/LAZ file
```bash
pysmrf classify input.las \
  --output classified.las \
  --dem bare_earth.tif \
  --cellsize 1.0 \
  --windows 1,2,3,5,10,15 \
  --slope 0.15 \
  --workers -1
```

### Generate a Bare-Earth DEM
```bash
pysmrf dem survey.laz -o dtm.tif --cellsize 1.0 --bin-type min
```

### Batch Process Multiple Files in Parallel
```bash
pysmrf batch "raw_tiles/*.las" -o classified_tiles/ --workers 8 --cellsize 1.0
```

### Inspect LiDAR File Metadata
```bash
pysmrf info input.las
```

### Print Formal Citation
```bash
pysmrf cite
```

---

## ⚙️ Parameter Guide

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cellsize` | `float` | `1.0` | Spatial resolution of the provisional grid in real-world map units (m or ft). |
| `windows` | `int` or `list` | `5` | Structuring element radii sequence. E.g., `5` evaluates `[1, 2, 3, 4, 5]`. Use larger radii (`15–30`) to remove large buildings; smaller radii (`2–5`) for small trees. |
| `slope_threshold` | `float` | `0.15` | Slope tolerance ($dz/dx$). `0.15` corresponds to 15% slope. Use higher values (`0.25–0.40`) in steep mountainous terrain. |
| `elevation_threshold`| `float` | `0.5` | Elevation distance tolerance from provisional DTM. Points within this distance are classified as ground. |
| `elevation_scaler` | `float` | `1.25` | Slope scaling factor: $\text{tol} = \text{elev\_thresh} + (\text{scaler} \times \text{slope})$. Set to `0` to disable slope scaling. |
| `low_filter_slope` | `float` | `5.0` | Threshold (500%) for identifying low outlier points from inverted surface. |
| `low_outlier_fill` | `bool` | `False` | Whether to inpaint and remove low outlier grid cells prior to progressive filtering. |
| `workers` | `int` | `1` | Number of worker threads. `-1` uses all available CPU cores. |
| `inpaint_method` | `str` | `'spring'`| Void inpainting algorithm: `'spring'` (Pingel 2013), `'fda'`, or `'nearest'`. |
| `interp_method` | `str` | `'spline'`| Surface interpolation method: `'spline'` (bivariate cubic spline) or `'linear'` (fast bilinear). |

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest -v tests/
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

Original Algorithm & Prototype Copyright (c) 2013–2021 Thomas Pingel.
PySMRF Modern Rewrite Copyright (c) 2026 Aman Ranjan.

---

## 👤 Author & Maintainer

- **Aman Ranjan** ([er.amanranjan@gmail.com](mailto:er.amanranjan@gmail.com))
- **Thomas J. Pingel** (Original Algorithm & Prototype)
