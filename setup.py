"""Setup and installation script for PySMRF.

Cites:
    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
    An improved simple morphological filter for producing ground surfaces from
    LiDAR point clouds. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysmrf",
    version="1.0.0",
    description="High-performance parallel Simple Morphological Filter (SMRF) for LiDAR ground classification citing Pingel et al. (2013)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://doi.org/10.1016/j.isprsjprs.2012.12.002",
    author="Aman Ranjan, Thomas Pingel (original algorithm)",
    author_email="er.amanranjan@gmail.com, thomas.pingel@gmail.com",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "rasterio>=1.3.0",
        "pandas>=1.4.0",
        "laspy>=2.0.0",
        "lazrs>=0.5.0",
        "click>=8.0.0",
    ],
    extras_require={
        "laz": ["lazrs>=0.5.0"],
        "viz": ["matplotlib>=3.5.0"],
        "all": ["lazrs>=0.5.0", "matplotlib>=3.5.0", "scikit-image>=0.19.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "lazrs>=0.5.0"],
    },
    entry_points={
        "console_scripts": [
            "pysmrf=pysmrf.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Geographic Information Science",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="GIS lidar remote-sensing morphological-filter DEM DTM ground-classification parallelism",
)
