"""Terrain visualization, hillshading, and Pyramid Surface Shading Model (PSSM)."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np


def pssm(
    Z: np.ndarray,
    cellsize: float = 1.0,
    ve: float = 2.3,
    reverse: bool = False,
    apply_colormap: bool = True,
) -> np.ndarray:
    """Pyramid Surface Shading Model (PSSM) slope-based terrain visualization.

    Parameters
    ----------
    Z : np.ndarray
        2D elevation grid.
    cellsize : float
        Grid cell size.
    ve : float
        Vertical exaggeration factor. Default is 2.3.
    reverse : bool
        If True, reverses colormap direction.
    apply_colormap : bool
        If True, applies matplotlib 'bone_r' (or 'bone') colormap to produce RGBA image.

    Returns
    -------
    shaded : np.ndarray
        8-bit grayscale intensity grid or RGBA colormapped float/uint8 image.
    """
    gy, gx = np.gradient(Z, cellsize)
    slope = np.sqrt(gx**2 + gy**2)

    # Slope angle in degrees divided by 90
    intensity = np.rad2deg(np.arctan(ve * slope)) / 90.0
    intensity = np.clip(np.round(255.0 * intensity), 0, 255).astype(np.uint8)

    if not apply_colormap:
        return intensity

    try:
        import matplotlib.pyplot as plt

        cmap = plt.cm.bone if reverse else plt.cm.bone_r
        return cmap(intensity)
    except ImportError:
        # Fallback to normalized grayscale float
        return intensity.astype(np.float32) / 255.0


def hillshade(
    Z: np.ndarray,
    cellsize: float = 1.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Analytical shaded relief (hillshade) from elevation grid.

    Parameters
    ----------
    Z : np.ndarray
        2D elevation grid.
    cellsize : float
        Grid cell size.
    azimuth : float
        Sun illumination angle in degrees (0-360, 315=NW).
    altitude : float
        Sun elevation angle in degrees above horizon (0-90).
    z_factor : float
        Vertical exaggeration multiplier.

    Returns
    -------
    shade : np.ndarray
        Normalized illumination intensity [0, 255] as uint8.
    """
    azimuth_rad = np.deg2rad(360.0 - azimuth + 90.0)
    altitude_rad = np.deg2rad(altitude)

    gy, gx = np.gradient(Z * z_factor, cellsize)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gx, gy)

    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(
        slope
    ) * np.cos(azimuth_rad - aspect)

    shaded = np.clip(255.0 * (shaded + 1.0) / 2.0, 0, 255).astype(np.uint8)
    return shaded
