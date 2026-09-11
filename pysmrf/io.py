"""LiDAR LAS/LAZ and raster GeoTIFF file I/O utilities with pure-Python fallback support."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import Affine


def read_las(filename: Union[str, Path]) -> Tuple[Dict[str, Any], Any]:
    """Read an ASPRS LAS/LAZ file into a header dictionary and pandas DataFrame or dict.

    Uses `laspy` when available for maximum speed and LAZ compression support.
    Falls back to an internal pure-Python binary decoder if `laspy` is unavailable.

    Parameters
    ----------
    filename : str or Path
        Path to .las or .laz file.

    Returns
    -------
    header : dict
        Header metadata.
    data : pandas.DataFrame or dict
        Point record columns including 'x', 'y', 'z', 'intensity', etc.
    """
    filename = Path(filename)
    try:
        import laspy
        import pandas as pd

        las = laspy.read(str(filename))
        header = {
            "version": float(f"{las.header.version.major}.{las.header.version.minor}"),
            "point_format_id": las.header.point_format.id,
            "num_point_records": las.header.point_count,
            "scale": list(las.header.scales),
            "offset": list(las.header.offsets),
            "minmax": [
                las.header.x_max,
                las.header.x_min,
                las.header.y_max,
                las.header.y_min,
                las.header.z_max,
                las.header.z_min,
            ],
            "crs": las.header.parse_crs(),
        }

        # Convert to DataFrame
        df_dict = {
            "x": np.array(las.x, dtype=np.float64),
            "y": np.array(las.y, dtype=np.float64),
            "z": np.array(las.z, dtype=np.float64),
        }
        for dim_name in las.point_format.dimension_names:
            if dim_name.lower() not in ("x", "y", "z"):
                try:
                    df_dict[dim_name] = np.array(getattr(las, dim_name))
                except Exception:
                    pass

        df = pd.DataFrame(df_dict)
        return header, df

    except ImportError:
        # Fallback pure-Python binary LAS reader
        return _pure_python_read_las(filename)


def _pure_python_read_las(filename: Union[str, Path]) -> Tuple[Dict[str, Any], Any]:
    """Pure-Python binary LAS decoder for environments without laspy."""
    import pandas as pd

    with open(filename, mode="rb") as file:
        data = file.read()

    point_data_format_key = {
        0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63,
        6: 30, 7: 36, 8: 38, 9: 59, 10: 67,
    }

    header: Dict[str, Any] = {}
    header["file_signature"] = struct.unpack("<4s", data[0:4])[0].decode("latin1")
    if header["file_signature"] != "LASF":
        raise ValueError(f"Invalid LAS file signature: {header['file_signature']}")

    header["file_source_id"] = struct.unpack("<H", data[4:6])[0]
    header["global_encoding"] = struct.unpack("<H", data[6:8])[0]
    header["version_major"] = struct.unpack("<B", data[24:25])[0]
    header["version_minor"] = struct.unpack("<B", data[25:26])[0]
    header["version"] = header["version_major"] + header["version_minor"] / 10.0
    header["system_id"] = struct.unpack("32s", data[26:58])[0].decode("latin1").rstrip("\x00")
    header["generating_software"] = struct.unpack("32s", data[58:90])[0].decode("latin1").rstrip("\x00")
    header["header_size"] = struct.unpack("H", data[94:96])[0]
    header["point_data_offset"] = struct.unpack("<L", data[96:100])[0]
    header["point_data_format_id"] = struct.unpack("<B", data[104:105])[0]

    fmt_id = header["point_data_format_id"]
    if fmt_id >= 128:
        raise ValueError("LAZ format requires 'laspy' and 'lazrs' installed.")

    header["point_data_record_length"] = struct.unpack("<H", data[105:107])[0]
    header["num_point_records"] = struct.unpack("<L", data[107:111])[0]
    header["scale"] = struct.unpack("<3d", data[131:155])
    header["offset"] = struct.unpack("<3d", data[155:179])
    header["minmax"] = struct.unpack("<6d", data[179:227])

    point_data = data[header["point_data_offset"]:]
    dt_base = [("x", "i4"), ("y", "i4"), ("z", "i4"), ("intensity", "u2"),
               ("return_byte", "u1"), ("class", "u1"), ("scan_angle", "u1"),
               ("user_data", "u1"), ("point_source_id", "u2")]

    if fmt_id in (1, 3, 4, 5):
        dt_base.append(("gpstime", "f8"))
    if fmt_id in (2, 3, 5):
        dt_base.extend([("red", "u2"), ("green", "u2"), ("blue", "u2")])

    dt = np.dtype(dt_base)
    record_len = header["point_data_record_length"]
    num_points = header["num_point_records"]

    # Slice exact byte buffer
    point_bytes = point_data[: num_points * record_len]
    arr = np.frombuffer(point_bytes, dtype=dt)

    df = pd.DataFrame(arr)
    df["x"] = df["x"] * header["scale"][0] + header["offset"][0]
    df["y"] = df["y"] * header["scale"][1] + header["offset"][1]
    df["z"] = df["z"] * header["scale"][2] + header["offset"][2]

    return header, df


def read_point_cloud(
    source: Union[str, Path, np.ndarray, Dict[str, Any], Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Universal point cloud loader accepting file paths, LAS objects, DataFrames, or arrays.

    Returns
    -------
    x, y, z : np.ndarray
        1D coordinate arrays.
    metadata : dict, optional
        File header or coordinate system info.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower()
        if ext in (".las", ".laz"):
            hdr, df = read_las(path)
            return (
                np.asarray(df["x"], dtype=np.float64),
                np.asarray(df["y"], dtype=np.float64),
                np.asarray(df["z"], dtype=np.float64),
                hdr,
            )
        elif ext in (".txt", ".csv", ".xyz", ".pts"):
            data = np.loadtxt(path)
            return (
                np.asarray(data[:, 0], dtype=np.float64),
                np.asarray(data[:, 1], dtype=np.float64),
                np.asarray(data[:, 2], dtype=np.float64),
                {"source_path": str(path)},
            )
        else:
            raise ValueError(f"Unsupported point cloud file extension: '{ext}'")

    if hasattr(source, "x") and hasattr(source, "y") and hasattr(source, "z"):
        return (
            np.asarray(source.x, dtype=np.float64),
            np.asarray(source.y, dtype=np.float64),
            np.asarray(source.z, dtype=np.float64),
            getattr(source, "header", None),
        )

    if isinstance(source, dict) and "x" in source and "y" in source and "z" in source:
        return (
            np.asarray(source["x"], dtype=np.float64),
            np.asarray(source["y"], dtype=np.float64),
            np.asarray(source["z"], dtype=np.float64),
            None,
        )

    if isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] >= 3:
        return (
            np.asarray(source[:, 0], dtype=np.float64),
            np.asarray(source[:, 1], dtype=np.float64),
            np.asarray(source[:, 2], dtype=np.float64),
            None,
        )

    raise TypeError(f"Cannot extract point coordinates from source of type {type(source)}")


def write_geotiff(
    filepath: Union[str, Path],
    raster: np.ndarray,
    transform: Affine,
    crs: Optional[Any] = None,
    nodata: float = -9999.0,
) -> None:
    """Export a 2D elevation or slope raster to a standard GeoTIFF file.

    Parameters
    ----------
    filepath : str or Path
        Destination TIFF path.
    raster : np.ndarray
        2D raster grid.
    transform : Affine
        Rasterio affine transform.
    crs : Any, optional
        Coordinate Reference System (EPSG code, WKT, or rasterio CRS).
    nodata : float
        NoData fill value. Default is -9999.0.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(raster, dtype=np.float32)
    # Replace NaNs with nodata
    if np.any(np.isnan(data)):
        data = np.where(np.isnan(data), np.float32(nodata), data)

    height, width = data.shape

    with rasterio.open(
        str(filepath),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)


def write_classified_las(
    filepath: Union[str, Path],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    source_las_path: Optional[Union[str, Path]] = None,
) -> None:
    """Export points with SMRF classification (2=ground, 1=unassigned) to a LAS/LAZ file.

    If `source_las_path` is a valid LAS/LAZ file, original point attributes (intensity,
    returns, colors, GPS time) and header metadata are preserved.

    Parameters
    ----------
    filepath : str or Path
        Destination .las or .laz path.
    x, y, z : np.ndarray
        Point coordinates.
    classification : np.ndarray
        Classification codes (ASPRS format: 2 = Ground, 1 = Unassigned / Object).
    source_las_path : str or Path, optional
        Source LAS file to clone headers and attributes from.
    """
    import laspy

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    is_las_source = (
        source_las_path is not None
        and Path(source_las_path).exists()
        and Path(source_las_path).suffix.lower() in (".las", ".laz")
    )

    if is_las_source:
        # Clone from source LAS
        las = laspy.read(str(source_las_path))
        las.classification = np.asarray(classification, dtype=np.uint8)
        las.write(str(filepath))
    else:
        # Create fresh LAS file
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
        header.scales = [0.01, 0.01, 0.01]

        las = laspy.LasData(header)
        las.x = np.asarray(x, dtype=np.float64)
        las.y = np.asarray(y, dtype=np.float64)
        las.z = np.asarray(z, dtype=np.float64)
        las.classification = np.asarray(classification, dtype=np.uint8)
        las.write(str(filepath))
