from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


def _as_path_list(paths, name):
    """
    Convert one path or an iterable of paths into a list of Path objects.

    Parameters
    ----------
    paths : str, pathlib.Path, or iterable
        A single raster path or a collection of raster paths.
    name : str
        The input name to use in error messages.

    Returns
    -------
    list[pathlib.Path]
        Normalized raster paths.
    """
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]

    if not path_list:
        raise ValueError(f"{name} must contain at least one path")

    return path_list


def _read_band(path):
    """
    Read the first band of a raster and collect alignment metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a single-band raster file.

    Returns
    -------
    tuple
        A NumPy array with nodata converted to NaN, and a metadata dictionary
        containing shape, transform, CRS, and nodata values.
    """
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = {
            "shape": array.shape,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
        }

    if profile["nodata"] is not None:
        array = np.where(array == profile["nodata"], np.nan, array)

    return array, profile


def optram_ndvi_str(
    ndvi_paths,
    str_paths,
    output_csv=None,
):
    """
    Build a dataframe of paired NDVI and STR pixel values.

    Parameters
    ----------
    ndvi_paths : str, pathlib.Path, or iterable
        One or more NDVI raster paths.
    str_paths : str, pathlib.Path, or iterable
        One or more STR raster paths. The order must match ``ndvi_paths``.
    output_csv : str or pathlib.Path, optional
        If provided, save the dataframe to this CSV path.

    Returns
    -------
    pandas.DataFrame
        One row per valid paired pixel, with coordinates, NDVI, STR, source
        index, raster row/column, and source file paths.

    Raises
    ------
    ValueError
        If the path lists have different lengths, or paired rasters do not
        have matching shape, transform, or CRS.
    """
    ndvi_path_list = _as_path_list(ndvi_paths, "ndvi_paths")
    str_path_list = _as_path_list(str_paths, "str_paths")

    if len(ndvi_path_list) != len(str_path_list):
        raise ValueError("ndvi_paths and str_paths must have the same length")

    frames = []

    for source_index, (ndvi_path, str_path) in enumerate(
        zip(ndvi_path_list, str_path_list)
    ):
        ndvi, ndvi_profile = _read_band(ndvi_path)
        str_array, str_profile = _read_band(str_path)

        if ndvi_profile["shape"] != str_profile["shape"]:
            raise ValueError(f"Shape mismatch: {ndvi_path} and {str_path}")

        if ndvi_profile["transform"] != str_profile["transform"]:
            raise ValueError(f"Transform mismatch: {ndvi_path} and {str_path}")

        if ndvi_profile["crs"] != str_profile["crs"]:
            raise ValueError(f"CRS mismatch: {ndvi_path} and {str_path}")

        valid = (
            np.isfinite(ndvi)
            & np.isfinite(str_array)
            & (ndvi >= -1)
            & (ndvi <= 1)
            & (str_array >= 0)
        )

        rows, cols = np.where(valid)
        xs, ys = xy(ndvi_profile["transform"], rows, cols)

        frame = pd.DataFrame(
            {
                "X": xs,
                "Y": ys,
                "NDVI": ndvi[valid],
                "STR": str_array[valid],
                "source_index": source_index,
                "row": rows,
                "col": cols,
                "ndvi_path": str(ndvi_path),
                "str_path": str(str_path),
            }
        )
        frames.append(frame)

    dataframe = pd.concat(frames, ignore_index=True)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_csv, index=False)

    return dataframe
