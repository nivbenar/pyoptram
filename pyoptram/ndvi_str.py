from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


def _as_path_list(paths, name):
    # Accept one path or a list of paths.
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError(f"{name} must contain at least one path")

    return path_list


def _read_band(path):
    # Read raster band 1 and convert NoData to NaN.
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


def _check_same_grid(ndvi_profile, str_profile, ndvi_path, str_path):
    # NDVI and STR must describe the same pixels.
    for key, label in [("shape", "Shape"), ("transform", "Transform"), ("crs", "CRS")]:
        if ndvi_profile[key] != str_profile[key]:
            raise ValueError(f"{label} mismatch: {ndvi_path} and {str_path}")


def _file_metadata(path):
    # Pull timestamp and Sentinel tile from the pyoptram filename.
    name = Path(path).stem
    timestamp_match = re.search(
        r"_(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2}(?:\.\d+)?)",
        name,
    )
    tile_match = re.search(r"_T(\d{2}[A-Z]{3})_", name)

    timestamp = pd.NaT
    if timestamp_match:
        date_text = timestamp_match.group(1)
        time_text = timestamp_match.group(2).replace("-", ":")
        timestamp = pd.to_datetime(f"{date_text} {time_text}", utc=True)

    month = timestamp.month if pd.notna(timestamp) else pd.NA
    tile = tile_match.group(1) if tile_match else pd.NA

    return timestamp, month, tile


def _remove_high_str(dataframe):
    # Remove high STR outliers with Q3 + 1.5 * IQR.
    q1 = dataframe["STR"].quantile(0.25)
    q3 = dataframe["STR"].quantile(0.75)
    cutoff = q3 + 1.5 * (q3 - q1)

    return dataframe[dataframe["STR"] < cutoff].copy()


def optram_ndvi_str(
    ndvi_paths,
    str_paths,
    output_csv=None,
    rm_low_vi=False,
    rm_hi_str=False,
    max_tbl_size=None,
    random_state=None,
):
    """Build a dataframe of paired NDVI and STR pixel values."""
    # Prepare input paths.
    ndvi_path_list = _as_path_list(ndvi_paths, "ndvi_paths")
    str_path_list = _as_path_list(str_paths, "str_paths")

    if len(ndvi_path_list) != len(str_path_list):
        raise ValueError("ndvi_paths and str_paths must have the same length")
    if max_tbl_size is not None and max_tbl_size < 1:
        raise ValueError("max_tbl_size must be a positive integer")

    frames = []

    for source_index, (ndvi_path, str_path) in enumerate(zip(ndvi_path_list, str_path_list)):
        # Read and validate one NDVI/STR raster pair.
        ndvi, ndvi_profile = _read_band(ndvi_path)
        str_array, str_profile = _read_band(str_path)
        _check_same_grid(ndvi_profile, str_profile, ndvi_path, str_path)

        # Keep only usable OPTRAM pixels.
        valid = (
            np.isfinite(ndvi)
            & np.isfinite(str_array)
            & (ndvi >= -1)
            & (ndvi <= 1)
            & (str_array > 0)
        )

        if rm_low_vi:
            valid &= ndvi > 0.005

        rows, cols = np.where(valid)
        if len(rows) == 0:
            continue

        # Convert raster pixels to dataframe rows.
        xs, ys = xy(ndvi_profile["transform"], rows, cols)
        timestamp, month, tile = _file_metadata(ndvi_path)

        frames.append(
            pd.DataFrame(
                {
                    "X": xs,
                    "Y": ys,
                    "TimestampUTC": timestamp,
                    "Month": month,
                    "Tile": tile,
                    "NDVI": ndvi[valid],
                    "STR": str_array[valid],
                    "source_index": source_index,
                    "row": rows,
                    "col": cols,
                    "ndvi_path": str(ndvi_path),
                    "str_path": str(str_path),
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "X",
                "Y",
                "TimestampUTC",
                "Month",
                "Tile",
                "NDVI",
                "STR",
                "source_index",
                "row",
                "col",
                "ndvi_path",
                "str_path",
            ]
        )

    dataframe = pd.concat(frames, ignore_index=True)

    if rm_hi_str:
        dataframe = _remove_high_str(dataframe)

    if max_tbl_size is not None and len(dataframe) > max_tbl_size:
        dataframe = dataframe.sample(n=max_tbl_size, random_state=random_state)
        dataframe = dataframe.reset_index(drop=True)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_csv, index=False)

    return dataframe
