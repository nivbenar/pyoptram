from pathlib import Path

import numpy as np
import rasterio


def calculate_str(swir):
    """
    Calculate STR values from SWIR reflectance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(swir > 0, ((1.0 - swir) ** 2) / (2.0 * swir), np.nan)


def prepare_str_inputs(boa_dir, str_dir=None):
    """
    Prepare BOA input files and the output STR folder.
    """
    boa_dir = Path(boa_dir)

    if not boa_dir.exists() or not boa_dir.is_dir():
        return None

    boa_list = sorted(path for path in boa_dir.glob("*.tif") if "BOA_" in path.name)

    if not boa_list:
        return None

    str_dir = boa_dir.parent / "STR" if str_dir is None else Path(str_dir)
    str_dir.mkdir(parents=True, exist_ok=True)

    return boa_list, str_dir


def process_boa_file(tif_path, str_dir, swir_band):
    """
    Read one BOA file, calculate STR, and save the result.
    """
    with rasterio.open(tif_path) as src:
        if swir_band < 1 or swir_band > src.count:
            raise ValueError(
                f"swir_band={swir_band} is out of range for file {tif_path.name}"
            )

        profile = src.profile.copy()
        swir = src.read(swir_band).astype(np.float32) / 10000.0
        str_arr = calculate_str(swir).astype(np.float32)

        profile.update(dtype="float32", count=1, nodata=np.nan)

        out_path = str_dir / tif_path.name.replace("BOA", "STR")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(str_arr, 1)

    return str(out_path)


def optram_calculate_str(boa_dir, str_dir=None, swir_band=11):
    """
    Create STR raster files from BOA raster files.
    """
    prepared = prepare_str_inputs(boa_dir, str_dir)

    if prepared is None:
        return None

    boa_list, str_dir = prepared
    str_list = []

    for tif_path in boa_list:
        str_list.append(process_boa_file(tif_path, str_dir, swir_band))

    print(f"Prepared: {len(str_list)} STR files")
    return str_list
