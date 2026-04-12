from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def calculate_str(swir: np.ndarray) -> np.ndarray:
    """
    Calculate SWIR Transformed Reflectance (STR) from SWIR reflectance values.

    Parameters
    ----------
    swir : np.ndarray
        SWIR reflectance values in native scale.

    Returns
    -------
    np.ndarray
        STR values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        str_arr = ((1.0 - swir) ** 2) / (2.0 * swir)
        return np.where(swir > 0, str_arr, np.nan)


def optram_calculate_str(
    boa_dir: str | Path,
    str_dir: str | Path | None = None,
    swir_band: int = 11,
) -> list[str] | None:
    """
    Create SWIR Transformed Reflectance (STR) rasters from BOA rasters.

    Parameters
    ----------
    boa_dir : str or Path
        Directory containing BOA GeoTIFF files.
    str_dir : str or Path, optional
        Output directory for STR rasters. If None, a sibling 'STR'
        directory will be created next to boa_dir.
    swir_band : int, default=11
        1-based raster band index to use as SWIR.

    Returns
    -------
    list[str] or None
        List of output raster paths, or None if the input directory
        does not exist or no matching files are found.
    """
    boa_dir = Path(boa_dir)

    if not boa_dir.exists() or not boa_dir.is_dir():
        return None

    boa_list = sorted(
        path for path in boa_dir.glob("*.tif")
        if "BOA_" in path.name
    )

    if not boa_list:
        return None

    if str_dir is None:
        str_dir = boa_dir.parent / "STR"
    else:
        str_dir = Path(str_dir)

    str_dir.mkdir(parents=True, exist_ok=True)

    str_list: list[str] = []

    for tif_path in boa_list:
        with rasterio.open(tif_path) as src:
            if swir_band < 1 or swir_band > src.count:
                raise ValueError(
                    f"swir_band={swir_band} is out of range for file "
                    f"{tif_path.name}. Raster has {src.count} band(s)."
                )

            profile = src.profile.copy()

            # rasterio uses 1-based band indexing, like R terra
            swir_dn = src.read(swir_band).astype(np.float32)

            # Convert scaled integer reflectance to native reflectance
            swir = swir_dn / 10000.0

            # Calculate STR using the separate mathematical function
            str_arr = calculate_str(swir).astype(np.float32)

            profile.update(dtype="float32", count=1, nodata=np.nan)

            out_name = tif_path.name.replace("BOA", "STR")נ
            out_path = str_dir / out_name

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(str_arr, 1)

            str_list.append(str(out_path))

    print(f"Prepared: {len(str_list)} STR files")
    return str_list
