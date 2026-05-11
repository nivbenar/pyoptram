# pyoptram

`pyoptram` is a Python implementation of core OPTRAM workflow steps inspired by [rOPTRAM](https://github.com/ropensci/rOPTRAM).

It currently focuses on:
- acquiring Sentinel-2 NDVI and STR inputs from Copernicus Data Space,
- preparing paired NDVI-STR pixel tables,
- fitting OPTRAM wet/dry edge coefficients,
- calculating soil moisture rasters from fitted coefficients,
- plotting VI-STR clouds.

## Install (editable)

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[geo]
```

## Quick start

```python
from pyoptram import (
    acquire_optram_inputs,
    optram_calculate_soil_moisture,
    optram_ndvi_str,
    optram_wetdry_coefficients,
)

acquired = acquire_optram_inputs(
    aoi="path/to/aoi.geojson",  # GeoJSON, vector file, dict, or bbox tuple
    from_date="2024-01-01",
    to_date="2024-03-31",
    output_dir="data/optram",
    client_id="YOUR_CDSE_CLIENT_ID",
    client_secret="YOUR_CDSE_CLIENT_SECRET",
    max_cloud=20,
    only_vi_str=True,
    download_scl=True,
)

df = optram_ndvi_str(
    ndvi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    scl_paths=acquired["SCL"],
    rm_low_vi=False,
    rm_hi_str=False,
)

rmse_df, coeffs_df, edges_df = optram_wetdry_coefficients(
    full_df=df,
    method="linear",
    return_outputs=True,
)

sm_paths = optram_calculate_soil_moisture(
    vi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    coefficients=coeffs_df,
    output_dir="data/optram/SM",
    porosity=0.4,
)
```

## rOPTRAM-like `optram_ndvi_str` options

The NDVI/STR table builder now supports quality masking, feature extraction, and size caps:

```python
features_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"ID": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[34.8, 31.4], [34.9, 31.4], [34.9, 31.5], [34.8, 31.5], [34.8, 31.4]]],
            },
        }
    ],
}

df = optram_ndvi_str(
    ndvi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    scl_paths=acquired["SCL"],        # optional SCL cloud-quality mask input
    scl_keep={4, 5, 6, 7},            # keep SCL classes (defaults to {4,5,6,7})
    features=features_geojson,        # optional polygon filtering
    feature_id_col="ID",              # creates Feature_ID column
    max_tbl_size=1_000_000,           # hard cap while assembling
    max_rows=250_000,                 # optional final downsample
    scene_metadata=acquired["scenes"] # robust datetime/tile metadata lookup
)
```

## Available API

- `get_cdse_token`
- `acquire_optram_inputs`
- `calculate_str`
- `optram_calculate_str`
- `optram_ndvi_str`
- `optram_wetdry_coefficients`
- `calculate_soil_moisture`
- `optram_calculate_soil_moisture`
- `plot_vi_str_cloud`

## Current status

This package is functional for coefficient generation, but still under active development toward fuller feature parity with rOPTRAM.

Planned additions include:
- a one-call `optram(...)` wrapper,
- broader documentation and tests.
