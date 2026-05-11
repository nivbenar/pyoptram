import numpy as np
import rasterio
from rasterio.transform import from_origin

from pyoptram import optram_ndvi_str


def _write_single_band_tif(path, array):
    transform = from_origin(10.0, 20.0, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def test_optram_ndvi_str_builds_dataframe_and_filters_zero_str(tmp_path):
    # NDVI and STR rasters become one dataframe with invalid STR zero removed.
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)

    dataframe = optram_ndvi_str([ndvi_path], [str_path])

    assert len(dataframe) == 3
    assert (dataframe["STR"] > 0).all()
    assert list(dataframe.columns) == [
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
    np.testing.assert_allclose(dataframe["NDVI"].to_numpy(), [0.2, 0.6, 0.8])
    np.testing.assert_allclose(dataframe["STR"].to_numpy(), [1.0, 2.0, 3.0])
