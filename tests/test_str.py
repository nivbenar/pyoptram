import numpy as np

from pyoptram import calculate_str


def test_calculate_str_formula_and_non_positive_values():
    # STR uses the OPTRAM formula and converts non-positive SWIR to NaN.
    swir = np.array([0.1, 0.2, 0.0, -0.1], dtype=np.float32)

    result = calculate_str(swir)

    expected = np.array([4.05, 1.6, np.nan, np.nan], dtype=np.float32)
    np.testing.assert_allclose(result[:2], expected[:2], rtol=1e-6)
    assert np.isnan(result[2])
    assert np.isnan(result[3])
