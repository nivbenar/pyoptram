import numpy as np
import pandas as pd

from pyoptram import optram_wetdry_coefficients


def test_optram_wetdry_coefficients_returns_rmse_coefficients_and_edges():
    # Wet/dry fitting returns RMSE, coefficients, and fitted edge points.
    rng = np.random.default_rng(42)
    ndvi = np.repeat(np.linspace(0.05, 0.85, 17), 80)

    wet_line = 0.8 + 2.0 * ndvi
    dry_line = 0.2 + 0.5 * ndvi
    position = rng.uniform(0.0, 1.0, size=ndvi.size)
    str_values = dry_line + position * (wet_line - dry_line)

    dataframe = pd.DataFrame({"NDVI": ndvi, "STR": str_values})

    rmse_df, coeffs_df, edges_df = optram_wetdry_coefficients(
        dataframe,
        method="linear",
        vi_step=0.1,
        min_bin_count=20,
        return_outputs=True,
    )

    assert list(rmse_df.columns) == ["RMSE wet", "RMSE dry"]
    assert set(coeffs_df["edge"]) == {"wet", "dry"}
    assert {"STR_wet", "STR_dry", "STR_wet_fit", "STR_dry_fit"}.issubset(
        edges_df.columns
    )
    assert rmse_df.loc[0, "RMSE wet"] >= 0
    assert rmse_df.loc[0, "RMSE dry"] >= 0
