"""
Notebook-style pyOPTRAM test workflow saved as a Python script.

Run this after setting CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in your
environment. Do not write credentials directly in this file.
"""

import os

import matplotlib.pyplot as plt
import pyoptram as op


# Get credentials from environment variables.
client_id = os.environ["CDSE_CLIENT_ID"]
client_secret = os.environ["CDSE_CLIENT_SECRET"]


# Download paired NDVI and STR rasters.
results = op.acquire_optram_inputs(
    aoi=[-114.75, 32.55, -114.45, 32.75],
    from_date="2025-12-01",
    to_date="2026-03-15",
    output_dir="outputs_yuma",
    client_id=client_id,
    client_secret=client_secret,
    veg_index="NDVI",
    swir_band=12,
    max_cloud=20,
    only_vi_str=True,
    limit=8,
    width=1024,
    height=1024,
)

print("NDVI files:", len(results["NDVI"]))
print("STR files:", len(results["STR"]))


# Build the VI-STR dataframe from the raster pairs.
df = op.optram_ndvi_str(
    results["NDVI"],
    results["STR"],
    output_csv="outputs_yuma/VI_STR_data.csv",
)

print(df[["NDVI", "STR"]].describe())
print("Zero STR rows:", (df["STR"] == 0).sum())


# Fit the wet and dry trapezoid edges.
rmse_df, coeffs_df, edges_df = op.optram_wetdry_coefficients(
    df,
    output_dir="outputs_yuma",
    method="linear",
    vi_step=0.05,
    rm_low_vi=True,
    return_outputs=True,
)

print(rmse_df)
print(coeffs_df)


# Plot the VI-STR cloud and fitted wet/dry edges.
op.plot_vi_str_cloud(
    df,
    edges_df,
    edge_points=True,
    plot_colors="density",
    output_path="outputs_yuma/vi_str_trapezoid.png",
)

plt.show()
