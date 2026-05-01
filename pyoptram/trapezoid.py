from pathlib import Path

import numpy as np
import pandas as pd


# Keep finite VI/STR values and optionally remove very low VI values.
def _clean_vi_str(dataframe, vi_col, str_col, rm_low_vi):
    data = dataframe[[vi_col, str_col]].dropna().copy()
    data = data[np.isfinite(data[vi_col]) & np.isfinite(data[str_col])]
    data = data[data[str_col] > 0]

    if rm_low_vi:
        data = data[data[vi_col] > 0.005]

    if data.empty:
        raise ValueError("No valid VI-STR rows to fit")

    return data


# Remove STR outliers inside one VI interval.
def _remove_interval_outliers(group, str_col):
    q1 = group[str_col].quantile(0.25)
    q3 = group[str_col].quantile(0.75)
    iqr = (q3 - q1) / 1.349
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return group[(group[str_col] > lower) & (group[str_col] < upper)]


# Extract wet and dry edge points using rOPTRAM-style VI intervals.
def _edge_points(
    data,
    vi_col,
    str_col,
    vi_step,
    wet_quantile,
    dry_quantile,
    min_bin_count,
    remove_outliers,
):
    vi_min, vi_max = data[vi_col].quantile([0.02, 0.99]).round(2)
    vi_series = np.arange(vi_min, vi_max, vi_step)
    rows = []

    for vi_start in vi_series:
        vi_end = vi_start + vi_step
        group = data[(data[vi_col] >= vi_start) & (data[vi_col] < vi_end)]

        if len(group) < min_bin_count:
            continue

        if remove_outliers:
            group = _remove_interval_outliers(group, str_col)

        if len(group) < min_bin_count:
            continue

        rows.append(
            {
                "VI": vi_start + vi_step / 2.0,
                "STR_wet": group[str_col].quantile(wet_quantile),
                "STR_dry": group[str_col].quantile(dry_quantile),
            }
        )

    edges = pd.DataFrame(rows)

    if edges.empty:
        raise ValueError("No edge points found. Try a larger vi_step.")

    if len(edges) < 0.5 * len(vi_series):
        raise ValueError("Too many edge points were dropped. Try a larger vi_step.")

    return edges


# Predict STR values for one fitted edge.
def _predict(x, coeffs, method):
    if method == "linear":
        return coeffs["slope"] * x + coeffs["intercept"]

    if method == "polynomial":
        return coeffs["alpha"] + coeffs["beta_1"] * x + coeffs["beta_2"] * x**2

    if method == "exponential":
        return coeffs["a"] * np.exp(coeffs["b"] * x)

    raise ValueError(f"Unknown method: {method}")


# Fit one wet or dry edge.
def _fit_edge(x, y, method):
    if method == "linear":
        slope, intercept = np.polyfit(x, y, 1)
        coeffs = {"slope": slope, "intercept": intercept}

    elif method == "polynomial":
        beta_2, beta_1, alpha = np.polyfit(x, y, 2)
        coeffs = {"alpha": alpha, "beta_1": beta_1, "beta_2": beta_2}

    elif method == "exponential":
        positive = y > 0
        if positive.sum() < 2:
            raise ValueError("Exponential fitting needs at least two positive STR values")
        b, log_a = np.polyfit(x[positive], np.log(y[positive]), 1)
        coeffs = {"a": np.exp(log_a), "b": b}

    else:
        raise ValueError("method must be 'linear', 'polynomial', or 'exponential'")

    fit = _predict(x, coeffs, method)
    rmse = np.sqrt(np.mean((y - fit) ** 2))
    return coeffs, fit, rmse


# Derive wet/dry trapezoid coefficients from a VI-STR dataframe.
def optram_wetdry_coefficients(
    full_df,
    output_dir=None,
    vi_col="NDVI",
    str_col="STR",
    method="linear",
    vi_step=0.05,
    wet_quantile=0.95,
    dry_quantile=0.05,
    min_bin_count=20,
    rm_low_vi=False,
    remove_outliers=True,
    return_outputs=False,
):
    data = _clean_vi_str(full_df, vi_col, str_col, rm_low_vi)
    edges_df = _edge_points(
        data,
        vi_col,
        str_col,
        vi_step,
        wet_quantile,
        dry_quantile,
        min_bin_count,
        remove_outliers,
    )

    x = edges_df["VI"].to_numpy()
    wet = edges_df["STR_wet"].to_numpy()
    dry = edges_df["STR_dry"].to_numpy()

    wet_coeffs, wet_fit, rmse_wet = _fit_edge(x, wet, method)
    dry_coeffs, dry_fit, rmse_dry = _fit_edge(x, dry, method)

    edges_df["STR_wet_fit"] = wet_fit
    edges_df["STR_dry_fit"] = dry_fit

    coeff_rows = []
    for edge, coeffs in (("wet", wet_coeffs), ("dry", dry_coeffs)):
        row = {"edge": edge, "method": method}
        row.update(coeffs)
        coeff_rows.append(row)

    coeffs_df = pd.DataFrame(coeff_rows)
    rmse_df = pd.DataFrame([{"RMSE wet": rmse_wet, "RMSE dry": rmse_dry}])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_dir / "trapezoid_points.csv", index=False)
        coeffs_df.to_csv(output_dir / "wetdry_coefficients.csv", index=False)
        rmse_df.to_csv(output_dir / "wetdry_rmse.csv", index=False)

    if return_outputs:
        return rmse_df, coeffs_df, edges_df

    return rmse_df


# Plot VI-STR points with fitted wet and dry trapezoid edges.
def plot_vi_str_cloud(
    full_df,
    edges_df,
    vi_col="NDVI",
    str_col="STR",
    edge_points=False,
    plot_colors="none",
    sample=True,
    ax=None,
    output_path=None,
):
    import matplotlib.pyplot as plt

    plot_df = full_df.copy()

    if sample and len(plot_df) >= 400000:
        sample_size = int(len(plot_df) / np.log(len(plot_df)))
        plot_df = plot_df.sample(n=sample_size, random_state=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if plot_colors == "density":
        points = ax.hexbin(plot_df[vi_col], plot_df[str_col], gridsize=120, cmap="viridis")
        plt.colorbar(points, ax=ax, label="count")
    elif plot_colors == "contour":
        ax.scatter(plot_df[vi_col], plot_df[str_col], color="green", alpha=0.2, s=0.3)
        ax.tricontour(plot_df[vi_col], plot_df[str_col], plot_df[str_col], colors="gray")
    elif plot_colors in ("month", "months") and "Month" in plot_df.columns:
        points = ax.scatter(plot_df[vi_col], plot_df[str_col], c=plot_df["Month"], s=0.3)
        plt.colorbar(points, ax=ax, label="Month")
    elif plot_colors in ("feature", "features") and "Feature_ID" in plot_df.columns:
        points = ax.scatter(
            plot_df[vi_col],
            plot_df[str_col],
            c=plot_df["Feature_ID"],
            s=0.3,
        )
        plt.colorbar(points, ax=ax, label="Feature_ID")
    else:
        ax.scatter(plot_df[vi_col], plot_df[str_col], color="green", alpha=0.1, s=0.3)

    ax.plot(edges_df["VI"], edges_df["STR_dry_fit"], color="orange", linewidth=2)
    ax.plot(edges_df["VI"], edges_df["STR_wet_fit"], color="blue", linewidth=2)

    if edge_points:
        ax.scatter(edges_df["VI"], edges_df["STR_wet"], color="black", s=16, marker="^")
        ax.scatter(edges_df["VI"], edges_df["STR_dry"], color="black", s=16, marker="v")

    ax.set_xlim(0.0, max(plot_df[vi_col].max(), edges_df["VI"].max()))
    ax.set_ylim(0.05, plot_df[str_col].quantile(0.98))
    ax.set_xlabel(vi_col)
    ax.set_ylabel("SWIR Transformed")
    ax.set_title("VI-STR Cloud")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")

    return ax
