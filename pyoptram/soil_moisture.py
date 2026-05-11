from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _as_path_list(paths, name):
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError(f"{name} must contain at least one path")
    return path_list


def _predict_edge(vi, coeffs, method):
    if method == "linear":
        return coeffs["slope"] * vi + coeffs["intercept"]
    if method == "polynomial":
        return coeffs["alpha"] + coeffs["beta_1"] * vi + coeffs["beta_2"] * vi**2
    if method == "exponential":
        return coeffs["a"] * np.exp(coeffs["b"] * vi)
    raise ValueError("method must be 'linear', 'polynomial', or 'exponential'")


def _coeffs_from_row(row, method):
    if method == "linear":
        return {
            "slope": float(row["slope"]),
            "intercept": float(row["intercept"]),
        }
    if method == "polynomial":
        return {
            "alpha": float(row["alpha"]),
            "beta_1": float(row["beta_1"]),
            "beta_2": float(row["beta_2"]),
        }
    if method == "exponential":
        return {
            "a": float(row["a"]),
            "b": float(row["b"]),
        }
    raise ValueError("method must be 'linear', 'polynomial', or 'exponential'")


def _read_coeffs_table(coefficients):
    if isinstance(coefficients, pd.DataFrame):
        return coefficients.copy()

    if isinstance(coefficients, (str, Path)):
        path = Path(coefficients)
        if not path.exists():
            raise FileNotFoundError(f"Coefficient table not found: {path}")
        return pd.read_csv(path)

    raise TypeError("coefficients must be a pandas DataFrame or path to a CSV table")


def _parse_coefficients(coefficients, method=None):
    if isinstance(coefficients, dict):
        if "wet" not in coefficients or "dry" not in coefficients:
            raise ValueError("Coefficient dict must include 'wet' and 'dry' keys")
        if method is None:
            method = coefficients.get("method", "linear")
        return {
            "method": method,
            "wet": dict(coefficients["wet"]),
            "dry": dict(coefficients["dry"]),
        }

    table = _read_coeffs_table(coefficients)

    required = {"edge"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Coefficient table missing required columns: {sorted(missing)}")

    if method is None:
        if "method" in table.columns and table["method"].notna().any():
            method = str(table["method"].dropna().iloc[0])
        else:
            method = "linear"

    method = method.lower()
    if method not in {"linear", "polynomial", "exponential"}:
        raise ValueError("method must be 'linear', 'polynomial', or 'exponential'")

    wet_rows = table[table["edge"] == "wet"]
    dry_rows = table[table["edge"] == "dry"]
    if wet_rows.empty or dry_rows.empty:
        raise ValueError("Coefficient table must contain one row for each edge: wet, dry")

    wet = _coeffs_from_row(wet_rows.iloc[0], method)
    dry = _coeffs_from_row(dry_rows.iloc[0], method)
    return {"method": method, "wet": wet, "dry": dry}


def calculate_soil_moisture(vi, str_array, coefficients, method=None, porosity=1.0, clip=True):
    vi = np.asarray(vi, dtype=np.float32)
    str_array = np.asarray(str_array, dtype=np.float32)

    if vi.shape != str_array.shape:
        raise ValueError("vi and str_array must have the same shape")

    if porosity <= 0:
        raise ValueError("porosity must be positive")

    parsed = _parse_coefficients(coefficients, method=method)
    method = parsed["method"]

    str_wet = _predict_edge(vi, parsed["wet"], method)
    str_dry = _predict_edge(vi, parsed["dry"], method)

    valid = (
        np.isfinite(vi)
        & np.isfinite(str_array)
        & np.isfinite(str_wet)
        & np.isfinite(str_dry)
    )

    denominator = str_dry - str_wet
    valid &= np.isfinite(denominator) & (np.abs(denominator) > 1e-12)

    sm = np.full(vi.shape, np.nan, dtype=np.float32)
    sm[valid] = porosity * (str_dry[valid] - str_array[valid]) / denominator[valid]

    if clip:
        sm = np.clip(sm, 0.0, porosity)

    return sm


def optram_calculate_soil_moisture(
    vi_paths,
    str_paths,
    coefficients,
    output_dir,
    method=None,
    porosity=1.0,
    clip=True,
):
    vi_path_list = _as_path_list(vi_paths, "vi_paths")
    str_path_list = _as_path_list(str_paths, "str_paths")

    if len(vi_path_list) != len(str_path_list):
        raise ValueError("vi_paths and str_paths must have the same length")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    for vi_path, str_path in zip(vi_path_list, str_path_list):
        with rasterio.open(vi_path) as vi_src, rasterio.open(str_path) as str_src:
            if vi_src.shape != str_src.shape:
                raise ValueError(f"Shape mismatch: {vi_path} and {str_path}")
            if vi_src.transform != str_src.transform:
                raise ValueError(f"Transform mismatch: {vi_path} and {str_path}")
            if vi_src.crs != str_src.crs:
                raise ValueError(f"CRS mismatch: {vi_path} and {str_path}")

            vi = vi_src.read(1).astype(np.float32)
            str_array = str_src.read(1).astype(np.float32)

            if vi_src.nodata is not None:
                vi = np.where(vi == vi_src.nodata, np.nan, vi)
            if str_src.nodata is not None:
                str_array = np.where(str_array == str_src.nodata, np.nan, str_array)

            sm = calculate_soil_moisture(
                vi=vi,
                str_array=str_array,
                coefficients=coefficients,
                method=method,
                porosity=porosity,
                clip=clip,
            )

            profile = vi_src.profile.copy()
            profile.update(dtype="float32", count=1, nodata=np.nan)

            base_name = str_path.name
            if "STR_" in base_name:
                out_name = base_name.replace("STR_", "SM_")
            else:
                out_name = f"SM_{base_name}"

            out_path = output_dir / out_name
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(sm.astype(np.float32), 1)

            outputs.append(str(out_path))

    return outputs
