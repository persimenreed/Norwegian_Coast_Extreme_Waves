import argparse

import numpy as np
import pandas as pd

from src.extreme_value_modelling.common import (
    DECLUSTER_HOURS,
    HS_COLUMN,
    THRESHOLD_QUANTILE,
    TIME_COLUMN,
    dataset_name,
)
from src.extreme_value_modelling.paths import resolve_input_path, resolve_preprocessing_dir


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    if TIME_COLUMN not in df.columns or HS_COLUMN not in df.columns:
        raise ValueError(f"Missing required columns: {TIME_COLUMN}, {HS_COLUMN}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df[HS_COLUMN] = pd.to_numeric(df[HS_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIME_COLUMN, HS_COLUMN])
    df = df[df[HS_COLUMN] > 0].sort_values(TIME_COLUMN)
    return df.set_index(TIME_COLUMN)


def compute_annual_maxima(df):
    annual_maxima = df[HS_COLUMN].resample("YE").max().dropna()
    first_year = df.index.min().year
    last_year = df.index.max().year
    annual_maxima = annual_maxima[
        (annual_maxima.index.year > first_year) & (annual_maxima.index.year < last_year)
    ]

    if len(annual_maxima) < 3:
        raise ValueError("Too few full years for GEV")
    return annual_maxima


def decluster_clustermax(exceed_df: pd.DataFrame, window_hours: float):
    if exceed_df.empty:
        return pd.Series(dtype=float)

    peaks = []
    cluster = [float(exceed_df.iloc[0][HS_COLUMN])]
    for index in range(1, len(exceed_df)):
        dt = (exceed_df.index[index] - exceed_df.index[index - 1]).total_seconds() / 3600.0
        if dt <= window_hours:
            cluster.append(float(exceed_df.iloc[index][HS_COLUMN]))
        else:
            peaks.append(max(cluster))
            cluster = [float(exceed_df.iloc[index][HS_COLUMN])]

    peaks.append(max(cluster))
    return pd.Series(peaks, dtype=float)


def compute_pot(df: pd.DataFrame, quantile: float = THRESHOLD_QUANTILE, decluster_hours: float = DECLUSTER_HOURS):
    threshold = float(df[HS_COLUMN].quantile(float(quantile)))
    peaks = decluster_clustermax(df[df[HS_COLUMN] > threshold], decluster_hours)

    total_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
    total_years = float(max(total_years, 1e-9))
    return peaks, threshold, float(len(peaks) / total_years), total_years


def _update_wide_csv(path, index_name, series, column_name):
    series = pd.to_numeric(series, errors="coerce").dropna()
    series.name = column_name

    if path.exists():
        existing = pd.read_csv(path)
        if index_name not in existing.columns:
            alias = "Unnamed: 0" if "Unnamed: 0" in existing.columns else "time" if "time" in existing.columns else None
            if alias is not None:
                existing = existing.rename(columns={alias: index_name})
            else:
                existing[index_name] = np.arange(len(existing))
        if index_name == TIME_COLUMN:
            existing[index_name] = pd.to_datetime(existing[index_name], errors="coerce")
            existing = existing.dropna(subset=[index_name])
        existing = existing.set_index(index_name)
        existing = existing.reindex(existing.index.union(series.index))
        existing.index.name = index_name
        existing[column_name] = series
        existing[sorted(existing.columns)].reset_index().to_csv(path, index=False)
        return

    out = series.to_frame().reset_index().rename(columns={series.index.name or "index": index_name})
    out.to_csv(path, index=False)


def run(location: str, mode: str, corr_method: str = "pqm", transfer_source: str | None = None):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)
    df = load_data(resolve_input_path(location, mode, corr_method=corr_method, transfer_source=transfer_source))

    annual_maxima = compute_annual_maxima(df)
    pot_peaks, threshold, lambda_year, total_years = compute_pot(df)

    out_dir = resolve_preprocessing_dir(location)
    annual_path = out_dir / "annual_maxima.csv"
    pot_path = out_dir / "pot_peaks.csv"

    _update_wide_csv(annual_path, TIME_COLUMN, annual_maxima, dataset)
    _update_wide_csv(
        pot_path,
        "event_id",
        pd.Series(pot_peaks.to_numpy(), index=np.arange(len(pot_peaks)), dtype=float),
        dataset,
    )

    return {
        "dataset": dataset,
        "threshold": threshold,
        "lambda_year": lambda_year,
        "total_years": total_years,
        "annual_path": str(annual_path),
        "pot_path": str(pot_path),
    }


def main():
    parser = argparse.ArgumentParser(description="EVT preprocessing: annual maxima + POT declustering.")
    parser.add_argument("--location", default="fauskane")
    parser.add_argument("--mode", default="corrected", choices=["raw", "corrected"])
    parser.add_argument("--corr-method", default="pqm")
    parser.add_argument("--transfer-source", default=None)
    args = parser.parse_args()

    result = run(
        location=args.location,
        mode=args.mode,
        corr_method=args.corr_method,
        transfer_source=args.transfer_source,
    )
    print(f"Updated preprocessing tables for {args.location} / {result['dataset']}")
    print(f"  annual_maxima: {result['annual_path']}")
    print(f"  pot_peaks:     {result['pot_path']}")


if __name__ == "__main__":
    main()
