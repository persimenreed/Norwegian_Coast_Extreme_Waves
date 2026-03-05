import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.extreme_value_modelling.paths import resolve_input_path
from src.extreme_value_modelling.common import dataset_name
from src.settings import get_columns, get_thresholds, get_path_template

_COLUMNS = get_columns()
HS_COLUMN = _COLUMNS.get("hs_model", "hs")
TIME_COLUMN = _COLUMNS.get("time", "time")

_THRESHOLDS = get_thresholds()
THRESHOLD_QUANTILE = float(_THRESHOLDS.get("evt_threshold_quantile", 0.95))
DECLUSTER_HOURS = float(_THRESHOLDS.get("decluster_hours", 48.0))


def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, comment="#")

    if "Significant_Wave_Height_Hm0" in df.columns and HS_COLUMN not in df.columns:
        df[HS_COLUMN] = pd.to_numeric(df["Significant_Wave_Height_Hm0"], errors="coerce")

    if TIME_COLUMN not in df.columns or HS_COLUMN not in df.columns:
        raise ValueError(f"Missing required columns: {TIME_COLUMN}, {HS_COLUMN}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df[HS_COLUMN] = pd.to_numeric(df[HS_COLUMN], errors="coerce")

    df = df.dropna(subset=[TIME_COLUMN, HS_COLUMN])
    df = df[df[HS_COLUMN] > 0]
    df = df.sort_values(TIME_COLUMN)

    return df.set_index(TIME_COLUMN)


def compute_annual_maxima(df):

    annual_max = df[HS_COLUMN].resample("YE").max().dropna()

    first_year = df.index.min().year
    last_year = df.index.max().year

    annual_max = annual_max[
        (annual_max.index.year > first_year) &
        (annual_max.index.year < last_year)
    ]

    if len(annual_max) < 3:
        raise ValueError("Too few full years for GEV")

    return annual_max


def decluster_clustermax(exceed_df: pd.DataFrame, window_hours: float) -> pd.Series:
    if exceed_df.empty:
        return pd.Series(dtype=float)

    peaks = []
    cluster = [exceed_df.iloc[0]]

    for i in range(1, len(exceed_df)):
        dt = (exceed_df.index[i] - exceed_df.index[i - 1]).total_seconds() / 3600.0
        if dt <= window_hours:
            cluster.append(exceed_df.iloc[i])
        else:
            peaks.append(pd.DataFrame(cluster)[HS_COLUMN].max())
            cluster = [exceed_df.iloc[i]]

    peaks.append(pd.DataFrame(cluster)[HS_COLUMN].max())
    return pd.Series(peaks, dtype=float)


def compute_pot(df: pd.DataFrame, quantile: float, decluster_hours: float):
    threshold = float(df[HS_COLUMN].quantile(quantile))
    exceed = df[df[HS_COLUMN] > threshold]

    peaks = decluster_clustermax(exceed, decluster_hours)

    total_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
    total_years = float(max(total_years, 1e-9))
    lambda_year = float(len(peaks) / total_years)

    return peaks, threshold, lambda_year, total_years


def _preprocessing_dir(location: str) -> Path:
    root = Path(get_path_template("evt_results_root"))
    out = root / location / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _update_wide_csv(path: Path, index_name: str, series: pd.Series, col_name: str):
    s = pd.to_numeric(series, errors="coerce").dropna()
    s.name = col_name

    if path.exists():
        existing = pd.read_csv(path)

        if index_name not in existing.columns:
            if "Unnamed: 0" in existing.columns:
                existing = existing.rename(columns={"Unnamed: 0": index_name})
            elif "time" in existing.columns and index_name != "time":
                existing = existing.rename(columns={"time": index_name})
            else:
                existing[index_name] = np.arange(len(existing))

        if index_name == "time":
            existing[index_name] = pd.to_datetime(existing[index_name], errors="coerce")
            existing = existing.dropna(subset=[index_name])

        existing = existing.set_index(index_name)
        existing[col_name] = s

        cols = sorted([c for c in existing.columns])
        out = existing[cols].reset_index()
        out.to_csv(path, index=False)
        return

    out = s.to_frame().reset_index()
    out = out.rename(columns={out.columns[0]: index_name})
    out.to_csv(path, index=False)


def run(location: str, mode: str, corr_method: str = "qm", pooling: bool = False, transfer: bool = False):
    dataset = dataset_name(mode, corr_method=corr_method, pooling=pooling, transfer=transfer)

    input_path = resolve_input_path(location, mode, corr_method=corr_method, pooling=pooling, transfer=transfer)
    df = load_data(str(input_path))

    annual_max = compute_annual_maxima(df)
    pot_peaks, threshold, lambda_year, total_years = compute_pot(df, THRESHOLD_QUANTILE, DECLUSTER_HOURS)

    out_dir = _preprocessing_dir(location)

    annual_path = out_dir / "annual_maxima.csv"
    _update_wide_csv(annual_path, index_name="time", series=annual_max, col_name=dataset)

    pot_path = out_dir / "pot_peaks.csv"
    pot_series = pd.Series(pot_peaks.values, index=np.arange(len(pot_peaks)), dtype=float)
    _update_wide_csv(pot_path, index_name="event_id", series=pot_series, col_name=dataset)

    return {
        "dataset": dataset,
        "threshold": float(threshold),
        "lambda_year": float(lambda_year),
        "total_years": float(total_years),
        "annual_path": str(annual_path),
        "pot_path": str(pot_path),
        "input_path": str(input_path),
    }


def main():
    parser = argparse.ArgumentParser(description="EVT preprocessing: annual maxima + POT declustering (wide tables).")
    parser.add_argument("--location", default="fauskane")
    parser.add_argument("--mode", default="corrected", choices=["raw", "corrected"])
    parser.add_argument("--corr-method", default="qm")
    parser.add_argument("--pooling", action="store_true")
    args = parser.parse_args()

    res = run(args.location, args.mode, args.corr_method, args.pooling)
    print(f"Updated preprocessing tables for {args.location} / {res['dataset']}")
    print(f"  annual_maxima: {res['annual_path']}")
    print(f"  pot_peaks:     {res['pot_path']}")


if __name__ == "__main__":
    main()