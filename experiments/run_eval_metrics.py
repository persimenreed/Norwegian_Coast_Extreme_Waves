import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.settings import format_path
from src.eval_metrics.core import compute_metrics
from src.eval_metrics.plot_diagnostics import (
    plot_pdf,
    plot_cdf,
    plot_qq,
    plot_residuals,
)

TIME_COL = "time"
OBS_COL = "Significant_Wave_Height_Hm0"
MODEL_COL = "hs"
MODEL_CORR_COL = "hs_corrected"


def _read_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def load_pairs(location):
    path = Path(format_path("pairs", location=location))
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def load_validation(location, corr_method):
    path = Path(format_path("validation", location=location, corr_method=corr_method))
    return _read_csv(path)


def load_corrected(location, corr_method):
    path = Path(format_path("corrected", location=location, corr_method=corr_method))
    return _read_csv(path)


def _extract_model_obs_from_validation(df):
    if df is None:
        return None, None

    if MODEL_CORR_COL not in df.columns:
        raise ValueError(f"Validation file missing column '{MODEL_CORR_COL}'.")

    if OBS_COL not in df.columns:
        raise ValueError(f"Validation file missing column '{OBS_COL}'.")

    model = df[MODEL_CORR_COL].values
    obs = df[OBS_COL].values
    return model, obs


def _extract_raw_from_pairs(df_pairs):
    if df_pairs is None:
        return None, None

    if MODEL_COL not in df_pairs.columns or OBS_COL not in df_pairs.columns:
        raise ValueError("Pairs file must contain raw model hs and observed Hs columns.")

    return df_pairs[MODEL_COL].values, df_pairs[OBS_COL].values


def _metrics_from_validation_file(name, df):
    model, obs = _extract_model_obs_from_validation(df)
    if model is None:
        return None
    return compute_metrics(name, model, obs)


def _series_from_validation_file(df):
    model, obs = _extract_model_obs_from_validation(df)
    return model, obs


def discover_validation_runs(location):
    """
    Find all validation_*.csv files for a location and return
    {display_name: dataframe}.
    """
    sample_path = Path(format_path("validation", location=location, corr_method="__dummy__"))
    out_dir = sample_path.parent
    discovered = {}

    if not out_dir.exists():
        return discovered

    for path in sorted(out_dir.glob("validation_*.csv")):
        stem = path.stem  # e.g. validation_localcv_pqm
        corr_method = stem.replace("validation_", "", 1)
        df = _read_csv(path)
        if df is not None:
            discovered[corr_method] = df

    return discovered


def run(location):
    print(f"\nRunning evaluation for {location}")

    out_dir = Path(f"results/eval_metrics/{location}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pairs = load_pairs(location)

    results = []
    series = {}
    obs_for_plots = None

    # -----------------------------
    # RAW (only for buoy/external buoy locations with pairs)
    # -----------------------------
    if df_pairs is not None:
        raw, obs = _extract_raw_from_pairs(df_pairs)
        results.append(compute_metrics("raw", raw, obs))
        series["raw"] = raw
        obs_for_plots = obs

    # -----------------------------
    # VALIDATION FILES
    # -----------------------------
    discovered = discover_validation_runs(location)

    for corr_method, df_val in discovered.items():
        metric_row = _metrics_from_validation_file(corr_method, df_val)
        if metric_row is not None:
            results.append(metric_row)

        model, obs = _series_from_validation_file(df_val)
        series[corr_method] = model

        if obs_for_plots is None:
            obs_for_plots = obs

    if not results:
        raise ValueError(
            f"No evaluable data found for {location}. "
            f"Expected pairs and/or validation_*.csv files."
        )

    metrics = pd.DataFrame(results).set_index("method").sort_index()
    metrics.to_csv(out_dir / "metrics.csv")

    if obs_for_plots is not None and len(series) > 0:
        plot_pdf(obs_for_plots, series, out_dir)
        plot_cdf(obs_for_plots, series, out_dir)
        plot_qq(obs_for_plots, series, out_dir)
        plot_residuals(obs_for_plots, series, out_dir)

    print("\nEvaluation metrics:\n")
    print(metrics)

    print(f"\nSaved results in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bias correction methods using validation outputs."
    )
    parser.add_argument("--location", required=True)
    args = parser.parse_args()
    run(args.location)


if __name__ == "__main__":
    main()