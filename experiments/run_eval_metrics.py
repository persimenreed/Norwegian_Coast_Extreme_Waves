import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import format_path

TIME_COL = "time"
OBS_COL = "Significant_Wave_Height_Hm0"
MODEL_COL = "hs"
MODEL_CORR_COL = "hs_corrected"


def _read_csv(path):
    import pandas as pd

    if not path.exists():
        return None

    df = pd.read_csv(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def _validation_series(df, path):
    if MODEL_CORR_COL not in df.columns:
        raise ValueError(f"Validation file {path} is missing '{MODEL_CORR_COL}'.")
    if OBS_COL not in df.columns:
        raise ValueError(f"Validation file {path} is missing '{OBS_COL}'.")
    return df[MODEL_CORR_COL].to_numpy(), df[OBS_COL].to_numpy()


def _pairs_series(location):
    from src.bias_correction.data import load_pairs

    path = Path(format_path("pairs", location=location))
    if not path.exists():
        return None, None

    df = load_pairs(location)
    if MODEL_COL not in df.columns or OBS_COL not in df.columns:
        raise ValueError("Pairs file must contain raw model hs and observed Hs columns.")
    return df[MODEL_COL].to_numpy(), df[OBS_COL].to_numpy()


def _validation_runs(location):
    root = Path(format_path("validation", location=location, corr_method="__dummy__")).parent
    if not root.exists():
        return []

    runs = []
    for path in sorted(root.glob("validation_*.csv")):
        df = _read_csv(path)
        if df is not None:
            runs.append((path.stem.replace("validation_", "", 1),) + _validation_series(df, path))
    return runs


def run(location):
    import pandas as pd

    from src.eval_metrics.core import compute_metrics
    from src.eval_metrics.plot_diagnostics import plot_cdf, plot_pdf, plot_qq, plot_residuals

    print(f"\nRunning evaluation for {location}")

    out_dir = Path(f"results/eval_metrics/{location}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    series = {}
    obs_for_plots = None

    raw, raw_obs = _pairs_series(location)
    if raw is not None:
        results.append(compute_metrics("raw", raw, raw_obs))
        series["raw"] = raw
        obs_for_plots = raw_obs

    for corr_method, model, obs in _validation_runs(location):
        results.append(compute_metrics(corr_method, model, obs))
        series[corr_method] = model
        if obs_for_plots is None:
            obs_for_plots = obs

    if not results:
        raise ValueError(
            f"No evaluable data found for {location}. Expected pairs and/or validation_*.csv files."
        )

    metrics = pd.DataFrame(results).set_index("method").sort_index()
    metrics.to_csv(out_dir / "metrics.csv")

    if obs_for_plots is not None and series:
        plot_pdf(obs_for_plots, series, out_dir)
        plot_cdf(obs_for_plots, series, out_dir)
        plot_qq(obs_for_plots, series, out_dir)
        plot_residuals(obs_for_plots, series, out_dir)

    print("\nEvaluation metrics:\n")
    print(metrics)
    print(f"\nSaved results in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate correction methods using saved validation outputs."
    )
    parser.add_argument("--location", required=True)
    args = parser.parse_args()
    run(args.location)


if __name__ == "__main__":
    main()
