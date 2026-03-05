import argparse
import sys
from pathlib import Path

# allow src imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.settings import format_path, get_methods
from src.eval_metrics.core import compute_metrics
from src.eval_metrics.plots_diagnostics import (
    plot_pdf,
    plot_cdf,
    plot_qq,
    plot_residuals,
)

OBS_COL = "Significant_Wave_Height_Hm0"
TIME_COL = "time"


def load_corrected(location, method, prefix):

    path = Path(
        f"data/output/{location}/hindcast_corrected_{prefix}_{method}.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    return df


def align_with_pairs(df_pairs, df_model):

    df = df_pairs[[TIME_COL, OBS_COL]].merge(
        df_model[[TIME_COL, "hs"]],
        on=TIME_COL,
        how="inner",
    )

    obs = df[OBS_COL].values
    model = df["hs"].values

    return model, obs


def run(location):

    print(f"\nRunning evaluation for {location}")

    # ----------------------------------
    # Load buoy overlap pairs
    # ----------------------------------

    pairs_path = format_path("pairs", location=location)
    df_pairs = pd.read_csv(pairs_path)

    df_pairs[TIME_COL] = pd.to_datetime(df_pairs[TIME_COL])

    obs = df_pairs[OBS_COL].values
    raw = df_pairs["hs"].values

    methods = get_methods()

    results = []
    series = {}

    # ----------------------------------
    # RAW
    # ----------------------------------

    results.append(compute_metrics("raw", raw, obs))
    series["raw"] = raw

    # ----------------------------------
    # LOCAL + POOLED + TRANSFER
    # ----------------------------------

    for m in methods:

        # -------- local --------

        df_local = load_corrected(location, m, "local")

        if df_local is not None:
            model, obs_local = align_with_pairs(df_pairs, df_local)
            name = f"local_{m}"
            results.append(
                compute_metrics(name, model, obs_local)
            )

            series[name] = model

        # -------- transfer --------

        df_transfer = load_corrected(location, m, "transfer")

        if df_transfer is not None:
            model, obs_transfer = align_with_pairs(df_pairs, df_transfer)
            name = f"transfer_{m}"
            results.append(
                compute_metrics(name, model, obs_transfer)
            )

            series[name] = model

        # -------- pooled --------

        df_pool = load_corrected(location, m, "pooled")

        if df_pool is not None:
            model, obs_pool = align_with_pairs(df_pairs, df_pool)
            name = f"pooled_{m}"
            results.append(
                compute_metrics(name, model, obs_pool)
            )

            series[name] = model

    # ----------------------------------
    # Build results dataframe
    # ----------------------------------

    metrics = pd.DataFrame(results).set_index("method")

    out_dir = Path(f"results/eval_metrics/{location}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out_dir / "metrics.csv")

    # ----------------------------------
    # plots
    # ----------------------------------

    plot_pdf(obs, series, out_dir)
    plot_cdf(obs, series, out_dir)
    plot_qq(obs, series, out_dir)
    plot_residuals(obs, series, out_dir)

    print("\nEvaluation metrics:\n")
    print(metrics)

    print(f"\nSaved results in {out_dir}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate bias correction methods"
    )

    parser.add_argument("--location", required=True)

    args = parser.parse_args()

    run(args.location)


if __name__ == "__main__":
    main()