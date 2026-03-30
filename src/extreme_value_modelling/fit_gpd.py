import argparse

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from src.extreme_value_modelling.common import (
    BOOTSTRAP_SAMPLES,
    CONF_LEVEL,
    RETURN_PERIOD_GRID,
    DECLUSTER_HOURS,
    THRESHOLD_QUANTILE,
    bootstrap_confidence_interval,
    dataset_name,
    return_level_table,
)
from src.extreme_value_modelling.distribution_plots import gpd_plots
from src.extreme_value_modelling.extreme_preprocessing import compute_pot, load_data
from src.extreme_value_modelling.parameter_summary import update_parameter_summary
from src.extreme_value_modelling.paths import (
    resolve_input_path,
    resolve_output_dir,
    resolve_preprocessing_dir,
)


def gpd_return_level(return_periods, xi, sigma, threshold, lambda_rate):
    z = np.clip(lambda_rate * np.asarray(return_periods, dtype=float), 1e-12, None)
    if abs(xi) < 1e-6:
        return threshold + sigma * np.log(z)
    return threshold + (sigma / xi) * (z ** xi - 1)


def run(
    location,
    mode,
    corr_method="pqm",
    transfer_source=None,
    n_bootstrap=BOOTSTRAP_SAMPLES,
    conf_level=CONF_LEVEL,
):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)
    df = pd.read_csv(resolve_preprocessing_dir(location) / "pot_peaks.csv")

    if dataset not in df.columns:
        raise ValueError(f"{dataset} not found in pot_peaks.csv")

    peaks = pd.to_numeric(df[dataset], errors="coerce").dropna().to_numpy()
    if len(peaks) < 10:
        raise ValueError("Too few POT peaks")

    input_path = resolve_input_path(
        location,
        mode,
        corr_method=corr_method,
        transfer_source=transfer_source,
    )
    source = load_data(input_path)
    _, threshold, lambda_rate, total_years = compute_pot(
        source,
        quantile=THRESHOLD_QUANTILE,
        decluster_hours=DECLUSTER_HOURS,
    )

    excess = peaks - threshold
    excess = excess[excess > 0]

    if len(excess) < 10:
        raise ValueError("Too few excesses")

    shape, _, scale = genpareto.fit(excess, floc=0)
    return_levels = gpd_return_level(RETURN_PERIOD_GRID, shape, scale, threshold, lambda_rate)
    ci_low, ci_high, n_success = bootstrap_confidence_interval(
        n_bootstrap,
        make_levels=lambda: gpd_return_level(
            RETURN_PERIOD_GRID,
            *genpareto.fit(genpareto.rvs(shape, scale=scale, size=len(excess)), floc=0)[::2],
            threshold,
            lambda_rate,
        ),
        desc="GPD bootstrap",
        conf_level=conf_level,
    )

    out_dir = resolve_output_dir(location, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpd_plots(
        excess=excess,
        shape=shape,
        scale=scale,
        threshold=threshold,
        out_dir=out_dir,
        dataset=dataset
    )

    update_parameter_summary(
        {
            "location": location,
            "dataset": dataset,
            "model": "GPD",
            "xi": float(shape),
            "sigma": float(scale),
            "mu": None,
            "lambda": float(lambda_rate),
            "threshold": float(threshold),
            "years": float(total_years),
        }
    )

    return return_level_table(
        RETURN_PERIOD_GRID,
        return_levels,
        ci_low,
        ci_high,
        n_success,
        conf_level=conf_level,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location")
    parser.add_argument("--mode", choices=["raw", "corrected"])
    parser.add_argument("--corr-method", default="pqm")
    parser.add_argument("--transfer-source", default=None)
    args = parser.parse_args()

    run(
        args.location,
        args.mode,
        args.corr_method,
        args.transfer_source,
    )


if __name__ == "__main__":
    main()
