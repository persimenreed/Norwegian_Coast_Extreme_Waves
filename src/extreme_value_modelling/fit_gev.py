import argparse
import numpy as np
import pandas as pd
from scipy.stats import genextreme

from src.extreme_value_modelling.common import (
    BOOTSTRAP_SAMPLES,
    CONF_LEVEL,
    RETURN_PERIOD_GRID,
    bootstrap_confidence_interval,
    dataset_name,
    return_level_table,
)
from src.extreme_value_modelling.distribution_plots import gev_plots
from src.extreme_value_modelling.parameter_summary import update_parameter_summary
from src.extreme_value_modelling.paths import resolve_output_dir, resolve_preprocessing_dir

EPS = 1e-6


def gev_return_level(return_periods, shape, loc, scale):
    periods = np.asarray(return_periods, dtype=float)
    p = np.clip(1.0 - 1.0 / periods, EPS, 1.0 - EPS)
    return genextreme.ppf(p, shape, loc=loc, scale=scale)


def run(
    location,
    mode,
    corr_method="pqm",
    transfer_source=None,
    n_bootstrap=BOOTSTRAP_SAMPLES,
    conf_level=CONF_LEVEL,
):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)
    df = pd.read_csv(resolve_preprocessing_dir(location) / "annual_maxima.csv")

    if dataset not in df.columns:
        raise ValueError(f"{dataset} not found in annual_maxima.csv")

    data = pd.to_numeric(df[dataset], errors="coerce").dropna().to_numpy()
    if len(data) < 5:
        raise ValueError("Too few annual maxima")

    shape, loc, scale = genextreme.fit(data)
    return_levels = gev_return_level(RETURN_PERIOD_GRID, shape, loc, scale)
    ci_low, ci_high, n_success = bootstrap_confidence_interval(
        n_bootstrap,
        make_levels=lambda: gev_return_level(
            RETURN_PERIOD_GRID,
            *genextreme.fit(genextreme.rvs(shape, loc=loc, scale=scale, size=len(data))),
        ),
        desc="GEV bootstrap",
        conf_level=conf_level,
    )

    out_dir = resolve_output_dir(location, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    gev_plots(data=data, shape=shape, loc=loc, scale=scale, out_dir=out_dir, dataset=dataset)

    update_parameter_summary(
        {
            "location": location,
            "dataset": dataset,
            "model": "GEV",
            "xi": float(-shape),
            "sigma": float(scale),
            "mu": float(loc),
            "lambda": None,
            "threshold": None,
            "years": None,
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
    run(args.location, args.mode, args.corr_method, args.transfer_source)


if __name__ == "__main__":
    main()
