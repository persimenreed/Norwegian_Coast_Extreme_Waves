import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme
from tqdm import tqdm

from src.extreme_value_modelling.paths import resolve_output_dir
from src.extreme_value_modelling.common import dataset_name
from src.extreme_value_modelling.parameter_summary import update_parameter_summary
from src.extreme_value_modelling.distribution_plots import gev_plots
from src.settings import get_path_template, get_evt_bootstrap_samples

# ==========================
# CONFIG
# ==========================

T_VALUES = np.arange(1, 51, dtype=float)
N_BOOTSTRAP = get_evt_bootstrap_samples()
CONF_LEVEL = 0.95
EPS = 1e-6


def gev_return_level(T, c, loc, scale):

    p = np.clip(1 - 1 / np.asarray(T, dtype=float), EPS, 1 - EPS)

    return genextreme.ppf(p, c, loc=loc, scale=scale)


def run(location, mode, corr_method="qm", pooling=False, transfer=False,
        n_bootstrap=N_BOOTSTRAP, conf_level=CONF_LEVEL):

    dataset = dataset_name(mode, corr_method, pooling, transfer)
    root = Path(get_path_template("evt_results_root"))
    annual_path = root / location / "preprocessing" / "annual_maxima.csv"
    df = pd.read_csv(annual_path)

    if dataset not in df.columns:
        raise ValueError(f"{dataset} not found in annual_maxima.csv")

    data = pd.to_numeric(df[dataset], errors="coerce").dropna().values

    if len(data) < 5:
        raise ValueError("Too few annual maxima")

    # ==========================
    # FIT GEV
    # ==========================

    shape, loc, scale = genextreme.fit(data)
    xi = -shape
    rl_full = gev_return_level(T_VALUES, shape, loc, scale)

    # ==========================
    # BOOTSTRAP
    # ==========================

    boot_rl = np.full((n_bootstrap, len(T_VALUES)), np.nan)

    for b in tqdm(range(n_bootstrap), desc="GEV bootstrap"):
        synthetic = genextreme.rvs(shape, loc=loc, scale=scale, size=len(data))

        try:
            b_shape, b_loc, b_scale = genextreme.fit(synthetic)
            boot_rl[b] = gev_return_level(T_VALUES, b_shape, b_loc, b_scale)

        except Exception:
            continue

    boot_rl = boot_rl[~np.isnan(boot_rl).any(axis=1)]
    alpha = 1 - conf_level
    ci_low = np.quantile(boot_rl, alpha / 2, axis=0)
    ci_high = np.quantile(boot_rl, 1 - alpha / 2, axis=0)

    # ==========================
    # OUTPUT DIRECTORY
    # ==========================

    out_dir = resolve_output_dir(location, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # DISTRIBUTION DIAGNOSTICS
    # ==========================

    gev_plots(
        data=data,
        shape=shape,
        loc=loc,
        scale=scale,
        out_dir=out_dir,
        dataset=dataset
    )

    # ==========================
    # PARAMETER SUMMARY
    # ==========================

    update_parameter_summary({
        "location": location,
        "dataset": dataset,
        "model": "GEV",
        "xi": float(xi),
        "sigma": float(scale),
        "mu": float(loc),
        "lambda": None,
        "threshold": None,
        "years": None,
    })

    # ==========================
    # RETURN LEVEL TABLE
    # ==========================

    table = pd.DataFrame({
        "return_period": T_VALUES.astype(int),
        "return_level": rl_full,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_width": ci_high - ci_low,
        "n_bootstrap": len(boot_rl),
        "conf_level": conf_level
    })

    return table


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--location")
    parser.add_argument("--mode", choices=["raw", "corrected"])
    parser.add_argument("--corr-method", default="qm")
    parser.add_argument("--pooling", action="store_true")

    args = parser.parse_args()

    run(
        args.location,
        args.mode,
        args.corr_method,
        args.pooling
    )


if __name__ == "__main__":
    main()