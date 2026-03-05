import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from tqdm import tqdm

from src.extreme_value_modelling.paths import resolve_output_dir, resolve_input_path
from src.extreme_value_modelling.common import dataset_name
from src.extreme_value_modelling.parameter_summary import update_parameter_summary
from src.extreme_value_modelling.distribution_plots import gpd_plots
from src.extreme_value_modelling.extreme_preprocessing import compute_pot, load_data
from src.settings import get_thresholds, get_evt_bootstrap_samples

# ==========================
# CONFIG
# ==========================

T_VALUES = np.arange(1, 51, dtype=float)
N_BOOTSTRAP = get_evt_bootstrap_samples()
CONF_LEVEL = 0.95


def gpd_return_level(T, xi, sigma, threshold, lambda_rate):

    z = np.clip(lambda_rate * np.asarray(T, dtype=float), 1e-12, None)
    if abs(xi) < 1e-6:
        return threshold + sigma * np.log(z)

    return threshold + (sigma / xi) * (z ** xi - 1)


def run(location, mode, corr_method="qm", pooling=False, transfer=False,
        n_bootstrap=N_BOOTSTRAP, conf_level=CONF_LEVEL):

    dataset = dataset_name(mode, corr_method, pooling, transfer)
    thresholds = get_thresholds()
    q = thresholds.get("evt_threshold_quantile", 0.95)
    decluster = thresholds.get("decluster_hours", 48)

    # ==========================
    # LOAD POT PEAKS
    # ==========================

    evt_root = Path(__import__("src.settings", fromlist=["get_path_template"]).get_path_template("evt_results_root"))
    pot_path = evt_root / location / "preprocessing" / "pot_peaks.csv"
    df = pd.read_csv(pot_path)
    if dataset not in df.columns:
        raise ValueError(f"{dataset} not found in pot_peaks.csv")

    peaks = pd.to_numeric(df[dataset], errors="coerce").dropna().values

    if len(peaks) < 10:
        raise ValueError("Too few POT peaks")

    # ==========================
    # COMPUTE THRESHOLD + RATE
    # ==========================

    input_path = resolve_input_path(location, mode, corr_method, pooling)
    source = load_data(str(input_path))
    _, threshold, lambda_rate, total_years = compute_pot(source, q, decluster)
    excess = peaks - threshold
    excess = excess[excess > 0]

    if len(excess) < 10:
        raise ValueError("Too few excesses")

    # ==========================
    # FIT GPD
    # ==========================

    shape, _, scale = genpareto.fit(excess, floc=0)
    rl_full = gpd_return_level(T_VALUES, shape, scale, threshold, lambda_rate)

    # ==========================
    # BOOTSTRAP
    # ==========================

    boot_rl = np.full((n_bootstrap, len(T_VALUES)), np.nan)

    for b in tqdm(range(n_bootstrap), desc="GPD bootstrap"):
        synthetic = genpareto.rvs(shape, scale=scale, size=len(excess))

        try:
            b_shape, _, b_scale = genpareto.fit(synthetic, floc=0)
            boot_rl[b] = gpd_return_level(T_VALUES, b_shape, b_scale, threshold, lambda_rate)
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

    gpd_plots(
        excess=excess,
        shape=shape,
        scale=scale,
        threshold=threshold,
        out_dir=out_dir,
        dataset=dataset
    )

    # ==========================
    # PARAMETER SUMMARY
    # ==========================

    update_parameter_summary({
        "location": location,
        "dataset": dataset,
        "model": "GPD",
        "xi": float(shape),
        "sigma": float(scale),
        "mu": None,
        "lambda": float(lambda_rate),
        "threshold": float(threshold),
        "years": float(total_years),
    })

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