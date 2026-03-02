import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EPS = 1e-12

PAIRS_PATH = "TESTING/dataset/NORA3_fauskane_pairs.csv"
FULL_PATH = "TESTING/dataset/NORA3_wind_wave_fauskane_1969_2025.csv"


def clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def fit_weibull(data):
    data = clean_positive(data)
    c, loc, scale = stats.weibull_min.fit(data, floc=0)
    return stats.weibull_min(c, loc=loc, scale=scale)


def apply_qm(values, source_dist, target_dist):
    q = source_dist.cdf(values)
    q = np.clip(q, EPS, 1 - EPS)
    return target_dist.ppf(q)


def print_upper_quantiles(label, arr, probs):
    arr = clean_positive(arr)
    if len(arr) == 0:
        print(f"{label}: (no valid data)")
        return None
    qs = np.quantile(arr, probs)
    print(f"\n{label}")
    for p, v in zip(probs, qs):
        print(f"  q{p*100:6.2f}% = {v:8.3f} m")
    return qs


def main():

    # --------------------
    # Load paired dataset
    # --------------------
    df_pairs = pd.read_csv(PAIRS_PATH)

    hs_model = df_pairs["hs"].values
    hs_obs = df_pairs["Significant_Wave_Height_Hm0"].values

    # --------------------
    # Fit Weibull distributions
    # --------------------
    model_dist = fit_weibull(hs_model)
    obs_dist = fit_weibull(hs_obs)

    print("\n--- Weibull parameters ---")
    print("Model (NORA3):", model_dist.args, model_dist.kwds)
    print("Obs (Buoy):   ", obs_dist.args, obs_dist.kwds)

    # --------------------
    # Extremes (simple)
    # --------------------
    print("\n--- Extremes ---")
    print("Max NORA3 overlap:", float(np.nanmax(clean_positive(hs_model))))
    print("Max Buoy overlap:", float(np.nanmax(clean_positive(hs_obs))))

    # --------------------
    # Load full hindcast
    # --------------------
    df_full = pd.read_csv(FULL_PATH, comment="#")
    hs_full = df_full["hs"].values

    # --------------------
    # Apply QM to full hindcast
    # --------------------
    hs_full_qm = apply_qm(hs_full, model_dist, obs_dist)

    print("\n--- Full dataset extremes ---")
    print("Max NORA3 full:", float(np.nanmax(clean_positive(hs_full))))
    print("Max QM corrected:", float(np.nanmax(clean_positive(hs_full_qm))))

    # --------------------
    # Upper-quantile diagnostics (important for EVT)
    # --------------------
    probs = np.array([0.99, 0.995, 0.999, 0.9995, 0.9999])

    q_overlap_model = print_upper_quantiles("Overlap NORA3 (raw)", hs_model, probs)
    q_overlap_obs = print_upper_quantiles("Overlap Buoy (obs)", hs_obs, probs)
    q_full_raw = print_upper_quantiles("Full hindcast NORA3 (raw)", hs_full, probs)
    q_full_qm = print_upper_quantiles("Full hindcast (QM corrected)", hs_full_qm, probs)

    if q_full_raw is not None and q_full_qm is not None:
        print("\nFull hindcast: QM effect at upper quantiles (QM - raw)")
        for p, dv in zip(probs, (q_full_qm - q_full_raw)):
            print(f"  Δq{p*100:6.2f}% = {dv:8.3f} m")

    # Extra: how many values are in the top tail (sanity)
    hs_full_c = clean_positive(hs_full)
    if len(hs_full_c) > 0:
        thresh = np.quantile(hs_full_c, 0.999)
        frac = float(np.mean(hs_full_c >= thresh))
        print(f"\nSanity: fraction >= raw q99.9% threshold ({thresh:.3f} m): {frac*100:.3f}%")

    # --------------------
    # Plot distributions
    # --------------------
    x = np.linspace(
        0,
        max(np.nanmax(clean_positive(hs_model)), np.nanmax(clean_positive(hs_obs))) * 1.2,
        500
    )

    plt.figure()
    plt.hist(clean_positive(hs_model), bins=50, density=True, alpha=0.5, label="NORA3 (overlap)")
    plt.hist(clean_positive(hs_obs), bins=50, density=True, alpha=0.5, label="Buoy (overlap)")

    plt.plot(x, model_dist.pdf(x), label="Weibull fit NORA3")
    plt.plot(x, obs_dist.pdf(x), label="Weibull fit Buoy")

    plt.xlabel("Significant Wave Height (m)")
    plt.ylabel("Density")
    plt.title("Weibull Distribution Fits")
    plt.legend()
    plt.savefig("TESTING/output/qm_weibull_fits.png", dpi=300)


if __name__ == "__main__":
    main()