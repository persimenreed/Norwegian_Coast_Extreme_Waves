import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAIRS_PATH = "TESTING/dataset/NORA3_fauskane_pairs.csv"
FULL_PATH = "TESTING/dataset/NORA3_wind_wave_fauskane_1969_2025.csv"


def clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def empirical_qm_map(x, model_train, obs_train, method="clip"):
    """
    Empirical quantile mapping.

    method:
      - "clip": values above/below model_train range are mapped to obs_train max/min (hard cap).
      - "linear": linear extrapolation beyond ends (risky, but avoids hard cap).
    """
    x = np.asarray(x, float)

    m = clean_positive(model_train)
    o = clean_positive(obs_train)

    if len(m) < 50 or len(o) < 50:
        raise ValueError("Too few samples for empirical QM (need ~50+).")

    m_sorted = np.sort(m)
    o_sorted = np.sort(o)

    # plotting positions in (0,1) including ends
    n_m = len(m_sorted)
    n_o = len(o_sorted)
    p_m = np.linspace(0.0, 1.0, n_m)
    p_o = np.linspace(0.0, 1.0, n_o)

    # Step 1: x -> q using empirical CDF of model (invert using interpolation)
    if method == "clip":
        q = np.interp(x, m_sorted, p_m, left=0.0, right=1.0)
        # Step 2: q -> y using empirical inverse CDF of obs
        y = np.interp(q, p_o, o_sorted, left=o_sorted[0], right=o_sorted[-1])
        return y

    if method == "linear":
        # allow linear extrapolation beyond ends (can explode in tail)
        q = np.interp(x, m_sorted, p_m, left=p_m[0], right=p_m[-1])
        y = np.interp(q, p_o, o_sorted, left=o_sorted[0], right=o_sorted[-1])
        return y

    raise ValueError("method must be 'clip' or 'linear'")


def main():
    df_pairs = pd.read_csv(PAIRS_PATH)
    hs_model = df_pairs["hs"].values
    hs_obs = df_pairs["Significant_Wave_Height_Hm0"].values

    df_full = pd.read_csv(FULL_PATH, comment="#")
    hs_full = df_full["hs"].values

    hs_model_c = clean_positive(hs_model)
    hs_obs_c = clean_positive(hs_obs)
    hs_full_c = clean_positive(hs_full)

    print("\n--- Overlap extremes ---")
    print("Max NORA3 overlap:", float(np.max(hs_model_c)))
    print("Max Buoy overlap:", float(np.max(hs_obs_c)))

    print("\n--- Full dataset extremes ---")
    print("Max NORA3 full:", float(np.max(hs_full_c)))

    # Empirical QM (clip = hard cap)
    hs_full_qm = empirical_qm_map(hs_full, hs_model, hs_obs, method="clip")
    hs_full_qm_c = clean_positive(hs_full_qm)

    print("Max empirical-QM corrected (clip):", float(np.max(hs_full_qm_c)))

    # Diagnostics: how many full hindcast values exceed overlap model max?
    mmax = float(np.max(hs_model_c))
    frac_above = float(np.mean(hs_full_c > mmax))
    print("\n--- Diagnostics ---")
    print(f"Fraction of full hindcast hs > overlap max model hs ({mmax:.2f} m): {frac_above*100:.2f}%")

    # If empirical QM is clipping, those points will map to (near) obs max
    omax = float(np.max(hs_obs_c))
    near_cap = np.mean(hs_full_qm_c >= (0.999 * omax))
    print(f"Fraction of corrected values >= 0.999 * obs max ({omax:.2f} m): {near_cap*100:.2f}%")

    # Plot distributions (overlap + corrected full)
    plt.figure()
    plt.hist(hs_model_c, bins=60, density=True, alpha=0.45, label="NORA3 (overlap)")
    plt.hist(hs_obs_c, bins=60, density=True, alpha=0.45, label="Buoy (overlap)")
    plt.hist(hs_full_qm_c, bins=60, density=True, alpha=0.35, label="Full hindcast (empirical QM, clip)")

    plt.xlabel("Significant Wave Height (m)")
    plt.ylabel("Density")
    plt.title("Empirical Quantile Mapping Distributions (with clipping)")
    plt.legend()
    plt.savefig("TESTING/output/empirical_qm_distributions.png", dpi=300)


if __name__ == "__main__":
    main()