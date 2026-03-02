"""
Threshold Stability Comparison: RAW vs CORRECTED
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from tqdm import tqdm


# ==========================
# CONFIG
# ==========================

LOCATION = "fauskane"
CORR_METHOD = "qm"
N_BOOT = 300
CONF_LEVEL = 0.95

RAW_PATH = f"DATA_EXTRACTION/nora3_locations/NORA3_wind_wave_{LOCATION}_1969_2025.csv"
CORR_PATH = f"BIAS_CORRECTION_V1/output/{LOCATION}/hindcast_corrected_{CORR_METHOD}.csv"

OUTPUT_DIR = f"EXTREME_VALUE_MODELLING/output/{LOCATION}/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

alpha = (1 - CONF_LEVEL) / 2


# ==========================
# FUNCTION
# ==========================

def compute_threshold_stability(path):

    df = pd.read_csv(path, comment="#")
    df["hs"] = pd.to_numeric(df["hs"], errors="coerce")
    df = df.dropna(subset=["hs"])
    hs = df["hs"].values

    thresholds = np.linspace(
        np.percentile(hs, 90),
        np.percentile(hs, 99),
        30
    )

    xis, lower_ci, upper_ci = [], [], []

    for u in tqdm(thresholds, leave=False):
        excess = hs[hs > u] - u
        
        if len(excess) > 30:
            xi_hat, _, _ = genpareto.fit(excess, floc=0)
            xis.append(xi_hat)

            boot = []
            for _ in range(N_BOOT):
                sample = np.random.choice(excess, len(excess), replace=True)
                try:
                    xi_b, _, _ = genpareto.fit(sample, floc=0)
                    boot.append(xi_b)
                except:
                    continue

            if len(boot) > 20:
                lower_ci.append(np.percentile(boot, 100*alpha))
                upper_ci.append(np.percentile(boot, 100*(1-alpha)))
            else:
                lower_ci.append(np.nan)
                upper_ci.append(np.nan)
        else:
            xis.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)

    return thresholds, np.array(xis), np.array(lower_ci), np.array(upper_ci)


# ==========================
# COMPUTE BOTH
# ==========================

print("Processing RAW...")
thr_raw, xi_raw, lo_raw, hi_raw = compute_threshold_stability(RAW_PATH)

print("Processing CORRECTED...")
thr_corr, xi_corr, lo_corr, hi_corr = compute_threshold_stability(CORR_PATH)


# ==========================
# PLOT
# ==========================

plt.figure(figsize=(9,6))

# RAW
plt.plot(thr_raw, xi_raw, color="red", label="RAW ξ")
plt.fill_between(thr_raw, lo_raw, hi_raw, color="red", alpha=0.2)

# CORRECTED
plt.plot(thr_corr, xi_corr, color="blue", label="CORRECTED ξ")
plt.fill_between(thr_corr, lo_corr, hi_corr, color="blue", alpha=0.2)

plt.axhline(0, linestyle="--", color="black", linewidth=1)

plt.xlabel("Threshold u (m)")
plt.ylabel("Shape parameter ξ")
plt.title("Threshold Stability Comparison - Raw vs QM")
plt.legend()
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/threshold_stability_comparison.png", dpi=300)
plt.close()

print("Saved comparison plot.")