"""
Threshold Stability Plot
Auto-detect RAW vs CORRECTED
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto


# ==========================
# CONFIG
# ==========================

LOCATION = "fauskane"
MODE = "raw"        # "raw" or "corrected"
CORR_METHOD = "qm"

if MODE == "raw":
    INPUT_PATH = f"DATA_EXTRACTION/nora3_locations/NORA3_wind_wave_{LOCATION}_1969_2025.csv"
elif MODE == "corrected":
    INPUT_PATH = f"BIAS_CORRECTION_V1/output/{LOCATION}/hindcast_corrected_{CORR_METHOD}.csv"
else:
    raise ValueError("MODE must be 'raw' or 'corrected'")

OUTPUT_DIR = f"EXTREME_VALUE_MODELLING/output/{LOCATION}/{MODE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(INPUT_PATH, comment="#")

df["hs"] = pd.to_numeric(df["hs"], errors="coerce")
df = df.dropna(subset=["hs"])

hs = df["hs"].values


# ==========================
# COMPUTE THRESHOLD STABILITY
# ==========================

thresholds = np.linspace(
    np.percentile(hs, 90),
    np.percentile(hs, 99),
    30
)

xis = []

for u in thresholds:
    excess = hs[hs > u] - u
    if len(excess) > 30:
        xi, _, _ = genpareto.fit(excess, floc=0)
        xis.append(xi)
    else:
        xis.append(np.nan)


# ==========================
# PLOT
# ==========================

plt.figure(figsize=(8,5))
plt.plot(thresholds, xis, marker='o')
plt.xlabel("Threshold u (m)")
plt.ylabel("Shape parameter ξ")
plt.title(f"Threshold Stability Plot ({MODE})")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/threshold_stability.png", dpi=300)
plt.close()

print(f"Saved threshold stability plot to {OUTPUT_DIR}")