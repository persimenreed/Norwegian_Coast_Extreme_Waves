"""
Mean Residual Life Plot
Auto-detect RAW vs CORRECTED
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# CONFIG
# ==========================

LOCATION = "fauskane"
MODE = "corrected"        # "raw" or "corrected"
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

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["hs"] = pd.to_numeric(df["hs"], errors="coerce")

df = df.dropna(subset=["time", "hs"]).sort_values("time")

hs = df["hs"].values


# ==========================
# COMPUTE MRL
# ==========================

thresholds = np.linspace(
    np.percentile(hs, 90),
    np.percentile(hs, 99.5),
    40
)

mean_excess = []

for u in thresholds:
    exceed = hs[hs > u]
    if len(exceed) > 0:
        mean_excess.append(np.mean(exceed - u))
    else:
        mean_excess.append(np.nan)


# ==========================
# PLOT
# ==========================

plt.figure(figsize=(8,5))
plt.plot(thresholds, mean_excess, marker='o')
plt.xlabel("Threshold u (m)")
plt.ylabel("Mean Excess")
plt.title(f"Mean Residual Life Plot ({MODE})")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/mean_residual_life.png", dpi=300)
plt.close()

print(f"Saved MRL plot to {OUTPUT_DIR}")