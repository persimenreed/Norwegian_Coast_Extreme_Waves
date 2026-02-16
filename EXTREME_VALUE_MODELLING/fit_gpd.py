"""
Fit GPD to POT peaks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto


INPUT_PATH = "EXTREME_VALUE_MODELLING/output/pot_peaks.csv"
THRESHOLD_PATH = "EXTREME_VALUE_MODELLING/output/threshold.txt"


# ==============================
# Load data
# ==============================

df = pd.read_csv(INPUT_PATH, index_col=0)
df.index = pd.to_datetime(df.index)
peaks = df.iloc[:, 0].values

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read())

print(f"Using threshold: {THRESHOLD:.2f} m")


# ==============================
# Compute excesses
# ==============================

excess = peaks - THRESHOLD


# ==============================
# Fit GPD
# ==============================

shape, loc, scale = genpareto.fit(excess, floc=0)

print("\nGPD parameters:")
print(f"Shape (xi): {shape:.4f}")
print(f"Scale (sigma): {scale:.4f}")


# ==============================
# Return levels
# ==============================

years = (df.index.year.max() - df.index.year.min()) + 1
lambda_rate = len(peaks) / years

RETURN_PERIODS = [10, 20, 50]

print("\nReturn levels:")

for T in RETURN_PERIODS:
    rl = THRESHOLD + (scale/shape) * (
        (lambda_rate*T)**shape - 1
    )
    print(f"{T}-year return level: {rl:.2f} m")


# ==============================
# Plot tail fit
# ==============================

x = np.linspace(0, excess.max(), 200)
cdf = genpareto.cdf(x, shape, loc=0, scale=scale)

plt.figure()
plt.plot(x, 1 - cdf)
plt.xlabel("Excess (m)")
plt.ylabel("Survival function")
plt.title("GPD Tail Fit")
plt.grid()
plt.savefig("EXTREME_VALUE_MODELLING/output/gpd_tail_fit.png", dpi=300)
