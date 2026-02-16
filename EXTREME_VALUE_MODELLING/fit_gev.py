"""
Fit GEV to annual maxima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme


INPUT_PATH = "EXTREME_VALUE_MODELLING/output/annual_maxima.csv"
RETURN_PERIODS = [10, 20, 50]


# ==============================
# Load data
# ==============================

df = pd.read_csv(INPUT_PATH, index_col=0)
data = df.values.flatten()


# ==============================
# Fit GEV
# ==============================

shape, loc, scale = genextreme.fit(data)

print("GEV parameters:")
print(f"Shape (xi): {-shape:.4f}")  # scipy uses reversed sign
print(f"Location (mu): {loc:.4f}")
print(f"Scale (sigma): {scale:.4f}")


# ==============================
# Return levels
# ==============================

return_levels = {}

for T in RETURN_PERIODS:
    prob = 1 - 1/T
    rl = genextreme.ppf(prob, shape, loc=loc, scale=scale)
    return_levels[T] = rl
    print(f"{T}-year return level: {rl:.2f} m")


# ==============================
# Plot return level curve
# ==============================

T = np.linspace(1.1, 100, 200)
prob = 1 - 1/T
rl_curve = genextreme.ppf(prob, shape, loc=loc, scale=scale)

plt.figure()
plt.plot(T, rl_curve)
plt.scatter(RETURN_PERIODS, [return_levels[t] for t in RETURN_PERIODS])
plt.xlabel("Return period (years)")
plt.ylabel("Return level Hs (m)")
plt.title("GEV Return Level Curve")
plt.grid()
plt.show()
