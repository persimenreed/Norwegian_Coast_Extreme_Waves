import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
LOCATION = "fedjeosen"
MODE = "corrected"  # "raw" or "corrected"

BASE_DIR = f"EXTREME_VALUE_MODELLING/output/{LOCATION}/{MODE}"
OUT_DIR = f"{BASE_DIR}/GPD"
INPUT_PATH = f"{BASE_DIR}/pot_peaks.csv"
THRESHOLD_PATH = f"{BASE_DIR}/threshold.txt"

T_VALUES = np.arange(1, 51, dtype=float)   # 1..50 years
REPORT_PERIODS = [10, 20, 50]
N_BOOTSTRAP = 2000
CONF_LEVEL = 0.95

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv(INPUT_PATH, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")
df = df[~df.index.isna()]

peaks = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
if len(peaks) < 10:
    raise ValueError("Not enough POT peaks to fit GPD reliably.")

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read())

excess = peaks - THRESHOLD
if np.any(excess <= 0):
    raise ValueError("POT peaks contain values at or below threshold.")

with open(f"{BASE_DIR}/lambda_year.txt") as f:
    lambda_rate = float(f.read())

# ======================================
# FIT ORIGINAL
# ======================================
shape, loc, scale = genpareto.fit(excess, floc=0)

print(f"\nMODE: {MODE.upper()}, Location: {LOCATION}")
print("GPD parameters:")
print(f"Shape (xi): {shape:.4f}")
print(f"Scale (sigma): {scale:.4f}")
print(f"Exceedance rate (lambda): {lambda_rate:.4f} / year")

def gpd_return_level(T, xi, sigma):
    z = np.clip(lambda_rate * np.asarray(T, dtype=float), 1e-12, None)
    if abs(xi) < 1e-6:
        return THRESHOLD + sigma * np.log(z)
    return THRESHOLD + (sigma / xi) * (z**xi - 1)

rl_full = gpd_return_level(T_VALUES, shape, scale)

# ======================================
# BOOTSTRAP
# ======================================
boot_rl = np.full((N_BOOTSTRAP, len(T_VALUES)), np.nan)
boot_shape = np.full(N_BOOTSTRAP, np.nan)
boot_scale = np.full(N_BOOTSTRAP, np.nan)

for b in tqdm(range(N_BOOTSTRAP)):
    synthetic = genpareto.rvs(shape, loc=0, scale=scale, size=len(excess))
    try:
        b_shape, _, b_scale = genpareto.fit(synthetic, floc=0)
        boot_shape[b] = b_shape
        boot_scale[b] = b_scale
        boot_rl[b, :] = gpd_return_level(T_VALUES, b_shape, b_scale)
    except Exception:
        continue

mask = ~np.isnan(boot_rl).any(axis=1)
mask = mask & np.isfinite(boot_shape) & np.isfinite(boot_scale)
boot_rl = boot_rl[mask]
boot_shape = boot_shape[mask]
boot_scale = boot_scale[mask]

if len(boot_rl) == 0:
    raise RuntimeError("All bootstrap fits failed.")

alpha = 1 - CONF_LEVEL
ci_low = np.quantile(boot_rl, alpha / 2, axis=0)
ci_high = np.quantile(boot_rl, 1 - alpha / 2, axis=0)

# ======================================
# SAVE TABLE (1..50)
# ======================================
out = pd.DataFrame({
    "return_period": T_VALUES.astype(int),
    "return_level": rl_full,
    "ci_lower": ci_low,
    "ci_upper": ci_high,
    "ci_width": ci_high - ci_low,
    "xi_hat_full": shape,
    "sigma_hat_full": scale,
    "threshold": THRESHOLD,
    "lambda_rate": lambda_rate,
    "xi_boot_mean": np.mean(boot_shape),
    "xi_boot_std": np.std(boot_shape),
    "sigma_boot_mean": np.mean(boot_scale),
    "sigma_boot_std": np.std(boot_scale),
    "n_bootstrap": len(boot_rl),
    "conf_level": CONF_LEVEL
})
out.to_csv(f"{BASE_DIR}/gpd_return_levels_1_50.csv", index=False)

print("\nReturn levels (selected):")
for T in REPORT_PERIODS:
    i = np.where(T_VALUES == T)[0][0]
    print(f"{T}-year: {rl_full[i]:.2f} m [{ci_low[i]:.2f}, {ci_high[i]:.2f}] width={ci_high[i]-ci_low[i]:.2f}")

# ======================================
# RETURN LEVEL PLOT
# ======================================
plt.figure(figsize=(8, 5))
plt.plot(T_VALUES, rl_full, label="GPD Return Level")
plt.fill_between(T_VALUES, ci_low, ci_high, alpha=0.25, label=f"{int(CONF_LEVEL*100)}% CI")
plt.scatter(REPORT_PERIODS,
            [rl_full[np.where(T_VALUES == t)[0][0]] for t in REPORT_PERIODS],
            color="red", zorder=3)
plt.xlabel("Return period (years)")
plt.ylabel("Return level Hs (m)")
plt.title(f"GPD Return Levels ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gpd_return_levels_ci_1_50.png", dpi=300)
plt.close()

# ======================================
# QQ-PLOT
# ======================================
sorted_excess = np.sort(excess)
p = (np.arange(1, len(excess) + 1) - 0.5) / len(excess)
theoretical = genpareto.ppf(p, shape, loc=0, scale=scale)

plt.figure(figsize=(6, 6))
plt.scatter(theoretical, sorted_excess)
m = max(np.max(theoretical), np.max(sorted_excess))
plt.plot([0, m], [0, m], "r--")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Observed Excess (m)")
plt.title(f"GPD QQ-Plot ({MODE})")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/gpd_qqplot.png", dpi=300)
plt.close()

# ======================================
# GPD DISTRIBUTION (Empirical + Fitted)
# ======================================

# Grid over excess range
x_min = 0
x_max = max(excess) * 1.2
x_grid = np.linspace(x_min, x_max, 500)

pdf_vals = genpareto.pdf(x_grid, shape, loc=0, scale=scale)
cdf_vals = genpareto.cdf(x_grid, shape, loc=0, scale=scale)
sf_vals  = genpareto.sf(x_grid, shape, loc=0, scale=scale)

# ======================================
# EMPIRICAL CDF (EXCESSES)
# ======================================
sorted_excess = np.sort(excess)
n = len(sorted_excess)
ecdf = np.arange(1, n + 1) / n
empirical_sf = 1 - ecdf

# ======================================
# PDF: Empirical (histogram) + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.hist(excess, bins="auto", density=True, alpha=0.4,
         label="Empirical (Histogram)")
plt.plot(x_grid, pdf_vals, linewidth=2, label="Fitted GPD PDF")
plt.xlabel("Excess (m)")
plt.ylabel("Density")
plt.title(f"GPD PDF ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gpd_pdf.png", dpi=300)
plt.close()

# ======================================
# CDF: Empirical + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.step(sorted_excess, ecdf, where="post", label="Empirical CDF")
plt.plot(x_grid, cdf_vals, linewidth=2, label="Fitted GPD CDF")
plt.xlabel("Excess (m)")
plt.ylabel("Cumulative Probability")
plt.title(f"GPD CDF ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gpd_cdf.png", dpi=300)
plt.close()

# ======================================
# Survival Function: Empirical + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.step(sorted_excess, empirical_sf, where="post",
         label="Empirical Survival")
plt.semilogy(x_grid, sf_vals, linewidth=2,
             label="Fitted GPD Survival")
plt.xlabel("Excess (m)")
plt.ylabel("Exceedance Probability")
plt.title(f"GPD Survival Function ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gpd_survival.png", dpi=300)
plt.close()