import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, probplot
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
LOCATION = "fauskane"
MODE = "raw"  # "raw" or "corrected"

BASE_DIR = f"EXTREME_VALUE_MODELLING/output/{LOCATION}/{MODE}"
OUT_DIR = f"{BASE_DIR}/GEV"
INPUT_PATH = f"{BASE_DIR}/annual_maxima.csv"

T_VALUES = np.arange(1, 51, dtype=float)   # 1..50 years
REPORT_PERIODS = [10, 20, 50]
N_BOOTSTRAP = 2000
CONF_LEVEL = 0.95
EPS = 1e-6

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv(INPUT_PATH, index_col=0)
data = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values

if len(data) < 5:
    raise ValueError("Not enough annual maxima to fit GEV reliably.")

# ======================================
# FIT ORIGINAL MODEL
# ======================================
shape, loc, scale = genextreme.fit(data)
xi = -shape

print(f"\nMODE: {MODE.upper()}, Location: {LOCATION}")
print("GEV parameters:")
print(f"Shape (xi): {xi:.4f}")
print(f"Location (mu): {loc:.4f}")
print(f"Scale (sigma): {scale:.4f}")

def gev_rl(T, c, l, s):
    p = np.clip(1 - 1 / np.asarray(T, dtype=float), EPS, 1 - EPS)
    return genextreme.ppf(p, c, loc=l, scale=s)

rl_full = gev_rl(T_VALUES, shape, loc, scale)

# ======================================
# BOOTSTRAP
# ======================================
boot_rl = np.full((N_BOOTSTRAP, len(T_VALUES)), np.nan)
boot_shape = np.full(N_BOOTSTRAP, np.nan)
boot_loc = np.full(N_BOOTSTRAP, np.nan)
boot_scale = np.full(N_BOOTSTRAP, np.nan)

for b in tqdm(range(N_BOOTSTRAP)):
    synthetic = genextreme.rvs(shape, loc=loc, scale=scale, size=len(data))
    try:
        b_shape, b_loc, b_scale = genextreme.fit(synthetic)
        boot_shape[b] = b_shape
        boot_loc[b] = b_loc
        boot_scale[b] = b_scale
        boot_rl[b, :] = gev_rl(T_VALUES, b_shape, b_loc, b_scale)
    except Exception:
        continue

mask = ~np.isnan(boot_rl).any(axis=1)
mask = mask & np.isfinite(boot_shape) & np.isfinite(boot_loc) & np.isfinite(boot_scale)
boot_rl = boot_rl[mask]
boot_shape = boot_shape[mask]
boot_loc = boot_loc[mask]
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
    "xi_hat_full": xi,
    "mu_hat_full": loc,
    "sigma_hat_full": scale,
    "xi_boot_mean": np.mean(-boot_shape),
    "xi_boot_std": np.std(-boot_shape),
    "mu_boot_mean": np.mean(boot_loc),
    "mu_boot_std": np.std(boot_loc),
    "sigma_boot_mean": np.mean(boot_scale),
    "sigma_boot_std": np.std(boot_scale),
    "n_bootstrap": len(boot_rl),
    "conf_level": CONF_LEVEL
})
out.to_csv(f"{BASE_DIR}/gev_return_levels_1_50.csv", index=False)

print("\nReturn levels (selected):")
for T in REPORT_PERIODS:
    i = np.where(T_VALUES == T)[0][0]
    print(f"{T}-year: {rl_full[i]:.2f} m [{ci_low[i]:.2f}, {ci_high[i]:.2f}] width={ci_high[i]-ci_low[i]:.2f}")

# ======================================
# PLOT
# ======================================
plt.figure(figsize=(8, 5))
plt.plot(T_VALUES, rl_full, label="GEV Return Level")
plt.fill_between(T_VALUES, ci_low, ci_high, alpha=0.25, label=f"{int(CONF_LEVEL*100)}% CI")
plt.scatter(REPORT_PERIODS,
            [rl_full[np.where(T_VALUES == t)[0][0]] for t in REPORT_PERIODS],
            color="red", zorder=3)
plt.xlabel("Return period (years)")
plt.ylabel("Return level Hs (m)")
plt.title(f"GEV Return Levels ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gev_return_levels_ci_1_50.png", dpi=300)
plt.close()

# ======================================
# QQ-PLOT
# ======================================
plt.figure(figsize=(6, 6))
probplot(data, dist=genextreme, sparams=(shape, loc, scale), plot=plt)
plt.ylabel("Observed Annual Maxima (m)")
plt.xlabel("Theoretical Quantiles")
plt.title(f"GEV QQ-Plot ({MODE})")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/gev_qqplot.png", dpi=300)
plt.close()

# ======================================
# GEV DISTRIBUTION (Empirical + Fitted)
# ======================================

# Grid over data range
x_min = min(data) * 0.9
x_max = max(data) * 1.2
x_grid = np.linspace(x_min, x_max, 500)

pdf_vals = genextreme.pdf(x_grid, shape, loc=loc, scale=scale)
cdf_vals = genextreme.cdf(x_grid, shape, loc=loc, scale=scale)
sf_vals  = genextreme.sf(x_grid, shape, loc=loc, scale=scale)

# ======================================
# EMPIRICAL CDF
# ======================================
sorted_data = np.sort(data)
n = len(sorted_data)
ecdf = np.arange(1, n + 1) / n
empirical_sf = 1 - ecdf

# ======================================
# PDF: Empirical (histogram) + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.hist(data, bins="auto", density=True, alpha=0.4, label="Empirical (Histogram)")
plt.plot(x_grid, pdf_vals, linewidth=2, label="Fitted GEV PDF")
plt.xlabel("Hs (m)")
plt.ylabel("Density")
plt.title(f"GEV PDF ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gev_pdf.png", dpi=300)
plt.close()

# ======================================
# CDF: Empirical + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.step(sorted_data, ecdf, where="post", label="Empirical CDF")
plt.plot(x_grid, cdf_vals, linewidth=2, label="Fitted GEV CDF")
plt.xlabel("Hs (m)")
plt.ylabel("Cumulative Probability")
plt.title(f"GEV CDF ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gev_cdf.png", dpi=300)
plt.close()

# ======================================
# Survival Function: Empirical + Fitted
# ======================================
plt.figure(figsize=(6, 4))
plt.step(sorted_data, empirical_sf, where="post", label="Empirical Survival")
plt.semilogy(x_grid, sf_vals, linewidth=2, label="Fitted GEV Survival")
plt.xlabel("Hs (m)")
plt.ylabel("Exceedance Probability")
plt.title(f"GEV Survival Function ({MODE})")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUT_DIR}/gev_survival.png", dpi=300)
plt.close()