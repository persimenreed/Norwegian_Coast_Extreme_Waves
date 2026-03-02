# Extreme Value Modelling – NORA3 Site Analysis

This folder contains scripts used to perform Extreme Value Theory (EVT) analysis on NORA3 significant wave height (Hs) data for the selected site.

The workflow follows classical EVT methodology:

- Block Maxima → GEV
- Peaks Over Threshold (POT) → GPD

---

# 1. extreme_preprocessing.py

Purpose:
- Loads NORA3 site dataset (1969–2025)
- Computes annual maxima (for GEV)
- Computes POT peaks using:
  - 95% quantile threshold
  - 48-hour declustering window
- Stores:
  - annual_maxima.csv
  - pot_peaks.csv
  - threshold.txt

This script must be run before any fitting.

---

# 2. fit_gev.py

Purpose:
- Fits Generalized Extreme Value (GEV) distribution to annual maxima.
- Estimates:
  - Shape (ξ)
  - Location (μ)
  - Scale (σ)
- Computes return levels (10, 20, 50 years).
- Computes 95% confidence intervals using parametric bootstrap.
- Produces:
  - Return level curve with CI
  - GEV QQ-plot

Used as block maxima baseline model.

---

# 3. fit_gpd.py

Purpose:
- Fits Generalized Pareto Distribution (GPD) to declustered excesses.
- Uses threshold from preprocessing step.
- Estimates:
  - Shape (ξ)
  - Scale (σ)
- Computes return levels (10, 20, 50 years).
- Computes 95% confidence intervals using parametric bootstrap.
- Produces:
  - Return level curve with CI
  - GPD QQ-plot

Used as POT baseline model.

---

# 4. mean_residual_life.py

Purpose:
- Produces Mean Residual Life (MRL) plot.
- Used to assess threshold suitability.
- Linear region indicates valid threshold for GPD modelling.

---

# 5. threshold_stability.py

Purpose:
- Evaluates GPD shape parameter ξ as function of threshold.
- Stable plateau indicates appropriate threshold region.
- Used to validate chosen 95% threshold.

---

