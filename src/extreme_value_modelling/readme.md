# Extreme Value Modelling

This module runs the EVT stage for one hindcast dataset at a time:

- annual maxima -> GEV
- POT exceedances -> GPD

The active pipeline is:

1. `extreme_preprocessing.py`
   Builds `annual_maxima.csv` and `pot_peaks.csv` from the input hindcast using fixed defaults:
   - wave-height column: `hs`
   - time column: `time`
   - POT threshold: 95th percentile
   - declustering window: 48 hours

2. `fit_gev.py`
   Fits a GEV model from `annual_maxima.csv`, writes plots, and returns 1-50 year return levels with bootstrap confidence intervals.

3. `fit_gpd.py`
   Fits a GPD model from `pot_peaks.csv`, recomputes threshold and exceedance rate from the source hindcast, writes plots, and returns 1-50 year return levels with bootstrap confidence intervals.

4. `diagnostics.py`
   Writes the threshold-diagnostic figure for the selected dataset when requested.

Use `experiments/run_extreme_value_modelling.py` to run the full stage for:

- one location and all methods
- one location and one method
- one method across all locations
