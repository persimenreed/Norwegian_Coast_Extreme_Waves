"""
EXTREME_VALUE_MODELLING/extreme_preprocessing.py

Best-practice preprocessing for EVT:
- Annual maxima
- POT with 48h declustering
- Fractional-year event rate
"""

import os
import numpy as np
import pandas as pd


# ==============================
# CONFIG
# ==============================

LOCATION = "fedjeosen"
MODE = "corrected"        # "raw" or "corrected"
CORR_METHOD = "qm"

if MODE == "raw":
    INPUT_PATH = f"DATA_EXTRACTION/nora3_locations/NORA3_wind_wave_{LOCATION}_1969_2025.csv"
elif MODE == "corrected":
    INPUT_PATH = f"BIAS_CORRECTION/output/{LOCATION}/hindcast_corrected_{CORR_METHOD}.csv"
else:
    raise ValueError("MODE must be 'raw' or 'corrected'")

OUTPUT_DIR = f"EXTREME_VALUE_MODELLING/output/{LOCATION}/{MODE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HS_COLUMN = "hs"
TIME_COLUMN = "time"

THRESHOLD_QUANTILE = 0.95
DECLUSTER_HOURS = 48


# ==============================
# LOAD DATA
# ==============================

def load_data(path):

    df = pd.read_csv(path, comment="#")

    if TIME_COLUMN not in df.columns or HS_COLUMN not in df.columns:
        raise ValueError("Missing required columns")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df[HS_COLUMN] = pd.to_numeric(df[HS_COLUMN], errors="coerce")

    df = df.dropna(subset=[TIME_COLUMN, HS_COLUMN])
    df = df[df[HS_COLUMN] > 0]
    df = df.sort_values(TIME_COLUMN)
    df = df.set_index(TIME_COLUMN)

    return df


# ==============================
# ANNUAL MAXIMA
# ==============================

def compute_annual_maxima(df):

    # Compute annual maxima (calendar year)
    annual_max = df[HS_COLUMN].resample("YE").max().dropna()

    if len(annual_max) < 3:
        raise ValueError("Not enough annual maxima.")

    # Identify full calendar years
    first_year = df.index.min().year
    last_year = df.index.max().year

    # Keep only fully observed years
    annual_max = annual_max[
        (annual_max.index.year > first_year) &
        (annual_max.index.year < last_year)
    ]

    if len(annual_max) < 3:
        raise ValueError("Too few full calendar years after removing partial years.")

    return annual_max


# ==============================
# DECLUSTERING (Cluster-max)
# ==============================

def decluster_clustermax(exceed_df, window_hours):

    if exceed_df.empty:
        return pd.Series(dtype=float)

    peaks = []
    cluster = [exceed_df.iloc[0]]

    for i in range(1, len(exceed_df)):
        dt = (exceed_df.index[i] - exceed_df.index[i-1]).total_seconds() / 3600

        if dt <= window_hours:
            cluster.append(exceed_df.iloc[i])
        else:
            cluster_df = pd.DataFrame(cluster)
            peaks.append(cluster_df[HS_COLUMN].max())
            cluster = [exceed_df.iloc[i]]

    # last cluster
    cluster_df = pd.DataFrame(cluster)
    peaks.append(cluster_df[HS_COLUMN].max())

    return pd.Series(peaks)


# ==============================
# POT
# ==============================

def compute_pot(df, quantile, decluster_hours):

    threshold = df[HS_COLUMN].quantile(quantile)

    exceed = df[df[HS_COLUMN] > threshold]

    if exceed.empty:
        raise ValueError("No exceedances found.")

    peaks = decluster_clustermax(exceed, decluster_hours)

    # Fractional-year calculation
    total_years = (df.index[-1] - df.index[0]).days / 365.25
    lambda_year = len(peaks) / total_years

    return peaks, threshold, lambda_year, total_years


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print(f"\nRunning preprocessing in {MODE.upper()} mode")
    print(f"Input: {INPUT_PATH}")

    df = load_data(INPUT_PATH)

    # Annual maxima
    annual_max = compute_annual_maxima(df)
    annual_max.to_csv(f"{OUTPUT_DIR}/annual_maxima.csv")

    # POT
    pot_peaks, threshold, lambda_year, total_years = compute_pot(
        df,
        THRESHOLD_QUANTILE,
        DECLUSTER_HOURS
    )

    pot_peaks.to_csv(f"{OUTPUT_DIR}/pot_peaks.csv")

    with open(f"{OUTPUT_DIR}/threshold.txt", "w") as f:
        f.write(str(threshold))

    with open(f"{OUTPUT_DIR}/lambda_year.txt", "w") as f:
        f.write(str(lambda_year))

    with open(f"{OUTPUT_DIR}/total_years.txt", "w") as f:
        f.write(str(total_years))

    print("\nDone")
    print(f"Threshold: {threshold:.2f} m")
    print(f"POT peaks: {len(pot_peaks)}")
    print(f"Fractional years: {total_years:.3f}")
    print(f"Lambda (clusters/year): {lambda_year:.4f}")