"""
EXTREME_VALUE_MODELLING/extreme_preprocessing.py

Prepares:
- Annual maxima (GEV-ready)
- POT storm peaks (GPD-ready)
"""

import pandas as pd
import numpy as np


# ==============================
# CONFIG
# ==============================

INPUT_PATH = "data/csv_raw_data/NORA3/NORA3_fauskane2_wind_wave_lon5.72659_lat62.5671_20200520_20210303.csv"

HS_COLUMN = "hs"
TIME_COLUMN = "time"

THRESHOLD_QUANTILE = 0.95
DECLUSTER_HOURS = 48


# ==============================
# LOAD DATA
# ==============================

def load_data(path):
    df = pd.read_csv(path, comment="#")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN)
    df = df[[TIME_COLUMN, HS_COLUMN]].dropna()
    df = df.set_index(TIME_COLUMN)
    return df


# ==============================
# ANNUAL MAXIMA
# ==============================

def compute_annual_maxima(df):
    return df[HS_COLUMN].resample("Y").max().dropna()


# ==============================
# POT + DECLUSTERING
# ==============================

def decluster(times, values, window_hours):
    peaks = []
    last_time = None

    for t, v in zip(times, values):
        if last_time is None:
            peaks.append((t, v))
            last_time = t
        else:
            delta = (t - last_time).total_seconds() / 3600
            if delta >= window_hours:
                peaks.append((t, v))
                last_time = t
            else:
                if v > peaks[-1][1]:
                    peaks[-1] = (t, v)
                    last_time = t

    return pd.Series(
        [p[1] for p in peaks],
        index=[p[0] for p in peaks]
    )


def compute_pot(df, quantile, decluster_hours):
    threshold = df[HS_COLUMN].quantile(quantile)
    exceed = df[df[HS_COLUMN] > threshold]

    if exceed.empty:
        raise ValueError("No exceedances found.")

    peaks = decluster(
        exceed.index,
        exceed[HS_COLUMN].values,
        decluster_hours
    )

    return peaks, threshold


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    df = load_data(INPUT_PATH)

    annual_max = compute_annual_maxima(df)
    annual_max.to_csv("EXTREME_VALUE_MODELLING/output/annual_maxima.csv")

    pot_peaks, threshold = compute_pot(
        df,
        THRESHOLD_QUANTILE,
        DECLUSTER_HOURS
    )
    pot_peaks.to_csv("EXTREME_VALUE_MODELLING/output/pot_peaks.csv")

    with open("EXTREME_VALUE_MODELLING/output/threshold.txt", "w") as f:
        f.write(str(threshold))


    print("Done")
    print(f"Threshold: {threshold:.2f} m")
    print(f"Annual maxima count: {len(annual_max)}")
    print(f"POT peaks count: {len(pot_peaks)}")
