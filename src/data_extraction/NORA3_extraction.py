import xarray as xr
import pandas as pd
import os
from tqdm.auto import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

ATM_BASE_URL = "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/atm_hourly_v2/"
WAV_BASE_URL = "https://thredds.met.no/thredds/dodsC/nora3_subset_wave/wave_tser/"

START_DATE = "1959-01-01"
END_DATE   = "2025-10-01"

LAT_MIN = 57.932274
LAT_MAX = 63.609142
LON_MIN = 2.7800689
LON_MAX = 8.895549

OUTPUT_DIR = "data/input/nora3_spatial"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# VARIABLES
# ==========================================

ATM_VARS = [
    "wind_speed",
    "wind_direction",
    "air_pressure_at_sea_level",
    "air_temperature_2m",
    "latitude",
    "longitude",
    "time"
]

WAV_VARS = [
    "hs",
    "hs_sea",
    "hs_swell",
    "tp",
    "thq",
    "model_depth",
    "latitude",
    "longitude",
    "time"
]

# ==========================================
# YEAR LOOP
# ==========================================

start_year = pd.to_datetime(START_DATE).year
end_year   = pd.to_datetime(END_DATE).year

for year in tqdm(range(start_year, end_year + 1), desc="Years"):

    atm_output = f"{OUTPUT_DIR}/nora3_atm_{year}.parquet"
    wav_output = f"{OUTPUT_DIR}/nora3_wave_{year}.parquet"

    # Skip if both exist
    if os.path.exists(atm_output) and os.path.exists(wav_output):
        print(f"Year {year} already exists — skipping.")
        continue

    print(f"\nProcessing year {year}")

    yearly_atm = []
    yearly_wav = []

    # Loop through months
    for month in tqdm(range(1, 13), desc=f"{year} months", leave=False):

        month_start = pd.Timestamp(year=year, month=month, day=1)

        if month_start < pd.to_datetime(START_DATE):
            continue
        if month_start >= pd.to_datetime(END_DATE):
            continue

        print(f"\nProcessing {year}-{month:02d}")

        # ==========================================
        # ATMOSPHERE
        # ==========================================

        try:
            atm_file = f"arome3km_1hr_{year}{month:02d}.nc"
            atm_url  = ATM_BASE_URL + atm_file

            ds_atm = xr.open_dataset(atm_url)[ATM_VARS]

            month_end = (
                month_start + pd.offsets.MonthBegin(1)
            ) - pd.Timedelta(seconds=1)

            ds_atm = ds_atm.sel(time=slice(month_start, month_end))

            mask_atm = (
                (ds_atm.latitude >= LAT_MIN) &
                (ds_atm.latitude <= LAT_MAX) &
                (ds_atm.longitude >= LON_MIN) &
                (ds_atm.longitude <= LON_MAX)
            )

            ds_atm = ds_atm.where(mask_atm, drop=True)

            atm_df = ds_atm.to_dataframe().reset_index()

            atm_df = atm_df.drop(
                columns=["x", "y", "height_above_msl", "height1", "height4"],
                errors="ignore"
            )

            atm_df = atm_df[
                atm_df["air_pressure_at_sea_level"].notna()
            ]

            atm_df["time"] = pd.to_datetime(atm_df["time"]).dt.floor("h")

            atm_df = atm_df[
                [
                    "time",
                    "wind_speed",
                    "wind_direction",
                    "air_pressure_at_sea_level",
                    "air_temperature_2m",
                    "latitude",
                    "longitude"
                ]
            ]

            atm_df = atm_df.sort_values(["time", "latitude", "longitude"])

            yearly_atm.append(atm_df)

            print(f"ATM rows: {len(atm_df)}")

        except Exception as e:
            print(f"Skipped ATM {year}-{month:02d}: {e}")

        # ==========================================
        # WAVE
        # ==========================================

        try:
            wav_file = f"{year}{month:02d}_NORA3wave_sub_time_unlimited.nc"
            wav_url  = WAV_BASE_URL + wav_file

            ds_wav = xr.open_dataset(wav_url)[WAV_VARS]

            month_end = (
                month_start + pd.offsets.MonthBegin(1)
            ) - pd.Timedelta(seconds=1)

            ds_wav = ds_wav.sel(time=slice(month_start, month_end))

            mask_wav = (
                (ds_wav.latitude >= LAT_MIN) &
                (ds_wav.latitude <= LAT_MAX) &
                (ds_wav.longitude >= LON_MIN) &
                (ds_wav.longitude <= LON_MAX)
            )

            ds_wav = ds_wav.where(mask_wav, drop=True)

            wav_df = ds_wav.to_dataframe().reset_index()

            wav_df = wav_df.drop(columns=["rlon", "rlat"], errors="ignore")

            wav_df = wav_df[wav_df["hs"].notna()]

            wav_df["time"] = pd.to_datetime(wav_df["time"]).dt.floor("h")

            wav_df = wav_df[
                [
                    "time",
                    "hs",
                    "hs_sea",
                    "hs_swell",
                    "tp",
                    "thq",
                    "model_depth",
                    "latitude",
                    "longitude"
                ]
            ]

            wav_df = wav_df.sort_values(["time", "latitude", "longitude"])

            yearly_wav.append(wav_df)

            print(f"WAVE rows: {len(wav_df)}")

        except Exception as e:
            print(f"Skipped WAVE {year}-{month:02d}: {e}")

    # ==========================================
    # SAVE YEARLY
    # ==========================================

    if yearly_atm:
        final_atm = pd.concat(yearly_atm, ignore_index=True)
        final_atm.to_parquet(atm_output, index=False, compression="snappy")
        print(f"\nSaved {atm_output}")
        print("Total ATM rows:", len(final_atm))

    if yearly_wav:
        final_wav = pd.concat(yearly_wav, ignore_index=True)
        final_wav.to_parquet(wav_output, index=False, compression="snappy")
        print(f"\nSaved {wav_output}")
        print("Total WAVE rows:", len(final_wav))

print("\nNORA3 yearly extraction finished.")
