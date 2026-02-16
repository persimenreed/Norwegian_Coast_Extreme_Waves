import xarray as xr
import pandas as pd
import numpy as np
import os

# ==================================
# USER INPUT
# ==================================

# location = "fedjeosen"
# target_lat = 60.732916
# target_lon = 4.662131
# START_DATE = "2023-11-01"
# END_DATE   = "2025-10-07"

location = "fauskane"
target_lat = 62.5672
target_lon = 5.72684
START_DATE = "2018-12-01"
END_DATE   = "2021-03-05"

URL_45 = "https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800c45vhf/"
URL_47 = "https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800vhf/"

OUTPUT_DIR = f"DATA_EXTRACTION/wam800_locations/wam800_{location}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================
# VARIABLES TO EXTRACT
# ==================================

VARS = [
    "hs",
    "tp",
    "thq",
    "ff",
    "dd",
    "hs_sea",
    "hs_swell",
    "tp_sea",
    "tp_swell",
    "depth",
    "latitude",
    "longitude",
    "time",
    "forecast_reference_time"
]

dates = pd.date_range(START_DATE, END_DATE, freq="1D")

# ==================================
# HELPER FUNCTION
# ==================================

def get_nearest_grid(ds, lat, lon):
    lat2d = ds["latitude"].values
    lon2d = ds["longitude"].values

    dist = np.sqrt((lat2d - lat)**2 + (lon2d - lon)**2)
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

    return iy, ix


# ==================================
# MAIN LOOP
# ==================================

all_data = []
skipped_dates = 0

for dt in dates:

    if dt < pd.Timestamp("2023-03-16"):
        base_url = URL_45
    else:
        base_url = URL_47

    filename = f"mywavewam800_vestlandet.an.{dt:%Y%m%d}18.nc"
    url = base_url + filename

    try:
        print(f"Opening {url}")

        ds = xr.open_dataset(url)[VARS]

        # Find nearest grid cell
        iy, ix = get_nearest_grid(ds, target_lat, target_lon)
        ds_point = ds.isel(rlat=iy, rlon=ix)

        # Compute lead time
        ref_time = pd.to_datetime(ds_point.forecast_reference_time.values)

        df = ds_point.to_dataframe().reset_index()

        df["lead_time_hours"] = (
            (pd.to_datetime(df["time"]) - ref_time)
            .dt.total_seconds() / 3600
        )

        # Keep only valid forecast window
        df = df[(df["lead_time_hours"] >= 0) &
                (df["lead_time_hours"] <= 17)]

        # Remove unnecessary columns
        df = df.drop(
            columns=["rlat", "rlon", "forecast_reference_time"],
            errors="ignore"
        )

        # Reorder columns to match main format
        df = df[
            [
                "time",
                "lead_time_hours",
                "latitude",
                "longitude",
                "depth",
                "hs",
                "tp",
                "thq",
                "ff",
                "dd",
                "hs_sea",
                "hs_swell",
                "tp_sea",
                "tp_swell"
            ]
        ]

        all_data.append(df)

    except Exception as e:
        skipped_dates += 1
        print(f"Skipped {dt}: {e}")
        continue

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    output_file = (
        f"{OUTPUT_DIR}/"
        f"wam800_{location}_{START_DATE[:4]}_{END_DATE[:4]}.csv"
    )

    final_df.to_csv(output_file, index=False)

    print(f"\nSaved to {output_file}")
    print(f"Total rows: {len(final_df)}")

print(f"Skipped dates: {skipped_dates}")
