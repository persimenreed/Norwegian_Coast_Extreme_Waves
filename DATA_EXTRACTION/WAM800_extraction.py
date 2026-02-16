import xarray as xr
import pandas as pd
import os

# ==================================
# CONFIGURATION
# ==================================

URL_45 = "https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800c45vhf/"
URL_47 = "https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800vhf/"

START_DATE = "2018-12-01"
END_DATE   = "2025-10-01"

OUTPUT_DIR = "DATA_EXTRACTION/wam800_partitioned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep only required variables
VARS = [
    "hs",
    "tp",
    "ff",
    "dd",
    "thq",
    "depth",
    "latitude",
    "longitude",
    "time",
    "forecast_reference_time"
]

dates = pd.date_range(START_DATE, END_DATE, freq="1D")

# ==================================
# MAIN LOOP (by year)
# ==================================

for year in range(pd.to_datetime(START_DATE).year,
                  pd.to_datetime(END_DATE).year + 1):

    year_path = f"{OUTPUT_DIR}/year_{year}"
    os.makedirs(year_path, exist_ok=True)

    output_file = f"{year_path}/wam800_{year}.parquet"

    # Skip if already processed
    if os.path.exists(output_file):
        print(f"Year {year} already exists — skipping.")
        continue

    print(f"\nProcessing year {year}")

    yearly_data = []

    for dt in dates[dates.year == year]:

        if dt < pd.Timestamp("2023-03-16"):
            base_url = URL_45
        else:
            base_url = URL_47

        filename = f"mywavewam800_vestlandet.an.{dt:%Y%m%d}18.nc"
        url = base_url + filename

        try:
            print(f"Opening {url}")

            ds = xr.open_dataset(url)[VARS]

            ref_time = pd.to_datetime(ds.forecast_reference_time.values)

            df = ds.to_dataframe().reset_index()
            df = df.drop(columns=["rlat", "rlon"], errors="ignore")

            # Compute lead time
            df["lead_time_hours"] = (
                (pd.to_datetime(df["time"]) - ref_time)
                .dt.total_seconds() / 3600
            )

            # Keep only valid forecast window
            df = df[(df["lead_time_hours"] >= 0) &
                    (df["lead_time_hours"] <= 17)]

            # Remove land points
            df = df[df["depth"].notna()]

            # Reorder
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
                    "dd"
                ]
            ]

            yearly_data.append(df)

            print(f"Rows extracted: {len(df)}")

        except Exception as e:
            print(f"Skipped {dt}: {e}")
            continue

    if yearly_data:
        final_df = pd.concat(yearly_data, ignore_index=True)

        final_df.to_parquet(
            output_file,
            index=False,
            compression="snappy"
        )

        print(f"\nSaved {output_file}")
        print("Total rows:", len(final_df))

    else:
        print(f"No data extracted for {year}")
