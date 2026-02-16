import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = "/home/coder/IKT590/Norwegian_Coast_Wave_Forecast/DATA_EXTRACTION/nora3_partitioned"
FILE_PATTERN = "nora3_wave_*.parquet"
VALUE_COL = "hs"

START_YEAR = 2025
END_YEAR = 2025

location = "Fauskane"

TARGET_LAT = 62.5672
TARGET_LON = 5.72684
LAT_HALF_WINDOW = 1.5
LON_HALF_WINDOW = 1.5
MARKER_SIZE = 50
MARKER_STYLE = "o"

OUT_DIR = "Analyse_NORA3/output"
os.makedirs(OUT_DIR, exist_ok=True)

if location == "Fauskane":
	TARGET_LAT = 62.5672
	TARGET_LON = 5.72684
elif location == "Fedjeosen":
	TARGET_LAT = 60.732916
	TARGET_LON = 4.662131

# ==========================================
# HELPERS
# ==========================================
def extract_year(path):
	base = os.path.basename(path)
	year_part = base.split("_")[-1].split(".")[0]
	return int(year_part)


def area_bounds(center_lat, center_lon, half_lat, half_lon):
	return (
		center_lat - half_lat,
		center_lat + half_lat,
		center_lon - half_lon,
		center_lon + half_lon,
	)


def subset_to_area(df, min_lat, max_lat, min_lon, max_lon):
	return df[
		df["latitude"].between(min_lat, max_lat)
		& df["longitude"].between(min_lon, max_lon)
	]


def plot_grid_scatter(grid, title, label, out_file):
	plt.figure(figsize=(10, 8))
	try:
		import cartopy.crs as ccrs
		import cartopy.feature as cfeature

		ax = plt.axes(projection=ccrs.PlateCarree())
		sc = ax.scatter(
			grid["longitude"],
			grid["latitude"],
			c=grid["value"],
			s=MARKER_SIZE,
			marker=MARKER_STYLE,
			cmap="viridis"
		)
		ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
		ax.add_feature(cfeature.BORDERS, linewidth=0.5)
		ax.set_title(title)
		plt.colorbar(sc, ax=ax, label=label)

	except Exception:
		sc = plt.scatter(
			grid["longitude"],
			grid["latitude"],
			c=grid["value"],
			s=MARKER_SIZE,
			marker=MARKER_STYLE,
			cmap="viridis"
		)
		plt.title(title)
		plt.xlabel("Longitude")
		plt.ylabel("Latitude")
		plt.colorbar(sc, label=label)

	plt.tight_layout()
	plt.savefig(out_file, dpi=200)
	plt.close()


# ==========================================
# LOAD FILES
# ==========================================
files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
files = [f for f in files if START_YEAR <= extract_year(f) <= END_YEAR]
if not files:
	raise FileNotFoundError(
		f"No files found in {DATA_DIR} for years {START_YEAR}-{END_YEAR}"
	)

min_lat, max_lat, min_lon, max_lon = area_bounds(
	TARGET_LAT,
	TARGET_LON,
	LAT_HALF_WINDOW,
	LON_HALF_WINDOW,
)

agg_all = None
for f in files:
	print(f"Reading {f}")
	df = pd.read_parquet(f, columns=["latitude", "longitude", VALUE_COL])
	df = subset_to_area(df, min_lat, max_lat, min_lon, max_lon)
	if df.empty:
		continue
	agg = df.groupby(["latitude", "longitude"])[VALUE_COL].agg(
		sum="sum",
		count="count"
	).reset_index()
	if agg_all is None:
		agg_all = agg
	else:
		agg_all = (
			pd.concat([agg_all, agg], ignore_index=True)
			.groupby(["latitude", "longitude"], as_index=False)[["sum", "count"]]
			.sum()
		)

if agg_all is None:
	raise ValueError("No grid cells found inside the selected area for the selected years.")

local_grid = agg_all[["latitude", "longitude"]].copy()
local_grid["value"] = agg_all["sum"] / agg_all["count"]

out_file = os.path.join(
	OUT_DIR, f"NORA3_{location}_hs_{START_YEAR}_{END_YEAR}.png"
)
plot_grid_scatter(
	local_grid,
	title=(
		f"NORA3 Mean Significant Wave Height (Hs) around {location}, {START_YEAR}-{END_YEAR}"
	),
	label="Hs (m)",
	out_file=out_file
)

print("Done. Output saved to Analyse_NORA3/output/")
