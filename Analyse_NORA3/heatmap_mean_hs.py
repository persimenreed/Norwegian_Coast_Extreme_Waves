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

START_YEAR = 2015
END_YEAR = 2020

OUT_DIR = "Analyse_NORA3/output"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# HELPERS
# ==========================================
def plot_grid_scatter(grid, title, label, out_file):
	plt.figure(figsize=(10, 8))
	try:
		import cartopy.crs as ccrs
		import cartopy.feature as cfeature

		ax = plt.axes(projection=ccrs.PlateCarree())
		sc = ax.scatter(
			grid["longitude"], grid["latitude"],
			c=grid["value"], s=8, cmap="viridis"
		)
		ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
		ax.add_feature(cfeature.BORDERS, linewidth=0.5)
		ax.set_title(title)
		plt.colorbar(sc, ax=ax, label=label)

	except Exception:
		sc = plt.scatter(
			grid["longitude"], grid["latitude"],
			c=grid["value"], s=8, cmap="viridis"
		)
		plt.title(title)
		plt.xlabel("Longitude")
		plt.ylabel("Latitude")
		plt.colorbar(sc, label=label)

	plt.tight_layout()
	plt.savefig(out_file, dpi=200)
	plt.close()


def extract_year(path):
	base = os.path.basename(path)
	year_part = base.split("_")[-1].split(".")[0]
	return int(year_part)


# ==========================================
# LOAD FILES
# ==========================================
files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
files = [f for f in files if START_YEAR <= extract_year(f) <= END_YEAR]
if not files:
	raise FileNotFoundError(
		f"No files found in {DATA_DIR} for years {START_YEAR}-{END_YEAR}"
	)

# ==========================================
# STREAMING AGGREGATION (MEAN)
# ==========================================
agg_all = None

for f in files:
	print(f"Reading {f}")
	df = pd.read_parquet(f, columns=["latitude", "longitude", VALUE_COL])

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

mean_grid = agg_all[["latitude", "longitude"]].copy()
mean_grid["value"] = agg_all["sum"] / agg_all["count"]

out_file = os.path.join(
	OUT_DIR, f"NORA3_mean_hs_{START_YEAR}_{END_YEAR}.png"
)
plot_grid_scatter(
	mean_grid,
	title=f"NORA3 Mean Significant Wave Height (Hs) {START_YEAR}-{END_YEAR}",
	label="Hs (m)",
	out_file=out_file
)

print("Done. Outputs saved to Analyse_NORA3/output/")
