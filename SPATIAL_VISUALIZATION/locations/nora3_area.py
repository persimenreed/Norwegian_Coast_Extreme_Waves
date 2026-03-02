import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

# Input files (created by DATA_EXTRACTION/NORA3_extraction.py)
BASE_DIR = Path(__file__).resolve().parent
ATM_FILE = BASE_DIR / "nora3_atm_2010.parquet"
FULL_NORA3_FILE = BASE_DIR / "arome3kmwind_1hr_202510_150m.nc"

# Map view controls (tune these to zoom/pan without changing data geometry)
WEST_PAD = -8.0
EAST_PAD = -40.0   # negative value trims space on the right side

#################
SOUTH_PAD = 0.0
NORTH_PAD = 3.0

################# smooth map
#SOUTH_PAD = 1.0
#NORTH_PAD = 5.0

# Optional hard limits (set to None to use auto from data bounds + pads)
VIEW_LON_MIN = None
VIEW_LON_MAX = None
VIEW_LAT_MIN = None
VIEW_LAT_MAX = None

# Global graticule spacing (drawn by Cartopy, independent of dataset bounds)
GRID_LON_STEP = 10
GRID_LAT_STEP = 5

# Background shading (land slightly darker than water)
OCEAN_FACECOLOR = "0.95"
LAND_FACECOLOR = "0.88"

ROTATED_OUT = BASE_DIR / "nora3_2010_atm_wave_coverage_rotated.png"
UNROTATED_OUT = BASE_DIR / "nora3_2010_atm_wave_coverage_unrotated.png"

# Load only coordinates from parquet files
atm_df = pd.read_parquet(ATM_FILE, columns=["latitude", "longitude"])

# Full NORA3 dataset extent from netCDF
ds_full = xr.open_dataset(FULL_NORA3_FILE)

# Keep polygon detail (avoid simplification to only a few corners)
mpl.rcParams["path.simplify"] = False
mpl.rcParams["path.simplify_threshold"] = 0.0

# Bounding box for ATM data
atm_lon_min, atm_lon_max = float(atm_df["longitude"].min()), float(atm_df["longitude"].max())
atm_lat_min, atm_lat_max = float(atm_df["latitude"].min()), float(atm_df["latitude"].max())

# Bounding box for full NORA3 dataset
full_lon_min, full_lon_max = float(ds_full.longitude.min()), float(ds_full.longitude.max())
full_lat_min, full_lat_max = float(ds_full.latitude.min()), float(ds_full.latitude.max())

# Build accurate NORA3 boundary from 2D lon/lat outer grid
full_lon = ds_full["longitude"].values
full_lat = ds_full["latitude"].values

top = np.column_stack([full_lon[0, :], full_lat[0, :]])
right = np.column_stack([full_lon[1:, -1], full_lat[1:, -1]])
bottom = np.column_stack([full_lon[-1, -2::-1], full_lat[-1, -2::-1]])
left = np.column_stack([full_lon[-2:0:-1, 0], full_lat[-2:0:-1, 0]])

full_boundary_ll = np.vstack([top, right, bottom, left])

ds_full.close()

# Print ranges
print("ATM longitude range:", atm_lon_min, "to", atm_lon_max)
print("ATM latitude range:", atm_lat_min, "to", atm_lat_max)
print("FULL NORA3 longitude range:", full_lon_min, "to", full_lon_max)
print("FULL NORA3 latitude range:", full_lat_min, "to", full_lat_max)

# Set map extent from full NORA3 bounds (with configurable asymmetric padding)
extent_lon_min = full_lon_min - WEST_PAD if VIEW_LON_MIN is None else VIEW_LON_MIN
extent_lon_max = full_lon_max + EAST_PAD if VIEW_LON_MAX is None else VIEW_LON_MAX
extent_lat_min = full_lat_min - SOUTH_PAD if VIEW_LAT_MIN is None else VIEW_LAT_MIN
extent_lat_max = full_lat_max + NORTH_PAD if VIEW_LAT_MAX is None else VIEW_LAT_MAX


def _degree_label(value, is_lon):
    direction = "E" if is_lon else "N"
    if value < 0:
        direction = "W" if is_lon else "S"
    return f"{abs(int(value))}° {direction}"


def _ticks_in_extent(vmin, vmax, step):
    start = np.ceil(vmin / step) * step
    end = np.floor(vmax / step) * step
    if start > end:
        return np.array([])
    return np.arange(start, end + 0.001, step)


def _add_forced_margin_labels(ax, lon_ticks, lat_ticks, lon_min, lon_max, lat_min, lat_max):
    for lat in lat_ticks:
        y = (lat - lat_min) / (lat_max - lat_min)
        ax.text(
            -0.035,
            y,
            _degree_label(lat, is_lon=False),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color="black",
        )

    for lon in lon_ticks:
        x = (lon - lon_min) / (lon_max - lon_min)
        ax.text(
            x,
            -0.035,
            _degree_label(lon, is_lon=True),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="black",
        )


def plot_map(projection, out_file):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": projection})

    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_linewidth(1.0)
    ax.spines['geo'].set_edgecolor('black')

    ax.set_extent(
        [extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_FACECOLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_FACECOLOR, edgecolor='none', zorder=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    lon_grid = np.arange(-180, 180 + 0.001, GRID_LON_STEP)
    lat_grid = np.arange(-90, 90 + 0.001, GRID_LAT_STEP)
    lon_ticks = _ticks_in_extent(extent_lon_min, extent_lon_max, GRID_LON_STEP)
    lat_ticks = _ticks_in_extent(extent_lat_min, extent_lat_max, GRID_LAT_STEP)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=lon_grid,
        ylocs=lat_grid,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        alpha=0.6,
        zorder=1,
    )
    gl.n_steps = 200

    _add_forced_margin_labels(
        ax,
        lon_ticks,
        lat_ticks,
        extent_lon_min,
        extent_lon_max,
        extent_lat_min,
        extent_lat_max,
    )

    full_poly = Polygon(
        full_boundary_ll,
        closed=True,
        linewidth=2,
        edgecolor='red',
        facecolor='red',
        alpha=0.15,
        label='NORA3',
        zorder=2,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(full_poly)

    atm_rect = Rectangle(
        (atm_lon_min, atm_lat_min),
        atm_lon_max - atm_lon_min,
        atm_lat_max - atm_lat_min,
        linewidth=2,
        edgecolor='green',
        facecolor='lightgreen',
        alpha=0.3,
        label='Extracted area',
        zorder=3,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(atm_rect)

    ax.legend(loc='lower left')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


rotated_projection = ccrs.LambertConformal(
    central_longitude=10,
    central_latitude=65,
    standard_parallels=(60, 66)
)

plot_map(rotated_projection, ROTATED_OUT)
plot_map(ccrs.PlateCarree(), UNROTATED_OUT)

print("Saved:", ROTATED_OUT)
print("Saved:", UNROTATED_OUT)
