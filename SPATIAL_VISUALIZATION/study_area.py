import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from pathlib import Path
import matplotlib.patheffects as pe

# ==========================================
# INPUT / OUTPUT
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
ATM_FILE = BASE_DIR / "nora3_atm_2010.parquet"

ROTATED_OUT = BASE_DIR / "study_area_rotated.png"
UNROTATED_OUT = BASE_DIR / "study_area_unrotated.png"

# Read extracted-area data once (used by both output maps)
ATM_DF = pd.read_parquet(ATM_FILE, columns=["latitude", "longitude"])
ATM_LON_MIN = float(ATM_DF["longitude"].min())
ATM_LON_MAX = float(ATM_DF["longitude"].max())
ATM_LAT_MIN = float(ATM_DF["latitude"].min())
ATM_LAT_MAX = float(ATM_DF["latitude"].max())

# ==========================================
# MAP STYLE (same grid system as nora3_area.py / buoys_location.py)
# ==========================================
GRID_LON_STEP = 10
GRID_LAT_STEP = 5

OCEAN_FACECOLOR = "0.95"
LAND_FACECOLOR = "0.88"

# Zoom controls (tune later if needed)
# This is intentionally more zoomed in than buoys_location.py while
# still keeping the extracted-area borders clearly visible.
LON_PAD_WEST = 1.5
LON_PAD_EAST = 2.4
LAT_PAD_SOUTH = 1.0
LAT_PAD_NORTH = 0.5

EXTENT_LON_MIN = ATM_LON_MIN - LON_PAD_WEST
EXTENT_LON_MAX = ATM_LON_MAX + LON_PAD_EAST
EXTENT_LAT_MIN = ATM_LAT_MIN - LAT_PAD_SOUTH
EXTENT_LAT_MAX = ATM_LAT_MAX + LAT_PAD_NORTH

# ==========================================
# POINTS
# ==========================================
# Red points requested
red_points = [
    {"name": "Fauskane", "lon": 5.72684, "lat": 62.5672},
    {"name": "Fedjeosen", "lon": 4.662131, "lat": 60.732916},
]

# Blue template points (nearby placeholders; replace later)
blue_points = [
    {"name": "Kristiansund", "lon": 7.570590, "lat": 63.189756},
    {"name": "Bergen", "lon": 4.8303750, "lat": 60.3809388},
    {"name": "Stavanger", "lon": 5.4146939, "lat": 58.9141135},
]


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


def _draw_labeled_points(ax, points, color, label, dx=0.18, dy=0.18):
    lons = [p["lon"] for p in points]
    lats = [p["lat"] for p in points]

    ax.scatter(
        lons,
        lats,
        color=color,
        s=45,
        label=label,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    for p in points:
        txt = ax.text(
            p["lon"] + dx,
            p["lat"] + dy,
            p["name"],
            fontsize=10,
            color=color,
            weight="bold",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=3, foreground="white"),
            pe.Normal(),
        ])


def plot_map(projection, out_file):
    fig, ax = plt.subplots(figsize=(8.2, 8.0), subplot_kw={"projection": projection})

    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_linewidth(1.0)
    ax.spines["geo"].set_edgecolor("black")

    ax.set_extent(
        [EXTENT_LON_MIN, EXTENT_LON_MAX, EXTENT_LAT_MIN, EXTENT_LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_FACECOLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_FACECOLOR, edgecolor="none", zorder=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    lon_grid = np.arange(-180, 180 + 0.001, GRID_LON_STEP)
    lat_grid = np.arange(-90, 90 + 0.001, GRID_LAT_STEP)
    lon_ticks = _ticks_in_extent(EXTENT_LON_MIN, EXTENT_LON_MAX, GRID_LON_STEP)
    lat_ticks = _ticks_in_extent(EXTENT_LAT_MIN, EXTENT_LAT_MAX, GRID_LAT_STEP)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=lon_grid,
        ylocs=lat_grid,
        linewidth=0.5,
        color="gray",
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )
    gl.n_steps = 80

    _add_forced_margin_labels(
        ax,
        lon_ticks,
        lat_ticks,
        EXTENT_LON_MIN,
        EXTENT_LON_MAX,
        EXTENT_LAT_MIN,
        EXTENT_LAT_MAX,
    )

    # Extracted area (green rectangle)
    atm_rect = Rectangle(
        (ATM_LON_MIN, ATM_LAT_MIN),
        ATM_LON_MAX - ATM_LON_MIN,
        ATM_LAT_MAX - ATM_LAT_MIN,
        linewidth=2,
        edgecolor="green",
        facecolor="lightgreen",
        alpha=0.3,
        label="Extracted area",
        zorder=3,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(atm_rect)

    # Point layers
    _draw_labeled_points(ax, red_points, color="red", label="Buoys")
    _draw_labeled_points(ax, blue_points, color="blue", label="Study Area")

    ax.legend(loc="lower left")
    plt.savefig(out_file, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rotated_projection = ccrs.LambertConformal(
        central_longitude=10,
        central_latitude=65,
        standard_parallels=(60, 66),
    )

    plot_map(rotated_projection, ROTATED_OUT)
    plot_map(ccrs.PlateCarree(), UNROTATED_OUT)

    print("Saved:", ROTATED_OUT)
    print("Saved:", UNROTATED_OUT)
