import os
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe




# ==========================================
# OUTPUT
# ==========================================
OUT_DIR = "testplot"
os.makedirs(OUT_DIR, exist_ok=True)

# Global graticule spacing (same as nora3_area.py)
GRID_LON_STEP = 10
GRID_LAT_STEP = 5

MAP_LON_MIN = 3
MAP_LON_MAX = 24
MAP_LAT_MIN = 57.6
MAP_LAT_MAX = 72

# Background shading (land slightly darker than water)
OCEAN_FACECOLOR = "0.95"
LAND_FACECOLOR = "0.88"

# ==========================================
# POINTS
# ==========================================
points = [
    {"name": "Fauskane", "lon": 5.72684, "lat": 62.5672},
    {"name": "Vestfjorden", "lon": 15.477234, "lat": 68.230576},
    {"name": "Fedjeosen", "lon": 4.662131, "lat": 60.732916},
]


def _degree_label(value, is_lon):
    direction = "E" if is_lon else "N"
    if value < 0:
        direction = "W" if is_lon else "S"
    return f"{abs(int(value))}° {direction}"


def _ticks_in_extent(vmin, vmax, step):
    start = (vmin // step) * step
    if start < vmin:
        start += step
    end = (vmax // step) * step
    if start > end:
        return []
    ticks = []
    value = start
    while value <= end + 1e-9:
        ticks.append(value)
        value += step
    return ticks


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

# ==========================================
# PLOT FUNCTION (MATCHES YOUR STYLE)
# ==========================================
def plot_points_map(points, title, out_file, projection):

    plt.figure(figsize=(6, 7.7))
    lons = [p["lon"] for p in points]
    lats = [p["lat"] for p in points]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = plt.axes(projection=projection)

        ax.set_extent([MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_FACECOLOR, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=LAND_FACECOLOR, edgecolor='none', zorder=0.5)

        lon_grid = list(range(-180, 181, GRID_LON_STEP))
        lat_grid = list(range(-90, 91, GRID_LAT_STEP))
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

        lon_ticks = _ticks_in_extent(MAP_LON_MIN, MAP_LON_MAX, GRID_LON_STEP)
        lat_ticks = _ticks_in_extent(MAP_LAT_MIN, MAP_LAT_MAX, GRID_LAT_STEP)
        _add_forced_margin_labels(
            ax,
            lon_ticks,
            lat_ticks,
            MAP_LON_MIN,
            MAP_LON_MAX,
            MAP_LAT_MIN,
            MAP_LAT_MAX,
        )

        # Plot points
        ax.scatter(
            lons,
            lats,
            color="red",
            s=40,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # Labels
        for p in points:
            txt = ax.text(
                p["lon"] + 0.25,
                p["lat"] + 0.25,
                p["name"],
                fontsize=11,              # slightly larger
                color="red",
                weight="bold",            # thicker font
                transform=ccrs.PlateCarree(),
                zorder=4
            )

            # Add white outline (halo)
            txt.set_path_effects([
                pe.Stroke(linewidth=3, foreground="white"),
                pe.Normal()
            ])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        # Remove the outer frame
        ax.spines['geo'].set_visible(False)

        ax.set_title(title)

    except Exception:
        plt.scatter(lons, lats, color="red", s=40)
        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()



# ==========================================
# RUN
# ==========================================
import cartopy.crs as ccrs

plot_points_map(
    points,
    title="",
    out_file=os.path.join(OUT_DIR, "NORA3_three_points_map_rotated.png"),
    projection=ccrs.LambertConformal(
        central_longitude=10,
        central_latitude=65,
        standard_parallels=(60, 66)
    )
)

plot_points_map(
    points,
    title="",
    out_file=os.path.join(OUT_DIR, "NORA3_three_points_map_unrotated.png"),
    projection=ccrs.PlateCarree()
)

print("Saved:", os.path.join(OUT_DIR, "NORA3_three_points_map_rotated.png"))
print("Saved:", os.path.join(OUT_DIR, "NORA3_three_points_map_unrotated.png"))
