import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bias_correction.data import load_pairs
from src.settings import get_buoy_locations


OBS_COL = "Significant_Wave_Height_Hm0"
NORA3_COL = "hs"
OBS_DIRECTION_COLS = [
    "Wave_Peak_Direction",
    "Wave_Mean_Direction",
    "Wave_Peak_Direction_Wind",
    "Wave_Peak_Direction_Swell",
]
DEFAULT_OUT_DIR = Path("results/MISC/existing_bias")
DEFAULT_LOCATIONS = ["fedjeosen", "fauskane", "vestfjorden"]
LOCATION_LABELS = {
    "fedjeosen": "Fedjeosen",
    "fauskane": "Fauskane",
    "vestfjorden": "Vestfjorden",
}
COMPASS_LABELS_8 = [
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
]
COLORS = {
    "nora3": "#e07a2f",
    "observed": "#2474a6",
    "error": "#2474a6",
    "scatter": "#2474a6",
    "zero": "#444444",
}


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)


def _style_boxed_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)


def _clean_pairs(location: str) -> pd.DataFrame:
    df = load_pairs(location)
    missing = [col for col in [NORA3_COL, OBS_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"{location} pairs file is missing columns: {missing}")

    out = df[["time", NORA3_COL, OBS_COL]].copy()
    out[NORA3_COL] = pd.to_numeric(out[NORA3_COL], errors="coerce")
    out[OBS_COL] = pd.to_numeric(out[OBS_COL], errors="coerce")
    out = out.dropna(subset=[NORA3_COL, OBS_COL])
    out = out[(out[NORA3_COL] >= 0) & (out[OBS_COL] >= 0)]
    if out.empty:
        raise ValueError(f"No valid paired wave-height rows for {location}")
    return out


def _observed_direction_column(df: pd.DataFrame) -> str:
    for col in OBS_DIRECTION_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        "No observed direction column found. Checked: "
        + ", ".join(OBS_DIRECTION_COLS)
    )


def _clean_observed_direction_pairs(location: str) -> tuple[pd.DataFrame, str]:
    df = load_pairs(location)
    direction_col = _observed_direction_column(df)
    columns = ["time", OBS_COL, direction_col]

    out = df[columns].copy()
    out[OBS_COL] = pd.to_numeric(out[OBS_COL], errors="coerce")
    out[direction_col] = pd.to_numeric(out[direction_col], errors="coerce") % 360.0
    required = [OBS_COL, direction_col]

    out = out.dropna(subset=required)
    out = out[(out[OBS_COL] >= 0) & (out[direction_col] >= 0)]
    if out.empty:
        raise ValueError(f"No valid observed Hs/direction rows for {location}")
    return out, direction_col


def _bin_edges(values: Iterable[pd.Series], bin_width: float) -> np.ndarray:
    max_value = max(float(series.max()) for series in values)
    upper = max(bin_width, np.ceil(max_value / bin_width) * bin_width)
    return np.arange(0.0, upper + bin_width, bin_width)


def _hist(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values.to_numpy(dtype=float), bins=edges)
    return counts


def _bar_labels(edges: np.ndarray) -> List[str]:
    return [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(edges) - 1)]


def _error_series(df: pd.DataFrame) -> pd.Series:
    return (df[NORA3_COL] - df[OBS_COL]).rename("error").dropna()


def _error_edges(error: pd.Series, bin_width: float) -> np.ndarray:
    values = error.to_numpy(dtype=float)
    lower = np.floor(float(np.nanmin(values)) / bin_width) * bin_width
    upper = np.ceil(float(np.nanmax(values)) / bin_width) * bin_width
    if lower == upper:
        upper = lower + bin_width
    return np.arange(lower, upper + bin_width, bin_width)


def _direction_sector_indices(direction: pd.Series, n_sectors: int = 8) -> pd.Series:
    sector_width = 360.0 / n_sectors
    return np.floor(((direction + sector_width / 2.0) % 360.0) / sector_width).astype(int)


def _circular_mean_deg(direction: pd.Series) -> float:
    radians = np.deg2rad(direction.to_numpy(dtype=float))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)


def _directional_concentration(direction: pd.Series) -> float:
    radians = np.deg2rad(direction.to_numpy(dtype=float))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    return float(np.hypot(mean_sin, mean_cos))


def _add_count_labels(ax, bars) -> None:
    max_height = max((bar.get_height() for bar in bars), default=0)
    pad = max(max_height * 0.012, 6)
    for bar in bars:
        count = int(bar.get_height())
        if count <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def plot_location_distribution(
    location: str,
    out_dir: Path,
    bin_width: float = 1.0,
) -> Path:
    df = _clean_pairs(location)
    edges = _bin_edges([df[NORA3_COL], df[OBS_COL]], bin_width)
    labels = _bar_labels(edges)
    nora3_counts = _hist(df[NORA3_COL], edges)
    obs_counts = _hist(df[OBS_COL], edges)

    group_spacing = 1.18
    x = np.arange(len(labels), dtype=float) * group_spacing
    width = 0.49

    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    nora3_bars = ax.bar(
        x - width / 2,
        nora3_counts,
        width=width,
        label="NORA3",
        color=COLORS["nora3"],
        edgecolor="#222222",
        linewidth=0.6,
    )
    obs_bars = ax.bar(
        x + width / 2,
        obs_counts,
        width=width,
        label="Observed",
        color=COLORS["observed"],
        edgecolor="#222222",
        linewidth=0.6,
    )
    _add_count_labels(ax, nora3_bars)
    _add_count_labels(ax, obs_bars)

    ymax = max(nora3_counts.max(initial=0), obs_counts.max(initial=0))
    ax.set_ylim(0, ymax * 1.14 if ymax > 0 else 1)
    ax.set_xlim(x[0] - 0.62, x[-1] + 0.62)
    ax.set_xlabel("Hs/Hm0")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right")
    _style_boxed_axes(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{location}_nora3_observed_hs_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scatter(location: str, out_dir: Path) -> Path:
    df = _clean_pairs(location)
    obs = df[OBS_COL].to_numpy(dtype=float)
    nora3 = df[NORA3_COL].to_numpy(dtype=float)
    limit = float(np.ceil(max(np.nanmax(obs), np.nanmax(nora3)) * 10.0) / 10.0)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(
        obs,
        nora3,
        s=10,
        color=COLORS["scatter"],
        alpha=0.42,
        linewidths=0,
    )
    ax.plot([0, limit], [0, limit], color="#111111", linestyle="--", linewidth=1.1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Observed Hm0 (m)")
    ax.set_ylabel("NORA3 Hs (m)")
    _style_boxed_axes(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{location}_nora3_observed_hs_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def paired_metrics(location: str) -> dict:
    df = _clean_pairs(location)
    error = _error_series(df)
    abs_error = error.abs()
    obs = df.loc[error.index, OBS_COL]
    nora3 = df.loc[error.index, NORA3_COL]
    slope, intercept = np.polyfit(obs.to_numpy(dtype=float), nora3.to_numpy(dtype=float), 1)
    pearson_r = float(obs.corr(nora3))
    return {
        "location": location,
        "n": int(error.count()),
        "obs_mean": float(obs.mean()),
        "nora3_mean": float(nora3.mean()),
        "obs_median": float(obs.median()),
        "nora3_median": float(nora3.median()),
        "obs_p90": float(obs.quantile(0.90)),
        "nora3_p90": float(nora3.quantile(0.90)),
        "obs_p95": float(obs.quantile(0.95)),
        "nora3_p95": float(nora3.quantile(0.95)),
        "obs_p99": float(obs.quantile(0.99)),
        "nora3_p99": float(nora3.quantile(0.99)),
        "obs_max": float(obs.max()),
        "nora3_max": float(nora3.max()),
        "mean_diff": float(nora3.mean() - obs.mean()),
        "p95_diff": float(nora3.quantile(0.95) - obs.quantile(0.95)),
        "bias": float(error.mean()),
        "median_error": float(error.median()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "std_error": float(error.std(ddof=1)),
        "min_error": float(error.min()),
        "max_error": float(error.max()),
        "p01_error": float(error.quantile(0.01)),
        "p05_error": float(error.quantile(0.05)),
        "p95_error": float(error.quantile(0.95)),
        "p99_error": float(error.quantile(0.99)),
        "over_pct": float((error > 0).mean() * 100.0),
        "under_pct": float((error < 0).mean() * 100.0),
        "pearson_r": pearson_r,
        "r2": pearson_r**2,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
    }


def plot_error_distribution(
    location: str,
    out_dir: Path,
    bin_width: float = 0.1,
) -> Path:
    df = _clean_pairs(location)
    error = _error_series(df)
    edges = _error_edges(error, bin_width)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(
        error.to_numpy(dtype=float),
        bins=edges,
        color=COLORS["error"],
        edgecolor="#222222",
        linewidth=0.55,
        alpha=0.82,
    )
    ax.axvline(0.0, color=COLORS["zero"], linestyle="--", linewidth=1.1)
    ax.set_xlabel("NORA3 Hs error (m)")
    ax.set_ylabel("Count")
    _style_boxed_axes(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{location}_nora3_hs_error_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def direction_metrics(location: str) -> dict:
    df, direction_col = _clean_observed_direction_pairs(location)
    sector_idx = _direction_sector_indices(df[direction_col])
    shares = sector_idx.value_counts(normalize=True).reindex(range(8), fill_value=0.0)
    top = shares.sort_values(ascending=False)
    dominant_idx = int(top.index[0])
    return {
        "location": location,
        "direction_col": direction_col,
        "n_total_direction": int(len(df)),
        "dominant_sector": COMPASS_LABELS_8[dominant_idx],
        "dominant_sector_share": float(shares.loc[dominant_idx] * 100.0),
        "top2_sector_share": float(top.iloc[:2].sum() * 100.0),
        "circular_mean_direction": _circular_mean_deg(df[direction_col]),
        "directional_concentration": _directional_concentration(df[direction_col]),
    }


def plot_direction_distribution(
    location: str,
    out_dir: Path,
) -> Path:
    df, direction_col = _clean_observed_direction_pairs(location)
    sector_idx = _direction_sector_indices(df[direction_col])
    counts = sector_idx.value_counts().reindex(range(8), fill_value=0).astype(int)
    shares = counts / max(int(counts.sum()), 1) * 100.0
    x = np.arange(len(COMPASS_LABELS_8), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(
        x,
        shares.to_numpy(dtype=float),
        width=0.78,
        color=COLORS["observed"],
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.set_xlabel("Observed peak wave direction")
    ax.set_ylabel("Share of observations (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPASS_LABELS_8)
    ax.set_ylim(0, max(float(shares.max()) * 1.18, 1.0))
    _style_boxed_axes(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{location}_observed_wave_direction.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _direction_shares(location: str) -> pd.Series:
    df, direction_col = _clean_observed_direction_pairs(location)
    sector_idx = _direction_sector_indices(df[direction_col])
    counts = sector_idx.value_counts().reindex(range(8), fill_value=0).astype(int)
    return counts / max(int(counts.sum()), 1) * 100.0


def plot_wave_rose_comparison(
    locations: List[str],
    out_dir: Path,
) -> Path:
    shares_by_location = {
        location: _direction_shares(location)
        for location in locations
    }
    rmax = max(float(shares.max()) for shares in shares_by_location.values())
    rmax = max(np.ceil(rmax / 10.0) * 10.0, 10.0)

    theta = np.deg2rad(np.arange(0, 360, 45))
    width = np.deg2rad(38)
    fig, axes = plt.subplots(
        1,
        len(locations),
        figsize=(3.5 * len(locations), 3.2),
        subplot_kw={"projection": "polar"},
    )
    if len(locations) == 1:
        axes = [axes]

    for ax, location in zip(axes, locations):
        shares = shares_by_location[location].to_numpy(dtype=float)
        ax.bar(
            theta,
            shares,
            width=width,
            bottom=0.0,
            color=COLORS["observed"],
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.88,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta)
        ax.set_xticklabels(COMPASS_LABELS_8)
        ax.set_ylim(0, rmax)
        ax.set_yticks(np.linspace(0, rmax, 4)[1:])
        ax.set_yticklabels([f"{tick:.0f}%" for tick in np.linspace(0, rmax, 4)[1:]], fontsize=8)
        ax.set_rlabel_position(22.5)
        ax.grid(alpha=0.35, linewidth=0.8)
        title = LOCATION_LABELS.get(location, location.title())
        ax.set_title(title, pad=16)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "observed_wave_rose_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _print_metric_table(title: str, columns: List[tuple], metrics: List[dict]) -> None:
    print(f"\n{title}")
    print(",".join(label for _, label, _ in columns))
    for row in metrics:
        values = []
        for key, _, fmt in columns:
            values.append(fmt.format(row[key]))
        print(",".join(values))


def print_metric_sections(metrics: List[dict]) -> None:
    error_columns = [
        ("location", "location", "{}"),
        ("n", "n", "{:d}"),
        ("bias", "bias_m", "{:+.3f}"),
        ("median_error", "median_error_m", "{:+.3f}"),
        ("mae", "mae_m", "{:.3f}"),
        ("rmse", "rmse_m", "{:.3f}"),
        ("std_error", "std_error_m", "{:.3f}"),
        ("min_error", "min_error_m", "{:+.3f}"),
        ("p01_error", "p01_error_m", "{:+.3f}"),
        ("p05_error", "p05_error_m", "{:+.3f}"),
        ("p95_error", "p95_error_m", "{:+.3f}"),
        ("p99_error", "p99_error_m", "{:+.3f}"),
        ("max_error", "max_error_m", "{:+.3f}"),
        ("over_pct", "over_pct", "{:.1f}"),
        ("under_pct", "under_pct", "{:.1f}"),
    ]
    hs_columns = [
        ("location", "location", "{}"),
        ("n", "n", "{:d}"),
        ("obs_mean", "obs_mean_m", "{:.3f}"),
        ("nora3_mean", "nora3_mean_m", "{:.3f}"),
        ("mean_diff", "mean_diff_m", "{:+.3f}"),
        ("obs_median", "obs_median_m", "{:.3f}"),
        ("nora3_median", "nora3_median_m", "{:.3f}"),
        ("obs_p90", "obs_p90_m", "{:.3f}"),
        ("nora3_p90", "nora3_p90_m", "{:.3f}"),
        ("obs_p95", "obs_p95_m", "{:.3f}"),
        ("nora3_p95", "nora3_p95_m", "{:.3f}"),
        ("p95_diff", "p95_diff_m", "{:+.3f}"),
        ("obs_p99", "obs_p99_m", "{:.3f}"),
        ("nora3_p99", "nora3_p99_m", "{:.3f}"),
        ("obs_max", "obs_max_m", "{:.3f}"),
        ("nora3_max", "nora3_max_m", "{:.3f}"),
    ]
    scatter_columns = [
        ("location", "location", "{}"),
        ("n", "n", "{:d}"),
        ("pearson_r", "pearson_r", "{:.3f}"),
        ("r2", "r2", "{:.3f}"),
        ("fit_slope", "fit_slope", "{:.3f}"),
        ("fit_intercept", "fit_intercept_m", "{:+.3f}"),
        ("bias", "mean_vertical_offset_m", "{:+.3f}"),
        ("rmse", "scatter_rmse_m", "{:.3f}"),
    ]

    _print_metric_table(
        "Error distribution metrics (error = NORA3 hs - observed Hm0)",
        error_columns,
        metrics,
    )
    _print_metric_table("Hs distribution metrics", hs_columns, metrics)
    _print_metric_table("Scatter metrics", scatter_columns, metrics)


def print_direction_metrics(metrics: List[dict]) -> None:
    columns = [
        ("location", "location", "{}"),
        ("direction_col", "direction_col", "{}"),
        ("n_total_direction", "n_total", "{:d}"),
        ("dominant_sector", "dominant_sector", "{}"),
        ("dominant_sector_share", "dominant_sector_share_pct", "{:.1f}"),
        ("top2_sector_share", "top2_sector_share_pct", "{:.1f}"),
        ("circular_mean_direction", "circular_mean_direction_deg", "{:.1f}"),
        ("directional_concentration", "directional_concentration", "{:.3f}"),
    ]
    _print_metric_table("Observed-wave direction metrics", columns, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired NORA3 vs observed wave-height distributions with 1 m bins."
        )
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=DEFAULT_LOCATIONS,
        help=(
            "Locations to plot. Defaults to fedjeosen fauskane vestfjorden. "
            f"Configured buoy locations: {', '.join(get_buoy_locations())}."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--error-bin-width", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if args.error_bin_width <= 0:
        raise ValueError("--error-bin-width must be positive")
    saved = []
    metrics = []
    direction_rows = []
    for location in args.locations:
        saved.append(plot_location_distribution(location, args.out_dir, args.bin_width))
        saved.append(plot_scatter(location, args.out_dir))
        saved.append(plot_error_distribution(location, args.out_dir, args.error_bin_width))
        saved.append(plot_direction_distribution(location, args.out_dir))
        metrics.append(paired_metrics(location))
        direction_rows.append(direction_metrics(location))
    saved.append(plot_wave_rose_comparison(args.locations, args.out_dir))

    for path in saved:
        print(path)
    print_metric_sections(metrics)
    print_direction_metrics(direction_rows)


if __name__ == "__main__":
    main()
