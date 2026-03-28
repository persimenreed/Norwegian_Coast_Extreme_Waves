import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extreme_value_modelling.extreme_preprocessing import (
    DECLUSTER_HOURS,
    THRESHOLD_QUANTILE,
    compute_pot,
    load_data,
)
from src.extreme_value_modelling.paths import resolve_input_path

OUT_ROOT = Path("results/extreme_value_analysis/diagnostics")


def _dataset_path(location: str, dataset: str) -> Path:
    name = str(dataset).strip()

    if name == "raw":
        return resolve_input_path(location=location, mode="raw")

    return Path(f"data/output/{location}/hindcast_corrected_{name}.csv")


def _display_name(dataset: str) -> str:
    name = str(dataset).strip()
    if name == "raw":
        return "raw"
    if name.startswith("local_"):
        return name.replace("local_", "local: ")
    if name.startswith("transfer_"):
        return name.replace("transfer_", "transfer: ")
    if name.startswith("ensemble_"):
        return name.replace("ensemble_", "ensemble: ")
    return name


def _threshold_grid(values: np.ndarray) -> np.ndarray:
    return np.linspace(np.percentile(values, 90), np.percentile(values, 99), 30)


def _diagnostic_series(hs: np.ndarray):
    thresholds = _threshold_grid(hs)
    mean_excess = []
    shape = []

    for threshold in thresholds:
        exceed = hs[hs > threshold]
        excess = exceed - threshold

        if len(excess) > 30:
            mean_excess.append(float(np.mean(excess)))
            fitted_shape, _, _ = genpareto.fit(excess, floc=0)
            shape.append(float(fitted_shape))
        else:
            mean_excess.append(np.nan)
            shape.append(np.nan)

    return (
        thresholds,
        np.asarray(mean_excess, dtype=float),
        np.asarray(shape, dtype=float),
    )


def _marker_y(thresholds: np.ndarray, values: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(thresholds) & np.isfinite(values)
    if np.sum(valid) == 0:
        return np.nan
    x = thresholds[valid]
    y = values[valid]
    if threshold <= x.min():
        return float(y[np.argmin(x)])
    if threshold >= x.max():
        return float(y[np.argmax(x)])
    return float(np.interp(threshold, x, y))


def _plot_single_metric(location: str, datasets, metric_key: str, diagnostics_map: dict):
    metric_meta = {
        "mean_excess": {
            "title": "Mean Residual Life",
            "ylabel": "Mean Excess",
            "filename": "mean_residual_life",
        },
        "shape": {
            "title": "Shape Stability",
            "ylabel": "Shape xi",
            "filename": "shape_stability",
        },
    }
    meta = metric_meta[metric_key]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    cmap = plt.get_cmap("tab10")

    for idx, dataset in enumerate(datasets):
        color = cmap(idx % 10)
        diag = diagnostics_map[dataset]
        thresholds = diag["thresholds"]
        values = diag[metric_key]
        used_threshold = diag["used_threshold"]
        marker_y = _marker_y(thresholds, values, used_threshold)

        ax.plot(
            thresholds,
            values,
            linewidth=2.0,
            color=color,
            label=_display_name(dataset),
        )

        if np.isfinite(marker_y):
            ax.scatter(
                [used_threshold],
                [marker_y],
                s=42,
                color="red",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

    threshold_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=7,
        linestyle="None",
        label=f"{THRESHOLD_QUANTILE:.0%}",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(threshold_handle)
    labels.append(threshold_handle.get_label())

    if metric_key == "shape":
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=0)

    ax.set_title(f"{location} - {meta['title']}")
    ax.set_xlabel("Threshold (m)")
    ax.set_ylabel(meta["ylabel"])
    ax.grid(alpha=0.3)
    ax.legend(handles, labels, frameon=True)

    out_dir = OUT_ROOT / location
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(d) for d in datasets)
    out_path = out_dir / f"{meta['filename']}_{tag}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def plot(location: str, datasets):
    diagnostics_map = {}

    for dataset in datasets:
        path = _dataset_path(location, dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"Input dataset not found for '{dataset}': {path}"
            )

        df = load_data(str(path))
        hs = df["hs"].to_numpy(dtype=float)
        thresholds, mean_excess, shape = _diagnostic_series(hs)
        _, used_threshold, _, _ = compute_pot(
            df,
            quantile=THRESHOLD_QUANTILE,
            decluster_hours=DECLUSTER_HOURS,
        )

        diagnostics_map[dataset] = {
            "thresholds": thresholds,
            "mean_excess": mean_excess,
            "shape": shape,
            "used_threshold": float(used_threshold),
        }

    _plot_single_metric(location, datasets, "mean_excess", diagnostics_map)
    _plot_single_metric(location, datasets, "shape", diagnostics_map)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay EVT threshold diagnostics for one location and one or more datasets."
        )
    )
    parser.add_argument("--location", required=True, help="Target location.")
    parser.add_argument(
        "--method",
        nargs="+",
        required=True,
        metavar="DATASET",
        help=(
            "One or more dataset names, e.g. raw ensemble_combined "
            "local_xgboost transfer_fedjeosen_xgboost."
        ),
    )

    args = parser.parse_args()
    plot(args.location, args.method)


if __name__ == "__main__":
    main()
