import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("results/extreme_value_modelling")
OUT = Path("results/extreme_value_analysis/return_level_comparison")

LOCATIONS = ["fauskane", "fedjeosen"]
MODELS = ["GEV", "GPD"]
PERIODS = [10, 20, 50]
STUDY_LOCATIONS = ["stavanger", "bergen", "kristiansund", "vestfjorden"]
STUDY_DATASETS = ["ensemble_combined", "ensemble_fauskane", "ensemble_fedjeosen"]
DATASET_GROUPS = {
    "local": "local_",
    "transfer": "transfer_",
}
OPPOSITE_LOCATION = {
    "fauskane": "fedjeosen",
    "fedjeosen": "fauskane",
}


def dataset_label(d):
    if d == "raw":
        return "raw"
    if d.startswith("local_"):
        return d.replace("local_", "local: ")
    if d.startswith("transfer_"):
        return d.replace("transfer_", "transfer: ")
    if d.startswith("pooled_"):
        return d.replace("pooled_", "pooled: ")
    if d.startswith("ensemble_"):
        return d.replace("ensemble_", "ensemble: ")
    return d


def plot(location, periods, group_name, prefix=None, fixed_datasets=None):

    df = pd.read_csv(ROOT / location / "summary_metrics.csv")
    df = df[df["model"].isin(MODELS)]

    if fixed_datasets is not None:
        df = df[df["dataset"].isin(fixed_datasets)]
    elif group_name == "transfer":
        opposite = OPPOSITE_LOCATION.get(location)
        transfer_mask = df["dataset"].astype(str).str.startswith(prefix)
        if opposite:
            transfer_mask = transfer_mask | (df["dataset"] == f"ensemble_{opposite}")
        df = df[transfer_mask]
    else:
        df = df[df["dataset"].astype(str).str.startswith(prefix)]

    obs_col = "rl_obs" if "rl_obs" in df.columns else "rl_raw"
    model_col = "rl_model"

    if obs_col not in df.columns or model_col not in df.columns:
        raise KeyError(
            f"Expected columns '{obs_col}' and '{model_col}' in {location}/summary_metrics.csv"
        )

    dataset_order = sorted(df["dataset"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    dataset_colors = {d: cmap(i % 10) for i, d in enumerate(dataset_order)}

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.2))

    period_bg = {
        10: "#cfe2ff",  # stronger light blue
        20: "#f7cfd2",  # stronger light red
        50: "#cfeecf",  # stronger light green
    }
    bands = []
    points = []
    all_x = []
    all_y = []

    for return_period in periods:
        subset = df[df.return_period == return_period].copy()
        subset = subset.dropna(subset=[obs_col, model_col])
        if subset.empty:
            continue

        obs = subset[obs_col].to_numpy(dtype=float)
        model_vals = subset[model_col].to_numpy(dtype=float)
        x_min_raw = float(np.min(obs))
        x_max_raw = float(np.max(obs))
        x_center = 0.5 * (x_min_raw + x_max_raw)
        x_half = max(0.82, 0.5 * (x_max_raw - x_min_raw) * 1.19)
        x_min = x_center - x_half
        x_max = x_center + x_half

        # Keep period backgrounds focused around each RP point cluster.
        band_pad = max(0.08, 0.20 * (x_max_raw - x_min_raw))
        band_left = x_min_raw - band_pad
        band_right = x_max_raw + band_pad
        bands.append((return_period, band_left, band_right))

        all_x.extend(obs.tolist())
        all_y.extend(model_vals.tolist())

        for _, row in subset.iterrows():
            points.append((float(row[obs_col]), float(row[model_col]), str(row.dataset), str(row.model)))

    if not points:
        ax.set_xlabel("Raw Hindcast (m)")
        ax.set_ylabel("Model (m)")
        ax.grid(alpha=0.3)
        fig.suptitle(f"{location} ({group_name})")
        OUT.mkdir(parents=True, exist_ok=True)
        period_tag = "_".join(str(p) for p in periods)
        path = OUT / f"{location}_{group_name}_{period_tag}_gev_gpd.png"
        plt.tight_layout(rect=(0, 0, 1, 0.86))
        plt.savefig(path, dpi=300)
        plt.close()
        print("Saved", path)
        return

    y_min_raw = float(np.min(all_y))
    y_max_raw = float(np.max(all_y))
    y_center = 0.5 * (y_min_raw + y_max_raw)
    y_half = max(0.90, 0.5 * (y_max_raw - y_min_raw) * 1.22)
    y_min = y_center - y_half
    y_max = y_center + y_half

    x_min = min(b[1] for b in bands)
    x_max = max(b[2] for b in bands)

    for rp, left, right in bands:
        ax.axvspan(left, right, color=period_bg.get(rp, "#f3f3f3"), alpha=0.42, zorder=0)

    for x, y, dataset, model in points:
        color = dataset_colors.get(dataset, "C0")
        is_hollow = model == "GPD"
        ax.scatter(
            x,
            y,
            s=44,
            facecolors="none" if is_hollow else color,
            edgecolors=color,
            linewidths=1.3,
            zorder=2,
        )

    line_min = min(x_min, y_min)
    line_max = max(x_max, y_max)
    ax.plot([line_min, line_max], [line_min, line_max], "k--", zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Raw Hindcast (m)")
    ax.set_ylabel("Model (m)")
    ax.grid(alpha=0.3)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_colors[d],
            markeredgecolor=dataset_colors[d],
            markersize=6,
            linestyle="None",
            label=dataset_label(d),
        )
        for d in dataset_order
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=6,
            linestyle="None",
            label="GEV (filled)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=6,
            linestyle="None",
            label="GPD (hollow)",
        ),
    ]

    period_handles = [
        Patch(facecolor=period_bg[10], edgecolor="none", label="RL10 background"),
        Patch(facecolor=period_bg[20], edgecolor="none", label="RL20 background"),
        Patch(facecolor=period_bg[50], edgecolor="none", label="RL50 background"),
    ]

    model_period_handles = model_handles + period_handles
    fig.legend(
        handles=model_period_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
    )

    dataset_cols = 1 if len(dataset_handles) <= 8 else 2
    fig.legend(
        handles=dataset_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=dataset_cols,
        frameon=True,
        title="Datasets",
    )

    fig.suptitle(f"{location} ({group_name})")

    OUT.mkdir(parents=True, exist_ok=True)

    period_tag = "_".join(str(p) for p in periods)
    path = OUT / f"{location}_{group_name}_{period_tag}_gev_gpd.png"

    if group_name == "transfer":
        top_margin = 0.76
    elif group_name == "local":
        top_margin = 0.80
    else:
        top_margin = 0.84
    plt.tight_layout(rect=(0, 0, 1, top_margin))
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)


def main():

    for loc in LOCATIONS:
        for group_name, prefix in DATASET_GROUPS.items():
            plot(loc, PERIODS, group_name, prefix)

    for loc in STUDY_LOCATIONS:
        plot(loc, PERIODS, "study_area", fixed_datasets=STUDY_DATASETS)


if __name__ == "__main__":
    main()