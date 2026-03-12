import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_ROOT = Path("results/extreme_value_modelling")
RESULT_DIR = Path("results/extreme_value_analysis/return_level")
RETURN_PERIODS = [10.0, 20.0, 50.0]


def load_summary(location):
    path = SUMMARY_ROOT / location / "summary_return_levels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dataset_label(d):
    if d == "raw":
        return "raw"
    if d.startswith("local_"):
        return d.replace("local_", "")
    if d.startswith("transfer_"):
        parts = d.split("_", 2)
        if len(parts) == 3:
            return parts[2]
        return d.replace("transfer_", "")
    if d.startswith("pooled_"):
        return d.replace("pooled_", "")
    return d


def dataset_group_tag(datasets):
    groups = set()
    for d in datasets:
        if d == "raw":
            continue
        if d.startswith("local_"):
            groups.add("local")
        elif d.startswith("transfer_"):
            groups.add("transfer")
        elif d.startswith("pooled_"):
            groups.add("pooled")
        else:
            groups.add("other")

    if not groups:
        return "raw"
    if len(groups) == 1:
        return next(iter(groups))
    return "mixed"


def plot_location(location, datasets):
    df = load_summary(location)
    group_tag = dataset_group_tag(datasets)

    # Keep one stable dataset order based on the lowest available 10-year RL across models.
    rp10 = df[np.isclose(df["return_period"], 10.0)][["dataset", "return_level"]].copy()
    rp10_map = (
        rp10.groupby("dataset", as_index=True)["return_level"]
        .min()
        .to_dict()
    )
    ordered_datasets = sorted(
        datasets,
        key=lambda d: (
            d not in rp10_map,
            rp10_map.get(d, float("inf")),
            d,
        ),
    )

    fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(16, 4.5), sharey=True)
    width = 0.38

    for ax, rp in zip(axes, RETURN_PERIODS):
        labels = []
        gev_vals, gev_err_lo, gev_err_hi = [], [], []
        gpd_vals, gpd_err_lo, gpd_err_hi = [], [], []

        for d in ordered_datasets:
            row_gev = df[
                (df["model"] == "GEV")
                & (df["dataset"] == d)
                & (np.isclose(df["return_period"], rp))
            ]
            row_gpd = df[
                (df["model"] == "GPD")
                & (df["dataset"] == d)
                & (np.isclose(df["return_period"], rp))
            ]

            if row_gev.empty and row_gpd.empty:
                continue

            labels.append(dataset_label(d))

            if row_gev.empty:
                gev_vals.append(np.nan)
                gev_err_lo.append(0.0)
                gev_err_hi.append(0.0)
            else:
                r = row_gev.iloc[0]
                rl = float(r["return_level"])
                lo = float(r["ci_lower"])
                hi = float(r["ci_upper"])
                gev_vals.append(rl)
                gev_err_lo.append(rl - lo)
                gev_err_hi.append(hi - rl)

            if row_gpd.empty:
                gpd_vals.append(np.nan)
                gpd_err_lo.append(0.0)
                gpd_err_hi.append(0.0)
            else:
                r = row_gpd.iloc[0]
                rl = float(r["return_level"])
                lo = float(r["ci_lower"])
                hi = float(r["ci_upper"])
                gpd_vals.append(rl)
                gpd_err_lo.append(rl - lo)
                gpd_err_hi.append(hi - rl)

        if not labels:
            ax.set_title(f"{int(rp)}-year")
            ax.grid(axis="y", alpha=0.3)
            continue

        x = np.arange(len(labels), dtype=float)
        x_gev = x - width / 2
        x_gpd = x + width / 2

        ax.bar(x_gev, gev_vals, width=width, label="GEV", color="#4C78A8")
        ax.errorbar(
            x_gev,
            gev_vals,
            yerr=[gev_err_lo, gev_err_hi],
            fmt="none",
            capsize=3,
            color="k",
            linewidth=1,
        )

        ax.bar(x_gpd, gpd_vals, width=width, label="GPD", color="#F58518")
        ax.errorbar(
            x_gpd,
            gpd_vals,
            yerr=[gpd_err_lo, gpd_err_hi],
            fmt="none",
            capsize=3,
            color="k",
            linewidth=1,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{int(rp)}-year")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Return level Hs (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
        )
    fig.suptitle(f"{location} — return levels ({group_tag})", y=0.995)

    out_path = RESULT_DIR / location / f"{location}_gev_gpd_return_level_bars.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    plot_location(args.location, args.datasets)


if __name__ == "__main__":
    main()