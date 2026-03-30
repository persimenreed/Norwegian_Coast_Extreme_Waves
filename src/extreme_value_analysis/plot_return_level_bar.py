import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SUMMARY_ROOT = Path("results/extreme_value_modelling")
RESULT_DIR = Path("results/extreme_value_analysis/return_level")
RETURN_PERIODS = [10.0, 20.0, 50.0]
RETURN_PERIOD_100 = [100.0]


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
    if d.startswith("ensemble_"):
        return d.replace("ensemble_", "")
    return d





def plot_location(location, datasets, return_periods=RETURN_PERIODS, output_name=None):
    df = load_summary(location)

    # Keep a preferred order for key ensemble comparisons, then stable fallback order.
    preferred_rank = {
        "raw": 0,
        "ensemble_fedjeosen": 1,
        "ensemble_fauskane": 2,
    }

    # Stable fallback order based on the lowest available 10-year RL across models.
    rp10 = df[np.isclose(df["return_period"], 10.0)][["dataset", "return_level"]].copy()
    rp10_map = (
        rp10.groupby("dataset", as_index=True)["return_level"]
        .min()
        .to_dict()
    )
    ordered_datasets = sorted(
        datasets,
        key=lambda d: (
            preferred_rank.get(d, 99),
            d not in rp10_map,
            rp10_map.get(d, float("inf")),
            d,
        ),
    )

    fig_width = max(8 if len(return_periods) == 1 else 11, 2.5 * len(ordered_datasets) + 4.5)
    fig, axes = plt.subplots(1, len(return_periods), figsize=(fig_width, 5.2), sharey=True)
    if len(return_periods) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    dataset_colors = {d: cmap(i % 10) for i, d in enumerate(ordered_datasets)}

    n = len(ordered_datasets)
    group_gap = 0.6
    x_gev = np.arange(n, dtype=float)
    x_gpd = x_gev + n + group_gap

    for ax, rp in zip(axes, return_periods):
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

        max_top = 0.0

        for i, d in enumerate(ordered_datasets):
            color = dataset_colors[d]

            if np.isfinite(gev_vals[i]):
                ax.bar(x_gev[i], gev_vals[i], width=0.8, color=color)
                ax.errorbar(
                    [x_gev[i]],
                    [gev_vals[i]],
                    yerr=[[gev_err_lo[i]], [gev_err_hi[i]]],
                    fmt="none",
                    capsize=3,
                    color="k",
                    linewidth=1,
                )
                max_top = max(max_top, gev_vals[i] + gev_err_hi[i])

            if np.isfinite(gpd_vals[i]):
                ax.bar(x_gpd[i], gpd_vals[i], width=0.8, color=color)
                ax.errorbar(
                    [x_gpd[i]],
                    [gpd_vals[i]],
                    yerr=[[gpd_err_lo[i]], [gpd_err_hi[i]]],
                    fmt="none",
                    capsize=3,
                    color="k",
                    linewidth=1,
                )
                max_top = max(max_top, gpd_vals[i] + gpd_err_hi[i])

        if max_top == 0:
            ax.set_title(f"{int(rp)}-year")
            ax.grid(axis="y", alpha=0.3)
            continue

        top_pad = max(0.5, 0.08 * max_top)
        ax.set_ylim(0, max_top + 2.2 * top_pad)

        # Subtle background split between GEV and GPD sections.
        gev_left = x_gev[0] - 0.5
        gev_right = x_gev[-1] + 0.5
        gpd_left = x_gpd[0] - 0.5
        gpd_right = x_gpd[-1] + 0.5
        gev_bg = "#eaf3ff"
        gpd_bg = "#fdecec"
        ax.axvspan(gev_left, gev_right, color=gev_bg, zorder=0)
        ax.axvspan(gpd_left, gpd_right, color=gpd_bg, zorder=0)

        for i in range(n):
            if np.isfinite(gev_vals[i]):
                ci_width = gev_err_lo[i] + gev_err_hi[i]
                ax.text(
                    x_gev[i],
                    gev_vals[i] + gev_err_hi[i] + 0.45 * top_pad,
                    f"{gev_vals[i]:.2f}\n({ci_width:.2f})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if np.isfinite(gpd_vals[i]):
                ci_width = gpd_err_lo[i] + gpd_err_hi[i]
                ax.text(
                    x_gpd[i],
                    gpd_vals[i] + gpd_err_hi[i] + 0.45 * top_pad,
                    f"{gpd_vals[i]:.2f}\n({ci_width:.2f})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        xticks = np.concatenate([x_gev, x_gpd])
        xticklabels = [dataset_label(d) for d in ordered_datasets] + [dataset_label(d) for d in ordered_datasets]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=30, ha="right")

        sep_x = (x_gev[-1] + x_gpd[0]) / 2
        ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(f"{int(rp)}-year")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Return level Hs (m)")

    bg_handles = [
        Patch(facecolor="#eaf3ff", edgecolor="none", label="GEV"),
        Patch(facecolor="#fdecec", edgecolor="none", label="GPD"),
    ]
    fig.legend(
        handles=bg_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )

    non_raw = [d for d in ordered_datasets if d != "raw"]
    if any(d.startswith("transfer_") for d in non_raw):
        correction_label = " — transfer bias correction"
    elif any(d.startswith("local_") for d in non_raw):
        correction_label = " — local bias correction"
    else:
        correction_label = ""
    fig.suptitle(f"{location}{correction_label} — return levels", y=0.98)

    if output_name:
        out_path = RESULT_DIR / location / output_name
    else:
        suffix = "100yr" if len(return_periods) == 1 and np.isclose(return_periods[0], 100.0) else "bars"
        out_path = RESULT_DIR / location / f"{location}_gev_gpd_return_level_{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.4)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Custom output filename (e.g., custom_name.png). If not provided, defaults to <location>_gev_gpd_return_level_bars.png",
    )
    args = parser.parse_args()

    plot_location(args.location, args.datasets, output_name=args.output_name)
    plot_location(
        args.location,
        args.datasets,
        return_periods=RETURN_PERIOD_100,
        output_name=f"{args.location}_gev_gpd_return_level_100yr.png",
    )


if __name__ == "__main__":
    main()
