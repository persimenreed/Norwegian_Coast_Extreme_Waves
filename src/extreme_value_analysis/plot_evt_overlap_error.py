import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/extreme_value_modelling")
OUT = Path("results/extreme_value_analysis/overlap_evt")

RETURN_PERIODS = [2, 5, 10]


def load_data(location):

    path = ROOT / location / "overlap_summary_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


# ----------------------------------------------------------
# 1: Skill plot (observed vs model)
# ----------------------------------------------------------

def plot_skill(location):

    df = load_data(location)

    fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(14, 4))

    for i, rp in enumerate(RETURN_PERIODS):

        ax = axes[i]

        subset = df[df.return_period == rp]

        for _, row in subset.iterrows():

            ax.scatter(
                row.rl_obs,
                row.rl_model,
                label=row.dataset,
                s=60
            )

        obs = subset.rl_obs.values
        model = subset.rl_model.values

        center = np.mean(obs)

        span = max(
            np.max(np.abs(obs - center)),
            np.max(np.abs(model - center)),
        ) * 1.15

        xmin = center - span
        xmax = center + span

        ax.plot([xmin, xmax], [xmin, xmax], "k--")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)

        ax.set_title(f"RL{rp}")
        ax.set_xlabel("Observed (m)")
        ax.set_ylabel("Model (m)")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=False,
    )

    fig.suptitle(f"Overlap EVT skill — {location}")

    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / f"{location}_overlap_skill.png"

    plt.tight_layout(rect=(0, 0.22, 1, 0.95))
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)


# ----------------------------------------------------------
# 2: Error bar chart (most useful)
# ----------------------------------------------------------

def plot_error_bars(location):

    df = load_data(location)

    fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(14, 4))

    for i, rp in enumerate(RETURN_PERIODS):

        ax = axes[i]

        subset = df[df.return_period == rp].copy()

        subset = subset.sort_values("error")

        ax.barh(
            subset.dataset,
            subset.error,
        )

        ax.axvline(0, color="k", linestyle="--")

        ax.set_title(f"RL{rp} error (m)")
        ax.set_xlabel("Model − Observed (m)")

        ax.grid(alpha=0.3)

    fig.suptitle(f"Overlap EVT error — {location}")

    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / f"{location}_overlap_error.png"

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)


# ----------------------------------------------------------
# 3: Absolute error ranking
# ----------------------------------------------------------

def plot_abs_error(location):

    df = load_data(location)

    fig, ax = plt.subplots(figsize=(6, 4))

    pivot = df.pivot(
        index="dataset",
        columns="return_period",
        values="abs_error"
    )

    sort_col = 2.0 if 2.0 in pivot.columns else (2 if 2 in pivot.columns else None)
    if sort_col is not None:
        pivot = pivot.sort_values(by=sort_col, ascending=True)

    pivot.plot(kind="bar", ax=ax)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_ylabel("Absolute error (m)")
    ax.set_title(f"Overlap EVT absolute error — {location}")

    ax.grid(alpha=0.3)

    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / f"{location}_overlap_abs_error.png"

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)

    # ----------------------------------------------------------
# 4: Improvement relative to raw hindcast
# ----------------------------------------------------------

def plot_improvement(location):

    df = load_data(location)

    # compute improvement if not present
    if "raw_error" not in df.columns:
        raise ValueError("raw_error column missing from summary_metrics")

    df["raw_abs_error"] = np.abs(df["raw_error"])
    df["improvement"] = df["raw_abs_error"] - df["abs_error"]

    fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(14,4))

    for i, rp in enumerate(RETURN_PERIODS):

        ax = axes[i]

        subset = df[df.return_period == rp].copy()

        subset = subset.sort_values("improvement", ascending=False)

        ax.barh(
            subset.dataset,
            subset.improvement
        )

        ax.axvline(0, color="k", linestyle="--")

        ax.set_title(f"RL{rp} improvement vs raw (m)")
        ax.set_xlabel("|raw error| − |method error|")

        ax.grid(alpha=0.3)

    fig.suptitle(f"Overlap EVT improvement vs raw — {location}")

    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / f"{location}_overlap_improvement.png"

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--locations",
        nargs="+",
        default=["fauskane", "fedjeosen"]
    )

    args = parser.parse_args()

    for loc in args.locations:

        plot_skill(loc)
        plot_error_bars(loc)
        plot_abs_error(loc)
        plot_improvement(loc)


if __name__ == "__main__":
    main()