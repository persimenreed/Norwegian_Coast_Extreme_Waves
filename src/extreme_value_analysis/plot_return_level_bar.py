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
        return d.replace("local_", "local: ")
    if d.startswith("transfer_"):
        return d.replace("transfer_", "transfer: ")
    if d.startswith("pooled_"):
        return d.replace("pooled_", "pooled: ")
    return d


def plot_location(location, datasets):
    df = load_summary(location)

    for model in ["GEV", "GPD"]:
        sub = df[df["model"] == model].copy()

        fig, axes = plt.subplots(1, len(RETURN_PERIODS), figsize=(14, 4), sharey=True)

        for ax, rp in zip(axes, RETURN_PERIODS):
            vals = []
            errs_low = []
            errs_high = []
            labels = []

            for d in datasets:
                row = sub[(sub["dataset"] == d) & (np.isclose(sub["return_period"], rp))]
                if row.empty:
                    continue

                row = row.iloc[0]
                rl = float(row["return_level"])
                lo = float(row["ci_lower"])
                hi = float(row["ci_upper"])

                vals.append(rl)
                errs_low.append(rl - lo)
                errs_high.append(hi - rl)
                labels.append(dataset_label(d))

            if not vals:
                ax.set_title(f"{int(rp)}-year")
                continue

            x = np.arange(len(vals))
            ax.bar(x, vals)
            ax.errorbar(x, vals, yerr=[errs_low, errs_high], fmt="none", capsize=4, color="k")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(f"{int(rp)}-year")
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel("Return level Hs (m)")
        fig.suptitle(f"{location} — {model} return levels")

        out_path = RESULT_DIR / location / f"{location}_{model.lower()}_return_level_bars.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
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