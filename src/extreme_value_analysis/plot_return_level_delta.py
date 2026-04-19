import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_ROOT = Path("results/extreme_value_modelling")
OUT = Path("results/extreme_value_analysis/return_level_delta")
MIN_RETURN_PERIOD = 1.0
MAX_RETURN_PERIOD = 50.0


def load_summary(location):
    path = SUMMARY_ROOT / location / "summary_return_levels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["return_period"] = pd.to_numeric(df["return_period"], errors="coerce")
    return df[
        df["return_period"].between(MIN_RETURN_PERIOD, MAX_RETURN_PERIOD, inclusive="both")
    ].copy()


def dataset_label(d):
    if d == "raw":
        return "raw"
    if d.startswith("local_"):
        return d.replace("local_", "local: ")
    if d.startswith("transfer_"):
        return d.replace("transfer_", "transfer: ")
    if d.startswith("ensemble_"):
        return d.replace("ensemble_", "ensemble: ")
    return d


def plot_delta(location, datasets):
    df = load_summary(location)

    for model in ["GEV", "GPD"]:
        sub = df[df["model"] == model].copy()
        raw = sub[sub["dataset"] == "raw"].copy().sort_values("return_period")

        if raw.empty:
            print(f"No raw dataset found for {location} / {model}")
            continue

        plt.figure(figsize=(8, 5))

        for d in datasets:
            if d == "raw":
                continue

            cur = sub[sub["dataset"] == d].copy().sort_values("return_period")
            if cur.empty:
                continue

            merged = raw[["return_period", "return_level"]].rename(
                columns={"return_level": "rl_raw"}
            ).merge(
                cur[["return_period", "return_level"]].rename(
                    columns={"return_level": "rl_cur"}
                ),
                on="return_period",
                how="inner",
            )

            if merged.empty:
                continue

            delta = merged["rl_cur"].values - merged["rl_raw"].values
            plt.plot(
                merged["return_period"].values,
                delta,
                marker="o",
                label=dataset_label(d),
            )

        non_raw = [d for d in datasets if d != "raw"]
        if any(d.startswith("transfer_") for d in non_raw):
            correction_label = " — transfer bias correction"
            file_suffix = "_transfer"
        elif any(d.startswith("local_") for d in non_raw):
            correction_label = " — local bias correction"
            file_suffix = "_local"
        else:
            correction_label = ""
            file_suffix = ""

        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Return period (years)")
        plt.ylabel("delta return level vs raw (m)")
        plt.title(f"{model} return-level change — {location}{correction_label}")
        plt.xlim(MIN_RETURN_PERIOD, MAX_RETURN_PERIOD)
        plt.grid(alpha=0.3)
        plt.legend()

        out_dir = OUT / location
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{location}_{model.lower()}_delta_vs_raw{file_suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    plot_delta(args.location, args.datasets)


if __name__ == "__main__":
    main()
