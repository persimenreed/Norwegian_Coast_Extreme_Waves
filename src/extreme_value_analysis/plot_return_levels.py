import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SUMMARY_ROOT = Path("results/extreme_value_modelling")
RESULT_DIR = Path("results/extreme_value_analysis")
YMIN_FLOOR = 8.0


def load_summary(location):
    path = SUMMARY_ROOT / location / "summary_return_levels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return pd.read_csv(path)


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


def dataset_tag(datasets):
    tags = []
    for d in datasets:
        if d == "raw":
            tags.append("raw")
        elif d.startswith("local_"):
            tags.append(d.replace("local_", "loc-"))
        elif d.startswith("transfer_"):
            tags.append(d.replace("transfer_", "tr-"))
        elif d.startswith("ensemble_"):
            tags.append(d.replace("ensemble_", "ens-"))
        else:
            tags.append(d)
    return "-".join(tags)


def plot_model(df, location, model, datasets, ymin):
    df_model = df[df["model"] == model].copy()

    plt.figure(figsize=(8, 5.5))

    for dataset in datasets:
        subset = df_model[df_model["dataset"] == dataset].copy()

        if subset.empty:
            print(f"WARNING: dataset '{dataset}' not found for {location}")
            continue

        subset = subset.sort_values("return_period")

        T = subset["return_period"].values
        rl = subset["return_level"].values
        ci_low = subset["ci_lower"].values
        ci_high = subset["ci_upper"].values

        line, = plt.plot(T, rl, label=dataset_label(dataset), linewidth=2)
        color = line.get_color()

        plt.fill_between(T, ci_low, ci_high, alpha=0.12, color=color)
        plt.plot(T, ci_low, linestyle=":", color=color, linewidth=1)
        plt.plot(T, ci_high, linestyle=":", color=color, linewidth=1)

    plt.xlabel("Return period (years)")
    plt.ylabel("Return level Hs (m)")
    plt.title(f"{model} return levels — {location}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(1, 50)

    bottom = YMIN_FLOOR if ymin is None else max(YMIN_FLOOR, ymin)
    plt.ylim(bottom=bottom)

    tag = dataset_tag(datasets)
    out_path = RESULT_DIR / "return_level" / location / f"{location}_{model.lower()}_rl_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--ymin", type=float, default=None)
    args = parser.parse_args()

    df = load_summary(args.location)
    plot_model(df, args.location, "GEV", args.datasets, args.ymin)
    plot_model(df, args.location, "GPD", args.datasets, args.ymin)


if __name__ == "__main__":
    main()
