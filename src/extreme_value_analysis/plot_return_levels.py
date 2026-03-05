import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SUMMARY_ROOT = Path("results/extreme_value_modelling")
RESULT_DIR = Path("results/extreme_value_analysis")


def load_summary(location):

    path = SUMMARY_ROOT / location / "summary_return_levels.csv"

    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    return pd.read_csv(path)


def dataset_tag(datasets):
    """
    Create short readable filename tag
    """

    tags = []

    for d in datasets:

        if d == "raw":
            tags.append("raw")

        elif d.startswith("local_"):
            tags.append(d.replace("local_", ""))

        elif d.startswith("pooled_"):
            tags.append(d.replace("pooled_", "") + "-pool")

        else:
            tags.append(d)

    return "-".join(tags)


def plot_model(df, location, model, datasets, ymin):

    df_model = df[df["model"] == model]

    if df_model.empty:
        print(f"No rows for model {model}")
        return

    plt.figure(figsize=(8,5))

    for dataset in datasets:

        subset = df_model[df_model["dataset"] == dataset]

        if subset.empty:
            print(f"WARNING: dataset '{dataset}' not found")
            continue

        subset = subset.sort_values("return_period")

        T = subset["return_period"].values
        rl = subset["return_level"].values
        ci_low = subset["ci_lower"].values
        ci_high = subset["ci_upper"].values

        line, = plt.plot(T, rl, label=dataset)

        color = line.get_color()

        # CI band
        plt.fill_between(
            T,
            ci_low,
            ci_high,
            alpha=0.1,
            color=color
        )

        # CI bounds
        plt.plot(T, ci_low, linestyle=":", color=color, linewidth=1)
        plt.plot(T, ci_high, linestyle=":", color=color, linewidth=1)

    plt.xlabel("Return period (years)")
    plt.ylabel("Return level Hs (m)")
    plt.title(f"{model} Return Levels (95% CI) — {location}")

    plt.grid(True)
    plt.legend()
    plt.xlim(1,50)

    if ymin is not None:
        plt.ylim(bottom=ymin)

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

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=None
    )

    args = parser.parse_args()

    df = load_summary(args.location)

    plot_model(df, args.location, "GEV", args.datasets, args.ymin)
    plot_model(df, args.location, "GPD", args.datasets, args.ymin)


if __name__ == "__main__":
    main()