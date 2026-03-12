import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/eval_metrics")
OUT = Path("results/extreme_value_analysis/eval_metrics")

DEFAULT_METRICS = [
    "rmse",
    "mae",
    "tail_rmse_95",
    "tail_rmse_99",
    "q95_bias",
    "q99_bias",
]


def load_metrics(location: str) -> pd.DataFrame:
    path = ROOT / location / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "method" not in df.columns:
        df = df.rename(columns={df.columns[0]: "method"})
    return df


def _method_sort_key(name: str):

    if name == "raw":
        return (0, name)

    if name.startswith("localcv_"):
        return (1, name)

    if name.startswith("transfer_"):
        return (2, name)

    if name.startswith("pooled_"):
        return (3, name)

    if name == "ensemble":
        return (4, name)

    return (5, name)


def _plot_single_heatmap(location: str, plot_df: pd.DataFrame, suffix: str):
    vals = plot_df.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(1.4 * len(plot_df.columns) + 2, 0.45 * len(plot_df) + 2.5)
    )
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    norm = im.norm

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Evaluation metrics ({suffix}) — {location}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{location}_metrics_heatmap_{suffix}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def _build_groups(df, location):
    """
    Build plotting groups depending on which methods exist.
    Special handling for Vestfjorden to avoid massive plots.
    """

    methods = df["method"].tolist()

    groups = {}

    transfer = [m for m in methods if m.startswith("transfer_")]
    pooled = [m for m in methods if m.startswith("pooled_")]
    localcv = [m for m in methods if m.startswith("localcv_")]

    ensemble = "ensemble" if "ensemble" in methods else None
    ensemble_xgb = "ensemble_xgboost" if "ensemble_xgboost" in methods else None

    # --------------------------------------------------
    # Vestfjorden special case
    # --------------------------------------------------

    if location == "vestfjorden":

        if pooled:
            groups["pooled"] = ["raw"] + pooled

            if ensemble:
                groups["pooled"].append("ensemble")

            if ensemble_xgb:
                groups["pooled"].append("ensemble_xgboost")

        return groups

    # --------------------------------------------------
    # Other locations
    # --------------------------------------------------

    if localcv:
        groups["localcv"] = ["raw"] + localcv

    if transfer:
        groups["transfer"] = ["raw"] + transfer

    if pooled:
        groups["pooled"] = ["raw"] + pooled

    if transfer and pooled:
        groups["combined"] = ["raw"] + transfer + pooled

    if transfer and localcv:
        groups["combined"] = ["raw"] + localcv + transfer

    return groups


def plot_heatmap(location: str, metrics=None):
    metrics = metrics or DEFAULT_METRICS

    df = load_metrics(location).copy()

    groups = _build_groups(df, location)

    for suffix, keep in groups.items():

        keep = sorted(set(keep), key=_method_sort_key)

        sub = df[df["method"].isin(keep)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("tail_rmse_99", ascending=True)

        plot_df = sub.set_index("method")[metrics].copy()

        # use absolute bias for heatmap
        for col in ["q95_bias", "q99_bias"]:
            if col in plot_df.columns:
                plot_df[col] = plot_df[col].abs()

        _plot_single_heatmap(location, plot_df, suffix)


def plot_improvement_vs_raw(location: str):
    df = load_metrics(location).copy()

    if "raw" not in df["method"].values:
        raise ValueError(f"No raw row found for {location}")

    raw = df[df["method"] == "raw"].iloc[0]

    metrics = ["rmse", "tail_rmse_95", "tail_rmse_99"]

    rows = []

    for _, row in df.iterrows():
        method = row["method"]

        if method == "raw":
            continue

        out = {"method": method}

        for m in metrics:
            raw_val = float(raw[m])
            val = float(row[m])

            if np.isfinite(raw_val) and raw_val != 0 and np.isfinite(val):
                out[m] = 100.0 * (raw_val - val) / raw_val
            else:
                out[m] = np.nan

        rows.append(out)

    imp = pd.DataFrame(rows).set_index("method")

    imp = imp.sort_values("rmse", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, metric in zip(axes, metrics):

        x = imp[metric].values
        y = np.arange(len(imp))

        ax.barh(y, x)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)

        ax.set_title(f"{metric} improvement")
        ax.set_xlabel("% vs raw")
        ax.set_yticks(y)
        ax.set_yticklabels(imp.index)

        ax.grid(alpha=0.3)

    fig.suptitle(f"Improvement relative to raw — {location}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{location}_improvement_vs_raw.png"

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--locations",
        nargs="+",
        default=["fedjeosen", "fauskane", "vestfjorden"],
    )

    args = parser.parse_args()

    for loc in args.locations:
        plot_heatmap(loc)
        plot_improvement_vs_raw(loc)


if __name__ == "__main__":
    main()