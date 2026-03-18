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

VESTFJORDEN_EXTENDED_METRICS = [
    "mae",
    "rmse",
    "corr",
    "scatter_index",
    "tail_rmse_95",
    "tail_rmse_99",
    "twrmse",
    "q95_bias",
    "q99_bias",
    "exceed_rate_bias_q95",
    "exceed_rate_bias_q99",
    "quantile_score_95",
    "quantile_score_99",
]

RAW_BIAS_SUMMARY_LOCATIONS = ["fedjeosen", "fauskane", "vestfjorden"]
RAW_BIAS_SUMMARY_METRICS = [
    "bias",
    "tail_bias_q50",
    "tail_bias_q75",
    "tail_bias_q95",
    "tail_bias_q995",
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

    if name.startswith("ensemble_"):
        return (3, name)

    if name.startswith("pooled_"):
        return (4, name)

    if name == "ensemble_transfer":
        return (5, name)

    if name == "ensemble_pooling":
        return (6, name)

    return (7, name)


def _opposite_location(location: str):
    mapping = {
        "fauskane": "fedjeosen",
        "fedjeosen": "fauskane",
    }
    return mapping.get(location)


def _best_method_with_prefix(df: pd.DataFrame, prefix: str):
    sub = df[df["method"].astype(str).str.startswith(prefix)].copy()
    if sub.empty:
        return None

    sort_cols = [c for c in ["tail_rmse_99", "rmse", "mae"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=True)

    return str(sub.iloc[0]["method"])


def _plot_single_heatmap(location: str, plot_df: pd.DataFrame, suffix: str):
    vals = plot_df.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(1.4 * len(plot_df.columns) + 2, 0.45 * len(plot_df) + 2.5)
    )
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(plot_df.columns)))
    display_cols = [c.replace("tail_bias_", "bias_") for c in plot_df.columns]
    ax.set_xticklabels(display_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    norm = im.norm

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Evaluation metrics - {location}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{location}_metrics_heatmap_{suffix}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def _plot_vestfjorden_raw_vs_ensemble_extensive(df: pd.DataFrame):
    methods = {"raw", "ensemble_pooling", "ensemble_transfer"}
    sub = df[df["method"].isin(methods)].copy()

    if sub["method"].nunique() < 2:
        return

    ordered = [
        "raw",
        "ensemble_transfer",
        "ensemble_pooling",
    ]
    ordered = [name for name in ordered if name in sub["method"].values]
    sub["method"] = pd.Categorical(sub["method"], categories=ordered, ordered=True)
    sub = sub.sort_values("method")

    metrics = [c for c in VESTFJORDEN_EXTENDED_METRICS if c in sub.columns]
    if not metrics:
        return

    plot_df = sub.set_index("method")[metrics].copy()

    # Absolute bias-like terms so color intensity reflects error magnitude.
    abs_cols = [
        "q95_bias",
        "q99_bias",
        "exceed_rate_bias_q95",
        "exceed_rate_bias_q99",
    ]
    for col in abs_cols:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].abs()

    _plot_single_heatmap("vestfjorden", plot_df, "raw_vs_ensembles_extended")


def _plot_vestfjorden_raw_vs_named_ensembles(df: pd.DataFrame, metrics):
    methods = ["raw", "ensemble_fauskane", "ensemble_fedjeosen", "ensemble_combined"]
    sub = df[df["method"].isin(methods)].copy()
    if sub.empty or sub["method"].nunique() < 2:
        return

    ordered = [m for m in methods if m in sub["method"].values]
    sub["method"] = pd.Categorical(sub["method"], categories=ordered, ordered=True)
    sub = sub.sort_values("method")

    keep_metrics = [c for c in metrics if c in sub.columns]
    if not keep_metrics:
        return

    plot_df = sub.set_index("method")[keep_metrics].copy()

    for col in ["q95_bias", "q99_bias"]:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].abs()

    _plot_single_heatmap("vestfjorden", plot_df, "raw_vs_ensemble_locations")


def _plot_raw_bias_summary_heatmap():
    rows = []

    for location in RAW_BIAS_SUMMARY_LOCATIONS:
        try:
            df = load_metrics(location)
        except FileNotFoundError:
            row = {"location": location}
            for metric in RAW_BIAS_SUMMARY_METRICS:
                row[metric] = np.nan
            rows.append(row)
            continue

        raw = df[df["method"] == "raw"]
        row = {"location": location}

        if raw.empty:
            for metric in RAW_BIAS_SUMMARY_METRICS:
                row[metric] = np.nan
        else:
            raw_row = raw.iloc[0]
            for metric in RAW_BIAS_SUMMARY_METRICS:
                row[metric] = float(raw_row[metric]) if metric in raw_row else np.nan

        rows.append(row)

    plot_df = pd.DataFrame(rows).set_index("location")
    plot_df = plot_df.reindex(index=RAW_BIAS_SUMMARY_LOCATIONS, columns=RAW_BIAS_SUMMARY_METRICS)

    vals = plot_df.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(1.4 * len(plot_df.columns) + 2, 0.45 * len(plot_df.index) + 2.5)
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

    ax.set_title("Raw bias summary")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "raw_bias_summary_heatmap.png"
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
    localcv = [m for m in methods if m.startswith("localcv_")]

    # For transfer heatmaps, include opposite-location ensemble if available.
    opposite = _opposite_location(location)
    ensemble_opposite = f"ensemble_{opposite}" if opposite else None

    if localcv:
        groups["localcv"] = ["raw"] + localcv
        ensemble_local = f"ensemble_{location}"
        if ensemble_local in methods:
            groups["localcv"].append(ensemble_local)

    if transfer:
        groups["transfer"] = ["raw"] + transfer
        if ensemble_opposite and ensemble_opposite in methods:
            groups["transfer"].append(ensemble_opposite)
        elif "ensemble_transfer" in methods:
            groups["transfer"].append("ensemble_transfer")

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

    # Additional ensemble-only comparison for buoy locations:
    # raw vs ensemble_fauskane vs ensemble_fedjeosen.
    if location in {"fauskane", "fedjeosen"}:
        keep = ["raw", "ensemble_fauskane", "ensemble_fedjeosen"]
        keep = sorted(set(keep), key=_method_sort_key)
        sub = df[df["method"].isin(keep)].copy()
        if not sub.empty and sub["method"].nunique() >= 2:
            sub = sub.sort_values("tail_rmse_99", ascending=True)
            plot_df = sub.set_index("method")[metrics].copy()

            for col in ["q95_bias", "q99_bias"]:
                if col in plot_df.columns:
                    plot_df[col] = plot_df[col].abs()

            _plot_single_heatmap(location, plot_df, "ensemble")

    if location == "vestfjorden":
        _plot_vestfjorden_raw_vs_named_ensembles(df, metrics)
        _plot_vestfjorden_raw_vs_ensemble_extensive(df)


def _plot_improvement(imp: pd.DataFrame, location: str, suffix: str, title_prefix: str):
    if imp.empty:
        return

    imp = imp.sort_values("rmse", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    metrics = ["rmse", "tail_rmse_95", "tail_rmse_99"]

    for ax, metric in zip(axes, metrics):

        x = imp[metric].values
        y = np.arange(len(imp))

        ax.barh(y, x)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)

        ax.set_title(f"{metric} improvement")
        ax.set_xlabel("% vs raw")
        ax.set_yticks(y)
        ax.set_yticklabels(imp.index)
        ax.invert_yaxis()

        ax.grid(alpha=0.3)

    fig.suptitle(f"{title_prefix} relative to raw {location}")

    OUT.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{location}_improvement_vs_raw.png"
        if not suffix
        else f"{location}_improvement_vs_raw_{suffix}.png"
    )
    path = OUT / filename

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved {path}")


def _build_improvement_table(df: pd.DataFrame, allowed):
    if "raw" not in df["method"].values:
        return pd.DataFrame()

    raw = df[df["method"] == "raw"].iloc[0]

    metrics = ["rmse", "tail_rmse_95", "tail_rmse_99"]
    rows = []

    for _, row in df.iterrows():
        method = row["method"]

        if method == "raw":
            continue

        if allowed is not None and method not in allowed:
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

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("method")


def plot_improvement_vs_raw(location: str):
    df = load_metrics(location).copy()

    if "raw" not in df["method"].values:
        raise ValueError(f"No raw row found for {location}")

    # Existing full comparison plot.
    all_methods = set(df["method"].tolist()) - {"raw"}
    imp_all = _build_improvement_table(df, all_methods)
    _plot_improvement(imp_all, location, "", "Improvement")

    # Local-only improvement.
    local_methods = {m for m in df["method"].tolist() if m.startswith("localcv_")}
    ensemble_local = f"ensemble_{location}"
    if ensemble_local in df["method"].values:
        local_methods.add(ensemble_local)
    imp_local = _build_improvement_table(df, local_methods)
    _plot_improvement(imp_local, location, "local", "Local improvement")

    # Transfer-only improvement.
    transfer_methods = {m for m in df["method"].tolist() if m.startswith("transfer_")}

    # For buoy locations, include opposite-location ensemble in transfer comparison.
    opposite = _opposite_location(location)
    ensemble_opposite = f"ensemble_{opposite}" if opposite else None
    if ensemble_opposite and ensemble_opposite in df["method"].values:
        transfer_methods.add(ensemble_opposite)

    # Keep legacy aggregate transfer ensemble when present.
    if "ensemble_transfer" in df["method"].values:
        transfer_methods.add("ensemble_transfer")

    imp_transfer = _build_improvement_table(df, transfer_methods)
    _plot_improvement(imp_transfer, location, "transfer", "Transfer improvement")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--locations",
        nargs="+",
        default=["fedjeosen", "fauskane", "vestfjorden"],
    )

    args = parser.parse_args()

    for loc in args.locations:
        metrics_path = ROOT / loc / "metrics.csv"
        if not metrics_path.exists():
            print(f"Skipping {loc}: {metrics_path} not found")
            continue

        plot_heatmap(loc)
        plot_improvement_vs_raw(loc)

    _plot_raw_bias_summary_heatmap()


if __name__ == "__main__":
    main()
