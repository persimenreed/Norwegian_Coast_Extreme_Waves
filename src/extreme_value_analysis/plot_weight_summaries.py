import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_SUMMARY_DIR = Path("results/ensemble/weight_summaries")
DEFAULT_OUT_DIR = Path("results/ensemble")
DEFAULT_APPLICATIONS = [
    "local_fauskane",
    "local_fedjeosen",
    "transfer_fauskane",
    "transfer_fedjeosen",
]

METHOD_ORDER = ["linear", "pqm", "dagqm", "gpr", "xgboost", "transformer"]
METHOD_LABELS = {
    "linear": "Linear",
    "pqm": "PQM",
    "dagqm": "DAGQM",
    "gpr": "GPR",
    "xgboost": "XGBoost",
    "transformer": "Transformer",
}
METHOD_COLORS = {
    "linear": "#8c564b",
    "pqm": "#e377c2",
    "dagqm": "#d62728",
    "gpr": "#7f7f7f",
    "xgboost": "#6b6ecf",
    "transformer": "#9467bd",
}
LOCATION_LABELS = {
    "fauskane": "Fauskane",
    "fedjeosen": "Fedjeosen",
}
LOCATION_HATCHES = {
    "fauskane": "",
    "fedjeosen": "///",
}
MODE_LABELS = {
    "local": "Local MoE",
    "transfer": "Transfer MoE",
}
MODE_HATCHES = {
    "local": "",
    "transfer": "///",
}
Y_MIN = 0.10
Y_MAX = 0.21
GROUPINGS = {
    "season": {
        "filename": "season",
        "bins": ["DJF", "MAM", "JJA", "SON"],
        "xlabel": "Season",
    },
    "wind_sector": {
        "filename": "wind_direction",
        "bins": ["North", "East", "South", "West"],
        "xlabel": "Wind sector",
    },
    "hs_percentile": {
        "filename": "hs_percentile",
        "bins": ["0-50", "50-75", "75-95", "95-100"],
        "xlabel": "Hs percentile bin",
    },
}


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)


def _load_summary(summary_dir: Path, application: str) -> pd.DataFrame:
    path = summary_dir / f"{application}_weight_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _application_location(application: str) -> str:
    for prefix in ("local_", "transfer_"):
        if application.startswith(prefix):
            return application.removeprefix(prefix)
    return application


def _application_out_dir(out_dir: Path, application: str) -> Path:
    return out_dir / _application_location(application)


def _pivot_group(df: pd.DataFrame, grouping: str, bins: List[str]) -> pd.DataFrame:
    sub = df[df["grouping"].astype(str) == grouping].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["expert"] = sub["expert"].astype(str)
    plot_df = sub.pivot_table(
        index="bin",
        columns="expert",
        values="mean_weight",
        aggfunc="mean",
        observed=True,
    )
    plot_df = plot_df.reindex(index=bins, columns=METHOD_ORDER)
    return plot_df


def _sample_counts(df: pd.DataFrame, grouping: str, bins: List[str]) -> Dict[str, int]:
    sub = df[df["grouping"].astype(str) == grouping].copy()
    if sub.empty:
        return {}
    counts = sub.drop_duplicates(["bin", "n"]).set_index("bin")["n"].to_dict()
    return {label: int(counts[label]) for label in bins if label in counts}


def _ordered_experts(plot_df: pd.DataFrame) -> List[str]:
    experts = [method for method in METHOD_ORDER if method in plot_df.columns]
    if plot_df.empty or not experts:
        return experts

    first_cluster = plot_df.iloc[0][experts]
    return (
        first_cluster.sort_values(ascending=True, na_position="last")
        .index.astype(str)
        .tolist()
    )


def _add_bar_labels(ax, bars, values, y_pad: float = 0.0015, fontsize: int = 7) -> None:
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_pad,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=fontsize,
        )


def plot_grouping(summary_dir: Path, out_dir: Path, application: str, grouping: str) -> Path:
    spec = GROUPINGS[grouping]
    df = _load_summary(summary_dir, application)
    plot_df = _pivot_group(df, grouping, spec["bins"])
    if plot_df.empty:
        raise ValueError(f"No rows for grouping '{grouping}' in {application}")

    counts = _sample_counts(df, grouping, spec["bins"])
    x = np.arange(len(plot_df.index), dtype=float) * 1.22
    experts = _ordered_experts(plot_df)
    width = min(0.18, 0.96 / max(len(experts), 1))
    offsets = (np.arange(len(experts), dtype=float) - (len(experts) - 1) / 2) * width * 1.08

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for offset, expert in zip(offsets, experts):
        values = plot_df[expert].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_LABELS.get(expert, expert.title()),
            color=METHOD_COLORS.get(expert, "#333333"),
            edgecolor="#222222",
            linewidth=0.55,
        )
        _add_bar_labels(ax, bars, values)

    labels = [
        f"{label}\nn={counts[label]:,}" if label in counts else str(label)
        for label in plot_df.index
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("Mean expert weight")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    _style_axes(ax)
    fig.tight_layout()

    app_out_dir = _application_out_dir(out_dir, application)
    app_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = app_out_dir / f"{application}_{spec['filename']}_expert_weights.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _overall_weights(df: pd.DataFrame) -> pd.Series:
    sub = df[df["grouping"].astype(str) == "season"].copy()
    if sub.empty:
        sub = df[df["grouping"].astype(str) == "wind_sector"].copy()
    if sub.empty:
        raise ValueError("No grouped rows available for overall weights")

    rows = []
    for expert, expert_df in sub.groupby("expert", sort=False):
        weights = pd.to_numeric(expert_df["n"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(expert_df["mean_weight"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
        if not np.any(mask):
            continue
        rows.append((expert, float(np.average(values[mask], weights=weights[mask]))))

    return pd.Series(dict(rows), dtype=float).reindex(METHOD_ORDER).dropna()


def plot_overall(summary_dir: Path, out_dir: Path, application: str) -> Path:
    df = _load_summary(summary_dir, application)
    weights = _overall_weights(df).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(weights), dtype=float) * 0.88
    colors = [METHOD_COLORS.get(expert, "#333333") for expert in weights.index]
    bars = ax.bar(
        x,
        weights.to_numpy(dtype=float),
        width=0.84,
        color=colors,
        edgecolor="#222222",
        linewidth=0.65,
    )
    _add_bar_labels(ax, bars, weights.to_numpy(dtype=float), fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(expert, str(expert).title()) for expert in weights.index],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean expert weight")
    ax.set_ylim(Y_MIN, Y_MAX)
    _style_axes(ax)
    fig.tight_layout()

    app_out_dir = _application_out_dir(out_dir, application)
    app_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = app_out_dir / f"{application}_overall_expert_weights.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overall_by_mode(summary_dir: Path, out_dir: Path, location: str) -> Path:
    modes = ["local", "transfer"]
    series_by_mode = {
        mode: _overall_weights(_load_summary(summary_dir, f"{mode}_{location}"))
        for mode in modes
    }
    overall_df = pd.DataFrame(series_by_mode).reindex(METHOD_ORDER).dropna(how="all")
    if overall_df.empty:
        raise ValueError(f"No overall weights available for {location}")

    order_by = "local" if "local" in overall_df.columns else overall_df.columns[0]
    order = overall_df[order_by].sort_values(ascending=True).index.tolist()
    overall_df = overall_df.reindex(order)

    x = np.arange(len(overall_df.index), dtype=float) * 0.92
    modes = [mode for mode in modes if mode in overall_df.columns]
    offset_step = 0.36 * 1.12
    width = 0.38
    offsets = (np.arange(len(modes), dtype=float) - (len(modes) - 1) / 2) * offset_step

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for offset, mode in zip(offsets, modes):
        values = overall_df[mode].to_numpy(dtype=float)
        colors = [METHOD_COLORS.get(expert, "#333333") for expert in overall_df.index]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=MODE_LABELS.get(mode, mode.title()),
            color=colors,
            edgecolor="#222222",
            linewidth=0.65,
            hatch=MODE_HATCHES.get(mode, ""),
        )
        _add_bar_labels(ax, bars, values, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(expert, str(expert).title()) for expert in overall_df.index],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean expert weight")
    ax.set_ylim(Y_MIN, Y_MAX)
    mode_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=MODE_HATCHES.get(mode, ""),
            label=MODE_LABELS.get(mode, mode.title()),
            linewidth=0.8,
        )
        for mode in modes
    ]
    ax.legend(
        handles=mode_handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    _style_axes(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{location}_overall_expert_weights_by_mode.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact MoE expert-weight summaries.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--applications", nargs="+", default=DEFAULT_APPLICATIONS)
    args = parser.parse_args()

    saved = []
    for location in ("fauskane", "fedjeosen"):
        saved.append(plot_overall_by_mode(args.summary_dir, args.out_dir, location))

    for application in args.applications:
        for grouping in GROUPINGS:
            saved.append(plot_grouping(args.summary_dir, args.out_dir, application, grouping))

    print("Saved plots:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
