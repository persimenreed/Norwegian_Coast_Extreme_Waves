import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/eval_metrics")
OUT = Path("results/extreme_value_analysis/eval_metrics")

RMSE_HEATMAP_METRICS = ["rmse", "rmse_q95", "rmse_q99"]
EXCEEDANCE_HEATMAP_METRICS = ["exceed_rate_bias_q95", "exceed_rate_bias_q99"]
RMSE_BAR_METRICS = ["rmse", "rmse_q95", "rmse_q99"]
EXCEEDANCE_BAR_METRICS = ["exceed_rate_bias_q95", "exceed_rate_bias_q99"]
RAW_LOCATION_RMSE_METRICS = [
    "rmse",
    "rmse_q25",
    "rmse_q50",
    "rmse_q75",
    "rmse_q95",
    "rmse_q99",
    "rmse_q995",
]
SUMMARY_LOCATIONS = ["fauskane", "fedjeosen", "vestfjorden"]


def load_metrics(location: str) -> pd.DataFrame:
    path = ROOT / location / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "method" not in df.columns:
        df = df.rename(columns={df.columns[0]: "method"})
    if "rmse_q5" in df.columns and "rmse_q50" not in df.columns:
        df = df.rename(columns={"rmse_q5": "rmse_q50"})
    return df


def _location_out_dir(location: str) -> Path:
    path = OUT / location
    path.mkdir(parents=True, exist_ok=True)
    return path


def _method_sort_key(name: str):

    if name == "raw":
        return (0, name)

    if name.startswith("localcv_"):
        return (1, name)

    if name.startswith("transfer_"):
        return (2, name)

    if name.startswith("ensemble_"):
        return (3, name)

    return (4, name)


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

    sort_cols = [c for c in ["rmse_q99", "rmse_q95", "rmse"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=True)

    return str(sub.iloc[0]["method"])


def _ensemble_method_for_group(df: pd.DataFrame, location: str, group: str):
    methods = set(df["method"].astype(str))

    if group == "localcv":
        preferred = f"ensemble_{location}"
        if preferred in methods:
            return preferred

    if group == "transfer":
        opposite = _opposite_location(location)
        preferred = f"ensemble_{opposite}" if opposite else None
        if preferred and preferred in methods:
            return preferred

    return _best_method_with_prefix(df, "ensemble_")


def _display_metric_name(metric: str) -> str:
    mapping = {
        "rmse": "RMSE",
        "rmse_q25": "RMSE q25",
        "rmse_q50": "RMSE q50",
        "rmse_q75": "RMSE q75",
        "rmse_q95": "RMSE q95",
        "rmse_q99": "RMSE q99",
        "rmse_q995": "RMSE q995",
        "exceed_rate_bias_q95": "Exceedance q95",
        "exceed_rate_bias_q99": "Exceedance q99",
    }
    return mapping.get(metric, metric)


def _display_method_name(name: str) -> str:
    name = str(name)
    if name == "ensemble":
        return "MoE"
    if name in {"ensemble_local", "ensemble_transfer"}:
        return name.replace("ensemble", "MoE")
    if name.startswith("MoE_"):
        return name
    if name.startswith("vestfjorden_transfer_"):
        rest = name.replace("vestfjorden_transfer_", "", 1)
        parts = rest.split("_", 1)
        if len(parts) == 2:
            return f"{parts[1]}_{parts[0]}"
        return rest
    if name.startswith("ensemble_"):
        return "MoE"
    if name.startswith("localcv_"):
        return name[len("localcv_"):]
    if name.startswith("transfer_"):
        rest = name[len("transfer_"):]
        parts = rest.split("_", 1)
        return parts[1] if len(parts) == 2 else rest
    return name


def _ensemble_local_transfer_methods(df: pd.DataFrame, location: str):
    methods = set(df["method"].astype(str))
    local = f"ensemble_{location}"
    opposite = _opposite_location(location)
    transfer = f"ensemble_{opposite}" if opposite else None
    return (
        local if local in methods else None,
        transfer if transfer in methods else None,
    )


def _moe_name_for_ensemble(name: str) -> str:
    return str(name).replace("ensemble_", "MoE_", 1)


def _vestfjorden_display_method(name: str) -> str:
    name = str(name)
    if name == "ensemble_combined":
        return "MoE_combined"
    if name.startswith("ensemble_"):
        return _moe_name_for_ensemble(name)
    if name.startswith("transfer_"):
        return f"vestfjorden_{name}"
    return name


def _wrap_label(label: str, width: int = 13) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) or str(label)


def _display_method_labels(methods) -> list[str]:
    return [_display_method_name(method) for method in methods]


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_raw_rmse_location_summary():
    rows = []

    for location in SUMMARY_LOCATIONS:
        path = ROOT / location / "metrics.csv"
        if not path.exists():
            continue

        df = load_metrics(location)
        raw = df[df["method"] == "raw"]
        if raw.empty:
            continue

        row = {"location": location}
        raw_row = raw.iloc[0]
        for metric in RAW_LOCATION_RMSE_METRICS:
            row[metric] = float(raw_row[metric]) if metric in raw_row else np.nan
        rows.append(row)

    if not rows:
        return

    plot_df = pd.DataFrame(rows).set_index("location")
    plot_df = plot_df.reindex(index=SUMMARY_LOCATIONS, columns=RAW_LOCATION_RMSE_METRICS)
    vals = plot_df.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(1.2 * len(plot_df.columns) + 2, 0.55 * len(plot_df.index) + 2.2)
    )
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_xticklabels([_display_metric_name(c) for c in plot_df.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    norm = im.norm
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "raw_rmse_by_location_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
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
    ensembles = [m for m in methods if m.startswith("ensemble_")]

    if localcv:
        groups["localcv"] = ["raw"] + localcv

    if transfer:
        groups["transfer"] = ["raw"] + transfer

    if ensembles:
        groups["ensemble"] = ["raw"] + ensembles

    return groups


def _sort_group(sub: pd.DataFrame, preferred_metric: str) -> pd.DataFrame:
    if preferred_metric in sub.columns:
        return sub.sort_values(preferred_metric, ascending=True, na_position="last")
    return sub.sort_values("method", key=lambda s: s.map(_method_sort_key))


def _plot_group_heatmaps(location: str, suffix: str, sub: pd.DataFrame):
    display_sub = sub.copy()
    if "_display_method" not in display_sub.columns:
        display_sub["_display_method"] = display_sub["method"].astype(str)

    rmse_metrics = [m for m in RMSE_HEATMAP_METRICS if m in sub.columns]
    exceedance_metrics = [m for m in EXCEEDANCE_HEATMAP_METRICS if m in sub.columns]
    if not rmse_metrics and not exceedance_metrics:
        return

    n_rows = len(display_sub)
    fig_width = 2.0 * len(rmse_metrics) + 1.8 * len(exceedance_metrics) + 4.5
    fig_height = 0.48 * n_rows + 2.8
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(8.5, fig_width), max(4.5, fig_height)),
        gridspec_kw={"width_ratios": [max(1, len(rmse_metrics)), max(1, len(exceedance_metrics))]},
    )

    def draw_heatmap(ax, plot_df, value_fmt, value_suffix="", show_methods=True):
        vals = plot_df.values.astype(float)
        im = ax.imshow(vals, aspect="auto")

        ax.set_xticks(np.arange(len(plot_df.columns)))
        ax.set_xticklabels([_display_metric_name(c) for c in plot_df.columns], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(plot_df.index)))
        if show_methods:
            ax.set_yticklabels([_display_method_name(idx) for idx in plot_df.index])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        norm = im.norm
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                txt = "nan" if not np.isfinite(v) else f"{format(v, value_fmt)}{value_suffix}"
                color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

        return im

    images = []
    if rmse_metrics:
        rmse_plot_df = display_sub.set_index("_display_method")[rmse_metrics].copy()
        images.append((axes[0], draw_heatmap(axes[0], rmse_plot_df, ".3f", show_methods=True), "RMSE (m)"))
    else:
        axes[0].axis("off")

    if exceedance_metrics:
        exceedance_plot_df = 100.0 * display_sub.set_index("_display_method")[exceedance_metrics].copy()
        images.append(
            (
                axes[1],
                draw_heatmap(axes[1], exceedance_plot_df, ".3f", value_suffix="%", show_methods=False),
                "Exceedance bias (%)",
            )
        )
    else:
        axes[1].axis("off")

    for ax, im, label in images:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)

    path = _location_out_dir(location) / f"{location}_metrics_heatmap_{suffix}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def plot_heatmap(location: str):
    df = load_metrics(location).copy()

    groups = _build_groups(df, location)

    for suffix, keep in groups.items():
        if suffix == "ensemble":
            if location == "vestfjorden":
                ensemble_methods = [m for m in keep if str(m).startswith("ensemble_")]
                ensemble_keep = sorted(set(["raw"] + ensemble_methods), key=_method_sort_key)
                sub = df[df["method"].isin(ensemble_keep)].copy()
                if sub.empty:
                    continue

                sub = _sort_group(sub, "rmse_q99")
                sub["_display_method"] = sub["method"].map(_vestfjorden_display_method)
                _plot_group_heatmaps(location, "ensemble", sub)
                continue

            ensemble_local, ensemble_transfer = _ensemble_local_transfer_methods(df, location)

            if ensemble_local and ensemble_transfer:
                ensemble_plots = [
                    ("ensemble_local", [ensemble_local], {ensemble_local: "ensemble"}),
                    ("ensemble_transfer", [ensemble_transfer], {ensemble_transfer: "ensemble"}),
                    (
                        "ensemble",
                        [ensemble_local, ensemble_transfer],
                        {
                            ensemble_local: "ensemble_local",
                            ensemble_transfer: "ensemble_transfer",
                        },
                    ),
                ]
            else:
                ensemble_methods = [m for m in keep if str(m).startswith("ensemble_")]
                ensemble_plots = [("ensemble", ensemble_methods, {})]

            for ensemble_suffix, ensemble_methods, display_names in ensemble_plots:
                ensemble_keep = sorted(set(["raw"] + ensemble_methods), key=_method_sort_key)
                sub = df[df["method"].isin(ensemble_keep)].copy()
                if sub.empty:
                    continue

                sub = _sort_group(sub, "rmse_q99")
                sub["_display_method"] = sub["method"].astype(str)
                for method, display_name in display_names.items():
                    sub.loc[sub["method"] == method, "_display_method"] = display_name
                _plot_group_heatmaps(location, ensemble_suffix, sub)

            continue

        if location == "vestfjorden" and suffix == "transfer":
            keep.extend([m for m in df["method"].astype(str) if m.startswith("ensemble_")])

        elif suffix in {"localcv", "transfer"}:
            ensemble_method = _ensemble_method_for_group(df, location, suffix)
            if ensemble_method:
                keep.append(ensemble_method)

        keep = sorted(set(keep), key=_method_sort_key)

        sub = df[df["method"].isin(keep)].copy()

        if sub.empty:
            continue

        sub = _sort_group(sub, "rmse_q99")
        sub["_display_method"] = sub["method"].astype(str)
        if location == "vestfjorden" and suffix == "transfer":
            sub["_display_method"] = sub["method"].map(_vestfjorden_display_method)
        if suffix in {"localcv", "transfer"}:
            ensemble_method = _ensemble_method_for_group(df, location, suffix)
            if ensemble_method and location != "vestfjorden":
                sub.loc[sub["method"] == ensemble_method, "_display_method"] = "ensemble"
        _plot_group_heatmaps(location, suffix, sub)


def _plot_rmse_vs_obs(imp: pd.DataFrame, location: str, suffix: str, title_prefix: str):
    if imp.empty:
        return

    imp = imp.sort_values("rmse_q99", ascending=True, na_position="last")

    metrics = RMSE_BAR_METRICS
    methods = imp.index.astype(str).tolist()
    labels = [_wrap_label(label) for label in _display_method_labels(methods)]
    x = np.arange(len(imp), dtype=float)
    width = 0.22
    offsets = (np.arange(len(metrics), dtype=float) - (len(metrics) - 1) / 2) * width
    metric_colors = {
        "rmse": "#4c78a8",
        "rmse_q95": "#f58518",
        "rmse_q99": "#54a24b",
    }

    fig, ax = plt.subplots(figsize=(max(8.5, 0.75 * len(imp) + 4.0), 5.8))

    max_y = 0.0
    for offset, metric in zip(offsets, metrics):
        values = imp[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=metric_colors.get(metric, "#4c78a8"),
            label=_display_metric_name(metric),
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            max_y = max(max_y, float(np.nanmax(finite)))
        ax.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "" for v in values], padding=2, fontsize=8)

    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(x)
    rotation = 35 if location == "vestfjorden" and suffix == "transfer" else 0
    ha = "right" if rotation else "center"
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.set_ylim(0, max_y * 1.20 if max_y > 0 else 1.0)
    _style_axes(ax)

    for tick, method in zip(ax.get_xticklabels(), methods):
        if method == "raw":
            tick.set_fontweight("bold")
            tick.set_color("#4d4d4d")

    path = _location_out_dir(location) / f"{location}_rmse_vs_obs_{suffix}.png"

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def _build_rmse_table(df: pd.DataFrame, allowed):
    metrics = RMSE_BAR_METRICS
    rows = []

    for _, row in df.iterrows():
        method = row["method"]

        if allowed is not None and method not in allowed:
            continue

        out = {"method": method}

        for m in metrics:
            out[m] = float(row[m]) if m in row and np.isfinite(float(row[m])) else np.nan

        rows.append(out)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("method")


def _plot_exceedance_bias(location: str, suffix: str, sub: pd.DataFrame, title_prefix: str):
    metrics = [m for m in EXCEEDANCE_BAR_METRICS if m in sub.columns]
    if len(metrics) < 2:
        return

    sub = sub.copy()
    if "exceed_rate_bias_q95" in sub.columns:
        sub["_sort_abs_q95"] = sub["exceed_rate_bias_q95"].abs()
        sub = sub.sort_values("_sort_abs_q95", ascending=True, na_position="last")
    else:
        sub["_sort_abs_exceedance"] = sub[metrics].abs().max(axis=1)
        sub = sub.sort_values("_sort_abs_exceedance", ascending=True, na_position="last")

    methods = sub["method"].astype(str).tolist()
    labels = [_wrap_label(label) for label in _display_method_labels(methods)]
    x = np.arange(len(sub), dtype=float)
    width = 0.32
    offsets = (np.arange(len(metrics), dtype=float) - (len(metrics) - 1) / 2) * width
    metric_colors = {
        "exceed_rate_bias_q95": "#4c78a8",
        "exceed_rate_bias_q99": "#e45756",
    }

    fig, ax = plt.subplots(figsize=(max(8.5, 0.75 * len(sub) + 4.0), 5.8))

    finite_values = []
    for offset, metric in zip(offsets, metrics):
        values = 100.0 * sub[metric].to_numpy(dtype=float)
        finite_values.extend(values[np.isfinite(values)].tolist())
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=metric_colors.get(metric, "#4c78a8"),
            label=_display_metric_name(metric),
        )
        ax.bar_label(
            bars,
            labels=[f"{v:+.2f}" if np.isfinite(v) else "" for v in values],
            padding=2,
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Bias vs observed exceedance (percentage points)")
    ax.set_xticks(x)
    rotation = 35 if location == "vestfjorden" and suffix == "transfer" else 0
    ha = "right" if rotation else "center"
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    if finite_values:
        y_min = min(min(finite_values), 0.0)
        y_max = max(max(finite_values), 0.0)
        span = max(y_max - y_min, 0.2)
        pad = max(0.08, 0.12 * span)
        ax.set_ylim(y_min - pad, y_max + pad)
    _style_axes(ax)

    for tick, method in zip(ax.get_xticklabels(), methods):
        if method == "raw":
            tick.set_fontweight("bold")
            tick.set_color("#4d4d4d")

    path = _location_out_dir(location) / f"{location}_exceedance_vs_obs_{suffix}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def plot_vs_obs(location: str):
    df = load_metrics(location).copy()

    if "raw" not in df["method"].values:
        raise ValueError(f"No raw row found for {location}")

    if location != "vestfjorden":
        local_methods = {m for m in df["method"].tolist() if m.startswith("localcv_")}
        ensemble_local = f"ensemble_{location}"
        if ensemble_local in df["method"].values:
            local_methods.add(ensemble_local)
        imp_local = _build_rmse_table(df, {"raw"} | local_methods)
        _plot_rmse_vs_obs(imp_local, location, "local", "Local RMSE")
        local_plot_methods = {"raw"} | local_methods
        local_exceedance_cols = ["method"] + [m for m in EXCEEDANCE_BAR_METRICS if m in df.columns]
        local_sub = df[df["method"].isin(sorted(local_plot_methods, key=_method_sort_key))][local_exceedance_cols].copy()
        if not local_sub.empty:
            _plot_exceedance_bias(location, "local", local_sub, "Local")

    transfer_methods = {m for m in df["method"].tolist() if m.startswith("transfer_")}

    opposite = _opposite_location(location)
    ensemble_opposite = f"ensemble_{opposite}" if opposite else None
    if ensemble_opposite and ensemble_opposite in df["method"].values:
        transfer_methods.add(ensemble_opposite)
    if location == "vestfjorden":
        transfer_methods |= {m for m in df["method"].tolist() if m.startswith("ensemble_")}

    imp_transfer = _build_rmse_table(df, {"raw"} | transfer_methods)
    if location == "vestfjorden" and not imp_transfer.empty:
        imp_transfer = imp_transfer.rename(index=_vestfjorden_display_method)
    _plot_rmse_vs_obs(imp_transfer, location, "transfer", "Transfer RMSE")
    transfer_plot_methods = {"raw"} | transfer_methods
    transfer_exceedance_cols = ["method"] + [m for m in EXCEEDANCE_BAR_METRICS if m in df.columns]
    transfer_sub = df[df["method"].isin(sorted(transfer_plot_methods, key=_method_sort_key))][transfer_exceedance_cols].copy()
    if location == "vestfjorden" and not transfer_sub.empty:
        transfer_sub["method"] = transfer_sub["method"].map(_vestfjorden_display_method)
    if not transfer_sub.empty:
        _plot_exceedance_bias(location, "transfer", transfer_sub, "Transfer")


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
        plot_vs_obs(loc)

    _plot_raw_rmse_location_summary()


if __name__ == "__main__":
    main()
