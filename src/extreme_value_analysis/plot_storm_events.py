import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VALIDATION_DIR = Path("data/output")
DEFAULT_HINDCAST_DIR = Path("data/output")
DEFAULT_NORA3_DIR = Path("data/input/nora3_locations")
DEFAULT_METRICS_DIR = Path("results/eval_metrics")
DEFAULT_OUT_DIR = Path("results/MISC/storm")

EVENT_WINDOW_DAYS = 1.5
CLASSICAL_METHODS = {"dagqm", "linear", "pqm"}
ML_METHODS = {"gpr", "transformer", "xgboost"}
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
    "observed": "#000000",
    "nora3": "#ff7f0e",
    "moe_local": "#2ca02c",
    "moe_transfer": "#4c78a8",
    "moe_combined": "#bcbd22",
    "linear": "#8c564b",
    "pqm": "#e377c2",
    "dagqm": "#d62728",
    "gpr": "#7f7f7f",
    "transformer": "#9467bd",
    "xgboost": "#6b6ecf",
}


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)


def _apply_headroom(ax, values: Iterable[pd.Series], pad_fraction: float = 0.26) -> None:
    finite_values = []
    for series in values:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite_values.extend(vals[np.isfinite(vals)].tolist())

    if not finite_values:
        return

    y_min = min(0.0, min(finite_values))
    y_max = max(finite_values)
    span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min, y_max + pad_fraction * span)


def _save_event_plot(
    df: pd.DataFrame,
    peak_time: pd.Timestamp,
    out_path: Path,
    lines: List[Dict],
    legend_columns: int,
) -> Path:
    window = df[
        (df["time"] >= peak_time - pd.Timedelta(days=EVENT_WINDOW_DAYS))
        & (df["time"] <= peak_time + pd.Timedelta(days=EVENT_WINDOW_DAYS))
    ].copy()
    if len(window) < 6:
        raise ValueError(f"Too few rows in storm window for {out_path.name}")

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    plotted_values = []

    for line in lines:
        column = line["column"]
        if column not in window.columns:
            continue
        plotted_values.append(window[column])
        ax.plot(
            window["time"],
            window[column],
            color=line["color"],
            linewidth=line.get("linewidth", 1.8),
            alpha=line.get("alpha", 1.0),
            linestyle=line.get("linestyle", "-"),
            label=line["label"],
        )

    ax.axvline(peak_time, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Significant wave height Hs (m)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y %H:%M"))
    _apply_headroom(ax, plotted_values)
    ax.legend(frameon=False, loc="upper left", ncol=legend_columns)
    _style_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_validation_series(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"time", "Significant_Wave_Height_Hm0", "hs", "hs_corrected"}
    if not required.issubset(df.columns):
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    keep = ["time", "Significant_Wave_Height_Hm0", "hs", "hs_corrected"]
    df = df[keep].rename(columns={"hs_corrected": label})
    df = df.dropna(subset=["time", "Significant_Wave_Height_Hm0", "hs", label])
    return df.sort_values("time").reset_index(drop=True)


def _merge_validation_series(
    validation_dir: Path,
    location: str,
    series: List[Tuple[str, str]],
) -> Optional[pd.DataFrame]:
    merged = None

    for filename_stem, label in series:
        path = validation_dir / location / f"validation_{filename_stem}.csv"
        df = _load_validation_series(path, label)
        if df is None:
            return None

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["time", label]], on="time", how="inner")

    return merged


def _observed_peak_time(df: pd.DataFrame) -> pd.Timestamp:
    obs = pd.to_numeric(df["Significant_Wave_Height_Hm0"], errors="coerce")
    return df.loc[obs.idxmax(), "time"]


def _plot_overlapping_event(
    validation_dir: Path,
    out_dir: Path,
    location: str,
    series: List[Tuple[str, str]],
    extra_lines: Optional[List[Dict]] = None,
    filename_suffix: str = "overlapping_event",
) -> Optional[Path]:
    df = _merge_validation_series(validation_dir, location, series)
    if df is None or df.empty:
        return None

    peak_time = _observed_peak_time(df)
    lines = [
        {
            "column": "Significant_Wave_Height_Hm0",
            "label": "Observed",
            "color": METHOD_COLORS["observed"],
            "linewidth": 2.1,
        },
        {
            "column": "hs",
            "label": "NORA3",
            "color": METHOD_COLORS["nora3"],
            "linewidth": 1.7,
        },
    ]
    if extra_lines:
        lines.extend(extra_lines)

    legend_columns = min(4, max(2, len(lines)))
    out_path = out_dir / f"{location}_{filename_suffix}.png"
    return _save_event_plot(df, peak_time, out_path, lines, legend_columns)


def _moe_lines(kind: str) -> List[Dict]:
    if kind == "combined":
        return [
            {
                "column": "MoE Combined",
                "label": "MoE Combined",
                "color": METHOD_COLORS["moe_combined"],
                "linewidth": 2.0,
            }
        ]
    return [
        {
            "column": "MoE Local",
            "label": "MoE Local",
            "color": METHOD_COLORS["moe_local"],
            "linewidth": 2.0,
        },
        {
            "column": "MoE Transfer",
            "label": "MoE Transfer",
            "color": METHOD_COLORS["moe_transfer"],
            "linewidth": 2.0,
        },
    ]


def _single_moe_line(column: str, color_key: str) -> Dict:
    return {
        "column": column,
        "label": "MoE",
        "color": METHOD_COLORS[color_key],
        "linewidth": 2.0,
    }


def plot_main_overlapping_events(validation_dir: Path, out_dir: Path) -> List[Path]:
    specs = [
        (
            "vestfjorden",
            [("ensemble_combined", "MoE Combined")],
            _moe_lines("combined"),
        ),
        (
            "fedjeosen",
            [("ensemble_fedjeosen", "MoE Local"), ("ensemble_fauskane", "MoE Transfer")],
            _moe_lines("local_transfer"),
        ),
        (
            "fauskane",
            [("ensemble_fauskane", "MoE Local"), ("ensemble_fedjeosen", "MoE Transfer")],
            _moe_lines("local_transfer"),
        ),
    ]

    saved = []
    for location, series, lines in specs:
        out = _plot_overlapping_event(
            validation_dir,
            out_dir,
            location,
            series,
            extra_lines=lines,
            filename_suffix="biggest_overlapping_event",
        )
        if out is not None:
            saved.append(out)
    return saved


def plot_method_overlapping_events(validation_dir: Path, metrics_dir: Path, out_dir: Path) -> List[Path]:
    saved = []

    for location, local_ensemble, transfer_ensemble, transfer_source in [
        ("fauskane", "ensemble_fauskane", "ensemble_fedjeosen", "fedjeosen"),
        ("fedjeosen", "ensemble_fedjeosen", "ensemble_fauskane", "fauskane"),
    ]:
        local_series = [(local_ensemble, "MoE")]
        local_lines = [
            {
                "column": "MoE",
                "label": "MoE",
                "color": METHOD_COLORS["moe_local"],
                "linewidth": 2.0,
            }
        ]
        for method in _best_methods_for_group(
            metrics_dir,
            location,
            "localcv_",
        ):
            label = METHOD_LABELS[method]
            local_series.append((f"localcv_{method}", label))
            local_lines.append(
                {
                    "column": label,
                    "label": label,
                    "color": METHOD_COLORS[method],
                    "linewidth": 1.45,
                    "alpha": 0.82,
                }
            )
        out = _plot_overlapping_event(
            validation_dir,
            out_dir,
            location,
            local_series,
            extra_lines=local_lines,
            filename_suffix="local_methods_overlapping_event",
        )
        if out is not None:
            saved.append(out)

        transfer_series = [(transfer_ensemble, "MoE")]
        transfer_lines = [
            {
                "column": "MoE",
                "label": "MoE",
                "color": METHOD_COLORS["moe_transfer"],
                "linewidth": 2.0,
            }
        ]
        for method in _best_methods_for_group(
            metrics_dir,
            location,
            f"transfer_{transfer_source}_",
        ):
            label = METHOD_LABELS[method]
            transfer_series.append((f"transfer_{transfer_source}_{method}", label))
            transfer_lines.append(
                {
                    "column": label,
                    "label": label,
                    "color": METHOD_COLORS[method],
                    "linewidth": 1.45,
                    "alpha": 0.82,
                }
            )
        out = _plot_overlapping_event(
            validation_dir,
            out_dir,
            location,
            transfer_series,
            extra_lines=transfer_lines,
            filename_suffix="transfer_methods_overlapping_event",
        )
        if out is not None:
            saved.append(out)

    return saved


def _best_methods_for_group(metrics_dir: Path, location: str, prefix: str) -> List[str]:
    path = metrics_dir / location / "metrics.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if "method" not in df.columns:
        return []

    sort_cols = [col for col in ["rmse_q99", "rmse_q95", "rmse"] if col in df.columns]
    selected = []
    for family in [CLASSICAL_METHODS, ML_METHODS]:
        sub = df[df["method"].astype(str).str.startswith(prefix)].copy()
        sub["base_method"] = sub["method"].astype(str).str.replace(prefix, "", regex=False)
        sub = sub[sub["base_method"].isin(family)]
        if sub.empty:
            continue
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=True, na_position="last")
        else:
            sub = sub.sort_values("base_method")
        selected.append(str(sub.iloc[0]["base_method"]))

    return selected


def _load_nora3_hindcast(nora3_dir: Path, location: str) -> Optional[pd.DataFrame]:
    path = nora3_dir / f"NORA3_wind_wave_{location}_1959_2025.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, comment="#")
    if not {"time", "hs"}.issubset(df.columns):
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[["time", "hs"]].rename(columns={"hs": "NORA3"})
    df = df.dropna(subset=["time", "NORA3"])
    return df.sort_values("time").reset_index(drop=True)


def _load_hindcast_series(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path, usecols=lambda col: col in {"time", "hs"})
    if not {"time", "hs"}.issubset(df.columns):
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.rename(columns={"hs": label}).dropna(subset=["time", label])
    return df.sort_values("time").reset_index(drop=True)


def _merge_hindcast_series(
    hindcast_dir: Path,
    nora3_dir: Path,
    location: str,
    series: List[Tuple[str, str]],
) -> Optional[pd.DataFrame]:
    merged = _load_nora3_hindcast(nora3_dir, location)
    if merged is None:
        return None

    for filename_stem, label in series:
        path = hindcast_dir / location / f"hindcast_corrected_{filename_stem}.csv"
        df = _load_hindcast_series(path, label)
        if df is None:
            return None
        merged = merged.merge(df, on="time", how="inner")

    return merged


def _nora3_peak_time(df: pd.DataFrame) -> pd.Timestamp:
    raw = pd.to_numeric(df["NORA3"], errors="coerce")
    return df.loc[raw.idxmax(), "time"]


def plot_hindcast_events(hindcast_dir: Path, nora3_dir: Path, out_dir: Path) -> List[Path]:
    specs = [
        (
            "fauskane",
            [("ensemble_fauskane", "MoE Local"), ("ensemble_fedjeosen", "MoE Transfer")],
            _moe_lines("local_transfer"),
        ),
        (
            "fedjeosen",
            [("ensemble_fedjeosen", "MoE Local"), ("ensemble_fauskane", "MoE Transfer")],
            _moe_lines("local_transfer"),
        ),
        ("vestfjorden", [("ensemble_combined", "MoE")], [_single_moe_line("MoE", "moe_combined")]),
        ("kristiansund", [("ensemble_combined", "MoE")], [_single_moe_line("MoE", "moe_combined")]),
        ("stavanger", [("ensemble_combined", "MoE")], [_single_moe_line("MoE", "moe_combined")]),
        ("bergen", [("ensemble_combined", "MoE")], [_single_moe_line("MoE", "moe_combined")]),
    ]

    saved = []
    for location, series, moe_lines in specs:
        df = _merge_hindcast_series(hindcast_dir, nora3_dir, location, series)
        if df is None or df.empty:
            continue

        peak_time = _nora3_peak_time(df)
        lines = [
            {
                "column": "NORA3",
                "label": "NORA3",
                "color": METHOD_COLORS["nora3"],
                "linewidth": 1.8,
            },
            *moe_lines,
        ]
        out_path = out_dir / f"{location}_largest_hindcast_event.png"
        saved.append(_save_event_plot(df, peak_time, out_path, lines, legend_columns=min(3, len(lines))))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot storm events for observed overlap and full hindcast cases.")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--hindcast-dir", type=Path, default=DEFAULT_HINDCAST_DIR)
    parser.add_argument("--nora3-dir", type=Path, default=DEFAULT_NORA3_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved.extend(plot_main_overlapping_events(args.validation_dir, args.out_dir))
    saved.extend(plot_method_overlapping_events(args.validation_dir, args.metrics_dir, args.out_dir))
    saved.extend(plot_hindcast_events(args.hindcast_dir, args.nora3_dir, args.out_dir))

    if saved:
        print("Saved plots:")
        for path in saved:
            print(f"  {path}")
    else:
        print("No storm plots were created. Check input paths and required columns.")


if __name__ == "__main__":
    main()
