import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


DEFAULT_SUMMARY_DIR = Path("results/ensemble")
DEFAULT_OUT_DIR = Path("results/ensemble")
DEFAULT_METRICS_DIR = Path("results/eval_metrics")

SUMMARY_PATTERN = "*_summary.txt"
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
PERFORMANCE_METRICS = ["rmse", "rmse_q95", "rmse_q99", "rmse_q995"]


def _clean_lines(text: str) -> List[str]:
    return [line.rstrip() for line in text.splitlines()]


def _split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if set(line.strip()) == {"="} and line.strip():
            if current:
                blocks.append(current)
                current = []
            continue

        if line.strip() or current:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks


def _parse_key_value(line: str) -> Tuple[Optional[str], Optional[str]]:
    if ":" not in line:
        return None, None
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def parse_summary_block(lines: List[str]) -> Dict:
    block = {
        "name": None,
        "training_cases": [],
        "application_member_family": None,
        "members": [],
        "top_features": {},
        "location": None,
        "input_families": [],
        "validation_mean_weights": {},
        "hindcast_mean_weights": {},
    }

    section = None
    subsection = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if not raw_line.startswith("  "):
            key, value = _parse_key_value(line)
            if key is None:
                continue

            if key in {"name", "application_member_family", "location"}:
                block[key] = value
                section = None
                subsection = None
            elif key == "training_cases":
                block[key] = [v.strip() for v in value.split("|") if v.strip()]
                section = None
                subsection = None
            elif key == "members":
                block[key] = [v.strip() for v in value.split("|") if v.strip()]
                section = None
                subsection = None
            elif key == "input_families":
                block[key] = [v.strip() for v in value.split("|") if v.strip()]
                section = None
                subsection = None
            elif key in {
                "top_features",
                "validation_mean_weights",
                "hindcast_mean_weights",
            }:
                section = key
                subsection = None
            else:
                section = None
                subsection = None
            continue

        if section == "top_features":
            if raw_line.startswith("  ") and not raw_line.startswith("    "):
                subsection = line.rstrip(":")
                block[section].setdefault(subsection, [])
            elif subsection is not None:
                key, value = _parse_key_value(line)
                if key is not None:
                    try:
                        block[section][subsection].append((key, float(value)))
                    except ValueError:
                        pass

        elif section in {"validation_mean_weights", "hindcast_mean_weights"}:
            key, value = _parse_key_value(line)
            if key is not None:
                try:
                    block[section][key] = float(value)
                except ValueError:
                    pass

    return block


def load_summary_blocks(summary_dir: Path) -> List[Dict]:
    blocks: List[Dict] = []
    for path in sorted(summary_dir.glob(SUMMARY_PATTERN)):
        lines = _clean_lines(path.read_text(encoding="utf-8", errors="ignore"))
        for raw_block in _split_blocks(lines):
            block = parse_summary_block(raw_block)
            if block.get("name"):
                block["source_file"] = path
                blocks.append(block)
    return blocks


def latest_blocks_by_name_location(blocks: List[Dict]) -> Dict[Tuple[str, Optional[str]], Dict]:
    latest: Dict[Tuple[str, Optional[str]], Dict] = {}
    for block in blocks:
        key = (str(block.get("name")), block.get("location"))
        latest[key] = block
    return latest


def _pretty_feature_label(name: str) -> str:
    text = str(name)

    replacements = {
        "member_": "",
        "delta_": "Delta ",
        "transfer_fauskane_": "Fauskane ",
        "transfer_fedjeosen_": "Fedjeosen ",
        "member_mean_minus_raw": "Mean - NORA3",
        "member_min_minus_raw": "Min - NORA3",
        "member_max_minus_raw": "Max - NORA3",
        "dir_cos": "Direction cosine",
        "hs_sea": "Sea-state Hs",
        "gpr": "GPR",
        "xgboost": "XGBoost",
        "dagqm": "DAGQM",
        "pqm": "PQM",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _latest_summary_block(summary_dir: Path, name: str) -> Optional[Dict]:
    blocks = load_summary_blocks(summary_dir)
    matching = [block for block in blocks if block.get("name") == name]
    if not matching:
        return None
    return matching[0]


def _feature_color(feature: str) -> str:
    text = str(feature)
    method = _method_from_expert(text)
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    if "raw" in text:
        return "#ff7f0e"
    if "wind" in text:
        return "#A0CBE8"
    if "dir" in text or "thq" in text:
        return "#B79A20"
    if "month" in text or "season" in text:
        return "#59A14F"
    return "#6B7280"


def plot_top_gate_features(summary_dir: Path, out_dir: Path, name: str, top_n: int = 10) -> Optional[Path]:
    block = _latest_summary_block(summary_dir, name)
    if block is None:
        return None
    features = block.get("top_features", {}).get("gate", [])[:top_n]
    if not features:
        return None

    labels = [_pretty_feature_label(name) for name, _ in features][::-1]
    values = [float(value) for _, value in features][::-1]
    colors = [_feature_color(name) for name, _ in features][::-1]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Importance")
    value_array = np.asarray(values, dtype=float)
    margin = 0.0025
    xmin = max(0.0, float(np.nanmin(value_array)) - margin)
    xmax = float(np.nanmax(value_array))
    ax.set_xlim(xmin, xmax + margin)
    _style_axes(ax, grid_axis="x")

    for y_pos, value in enumerate(values):
        ax.text(value, y_pos, f" {value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"moe_{name.replace('ensemble_', '')}_top_gate_features.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _method_from_expert(expert: str) -> str:
    for method in METHOD_ORDER:
        if expert == method or expert.endswith(f"_{method}"):
            return method
    return str(expert).split("_")[-1]


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def _display_metric_name(metric: str) -> str:
    mapping = {
        "rmse": "RMSE",
        "rmse_q95": "RMSE q95",
        "rmse_q99": "RMSE q99",
        "rmse_q995": "RMSE q99.5",
    }
    return mapping.get(metric, metric)


def _display_ensemble_case(location: str, method: str) -> str:
    key = (str(location), str(method))
    labels = {
        ("fauskane", "ensemble_fauskane"): "Fauskane_local",
        ("fauskane", "ensemble_fedjeosen"): "fauskane_transfer",
        ("fedjeosen", "ensemble_fedjeosen"): "fedjeosen_local",
        ("fedjeosen", "ensemble_fauskane"): "fedjeosen_transfer",
        ("vestfjorden", "ensemble_combined"): "vestfjorden_combined",
        ("vestfjorden", "ensemble_fauskane"): "vestfjorden_fauskane",
        ("vestfjorden", "ensemble_fedjeosen"): "vestfjorden_fedjeosen",
    }
    return labels.get(key, f"{location}_{method}")


def _case_sort_key(case: str) -> int:
    order = [
        "Fauskane_local",
        "fauskane_transfer",
        "fedjeosen_local",
        "fedjeosen_transfer",
        "vestfjorden_combined",
        "vestfjorden_fauskane",
        "vestfjorden_fedjeosen",
    ]
    return order.index(case) if case in order else len(order)


def _load_ensemble_metric_improvement(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*/metrics.csv")):
        location = path.parent.name
        df = pd.read_csv(path)
        if "method" not in df.columns or "raw" not in set(df["method"].astype(str)):
            continue

        raw = df[df["method"].astype(str) == "raw"].iloc[0]
        for _, row in df[df["method"].astype(str).str.startswith("ensemble_")].iterrows():
            out = {
                "location": location,
                "method": str(row["method"]),
                "case": _display_ensemble_case(location, str(row["method"])),
            }
            for metric in PERFORMANCE_METRICS:
                if metric not in df.columns:
                    continue
                raw_value = float(raw[metric])
                value = float(row[metric])
                out[metric] = 100.0 * (raw_value - value) / raw_value if raw_value > 0 else np.nan
            rows.append(out)

    return pd.DataFrame(rows)


def plot_moe_performance_vs_raw(metrics_dir: Path, out_dir: Path) -> Optional[Path]:
    df = _load_ensemble_metric_improvement(metrics_dir)
    if df.empty:
        return None

    df["case_order"] = df["case"].map(_case_sort_key)
    df = df.sort_values(["case_order", "case"]).reset_index(drop=True)
    available_metrics = [metric for metric in PERFORMANCE_METRICS if metric in df.columns]
    if not available_metrics:
        return None

    values = df[available_metrics].to_numpy(dtype=float)
    vmax = max(5.0, float(np.nanmax(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.6, max(4.5, 0.45 * len(df) + 1.6)))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_xticklabels([_display_metric_name(metric) for metric in available_metrics])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["case"])
    ax.tick_params(length=0)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("RMSE reduction vs NORA3 (%)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Validation case")
    fig.tight_layout()
    out_path = out_dir / "moe_performance_vs_raw_metrics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Create focused MoE interpretation plots for the thesis."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fn in [
        lambda: plot_top_gate_features(args.summary_dir, args.out_dir, "ensemble_combined"),
        lambda: plot_top_gate_features(args.summary_dir, args.out_dir, "ensemble_fedjeosen"),
        lambda: plot_top_gate_features(args.summary_dir, args.out_dir, "ensemble_fauskane"),
        lambda: plot_moe_performance_vs_raw(args.metrics_dir, args.out_dir),
    ]:
        out = fn()
        if out is not None:
            saved.append(out)

    if saved:
        print("Saved plots:")
        for path in saved:
            print(f"  {path}")
    else:
        print("No plots were created. Check summary files and metrics tables.")


if __name__ == "__main__":
    main()
