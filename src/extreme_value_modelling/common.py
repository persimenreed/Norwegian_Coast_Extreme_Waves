from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.settings import get_path_template


def dataset_name(
    mode: str,
    corr_method: str = "pqm",
    pooling: bool = False,
    transfer_source: str | None = None,
) -> str:
    mode = str(mode).strip().lower()

    if mode == "raw":
        return "raw"

    if mode != "corrected":
        raise ValueError("mode must be 'raw' or 'corrected'")

    if transfer_source:
        return f"transfer_{transfer_source}_{corr_method}"

    if pooling:
        return f"pooled_{corr_method}"

    return f"local_{corr_method}"


def summary_path(location: str) -> Path:
    root = Path(get_path_template("evt_results_root"))
    return root / location / "summary_return_levels.csv"


def append_return_level_summary(location: str, dataset: str, model: str, table: pd.DataFrame) -> Path:
    required = {"return_period", "return_level", "ci_lower", "ci_upper", "ci_width"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in return level table: {missing}")

    out = pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "return_period": table["return_period"].astype(float),
        "return_level": table["return_level"].astype(float),
        "ci_lower": table["ci_lower"].astype(float),
        "ci_upper": table["ci_upper"].astype(float),
        "ci_width": table["ci_width"].astype(float),
    })

    path = summary_path(location)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        prev = pd.read_csv(path)
        mask = ~((prev["dataset"] == dataset) & (prev["model"] == model))
        prev = prev[mask]
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(path, index=False)

    print(f"Updated summary table: {path}")
    return path


def build_evt_summary_metrics(location: str):
    """
    Create summary_metrics.csv containing EVT differences
    relative to the raw hindcast baseline.
    """

    path = summary_path(location)
    if not path.exists():
        return

    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        return

    raw = df[df["dataset"] == "raw"]
    if raw.empty:
        return

    rows = []

    for dataset in sorted(df["dataset"].unique()):
        if dataset == "raw":
            continue

        for model in ["GEV", "GPD"]:
            raw_ref = raw[raw["model"] == model]
            mod = df[(df["dataset"] == dataset) & (df["model"] == model)]

            if raw_ref.empty or mod.empty:
                continue

            for rp in [2, 5, 10, 20, 50]:
                r = raw_ref[raw_ref["return_period"] == rp]
                m = mod[mod["return_period"] == rp]

                if r.empty or m.empty:
                    continue

                rl_raw = float(r["return_level"].iloc[0])
                rl_mod = float(m["return_level"].iloc[0])

                ci_low = float(r["ci_lower"].iloc[0])
                ci_high = float(r["ci_upper"].iloc[0])

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "return_period": rp,
                    "rl_raw": rl_raw,
                    "rl_model": rl_mod,
                    "rle_vs_raw": rl_mod - rl_raw,
                    "arle_vs_raw": abs(rl_mod - rl_raw),
                    "rrle_pct_vs_raw": 100 * (rl_mod - rl_raw) / rl_raw if rl_raw != 0 else pd.NA,
                    "inside_raw_ci": int(ci_low <= rl_mod <= ci_high),
                })

    if not rows:
        return

    out = pd.DataFrame(rows)
    out_path = summary_path(location).parent / "summary_metrics.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved EVT summary metrics: {out_path}")