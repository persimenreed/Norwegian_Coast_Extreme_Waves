from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.settings import get_path_template

TIME_COLUMN = "time"
HS_COLUMN = "hs"
THRESHOLD_QUANTILE = 0.95
DECLUSTER_HOURS = 48.0
RETURN_PERIOD_GRID = np.arange(1, 51, dtype=float)
SUMMARY_RETURN_PERIODS = (10, 20, 50)
BOOTSTRAP_SAMPLES = 1000
CONF_LEVEL = 0.95


def dataset_name(mode: str, corr_method: str = "pqm", transfer_source: str | None = None) -> str:
    mode = str(mode).strip().lower()
    if mode == "raw":
        return "raw"
    if mode != "corrected":
        raise ValueError("mode must be 'raw' or 'corrected'")
    if str(corr_method).startswith("ensemble_"):
        return str(corr_method)
    return f"transfer_{transfer_source}_{corr_method}" if transfer_source else f"local_{corr_method}"


def evt_root() -> Path:
    return Path(get_path_template("evt_results_root"))


def summary_path(location: str) -> Path:
    return evt_root() / location / "summary_return_levels.csv"


def bootstrap_confidence_interval(n_bootstrap, make_levels, desc, conf_level=CONF_LEVEL):
    boot_levels = []
    for _ in tqdm(range(int(n_bootstrap)), desc=desc):
        try:
            levels = np.asarray(make_levels(), dtype=float)
            if np.all(np.isfinite(levels)):
                boot_levels.append(levels)
        except Exception:
            continue

    if not boot_levels:
        raise ValueError(f"All bootstrap fits failed for {desc}.")

    boot_levels = np.asarray(boot_levels, dtype=float)
    alpha = 1.0 - float(conf_level)
    return (
        np.quantile(boot_levels, alpha / 2, axis=0),
        np.quantile(boot_levels, 1 - alpha / 2, axis=0),
        len(boot_levels),
    )


def return_level_table(return_periods, return_levels, ci_low, ci_high, n_bootstrap, conf_level=CONF_LEVEL):
    return pd.DataFrame(
        {
            "return_period": np.asarray(return_periods, dtype=int),
            "return_level": np.asarray(return_levels, dtype=float),
            "ci_lower": np.asarray(ci_low, dtype=float),
            "ci_upper": np.asarray(ci_high, dtype=float),
            "ci_width": np.asarray(ci_high, dtype=float) - np.asarray(ci_low, dtype=float),
            "n_bootstrap": int(n_bootstrap),
            "conf_level": float(conf_level),
        }
    )


def append_return_level_summary(location: str, dataset: str, model: str, table: pd.DataFrame) -> Path:
    required = {"return_period", "return_level", "ci_lower", "ci_upper", "ci_width"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in return level table: {missing}")

    out = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "return_period": table["return_period"].astype(float),
            "return_level": table["return_level"].astype(float),
            "ci_lower": table["ci_lower"].astype(float),
            "ci_upper": table["ci_upper"].astype(float),
            "ci_width": table["ci_width"].astype(float),
        }
    )

    path = summary_path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[~((prev["dataset"] == dataset) & (prev["model"] == model))]
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(path, index=False)
    print(f"Updated summary table: {path}")
    return path


def build_evt_summary_metrics(location: str):
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

        for model in ("GEV", "GPD"):
            raw_ref = raw[raw["model"] == model]
            model_ref = df[(df["dataset"] == dataset) & (df["model"] == model)]
            if raw_ref.empty or model_ref.empty:
                continue

            for return_period in SUMMARY_RETURN_PERIODS:
                raw_row = raw_ref[raw_ref["return_period"] == return_period]
                model_row = model_ref[model_ref["return_period"] == return_period]
                if raw_row.empty or model_row.empty:
                    continue

                rl_raw = float(raw_row["return_level"].iloc[0])
                rl_model = float(model_row["return_level"].iloc[0])
                ci_low = float(raw_row["ci_lower"].iloc[0])
                ci_high = float(raw_row["ci_upper"].iloc[0])

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "return_period": return_period,
                        "rl_raw": rl_raw,
                        "rl_model": rl_model,
                        "rle_vs_raw": rl_model - rl_raw,
                        "arle_vs_raw": abs(rl_model - rl_raw),
                        "rrle_pct_vs_raw": 100 * (rl_model - rl_raw) / rl_raw if rl_raw else pd.NA,
                        "inside_raw_ci": int(ci_low <= rl_model <= ci_high),
                    }
                )

    if not rows:
        return

    out_path = path.parent / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved EVT summary metrics: {out_path}")
