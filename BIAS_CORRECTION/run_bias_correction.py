import argparse
import os

import numpy as np
import pandas as pd

from bc_methods.linear_scaling import apply_linear_scaling, fit_linear_scaling
from bc_methods.qm_parametric import apply_qm, fit_qm
from bc_methods.random_forest import apply_rf_residual, fit_rf_residual


COL = {
    "time": "time",
    "hs_model": "hs",
    "hs_obs": "Significant_Wave_Height_Hm0",
    "per_model": "tm2",
    "per_obs": "Wave_Mean_Period_Tm02",
}

RF_FEATURES = ["hs", "tp", "tm2", "wind_speed_10m", "wind_speed_20m", "Pdir", "month_sin", "month_cos"]

N_FOLDS = 4
TEST_FRACTION = 0.15
MIN_TRAIN_FRACTION = 0.50
RF_SEED = 42


def _read_csv_with_time(path):
    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip() for c in df.columns]
    if COL["time"] not in df.columns:
        cands = [c for c in df.columns if c.lstrip("#").strip().lower() == "time"]
        if not cands:
            raise ValueError(f"Missing '{COL['time']}' in {path}")
        df = df.rename(columns={cands[0]: COL["time"]})
    df[COL["time"]] = pd.to_datetime(df[COL["time"]], errors="coerce")
    return df


def _add_month_features(df):
    out = df.copy()
    m = out[COL["time"]].dt.month.astype(float)
    a = 2 * np.pi * (m - 1.0) / 12.0
    out["month_sin"] = np.sin(a)
    out["month_cos"] = np.cos(a)
    return out


def load_overlap_pairs(path):
    df = _read_csv_with_time(path)
    df = df.sort_values(COL["time"]).dropna(subset=[COL["time"], COL["hs_model"], COL["hs_obs"]]).copy()
    return _add_month_features(df)


def load_hindcast(path):
    df = _read_csv_with_time(path)
    df = df.sort_values(COL["time"]).dropna(subset=[COL["time"], COL["hs_model"]]).copy()
    return _add_month_features(df)


def make_forward_folds(n_samples, n_folds=N_FOLDS, test_fraction=TEST_FRACTION, min_train_fraction=MIN_TRAIN_FRACTION):
    test_size = max(1, int(round(n_samples * test_fraction)))
    min_train = max(1, int(round(n_samples * min_train_fraction)))
    last_start = n_samples - test_size
    if last_start <= min_train:
        raise ValueError("Not enough data for CV settings.")
    starts = np.unique(np.linspace(min_train, last_start, num=n_folds, dtype=int))
    folds = []
    for fold_id, start in enumerate(starts):
        end = min(start + test_size, n_samples)
        if start >= min_train and end > start:
            folds.append((fold_id, start, end))
    if not folds:
        raise ValueError("No valid folds.")
    return folds


def fit_all_methods(df_train, rf_seed=RF_SEED):
    linear_model = fit_linear_scaling(df_train, COL["hs_model"], COL["hs_obs"])

    use_period = COL["per_model"] in df_train.columns and COL["per_obs"] in df_train.columns
    qm_model = fit_qm(
        df_train,
        COL["hs_model"],
        COL["hs_obs"],
        COL["per_model"] if use_period else None,
        COL["per_obs"] if use_period else None,
    )
    rf_cols = [c for c in RF_FEATURES if c in df_train.columns]
    rf_model = fit_rf_residual(
        df_train,
        hs_model=COL["hs_model"],
        hs_obs=COL["hs_obs"],
        feature_cols=rf_cols,
        random_state=rf_seed,
    )
    return linear_model, qm_model, rf_model


def apply_all_methods(df_in, linear_model, qm_model, rf_model):
    out = df_in.copy()
    out["pred_raw"] = out[COL["hs_model"]]
    out = apply_linear_scaling(out, linear_model, COL["hs_model"], out_col="pred_linear")

    use_period = COL["per_model"] in out.columns
    out = apply_qm(out, qm_model, COL["hs_model"], COL["per_model"] if use_period else None)
    out["pred_qm"] = out["hs_qm"]
    drop_cols = ["hs_qm"] + (["tm2_qm"] if "tm2_qm" in out.columns else [])
    out = out.drop(columns=drop_cols)

    out = apply_rf_residual(out, rf_model, COL["hs_model"], out_col="pred_rf")
    return out


def _read_preamble(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                lines.append(line)
            else:
                break
    return lines


def save_corrected_hindcast_files(hindcast_path, df_hind_pred, location):
    preamble = _read_preamble(hindcast_path)
    df_src = pd.read_csv(hindcast_path, comment="#")
    df_src.columns = [str(c).strip() for c in df_src.columns]
    if COL["time"] not in df_src.columns:
        cands = [c for c in df_src.columns if c.lstrip("#").strip().lower() == "time"]
        if not cands:
            raise ValueError(f"Missing time column in {hindcast_path}")
        df_src = df_src.rename(columns={cands[0]: COL["time"]})

    src_time = pd.to_datetime(df_src[COL["time"]], errors="coerce")
    pred_time = pd.to_datetime(df_hind_pred[COL["time"]], errors="coerce")
    out_dir = f"BIAS_CORRECTION_V1/output/{location}"
    os.makedirs(out_dir, exist_ok=True)

    saved = {}
    for pred_col, method_name in [
        ("pred_linear", "linear"),
        ("pred_qm", "qm"),
        ("pred_rf", "rf"),
    ]:
        if pred_col not in df_hind_pred.columns:
            continue
        series = pd.Series(pd.to_numeric(df_hind_pred[pred_col], errors="coerce").values, index=pred_time.values)
        out_df = df_src.copy()
        mapped = src_time.map(series)
        mask = mapped.notna()
        out_df.loc[mask, COL["hs_model"]] = mapped.loc[mask].values

        out_path = f"{out_dir}/hindcast_corrected_{method_name}.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            for line in preamble:
                f.write(line)
            out_df.to_csv(f, index=False)
        saved[method_name] = out_path
    return saved


def main():
    parser = argparse.ArgumentParser(description="Fit correction methods and save corrected full hindcast files.")
    parser.add_argument("--location", default="fedjeosen")
    args = parser.parse_args()

    pairs_path = f"BIAS_CORRECTION_V1/dataset/NORA3_{args.location}_pairs.csv"
    hindcast_path = f"DATA_EXTRACTION/nora3_locations/NORA3_wind_wave_{args.location}_1969_2025.csv"

    df_pairs = load_overlap_pairs(pairs_path)
    df_hind = load_hindcast(hindcast_path)
    linear_model, qm_model, rf_model = fit_all_methods(df_pairs, rf_seed=RF_SEED)
    df_hind_pred = apply_all_methods(df_hind, linear_model, qm_model, rf_model)
    saved = save_corrected_hindcast_files(hindcast_path, df_hind_pred, args.location)

    print("Saved corrected full hindcast files:")
    for m in sorted(saved):
        print(f"  {m}: {saved[m]}")


if __name__ == "__main__":
    main()
