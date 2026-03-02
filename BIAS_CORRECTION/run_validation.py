import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto

from run_bias_correction import (
    apply_all_methods,
    fit_all_methods,
    load_hindcast,
    load_overlap_pairs,
    make_forward_folds,
    save_corrected_hindcast_files,
)
from plots_diagnostics import plot_cdf, plot_pdf, plot_qq, plot_residuals


COL = {"time": "time", "hs_obs": "Significant_Wave_Height_Hm0"}
METHOD_COLS = {"raw": "pred_raw", "linear": "pred_linear", "qm": "pred_qm", "rf": "pred_rf"}

N_FOLDS = 4
TEST_FRACTION = 0.15
MIN_TRAIN_FRACTION = 0.50
RF_SEED = 42
EVT_Q = 0.95
EVT_DECLUSTER_H = 48.0
EVT_MIN_EVENTS = 20
RL_PERIODS = [10.0, 20.0, 50.0]


def _rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def _tail_rmse(model, obs, q):
    thr = float(np.nanquantile(obs, q))
    m = obs >= thr
    if np.sum(m) < 20:
        return np.nan
    return _rmse(model[m], obs[m])


def _q_bias(model, obs, q):
    if np.isfinite(model).sum() < 20 or np.isfinite(obs).sum() < 20:
        return np.nan
    return float(np.nanquantile(model, q) - np.nanquantile(obs, q))


def _exceed_rate_bias(model, obs, q):
    if np.isfinite(obs).sum() < 20:
        return np.nan
    thr = float(np.nanquantile(obs, q))
    m_model = np.asarray(model, float)
    m_obs = np.asarray(obs, float)
    a = np.mean(m_model[np.isfinite(m_model)] >= thr)
    b = np.mean(m_obs[np.isfinite(m_obs)] >= thr)
    return float(a - b)


def _decluster_clustermax(times, values, threshold, window_hours):
    m = np.isfinite(values) & (values > threshold)
    if m.sum() == 0:
        return np.array([])
    t = pd.to_datetime(times[m]).to_numpy()
    v = np.asarray(values[m], dtype=float)
    peaks, cluster_max, prev_time = [], v[0], t[0]
    for i in range(1, len(v)):
        dt = (t[i] - prev_time) / np.timedelta64(1, "h")
        if dt <= window_hours:
            if v[i] > cluster_max:
                cluster_max = v[i]
        else:
            peaks.append(cluster_max)
            cluster_max = v[i]
        prev_time = t[i]
    peaks.append(cluster_max)
    return np.asarray(peaks, dtype=float)


def _fit_overlap_gpd(df, value_col, threshold_quantile=EVT_Q, decluster_hours=EVT_DECLUSTER_H, min_events=EVT_MIN_EVENTS, fixed_threshold=None):
    x = pd.to_numeric(df[value_col], errors="coerce").values
    t = pd.to_datetime(df[COL["time"]], errors="coerce").values
    m = np.isfinite(x) & pd.notna(t)
    x, t = x[m], t[m]
    if len(x) < 100:
        return None
    thr = float(np.nanquantile(x, threshold_quantile)) if fixed_threshold is None else float(fixed_threshold)
    peaks = _decluster_clustermax(t, x, threshold=thr, window_hours=decluster_hours)
    if len(peaks) < min_events:
        return None
    excess = peaks - thr
    excess = excess[excess > 0]
    if len(excess) < min_events:
        return None
    xi, _, sigma = genpareto.fit(excess, floc=0)
    years = (pd.Timestamp(t.max()) - pd.Timestamp(t.min())).total_seconds() / (365.25 * 24 * 3600)
    years = max(years, 1e-6)
    lam = len(peaks) / years
    z2, z5 = np.clip(lam * 2.0, 1e-12, None), np.clip(lam * 5.0, 1e-12, None)
    if abs(float(xi)) < 1e-6:
        rl2 = float(thr + sigma * np.log(z2))
        rl5 = float(thr + sigma * np.log(z5))
    else:
        rl2 = float(thr + (sigma / xi) * (z2 ** xi - 1.0))
        rl5 = float(thr + (sigma / xi) * (z5 ** xi - 1.0))
    return {"xi": float(xi), "sigma": float(sigma), "threshold": thr, "lambda_year": float(lam), "rl_2y": rl2, "rl_5y": rl5}


def _gpd_rl(df, value_col):
    fit = _fit_overlap_gpd(df, value_col)
    if fit is None:
        return {f"gpd_rl_{int(t)}y": np.nan for t in RL_PERIODS}
    out = {}
    for t in RL_PERIODS:
        z = np.clip(fit["lambda_year"] * t, 1e-12, None)
        xi, sigma, thr = fit["xi"], fit["sigma"], fit["threshold"]
        if abs(float(xi)) < 1e-6:
            out[f"gpd_rl_{int(t)}y"] = float(thr + sigma * np.log(z))
        else:
            out[f"gpd_rl_{int(t)}y"] = float(thr + (sigma / xi) * (z ** xi - 1.0))
    return out


def _gev_rl(df, value_col):
    tmp = df[[COL["time"], value_col]].copy()
    tmp[COL["time"]] = pd.to_datetime(tmp[COL["time"]], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[COL["time"], value_col]).sort_values(COL["time"]).set_index(COL["time"])
    tmp = tmp[tmp[value_col] > 0]
    if tmp.empty:
        return {f"gev_rl_{int(t)}y": np.nan for t in RL_PERIODS}
    annual_max = tmp[value_col].resample("YE").max().dropna()
    first_year, last_year = tmp.index.min().year, tmp.index.max().year
    annual_max = annual_max[(annual_max.index.year > first_year) & (annual_max.index.year < last_year)]
    if len(annual_max) < 5:
        return {f"gev_rl_{int(t)}y": np.nan for t in RL_PERIODS}
    c, loc, scale = genextreme.fit(annual_max.values)
    return {f"gev_rl_{int(t)}y": float(genextreme.ppf(1 - 1 / t, c, loc=loc, scale=scale)) for t in RL_PERIODS}


def _evt_score(summary):
    def norm(s):
        a, b = np.nanmin(s.values), np.nanmax(s.values)
        return np.zeros(len(s)) if (not np.isfinite(a) or not np.isfinite(b) or (b - a) < 1e-12) else (s.values - a) / (b - a)

    return (
        0.20 * norm(summary["abs_delta_xi"])
        + 0.15 * norm(summary["abs_delta_sigma"])
        + 0.15 * norm(summary["abs_delta_lambda_year"])
        + 0.20 * norm(summary["abs_delta_rl_2y"])
        + 0.20 * norm(summary["abs_delta_rl_5y"])
        + 0.10 * norm(summary["std_delta_rl_5y"])
    )


def _compute_cv_predictions(df_pairs):
    folds = make_forward_folds(len(df_pairs), N_FOLDS, TEST_FRACTION, MIN_TRAIN_FRACTION)
    parts = []
    for fold_id, start, end in folds:
        df_train = df_pairs.iloc[:start].copy()
        df_test = df_pairs.iloc[start:end].copy()
        linear_model, qm_model, rf_model = fit_all_methods(df_train, rf_seed=RF_SEED)
        pred = apply_all_methods(df_test, linear_model, qm_model, rf_model)
        pred["fold_id"] = fold_id
        parts.append(pred)
    return pd.concat(parts, ignore_index=True).sort_values(COL["time"])


def _global_metrics(df_cv):
    obs = df_cv[COL["hs_obs"]].values
    rows = []
    for method, col in METHOD_COLS.items():
        pred = df_cv[col].values
        rows.append(
            {
                "method": method,
                "tail_rmse_95": _tail_rmse(pred, obs, 0.95),
                "tail_rmse_99": _tail_rmse(pred, obs, 0.99),
                "q95_bias": _q_bias(pred, obs, 0.95),
                "q99_bias": _q_bias(pred, obs, 0.99),
                "q995_bias": _q_bias(pred, obs, 0.995),
                "exceed_rate_bias_q95": _exceed_rate_bias(pred, obs, 0.95),
                "exceed_rate_bias_q99": _exceed_rate_bias(pred, obs, 0.99),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def _evt_summary(df_pred):
    rows = []
    groups = [("all", df_pred)]
    if "fold_id" in df_pred.columns:
        groups.extend(list(df_pred.groupby("fold_id")))
    for fold_id, g in groups:
        obs_fit = _fit_overlap_gpd(g, COL["hs_obs"])
        if obs_fit is None:
            continue
        for method, col in METHOD_COLS.items():
            m_fit = _fit_overlap_gpd(g, col, fixed_threshold=obs_fit["threshold"])
            if m_fit is None:
                continue
            rows.append(
                {
                    "fold_id": fold_id,
                    "method": method,
                    "delta_xi": m_fit["xi"] - obs_fit["xi"],
                    "delta_sigma": m_fit["sigma"] - obs_fit["sigma"],
                    "delta_lambda_year": m_fit["lambda_year"] - obs_fit["lambda_year"],
                    "delta_rl_2y": m_fit["rl_2y"] - obs_fit["rl_2y"],
                    "delta_rl_5y": m_fit["rl_5y"] - obs_fit["rl_5y"],
                }
            )
    x = pd.DataFrame(rows)
    if x.empty:
        raise RuntimeError("No valid EVT overlap fits.")
    s = x.groupby("method").agg(
        abs_delta_xi=("delta_xi", lambda z: np.nanmean(np.abs(z))),
        abs_delta_sigma=("delta_sigma", lambda z: np.nanmean(np.abs(z))),
        abs_delta_lambda_year=("delta_lambda_year", lambda z: np.nanmean(np.abs(z))),
        abs_delta_rl_2y=("delta_rl_2y", lambda z: np.nanmean(np.abs(z))),
        abs_delta_rl_5y=("delta_rl_5y", lambda z: np.nanmean(np.abs(z))),
        std_delta_rl_5y=("delta_rl_5y", lambda z: np.nanstd(z)),
    )
    s["score_evt"] = _evt_score(s)
    return s.sort_values("score_evt")


def _annual_p99_stats(df, value_col):
    tmp = df[[COL["time"], value_col]].dropna().copy()
    tmp["year"] = pd.to_datetime(tmp[COL["time"]], errors="coerce").dt.year
    a = tmp.groupby("year")[value_col].quantile(0.99).dropna()
    if len(a) == 0:
        return np.nan, np.nan
    m, st = float(np.nanmean(a.values)), float(np.nanstd(a.values))
    cv = st / m if np.isfinite(m) and abs(m) > 1e-12 else np.nan
    tr = float(np.polyfit(a.index.values.astype(float), a.values, 1)[0]) if len(a) >= 2 else np.nan
    return cv, tr


def _full_hindcast_summary(df_pairs, df_hind_pred):
    obs = pd.to_numeric(df_pairs[COL["hs_obs"]], errors="coerce").values
    obs = obs[np.isfinite(obs)]
    p95_obs, p99_obs, p995_obs = float(np.nanquantile(obs, 0.95)), float(np.nanquantile(obs, 0.99)), float(np.nanquantile(obs, 0.995))
    rows = []
    for method, col in METHOD_COLS.items():
        x = pd.to_numeric(df_hind_pred[col], errors="coerce").values
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        cv, trend = _annual_p99_stats(df_hind_pred, col)
        row = {
            "method": method,
            "delta_p95_vs_overlap_obs": float(np.nanquantile(x, 0.95) - p95_obs),
            "delta_p99_vs_overlap_obs": float(np.nanquantile(x, 0.99) - p99_obs),
            "delta_p995_vs_overlap_obs": float(np.nanquantile(x, 0.995) - p995_obs),
            "annual_p99_cv": cv,
            "annual_p99_trend_per_year": trend,
        }
        row.update(_gev_rl(df_hind_pred[[COL["time"], col]], col))
        row.update(_gpd_rl(df_hind_pred[[COL["time"], col]], col))
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def _section(title, df):
    return f"{title}\n{df.to_string(float_format=lambda v: f'{v:.6f}')}\n"


def _clean_result_dir(path):
    os.makedirs(path, exist_ok=True)
    keep = {
        "summary.txt",
        "pdf.png",
        "cdf.png",
        "qq_plot.png",
        "residuals.png",
        "hindcast_corrected_linear.csv",
        "hindcast_corrected_qm.csv",
        "hindcast_corrected_rf.csv",
    }
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p) and name not in keep:
            os.remove(p)


def run(location):
    result_dir = f"BIAS_CORRECTION_V1/output/{location}"
    _clean_result_dir(result_dir)

    pairs_path = f"BIAS_CORRECTION_V1/dataset/NORA3_{location}_pairs.csv"
    hindcast_path = f"DATA_EXTRACTION/nora3_locations/NORA3_wind_wave_{location}_1969_2025.csv"

    df_pairs = load_overlap_pairs(pairs_path)
    df_hind = load_hindcast(hindcast_path)
    df_cv = _compute_cv_predictions(df_pairs)

    global_metrics = _global_metrics(df_cv)

    linear_model, qm_model, rf_model = fit_all_methods(df_pairs, rf_seed=RF_SEED)
    df_overlap_pred = apply_all_methods(df_pairs, linear_model, qm_model, rf_model)
    evt_summary = _evt_summary(df_overlap_pred)

    df_hind_pred = apply_all_methods(df_hind, linear_model, qm_model, rf_model)
    save_corrected_hindcast_files(hindcast_path, df_hind_pred, location)
    full_summary = _full_hindcast_summary(df_pairs, df_hind_pred)

    plot_pdf(df_cv, COL["hs_obs"], METHOD_COLS, result_dir)
    plot_cdf(df_cv, COL["hs_obs"], METHOD_COLS, result_dir)
    plot_qq(df_cv, COL["hs_obs"], METHOD_COLS, result_dir)
    plot_residuals(df_cv, COL["hs_obs"], METHOD_COLS, result_dir)

    evt_print = evt_summary[["score_evt", "abs_delta_xi", "abs_delta_sigma", "abs_delta_lambda_year", "abs_delta_rl_2y", "abs_delta_rl_5y"]]
    full_print = full_summary[
        [
            "delta_p95_vs_overlap_obs",
            "delta_p99_vs_overlap_obs",
            "delta_p995_vs_overlap_obs",
            "annual_p99_cv",
            "annual_p99_trend_per_year",
            "gev_rl_10y",
            "gev_rl_20y",
            "gev_rl_50y",
            "gpd_rl_10y",
            "gpd_rl_20y",
            "gpd_rl_50y",
        ]
    ]

    parts = [
        _section("Global validation metrics:", global_metrics),
        _section("EVT overlap summary (lower score is better):", evt_print),
        _section("Full hindcast summary (vs overlap observations):", full_print),
    ]

    text = "\n\n".join(parts).strip() + "\n"
    print(text)
    summary_path = f"{result_dir}/summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple one-command validation.")
    parser.add_argument("--location", default="fedjeosen")
    args = parser.parse_args()
    run(args.location)


if __name__ == "__main__":
    main()
