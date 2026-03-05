import argparse
from pathlib import Path
import sys

# allow src imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from src.settings import get_columns, get_thresholds, format_path, get_path_template


# ============================================================
# Helpers: loading
# ============================================================

def _read_time_hs_csv(path: Path, time_col: str, hs_col: str) -> pd.DataFrame:

    df = pd.read_csv(path, comment="#")

    # normalize column names (some files contain whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    if time_col not in df.columns or hs_col not in df.columns:
        raise ValueError(
            f"{path} missing required columns {time_col}, {hs_col}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[time_col, hs_col]].copy()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[hs_col] = pd.to_numeric(df[hs_col], errors="coerce")

    df = df.dropna(subset=[time_col, hs_col])
    df = df[df[hs_col] > 0]

    df = df.sort_values(time_col)

    return df.set_index(time_col)


def _discover_corrected_files(location: str) -> list[tuple[str, Path]]:
    """
    Finds files like:
      data/output/{location}/hindcast_corrected_{mode}_{method}.csv

    Returns list of (dataset_name, path)
    where dataset_name is "{mode}_{method}" e.g. "local_qm".
    """
    out_dir = Path("data/output") / location
    if not out_dir.exists():
        return []

    files = sorted(out_dir.glob("hindcast_corrected_*.csv"))
    out: list[tuple[str, Path]] = []

    for f in files:
        # expected: hindcast_corrected_{mode}_{method}.csv
        name = f.stem  # without .csv
        parts = name.split("_")
        # ["hindcast", "corrected", "{mode}", "{method}"] possibly more underscores in future
        if len(parts) < 4:
            continue
        if parts[0] != "hindcast" or parts[1] != "corrected":
            continue

        mode = parts[2]
        method = "_".join(parts[3:])  # robust if method ever contains underscores
        dataset = f"{mode}_{method}"
        out.append((dataset, f))

    return out


# ============================================================
# EVT (GPD POT) on a time-indexed series with declustering
# ============================================================

def _decluster_clustermax(exceed: pd.Series, window_hours: float) -> np.ndarray:
    """
    exceed: Series indexed by datetime, values = Hs above threshold
    Returns array of cluster maxima (Hs), one per cluster.
    """
    if exceed.empty:
        return np.array([], dtype=float)

    exceed = exceed.sort_index()
    peaks = []
    cluster_max = float(exceed.iloc[0])
    last_t = exceed.index[0]

    for t, v in exceed.iloc[1:].items():
        dt_hours = (t - last_t).total_seconds() / 3600.0
        if dt_hours <= window_hours:
            if float(v) > cluster_max:
                cluster_max = float(v)
        else:
            peaks.append(cluster_max)
            cluster_max = float(v)
        last_t = t

    peaks.append(cluster_max)
    return np.asarray(peaks, dtype=float)


def fit_gpd_pot_return_levels(
    s: pd.Series,
    threshold_quantile: float,
    decluster_hours: float,
    return_periods: list[float],
) -> dict:
    """
    Fits GPD on declustered POT excesses.
    Return levels computed with standard POT formula:

      RL(T) = u + (sigma/xi) * ( (lambda*T)^xi - 1 ) , xi != 0
      RL(T) = u + sigma * log(lambda*T)              , xi == 0

    where lambda is rate of declustered events per year.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    s = s.sort_index()

    if len(s) < 200:  # guard: too few points for stable POT
        return {
            "threshold": np.nan,
            "xi": np.nan,
            "sigma": np.nan,
            "n_points": int(len(s)),
            "n_events": 0,
            "years": np.nan,
            "return_levels": {rp: np.nan for rp in return_periods},
        }

    u = float(s.quantile(threshold_quantile))
    exceed = s[s > u]

    peaks = _decluster_clustermax(exceed, window_hours=float(decluster_hours))

    # duration in years based on actual time span (works with missing timestamps)
    dt_years = (s.index.max() - s.index.min()).total_seconds() / (365.25 * 24 * 3600.0)
    dt_years = float(max(dt_years, 1e-9))

    n_events = int(len(peaks))
    lam = float(n_events / dt_years)

    if n_events < 20:  # your settings has evt_min_events=20; respect that spirit here
        return {
            "threshold": u,
            "xi": np.nan,
            "sigma": np.nan,
            "n_points": int(len(s)),
            "n_events": n_events,
            "years": dt_years,
            "return_levels": {rp: np.nan for rp in return_periods},
        }

    excess = peaks - u
    excess = excess[excess > 0]

    if len(excess) < 10:
        return {
            "threshold": u,
            "xi": np.nan,
            "sigma": np.nan,
            "n_points": int(len(s)),
            "n_events": n_events,
            "years": dt_years,
            "return_levels": {rp: np.nan for rp in return_periods},
        }

    # Fit GPD with loc fixed at 0 (excesses)
    shape, _, scale = genpareto.fit(excess, floc=0)
    xi = float(shape)
    sigma = float(scale)

    rls = {}
    for T in return_periods:
        T = float(T)
        x = lam * T
        if x <= 0 or not np.isfinite(x):
            rls[T] = np.nan
            continue

        if abs(xi) < 1e-8:
            rl = u + sigma * np.log(x)
        else:
            rl = u + (sigma / xi) * (x**xi - 1.0)

        rls[T] = float(rl)

    return {
        "threshold": u,
        "xi": xi,
        "sigma": sigma,
        "n_points": int(len(s)),
        "n_events": n_events,
        "years": dt_years,
        "return_levels": rls,
    }


# ============================================================
# Main
# ============================================================

def run(location: str, return_periods: list[float] | None = None) -> Path | None:
    cols = get_columns()
    thr = get_thresholds()

    TIME = cols.get("time", "time")
    HS_MODEL = cols.get("hs_model", "hs")
    HS_OBS = cols.get("hs_obs", "Significant_Wave_Height_Hm0")

    q = float(thr.get("evt_threshold_quantile", 0.95))
    decluster_hours = float(thr.get("decluster_hours", 48.0))

    if return_periods is None:
        return_periods = [2, 5, 10]

    # --------------------------
    # Load observations
    # --------------------------
    obs_path = Path(f"data/input/buoys_data/buoy_{location}_max.csv")
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    obs = _read_time_hs_csv(obs_path, TIME, HS_OBS)

    # --------------------------
    # Load raw hindcast (from settings template)
    # --------------------------
    raw_path = Path(format_path("hindcast_raw", location=location))
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw hindcast file not found: {raw_path}")

    raw = _read_time_hs_csv(raw_path, TIME, HS_MODEL)

    # --------------------------
    # Discover corrected hindcasts
    # --------------------------
    corrected = _discover_corrected_files(location)

    datasets: list[tuple[str, pd.DataFrame]] = [("raw", raw)]
    for ds, p in corrected:
        try:
            df = _read_time_hs_csv(p, TIME, HS_MODEL)
            datasets.append((ds, df))
        except Exception:
            # skip unreadable / malformed files
            continue

    if len(datasets) == 0:
        print("No datasets found.")
        return None

    # --------------------------
    # MASTER timestamp intersection (exact overlaps)
    # --------------------------
    master_times = obs.index
    for _, df in datasets:
        master_times = master_times.intersection(df.index)

    master_times = master_times.sort_values()

    if len(master_times) < 500:
        raise ValueError(
            f"Too few overlapping timestamps after intersection: {len(master_times)}. "
            f"(Check time parsing / timezones / dataset coverage.)"
        )

    obs_aligned = obs.loc[master_times, HS_OBS]

    # Fit OBS once on the master overlap
    obs_evt = fit_gpd_pot_return_levels(
        obs_aligned,
        threshold_quantile=q,
        decluster_hours=decluster_hours,
        return_periods=return_periods,
    )

    # --------------------------
    # Fit each model on same master overlap & compute error vs OBS
    # --------------------------
    rows = []

    for ds_name, df in datasets:
        mod_aligned = df.loc[master_times, HS_MODEL]

        mod_evt = fit_gpd_pot_return_levels(
            mod_aligned,
            threshold_quantile=q,
            decluster_hours=decluster_hours,
            return_periods=return_periods,
        )

        for rp in return_periods:
            rp = float(rp)
            rl_obs = obs_evt["return_levels"].get(rp, np.nan)
            rl_mod = mod_evt["return_levels"].get(rp, np.nan)
            print(ds_name, rl_obs, rl_mod)

            rows.append({
                "location": location,
                "dataset": ds_name,
                "model": "GPD_overlap",
                "return_period": rp,
                "rl_obs": rl_obs,
                "rl_model": rl_mod,
                "error": rl_mod - rl_obs if np.isfinite(rl_obs) and np.isfinite(rl_mod) else np.nan,
                "abs_error": abs(rl_mod - rl_obs) if np.isfinite(rl_obs) and np.isfinite(rl_mod) else np.nan,
                "rrle_pct": 100.0 * (rl_mod - rl_obs) / rl_obs if np.isfinite(rl_obs) and np.isfinite(rl_mod) and rl_obs != 0 else np.nan,
                "n_overlap_points": int(len(master_times)),

                # diagnostics / transparency:
                "obs_threshold": obs_evt["threshold"],
                "obs_xi": obs_evt["xi"],
                "obs_sigma": obs_evt["sigma"],
                "obs_n_events": obs_evt["n_events"],
                "obs_years": obs_evt["years"],

                "model_threshold": mod_evt["threshold"],
                "model_xi": mod_evt["xi"],
                "model_sigma": mod_evt["sigma"],
                "model_n_events": mod_evt["n_events"],
                "model_years": mod_evt["years"],
            })

    out = pd.DataFrame(rows)

    # Add raw baseline error as explicit column on all rows
    raw_err = (
        out[out["dataset"] == "raw"][["return_period", "error"]]
        .rename(columns={"error": "raw_error"})
    )
    out = out.merge(raw_err, on="return_period", how="left")

    # ------------------------------------------------
    # Improvement relative to raw hindcast
    # ------------------------------------------------

    out["raw_abs_error"] = np.abs(out["raw_error"])

    out["improvement"] = out["raw_abs_error"] - out["abs_error"]

    # percent improvement relative to raw error
    out["improvement_pct"] = 100 * out["improvement"] / out["raw_abs_error"]

    # --------------------------
    # Save
    # --------------------------
    root = Path(get_path_template("evt_results_root"))
    out_dir = root / location
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "overlap_summary_metrics.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved overlap EVT metrics: {out_path}")
    print(f"Overlapping timestamps used: {len(master_times)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Overlap EVT (GPD POT) metrics vs buoy observations.")
    parser.add_argument("--location", required=True, help="Buoy location (e.g. fauskane).")
    parser.add_argument(
        "--return-periods",
        default="2,5,10",
        help="Comma-separated return periods (years), e.g. '2,5,10'. Default: 2,5,10",
    )
    args = parser.parse_args()

    rps = [float(x.strip()) for x in str(args.return_periods).split(",") if x.strip()]
    run(args.location, return_periods=rps)


if __name__ == "__main__":
    main()