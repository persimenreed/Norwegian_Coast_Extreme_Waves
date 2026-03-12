import numpy as np
import pandas as pd

from src.settings import get_columns, get_default_ml_features

_COLUMNS = get_columns()

TIME = _COLUMNS.get("time", "time")
HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")
DIR_MODEL = _COLUMNS.get("dir_model", "Pdir")


def clip_nonnegative(x, eps=0.0):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    return np.maximum(x, eps)


def finite_pair_mask(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.isfinite(x) & np.isfinite(y)


def add_time_features(df):
    out = df.copy()

    if TIME in out.columns:
        t = pd.to_datetime(out[TIME], errors="coerce")
        month = t.dt.month.fillna(1).astype(int)

        if "month_sin" not in out.columns:
            out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)

        if "month_cos" not in out.columns:
            out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    return out


def add_direction_features(df):
    out = df.copy()

    if DIR_MODEL in out.columns and "dir_sin" not in out.columns:
        ang = np.deg2rad(pd.to_numeric(out[DIR_MODEL], errors="coerce"))
        out["dir_sin"] = np.sin(ang)
        out["dir_cos"] = np.cos(ang)

    if "wind_direction_10m" in out.columns and "wind_dir_sin" not in out.columns:
        ang = np.deg2rad(pd.to_numeric(out["wind_direction_10m"], errors="coerce"))
        out["wind_dir_sin"] = np.sin(ang)
        out["wind_dir_cos"] = np.cos(ang)

    return out


def prepare_ml_dataframe(df):
    out = add_time_features(df)
    out = add_direction_features(out)
    return out


def resolve_feature_columns(df, requested=None):
    requested = list(requested or [])
    candidates = requested if requested else get_default_ml_features()
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise ValueError(
            f"No usable ML features found. Candidates were: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return cols


def chronological_train_val_split(
    n_samples,
    val_fraction=0.2,
    min_train=20,
    min_val=20,
):
    if n_samples < (min_train + min_val):
        return np.arange(n_samples), np.array([], dtype=int)

    n_val = max(min_val, int(round(n_samples * float(val_fraction))))
    n_val = min(n_val, n_samples - min_train)

    if n_val <= 0:
        return np.arange(n_samples), np.array([], dtype=int)

    split = n_samples - n_val
    train_idx = np.arange(split)
    val_idx = np.arange(split, n_samples)
    return train_idx, val_idx