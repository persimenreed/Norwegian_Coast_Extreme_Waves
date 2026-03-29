import numpy as np
import pandas as pd

from src.settings import get_columns, get_default_ml_features

_COLUMNS = get_columns()

TIME = _COLUMNS.get("time", "time")
HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")
DIR_MODEL = _COLUMNS.get("dir_model", "Pdir")
HS_QUANTILE = "hs_quantile"
HS_QUANTILE_BIAS = "hs_quantile_bias"
HS_QUANTILE_BASELINE = "hs_quantile_baseline"


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


def gumbel_quantile_grid(n=401, p_min=1e-3, p_max=0.999):
    n = max(int(n), 3)
    p = np.linspace(0.0, 1.0, n, dtype=float)
    g = -np.log(-np.log(np.clip(p, 1e-6, 1.0 - 1e-6)))
    g = (g - g.min()) / max(g.max() - g.min(), 1e-8)
    out = float(p_min) + g * (float(p_max) - float(p_min))
    return np.unique(np.clip(out, p_min, p_max)).astype(np.float32)


def empirical_percentiles(values, reference_values=None, p_min=1e-3, p_max=0.999):
    x = np.asarray(values, dtype=float)
    ref = x if reference_values is None else np.asarray(reference_values, dtype=float)
    ref = ref[np.isfinite(ref)]

    out = np.full(len(x), np.nan, dtype=np.float32)
    if len(ref) == 0:
        return out

    ref_sorted = np.sort(ref)
    m = np.isfinite(x)
    if not np.any(m):
        return out

    if len(ref_sorted) == 1:
        out[m] = np.float32(0.5)
        return out

    left = np.searchsorted(ref_sorted, x[m], side="left")
    right = np.searchsorted(ref_sorted, x[m], side="right")
    rank = 0.5 * (left + right)
    denom = max(len(ref_sorted) - 1, 1)
    pct = rank / float(denom)
    pct = np.clip(pct, float(p_min), float(p_max))
    out[m] = pct.astype(np.float32)
    return out


def build_quantile_bias_mapping(
    source_values,
    target_values,
    n_quantiles=401,
    p_min=1e-3,
    p_max=0.999,
):
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    valid = np.isfinite(source) & np.isfinite(target)

    if np.sum(valid) < 20:
        raise ValueError("Too few valid samples for quantile bias mapping.")

    probs = gumbel_quantile_grid(n=n_quantiles, p_min=p_min, p_max=p_max)
    qs = np.quantile(source[valid], probs).astype(np.float32)
    qt = np.quantile(target[valid], probs).astype(np.float32)

    return {
        "probabilities": probs.astype(np.float32),
        "source_quantiles": qs,
        "target_quantiles": qt,
        "additive_bias": (qt - qs).astype(np.float32),
        "p_min": float(p_min),
        "p_max": float(p_max),
    }


def quantile_bias_from_percentiles(percentiles, mapping, mode="additive", eps=1e-4):
    pct = np.asarray(percentiles, dtype=float)
    out = np.full(len(pct), np.nan, dtype=np.float32)
    m = np.isfinite(pct)
    if not np.any(m):
        return out

    probs = np.asarray(mapping["probabilities"], dtype=float)
    mode = str(mode).strip().lower()

    if mode == "additive":
        curve = np.asarray(mapping["additive_bias"], dtype=float)
    elif mode == "log_ratio":
        qs = np.maximum(
            np.asarray(mapping["source_quantiles"], dtype=float) + float(eps),
            float(eps),
        )
        qt = np.maximum(
            np.asarray(mapping["target_quantiles"], dtype=float) + float(eps),
            float(eps),
        )
        curve = np.log(qt) - np.log(qs)
    else:
        raise ValueError(f"Unsupported quantile bias mode: {mode}")

    out[m] = np.interp(
        pct[m],
        probs,
        curve,
        left=curve[0],
        right=curve[-1],
    ).astype(np.float32)
    return out


def quantile_bias_features(
    values,
    mapping,
    reference_values=None,
    mode="additive",
    eps=1e-4,
):
    x = np.asarray(values, dtype=float)
    pct = empirical_percentiles(
        x,
        reference_values=reference_values,
        p_min=float(mapping.get("p_min", 1e-3)),
        p_max=float(mapping.get("p_max", 0.999)),
    )
    bias = quantile_bias_from_percentiles(pct, mapping, mode=mode, eps=eps)

    baseline = np.full(len(x), np.nan, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(bias)

    mode = str(mode).strip().lower()
    if mode == "additive":
        baseline[m] = (x[m] + bias[m]).astype(np.float32)
    elif mode == "log_ratio":
        safe = np.maximum(x[m] + float(eps), float(eps))
        baseline[m] = (np.exp(np.log(safe) + bias[m]) - float(eps)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported quantile bias mode: {mode}")

    baseline = clip_nonnegative(baseline).astype(np.float32)
    return {
        HS_QUANTILE: pct.astype(np.float32),
        HS_QUANTILE_BIAS: bias.astype(np.float32),
        HS_QUANTILE_BASELINE: baseline,
    }
