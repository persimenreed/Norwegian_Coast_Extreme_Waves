import numpy as np
import pandas as pd

TIME = "time"
HS_MODEL = "hs"
HS_OBS = "Significant_Wave_Height_Hm0"
DIR_MODEL = "Pdir"
HS_QUANTILE = "hs_quantile"
HS_QUANTILE_BIAS = "hs_quantile_bias"
HS_QUANTILE_BASELINE = "hs_quantile_baseline"
ORIGINAL_INDEX = "_orig_index"

DEFAULT_FEATURE_COLUMNS = [
    "hs",
    "tp",
    "tm2",
    "hs_sea",
    "hs_swell",
    "wind_speed_10m",
    "wind_speed_100m",
    "month_sin",
    "month_cos",
    "dir_sin",
    "dir_cos",
    "wind_dir_sin",
    "wind_dir_cos",
]


def clip_nonnegative(x, eps=0.0):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    return np.maximum(x, eps)


def cfg_int(cfg, key, default):
    return int(cfg.get(key, default))


def cfg_float(cfg, key, default):
    return float(cfg.get(key, default))


def cfg_str(cfg, key, default):
    return str(cfg.get(key, default))


def finite_pair_mask(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.isfinite(x) & np.isfinite(y)


def numeric_values(df, column, dtype=float):
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=dtype)


def sort_frame(df, preserve_order=False):
    out = df.copy()
    if preserve_order:
        out = out.reset_index(drop=False).rename(columns={"index": ORIGINAL_INDEX})
    if TIME in out.columns:
        out[TIME] = pd.to_datetime(out[TIME], errors="coerce")
    sort_cols = [column for column in ("source", TIME) if column in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def restore_frame_order(df):
    if ORIGINAL_INDEX not in df.columns:
        return df.reset_index(drop=True)
    return df.sort_values(ORIGINAL_INDEX).drop(columns=[ORIGINAL_INDEX]).reset_index(drop=True)


def add_time_features(df):
    out = df.copy()
    if TIME not in out.columns:
        return out

    month = pd.to_datetime(out[TIME], errors="coerce").dt.month.fillna(1).astype(int)
    if "month_sin" not in out.columns:
        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    if "month_cos" not in out.columns:
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def add_direction_features(df):
    out = df.copy()

    if DIR_MODEL in out.columns and "dir_sin" not in out.columns:
        angle = np.deg2rad(pd.to_numeric(out[DIR_MODEL], errors="coerce"))
        out["dir_sin"] = np.sin(angle)
        out["dir_cos"] = np.cos(angle)

    if "wind_direction_10m" in out.columns and "wind_dir_sin" not in out.columns:
        angle = np.deg2rad(pd.to_numeric(out["wind_direction_10m"], errors="coerce"))
        out["wind_dir_sin"] = np.sin(angle)
        out["wind_dir_cos"] = np.cos(angle)

    return out


def prepare_ml_dataframe(df):
    return add_direction_features(add_time_features(df))


def quantile_feature_columns(features):
    out = list(features)
    for column in (HS_QUANTILE, HS_QUANTILE_BIAS, HS_QUANTILE_BASELINE):
        if column not in out:
            out.append(column)
    return out


def resolve_feature_columns(df, requested=None):
    candidates = list(requested or DEFAULT_FEATURE_COLUMNS)
    columns = [column for column in candidates if column in df.columns]
    if not columns:
        raise ValueError(
            f"No usable ML features found. Candidates were: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return columns


def fit_fill_values(df, feature_cols, mask=None):
    fill = {}
    row_mask = np.asarray(mask, dtype=bool) if mask is not None else None

    for column in feature_cols:
        values = pd.to_numeric(
            df.loc[row_mask, column] if row_mask is not None else df[column],
            errors="coerce",
        ).to_numpy(float)
        valid = values[np.isfinite(values)]
        fill[column] = float(np.median(valid)) if len(valid) else 0.0

    return fill


def fit_standard_scaler(df, feature_cols, mask=None):
    fill = fit_fill_values(df, feature_cols, mask=mask)
    mean = {}
    std = {}
    row_mask = np.asarray(mask, dtype=bool) if mask is not None else None

    for column in feature_cols:
        values = pd.to_numeric(
            df.loc[row_mask, column] if row_mask is not None else df[column],
            errors="coerce",
        ).to_numpy(float)
        values = np.where(np.isfinite(values), values, fill[column])
        mean[column] = float(np.mean(values)) if len(values) else 0.0
        sigma = float(np.std(values)) if len(values) else 1.0
        std[column] = sigma if np.isfinite(sigma) and sigma > 0 else 1.0

    return fill, mean, std


def feature_matrix(df, feature_cols, fill=None, mean=None, std=None, dtype=np.float32):
    if fill is None:
        fill = fit_fill_values(df, feature_cols)

    matrix = np.empty((len(df), len(feature_cols)), dtype=dtype)
    scaled = mean is not None and std is not None

    for index, column in enumerate(feature_cols):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        values = np.where(np.isfinite(values), values, fill[column])
        if scaled:
            values = (values - mean[column]) / std[column]
        matrix[:, index] = values.astype(dtype)

    return matrix, fill


def gumbel_quantile_grid(n=401, p_min=1e-3, p_max=0.999):
    n = max(int(n), 3)
    p_min = float(p_min)
    p_max = float(p_max)
    g_min = -np.log(-np.log(np.clip(p_min, 1e-6, 1.0 - 1e-6)))
    g_max = -np.log(-np.log(np.clip(p_max, 1e-6, 1.0 - 1e-6)))
    grid = np.linspace(g_min, g_max, n, dtype=float)
    return np.unique(np.clip(np.exp(-np.exp(-grid)), p_min, p_max)).astype(np.float32)


def empirical_percentiles(values, reference_values=None, p_min=1e-3, p_max=0.999):
    x = np.asarray(values, dtype=float)
    reference = x if reference_values is None else np.asarray(reference_values, dtype=float)
    reference = reference[np.isfinite(reference)]

    out = np.full(len(x), np.nan, dtype=np.float32)
    if not len(reference):
        return out

    mask = np.isfinite(x)
    if not np.any(mask):
        return out

    if len(reference) == 1:
        out[mask] = np.float32(0.5)
        return out

    reference = np.sort(reference)
    left = np.searchsorted(reference, x[mask], side="left")
    right = np.searchsorted(reference, x[mask], side="right")
    denom = max(len(reference) - 1, 1)
    out[mask] = np.clip(0.5 * (left + right) / float(denom), p_min, p_max).astype(np.float32)
    return out


def stabilize_quantile_mapping_tail(
    source_quantiles,
    target_quantiles,
    probabilities,
    *,
    enabled=False,
    blend_start=0.95,
    pool_start=0.95,
    pool_end=0.995,
):
    qs = np.asarray(source_quantiles, dtype=np.float32)
    qt = np.asarray(target_quantiles, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)
    bias = (qt - qs).astype(np.float32)
    meta = {
        "right_tail_bias": float(bias[-1]) if len(bias) else 0.0,
        "tail_bias_pool": np.nan,
    }

    if not enabled or len(probs) < 5:
        return qt, meta

    pool_mask = (probs >= float(pool_start)) & (probs <= float(pool_end))
    if np.sum(pool_mask) < 3:
        return qt, meta

    tail_probs = probs[pool_mask].astype(np.float64)
    weights = 1.0 + (tail_probs - tail_probs.min()) / max(tail_probs.max() - tail_probs.min(), 1e-6)
    pooled_bias = float(np.average(bias[pool_mask].astype(np.float64), weights=weights))

    tail_mask = probs >= float(blend_start)
    if np.any(tail_mask):
        alpha = np.clip(
            (probs[tail_mask] - float(blend_start))
            / max(float(pool_end) - float(blend_start), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        bias[tail_mask] = ((1.0 - alpha) * bias[tail_mask] + alpha * pooled_bias).astype(np.float32)
        qt = (qs + bias).astype(np.float32)

    meta["right_tail_bias"] = float(bias[-1]) if len(bias) else 0.0
    meta["tail_bias_pool"] = pooled_bias
    return qt, meta


def map_quantiles_by_value(values, mapping, eps=1e-4):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=np.float32)

    source = np.asarray(mapping["source_quantiles"], dtype=float)
    target = np.asarray(mapping["target_quantiles"], dtype=float)
    if len(source) < 2 or len(target) < 2:
        return out

    mask = np.isfinite(x)
    if not np.any(mask):
        return out

    values_in = x[mask]
    values_out = np.interp(values_in, source, target).astype(np.float32)

    left_mask = values_in < source[0]
    if np.any(left_mask):
        dx = source[1] - source[0]
        slope = 0.0 if abs(dx) < 1e-8 else (target[1] - target[0]) / dx
        values_out[left_mask] = (target[0] + slope * (values_in[left_mask] - source[0])).astype(np.float32)

    right_mask = values_in > source[-1]
    if np.any(right_mask):
        bias_end = float(mapping.get("right_tail_bias", target[-1] - source[-1]))
        values_out[right_mask] = (values_in[right_mask] + bias_end).astype(np.float32)

    out[mask] = values_out
    return clip_nonnegative(out).astype(np.float32)


def build_quantile_bias_mapping(
    source_values,
    target_values,
    n_quantiles=401,
    p_min=1e-3,
    p_max=0.999,
    tail_cfg=None,
):
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    valid = np.isfinite(source) & np.isfinite(target)

    if np.sum(valid) < 20:
        raise ValueError("Too few valid samples for quantile bias mapping.")

    probabilities = gumbel_quantile_grid(n=n_quantiles, p_min=p_min, p_max=p_max)
    source_quantiles = np.quantile(source[valid], probabilities).astype(np.float32)
    target_quantiles = np.quantile(target[valid], probabilities).astype(np.float32)

    tail_cfg = tail_cfg or {}
    target_quantiles, tail_meta = stabilize_quantile_mapping_tail(
        source_quantiles,
        target_quantiles,
        probabilities,
        enabled=bool(tail_cfg.get("tail_pool_enabled", False)),
        blend_start=float(tail_cfg.get("tail_blend_start", 0.95)),
        pool_start=float(tail_cfg.get("tail_pool_start", 0.95)),
        pool_end=float(tail_cfg.get("tail_pool_end", 0.995)),
    )

    return {
        "probabilities": probabilities.astype(np.float32),
        "source_quantiles": source_quantiles,
        "target_quantiles": target_quantiles,
        "additive_bias": (target_quantiles - source_quantiles).astype(np.float32),
        "right_tail_bias": float(tail_meta.get("right_tail_bias", (target_quantiles - source_quantiles)[-1])),
        "tail_bias_pool": float(tail_meta.get("tail_bias_pool", np.nan)),
        "p_min": float(p_min),
        "p_max": float(p_max),
    }


def quantile_bias_features(values, mapping, reference_values=None, eps=1e-4):
    x = np.asarray(values, dtype=float)
    percentiles = empirical_percentiles(
        x,
        reference_values=reference_values,
        p_min=float(mapping.get("p_min", 1e-3)),
        p_max=float(mapping.get("p_max", 0.999)),
    )
    baseline = map_quantiles_by_value(x, mapping, eps=eps)
    bias = np.full(len(x), np.nan, dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(baseline)

    bias[mask] = (baseline[mask] - x[mask]).astype(np.float32)

    return {
        HS_QUANTILE: percentiles.astype(np.float32),
        HS_QUANTILE_BIAS: bias.astype(np.float32),
        HS_QUANTILE_BASELINE: baseline,
    }


def augment_quantile_features(df, raw_values, transform_cfg, reference_values=None):
    out = df.copy()
    extras = quantile_bias_features(
        raw_values,
        transform_cfg["quantile_mapping"],
        reference_values=reference_values if reference_values is not None else raw_values,
        eps=float(transform_cfg.get("eps", 1e-4)),
    )
    for column, values in extras.items():
        out[column] = values
    return out, extras


def build_target_transform(obs_values, raw_values, cfg):
    obs = np.asarray(obs_values, dtype=np.float32)
    raw = np.asarray(raw_values, dtype=np.float32)
    eps = np.float32(cfg_float(cfg, "target_eps", 1e-4))

    mapping = build_quantile_bias_mapping(
        raw,
        obs,
        n_quantiles=int(cfg.get("quantile_grid_size", 401)),
        p_min=float(cfg.get("quantile_p_min", 1e-3)),
        p_max=float(cfg.get("quantile_p_max", 0.999)),
        tail_cfg=cfg,
    )
    extras = quantile_bias_features(
        raw,
        mapping,
        reference_values=raw,
        eps=float(eps),
    )
    baseline = extras[HS_QUANTILE_BASELINE]
    valid = np.isfinite(obs) & np.isfinite(baseline)
    return (
        (obs - baseline).astype(np.float32),
        valid,
        {
            "mode": "quantile_residual",
            "eps": float(eps),
            "quantile_mapping": mapping,
            "quantile_bias_mode": "additive",
            "tail_residual_protection_enabled": bool(
                cfg.get("tail_residual_protection_enabled", False)
            ),
            "tail_residual_protection_mode": cfg_str(
                cfg,
                "tail_residual_protection_mode",
                "sign_aware",
            ),
            "tail_residual_start": float(cfg.get("tail_residual_start", 0.95)),
            "tail_residual_end": float(cfg.get("tail_residual_end", 0.999)),
            "tail_residual_min_scale": float(cfg.get("tail_residual_min_scale", 0.25)),
        },
    )


def invert_target(pred_target, base_values, transform_cfg):
    pred = np.asarray(pred_target, dtype=np.float32)
    base = np.asarray(base_values, dtype=np.float32)

    corrected = np.full(len(base), np.nan, dtype=np.float32)
    mask = np.isfinite(base) & np.isfinite(pred)
    corrected[mask] = base[mask] + pred[mask]
    return clip_nonnegative(corrected)


def protect_tail_residuals(pred_target, percentiles, transform_cfg):
    pred = np.asarray(pred_target, dtype=np.float32).copy()
    pct = np.asarray(percentiles, dtype=np.float32)

    if not bool(transform_cfg.get("tail_residual_protection_enabled", False)):
        return pred

    mask = np.isfinite(pred) & np.isfinite(pct)
    if not np.any(mask):
        return pred

    mode = str(transform_cfg.get("tail_residual_protection_mode", "sign_aware")).strip().lower()
    if mode == "negative_only":
        mask &= pred < 0.0
    elif mode == "positive_only":
        mask &= pred > 0.0
    elif mode != "symmetric":
        mapping = transform_cfg.get("quantile_mapping", {}) or {}
        tail_bias = mapping.get("tail_bias_pool", np.nan)
        if not np.isfinite(tail_bias):
            tail_bias = mapping.get("right_tail_bias", np.nan)
        if not np.isfinite(tail_bias) or abs(float(tail_bias)) < 1e-8:
            return pred
        mask &= pred < 0.0 if float(tail_bias) > 0.0 else pred > 0.0

    if not np.any(mask):
        return pred

    start = float(transform_cfg.get("tail_residual_start", 0.95))
    end = float(transform_cfg.get("tail_residual_end", 0.999))
    min_scale = min(max(float(transform_cfg.get("tail_residual_min_scale", 0.25)), 0.0), 1.0)
    alpha = np.clip((pct[mask] - start) / max(end - start, 1e-6), 0.0, 1.0)
    pred[mask] = (pred[mask] * (1.0 - alpha * (1.0 - min_scale))).astype(np.float32)
    return pred


def build_tail_sample_weights(obs_values, cfg, dtype=np.float32):
    obs = np.asarray(obs_values, dtype=float)
    weights = np.ones(len(obs), dtype=float)
    valid = np.isfinite(obs)

    if np.sum(valid) >= 20:
        q90 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q90_quantile", 0.90)))
        q95 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q95_quantile", 0.95)))
        q99 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q99_quantile", 0.99)))
        weights[valid & (obs >= q90)] = cfg_float(cfg, "tail_weight_q90", 2.0)
        weights[valid & (obs >= q95)] = cfg_float(cfg, "tail_weight_q95", 3.0)
        weights[valid & (obs >= q99)] = cfg_float(cfg, "tail_weight_q99", 5.0)

    return weights.astype(dtype, copy=False)
