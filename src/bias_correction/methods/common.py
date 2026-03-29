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


def quantile_feature_columns(features):
    out = list(features)
    for col in (HS_QUANTILE, HS_QUANTILE_BIAS, HS_QUANTILE_BASELINE):
        if col not in out:
            out.append(col)
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
    p_min = float(p_min)
    p_max = float(p_max)
    g_min = -np.log(-np.log(np.clip(p_min, 1e-6, 1.0 - 1e-6)))
    g_max = -np.log(-np.log(np.clip(p_max, 1e-6, 1.0 - 1e-6)))
    g = np.linspace(g_min, g_max, n, dtype=float)
    out = np.exp(-np.exp(-g))
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


def stabilize_quantile_mapping_tail(
    source_quantiles,
    target_quantiles,
    probabilities,
    *,
    enabled=False,
    blend_start=0.95,
    pool_start=0.95,
    pool_end=0.995,
    monotone=True,
):
    qs = np.asarray(source_quantiles, dtype=np.float32)
    qt = np.asarray(target_quantiles, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)

    bias = (qt - qs).astype(np.float32)
    meta = {
        "right_tail_bias": float(bias[-1]) if len(bias) else 0.0,
        "right_tail_bias_slope": 0.0,
        "tail_bias_pool": np.nan,
    }

    if not enabled or len(probs) < 5:
        return qt, meta

    pool_start = float(pool_start)
    pool_end = float(pool_end)
    blend_start = float(blend_start)

    pool_mask = (probs >= pool_start) & (probs <= pool_end)
    if np.sum(pool_mask) < 3:
        return qt, meta

    tail_bias = bias[pool_mask].astype(np.float64)
    tail_probs = probs[pool_mask].astype(np.float64)
    ramp = tail_probs - tail_probs.min()
    weights = 1.0 + ramp / max(tail_probs.max() - tail_probs.min(), 1e-6)
    pooled_bias = float(np.average(tail_bias, weights=weights))

    tail_mask = probs >= blend_start
    if not np.any(tail_mask):
        meta["tail_bias_pool"] = pooled_bias
        return qt, meta

    alpha = np.clip(
        (probs[tail_mask] - blend_start) / max(pool_end - blend_start, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    bias_tail = ((1.0 - alpha) * bias[tail_mask] + alpha * pooled_bias).astype(np.float32)

    if monotone:
        if pooled_bias >= 0.0:
            bias_tail = np.maximum.accumulate(bias_tail)
        else:
            bias_tail = np.minimum.accumulate(bias_tail)

    bias[tail_mask] = bias_tail
    qt_adj = (qs + bias).astype(np.float32)

    meta["right_tail_bias"] = float(bias_tail[-1])
    meta["tail_bias_pool"] = pooled_bias
    return qt_adj, meta


def map_quantiles_by_value(values, mapping, mode="additive", eps=1e-4):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=np.float32)

    qs = np.asarray(mapping["source_quantiles"], dtype=float)
    qt = np.asarray(mapping["target_quantiles"], dtype=float)
    if len(qs) < 2 or len(qt) < 2:
        return out

    m = np.isfinite(x)
    if not np.any(m):
        return out

    xv = x[m]
    yv = np.interp(xv, qs, qt).astype(np.float32)

    left_mask = xv < qs[0]
    if np.any(left_mask):
        dx = qs[1] - qs[0]
        slope = 0.0 if abs(dx) < 1e-8 else (qt[1] - qt[0]) / dx
        yv[left_mask] = (qt[0] + slope * (xv[left_mask] - qs[0])).astype(np.float32)

    right_mask = xv > qs[-1]
    if np.any(right_mask):
        mode = str(mode).strip().lower()
        if mode == "additive":
            bias_end = float(mapping.get("right_tail_bias", qt[-1] - qs[-1]))
            bias_slope = float(mapping.get("right_tail_bias_slope", 0.0))
            dx = xv[right_mask] - qs[-1]
            yv[right_mask] = (xv[right_mask] + bias_end + bias_slope * dx).astype(np.float32)
        elif mode == "log_ratio":
            safe_qs = max(qs[-1] + float(eps), float(eps))
            safe_qt = max(qt[-1] + float(eps), float(eps))
            log_ratio = np.log(safe_qt) - np.log(safe_qs)
            yv[right_mask] = (
                np.exp(np.log(np.maximum(xv[right_mask] + float(eps), float(eps))) + log_ratio)
                - float(eps)
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported quantile mapping mode: {mode}")

    out[m] = yv
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

    probs = gumbel_quantile_grid(n=n_quantiles, p_min=p_min, p_max=p_max)
    qs = np.quantile(source[valid], probs).astype(np.float32)
    qt = np.quantile(target[valid], probs).astype(np.float32)

    tail_cfg = tail_cfg or {}
    qt, tail_meta = stabilize_quantile_mapping_tail(
        qs,
        qt,
        probs,
        enabled=bool(tail_cfg.get("tail_pool_enabled", False)),
        blend_start=float(tail_cfg.get("tail_blend_start", 0.95)),
        pool_start=float(tail_cfg.get("tail_pool_start", 0.95)),
        pool_end=float(tail_cfg.get("tail_pool_end", 0.995)),
        monotone=bool(tail_cfg.get("tail_monotone", True)),
    )

    return {
        "probabilities": probs.astype(np.float32),
        "source_quantiles": qs,
        "target_quantiles": qt,
        "additive_bias": (qt - qs).astype(np.float32),
        "right_tail_bias": float(tail_meta.get("right_tail_bias", (qt - qs)[-1])),
        "right_tail_bias_slope": float(tail_meta.get("right_tail_bias_slope", 0.0)),
        "tail_bias_pool": float(tail_meta.get("tail_bias_pool", np.nan)),
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
    baseline = map_quantiles_by_value(x, mapping, mode=mode, eps=eps)
    bias = np.full(len(x), np.nan, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(baseline)

    mode = str(mode).strip().lower()
    if mode == "additive":
        bias[m] = (baseline[m] - x[m]).astype(np.float32)
    elif mode == "log_ratio":
        safe_x = np.maximum(x[m] + float(eps), float(eps))
        safe_base = np.maximum(baseline[m] + float(eps), float(eps))
        bias[m] = (np.log(safe_base) - np.log(safe_x)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported quantile bias mode: {mode}")

    return {
        HS_QUANTILE: pct.astype(np.float32),
        HS_QUANTILE_BIAS: bias.astype(np.float32),
        HS_QUANTILE_BASELINE: baseline,
    }


def augment_quantile_features(df, raw_values, transform_cfg, reference_values=None):
    out = df.copy()
    extras = quantile_bias_features(
        raw_values,
        transform_cfg["quantile_mapping"],
        reference_values=reference_values if reference_values is not None else raw_values,
        mode=transform_cfg.get("quantile_bias_mode", "additive"),
        eps=float(transform_cfg.get("eps", 1e-4)),
    )
    for col, values in extras.items():
        out[col] = values
    return out, extras


def build_target_transform(obs_values, raw_values, cfg):
    obs = np.asarray(obs_values, dtype=np.float32)
    raw = np.asarray(raw_values, dtype=np.float32)

    mode = cfg_str(cfg, "target_transform", "log_ratio").strip().lower()
    eps = np.float32(cfg_float(cfg, "target_eps", 1e-4))

    if mode == "quantile_residual":
        qmap = build_quantile_bias_mapping(
            raw,
            obs,
            n_quantiles=int(cfg.get("quantile_grid_size", 401)),
            p_min=float(cfg.get("quantile_p_min", 1e-3)),
            p_max=float(cfg.get("quantile_p_max", 0.999)),
            tail_cfg=cfg,
        )
        extras = quantile_bias_features(
            raw,
            qmap,
            reference_values=raw,
            mode=cfg_str(cfg, "quantile_bias_mode", "additive"),
            eps=float(eps),
        )
        baseline = extras[HS_QUANTILE_BASELINE]
        y = obs - baseline
        valid = np.isfinite(obs) & np.isfinite(baseline)
        return (
            y.astype(np.float32),
            valid,
            {
                "mode": mode,
                "eps": float(eps),
                "quantile_mapping": qmap,
                "quantile_bias_mode": cfg_str(cfg, "quantile_bias_mode", "additive"),
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
                "tail_residual_min_scale": float(
                    cfg.get("tail_residual_min_scale", 0.25)
                ),
            },
        )

    if mode == "additive_residual":
        y = obs - raw
        valid = np.isfinite(obs) & np.isfinite(raw)
        return y.astype(np.float32), valid, {"mode": mode, "eps": float(eps)}

    if mode == "log_ratio":
        valid = np.isfinite(obs) & np.isfinite(raw) & (obs > -eps) & (raw > -eps)
        y = np.full(len(obs), np.nan, dtype=np.float32)
        y[valid] = np.log(obs[valid] + eps) - np.log(raw[valid] + eps)
        return y.astype(np.float32), valid, {"mode": mode, "eps": float(eps)}

    raise ValueError(f"Unsupported target_transform: {mode}")


def invert_target(pred_target, base_values, transform_cfg):
    pred = np.asarray(pred_target, dtype=np.float32)
    base = np.asarray(base_values, dtype=np.float32)

    mode = str(transform_cfg.get("mode", "log_ratio")).strip().lower()
    eps = np.float32(float(transform_cfg.get("eps", 1e-4)))

    corrected = np.full(len(base), np.nan, dtype=np.float32)

    if mode in {"additive_residual", "quantile_residual"}:
        m = np.isfinite(base) & np.isfinite(pred)
        corrected[m] = base[m] + pred[m]
        return clip_nonnegative(corrected)

    if mode == "log_ratio":
        m = np.isfinite(base) & np.isfinite(pred) & (base > -eps)
        corrected[m] = np.exp(np.log(base[m] + eps) + pred[m]) - eps
        return clip_nonnegative(corrected)

    raise ValueError(f"Unsupported inverse target transform: {mode}")


def protect_tail_residuals(pred_target, percentiles, transform_cfg):
    pred = np.asarray(pred_target, dtype=np.float32).copy()
    pct = np.asarray(percentiles, dtype=np.float32)

    if not bool(transform_cfg.get("tail_residual_protection_enabled", False)):
        return pred

    start = float(transform_cfg.get("tail_residual_start", 0.95))
    end = float(transform_cfg.get("tail_residual_end", 0.999))
    min_scale = float(transform_cfg.get("tail_residual_min_scale", 0.25))
    mode = str(
        transform_cfg.get("tail_residual_protection_mode", "sign_aware")
    ).strip().lower()
    min_scale = min(max(min_scale, 0.0), 1.0)

    m = np.isfinite(pred) & np.isfinite(pct)
    if not np.any(m):
        return pred

    if mode == "negative_only":
        m = m & (pred < 0.0)
    elif mode == "positive_only":
        m = m & (pred > 0.0)
    elif mode == "symmetric":
        pass
    else:
        qmap = transform_cfg.get("quantile_mapping", {}) or {}
        tail_bias = qmap.get("tail_bias_pool", np.nan)
        if not np.isfinite(tail_bias):
            tail_bias = qmap.get("right_tail_bias", np.nan)
        if not np.isfinite(tail_bias) or abs(float(tail_bias)) < 1e-8:
            return pred
        if float(tail_bias) > 0.0:
            m = m & (pred < 0.0)
        else:
            m = m & (pred > 0.0)

    if not np.any(m):
        return pred

    alpha = np.clip((pct[m] - start) / max(end - start, 1e-6), 0.0, 1.0)
    scale = 1.0 - alpha * (1.0 - min_scale)
    pred[m] = (pred[m] * scale).astype(np.float32)
    return pred


def build_tail_sample_weights(obs_values, cfg, dtype=np.float32):
    obs = np.asarray(obs_values, dtype=float)
    valid = np.isfinite(obs)
    weights = np.ones(len(obs), dtype=float)

    if np.sum(valid) >= 20:
        q90 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q90_quantile", 0.90)))
        q95 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q95_quantile", 0.95)))
        q99 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q99_quantile", 0.99)))
        weights[valid & (obs >= q90)] = cfg_float(cfg, "tail_weight_q90", 2.0)
        weights[valid & (obs >= q95)] = cfg_float(cfg, "tail_weight_q95", 3.0)
        weights[valid & (obs >= q99)] = cfg_float(cfg, "tail_weight_q99", 5.0)

    return weights.astype(dtype, copy=False)
