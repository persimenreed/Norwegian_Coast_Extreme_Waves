import numpy as np
import pandas as pd

from src.settings import get_method_settings
from src.bias_correction.methods.common import (
    TIME,
    HS_MODEL,
    HS_OBS,
    prepare_ml_dataframe,
    resolve_feature_columns,
    clip_nonnegative,
)


def _cfg_float(cfg, key, default):
    return float(cfg.get(key, default))


def _cfg_str(cfg, key, default):
    return str(cfg.get(key, default))


def _prepare_features(df, feature_cols, fill=None):
    X = prepare_ml_dataframe(df)[feature_cols].copy()

    if fill is None:
        fill = {}
        for col in feature_cols:
            m = float(np.nanmedian(pd.to_numeric(X[col], errors="coerce").values))
            if not np.isfinite(m):
                m = 0.0
            fill[col] = m

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill[col])

    return X.values.astype(np.float32), fill


def _build_target(obs_values, raw_values, cfg):
    obs = np.asarray(obs_values, dtype=np.float32)
    raw = np.asarray(raw_values, dtype=np.float32)

    mode = _cfg_str(cfg, "target_transform", "log_ratio").strip().lower()
    eps = np.float32(_cfg_float(cfg, "target_eps", 1e-4))

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


def _invert_target(pred_target, raw_values, transform_cfg):
    pred = np.asarray(pred_target, dtype=np.float32)
    raw = np.asarray(raw_values, dtype=np.float32)

    mode = str(transform_cfg.get("mode", "log_ratio")).strip().lower()
    eps = np.float32(float(transform_cfg.get("eps", 1e-4)))

    corrected = np.full(len(raw), np.nan, dtype=np.float32)

    if mode == "additive_residual":
        m = np.isfinite(raw) & np.isfinite(pred)
        corrected[m] = raw[m] + pred[m]
        return clip_nonnegative(corrected)

    if mode == "log_ratio":
        m = np.isfinite(raw) & np.isfinite(pred) & (raw > -eps)
        corrected[m] = np.exp(np.log(raw[m] + eps) + pred[m]) - eps
        return clip_nonnegative(corrected)

    raise ValueError(f"Unsupported inverse target transform: {mode}")


def _build_sample_weights(obs_values, cfg):
    obs = np.asarray(obs_values, dtype=float)
    m = np.isfinite(obs)
    w = np.ones(len(obs), dtype=float)

    if np.sum(m) >= 20:
        q90 = np.nanquantile(obs[m], float(cfg.get("tail_weight_q90_quantile", 0.90)))
        q95 = np.nanquantile(obs[m], float(cfg.get("tail_weight_q95_quantile", 0.95)))
        q99 = np.nanquantile(obs[m], float(cfg.get("tail_weight_q99_quantile", 0.99)))
        w[m & (obs >= q90)] = _cfg_float(cfg, "tail_weight_q90", 2.0)
        w[m & (obs >= q95)] = _cfg_float(cfg, "tail_weight_q95", 3.0)
        w[m & (obs >= q99)] = _cfg_float(cfg, "tail_weight_q99", 5.0)

    return w


def fit(df, settings_name=None):
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from e

    if not settings_name:
        raise ValueError("settings_name must be provided for XGBoost training.")

    cfg = get_method_settings(settings_name)
    if not cfg:
        raise ValueError(f"Missing XGBoost settings block '{settings_name}'.")

    work = df.copy()
    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        work = work.sort_values(TIME).reset_index(drop=True)

    work = prepare_ml_dataframe(work)
    features = resolve_feature_columns(work, cfg.get("features", []))

    obs = pd.to_numeric(work[HS_OBS], errors="coerce").values.astype(np.float32)
    raw = pd.to_numeric(work[HS_MODEL], errors="coerce").values.astype(np.float32)
    y, valid_target, transform_cfg = _build_target(obs, raw, cfg)

    valid_idx = np.flatnonzero(valid_target)

    min_train = int(cfg.get("min_train_samples", 50))
    if len(valid_idx) < min_train:
        raise ValueError("Too few valid samples for XGBoost.")

    X_all, fill = _prepare_features(work, features)
    weights_all = _build_sample_weights(obs, cfg)

    X_train = X_all[valid_idx]
    y_train = y[valid_idx]
    w_train = weights_all[valid_idx]

    model = XGBRegressor(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        subsample=float(cfg.get("subsample", 0.8)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        gamma=float(cfg.get("gamma", 0.0)),
        min_child_weight=float(cfg.get("min_child_weight", 1.0)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        random_state=int(cfg.get("random_state", 1)),
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "features": features,
        "fill": fill,
        "model": model,
        "feature_importance": importance,
        "target_transform": transform_cfg,
    }


def apply(df, bundle):
    out = df.copy()
    if TIME in out.columns:
        out[TIME] = pd.to_datetime(out[TIME], errors="coerce")
        out = out.sort_values(TIME).reset_index(drop=False).rename(columns={"index": "_orig_index"})
    else:
        out["_orig_index"] = np.arange(len(out))

    prepared = prepare_ml_dataframe(out.copy())
    X, _ = _prepare_features(prepared, bundle["features"], bundle["fill"])

    residual = bundle["model"].predict(X)
    hs = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(np.float32)
    prepared[HS_MODEL] = _invert_target(
        residual,
        hs,
        bundle.get("target_transform", {"mode": "additive_residual", "eps": 1e-4}),
    )

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared
