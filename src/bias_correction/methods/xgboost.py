import numpy as np
import pandas as pd

from src.settings import get_method_settings
from src.bias_correction.methods.common import (
    TIME,
    HS_MODEL,
    HS_OBS,
    HS_QUANTILE_BASELINE,
    prepare_ml_dataframe,
    resolve_feature_columns,
    quantile_feature_columns,
    augment_quantile_features,
    build_target_transform,
    invert_target,
    build_tail_sample_weights,
)


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

    obs = pd.to_numeric(work[HS_OBS], errors="coerce").values.astype(np.float32)
    raw = pd.to_numeric(work[HS_MODEL], errors="coerce").values.astype(np.float32)
    y, valid_target, transform_cfg = build_target_transform(obs, raw, cfg)

    if transform_cfg["mode"] == "quantile_residual":
        work, _ = augment_quantile_features(
            work,
            raw,
            transform_cfg,
            reference_values=raw,
        )

    features = resolve_feature_columns(work, cfg.get("features", []))
    if transform_cfg["mode"] == "quantile_residual":
        features = quantile_feature_columns(features)

    valid_idx = np.flatnonzero(valid_target)

    min_train = int(cfg.get("min_train_samples", 50))
    if len(valid_idx) < min_train:
        raise ValueError("Too few valid samples for XGBoost.")

    X_all, fill = _prepare_features(work, features)
    weights_all = build_tail_sample_weights(obs, cfg, dtype=float)

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
    transform_cfg = bundle.get(
        "target_transform",
        {"mode": "additive_residual", "eps": 1e-4},
    )

    if transform_cfg.get("mode") == "quantile_residual":
        raw = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(np.float32)
        prepared, extras = augment_quantile_features(
            prepared,
            raw,
            transform_cfg,
            reference_values=raw,
        )
        base_values = extras[HS_QUANTILE_BASELINE]
    else:
        base_values = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(np.float32)

    X, _ = _prepare_features(prepared, bundle["features"], bundle["fill"])

    residual = bundle["model"].predict(X)
    prepared[HS_MODEL] = invert_target(
        residual,
        base_values,
        transform_cfg,
    )

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared
