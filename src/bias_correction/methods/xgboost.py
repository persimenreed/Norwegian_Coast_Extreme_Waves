from copy import deepcopy

import pandas as pd

from src.model_profiles import resolve_profile
from src.bias_correction.methods.common import (
    HS_MODEL,
    HS_OBS,
    HS_QUANTILE,
    HS_QUANTILE_BASELINE,
    augment_quantile_features,
    build_tail_sample_weights,
    build_target_transform,
    feature_matrix,
    invert_target,
    numeric_values,
    prepare_ml_dataframe,
    protect_tail_residuals,
    quantile_feature_columns,
    resolve_feature_columns,
    restore_frame_order,
    sort_frame,
)

DEFAULT_XGBOOST_CONFIG = {
    "features": [],
    "min_train_samples": 50,
    "quantile_bias_mode": "additive",
    "target_eps": 1e-5,
    "tail_weight_q90": 3.0,
    "tail_weight_q95": 6.0,
    "tail_weight_q99": 14.0,
    "tail_pool_enabled": True,
    "tail_pool_start": 0.95,
    "tail_pool_end": 0.995,
    "tail_blend_start": 0.95,
    "tail_residual_protection_enabled": True,
    "tail_residual_protection_mode": "sign_aware",
    "tail_residual_start": 0.95,
    "tail_residual_end": 0.999,
    "tail_residual_min_scale": 0.20,
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.88,
    "gamma": 1.0,
    "min_child_weight": 1.0,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "random_state": 1,
}

def _resolve_config(settings_name=None):
    return resolve_profile(deepcopy(DEFAULT_XGBOOST_CONFIG), "xgboost", settings_name)


def fit(df, settings_name=None):
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from error

    cfg = _resolve_config(settings_name)
    work = prepare_ml_dataframe(sort_frame(df))

    obs = numeric_values(work, HS_OBS, dtype="float32")
    raw = numeric_values(work, HS_MODEL, dtype="float32")
    target, valid_target, transform_cfg = build_target_transform(obs, raw, cfg)

    work, _ = augment_quantile_features(work, raw, transform_cfg, reference_values=raw)

    features = resolve_feature_columns(work, cfg.get("features"))
    features = quantile_feature_columns(features)

    valid_idx = valid_target.nonzero()[0]
    if len(valid_idx) < int(cfg.get("min_train_samples", 50)):
        raise ValueError("Too few valid samples for XGBoost.")

    X_all, fill = feature_matrix(work, features)
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
    model.fit(
        X_all[valid_idx],
        target[valid_idx],
        sample_weight=build_tail_sample_weights(obs, cfg, dtype=float)[valid_idx],
        verbose=False,
    )

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "features": features,
        "fill": fill,
        "model": model,
        "feature_importance": importance,
        "target_transform": transform_cfg,
    }


def apply(df, bundle):
    prepared = prepare_ml_dataframe(sort_frame(df, preserve_order=True))
    transform_cfg = bundle["target_transform"]
    raw = numeric_values(prepared, HS_MODEL, dtype="float32")
    prepared, extras = augment_quantile_features(
        prepared,
        raw,
        transform_cfg,
        reference_values=raw,
    )
    base_values = extras[HS_QUANTILE_BASELINE]
    quantiles = extras[HS_QUANTILE]

    X, _ = feature_matrix(prepared, bundle["features"], fill=bundle["fill"])
    residual = bundle["model"].predict(X)
    residual = protect_tail_residuals(residual, quantiles, transform_cfg)
    prepared[HS_MODEL] = invert_target(residual, base_values, transform_cfg)
    return restore_frame_order(prepared)
