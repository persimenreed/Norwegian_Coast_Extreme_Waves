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


def _build_sample_weights(df):
    hs = pd.to_numeric(df[HS_MODEL], errors="coerce").values.astype(float)
    m = np.isfinite(hs)
    w = np.ones(len(df), dtype=float)

    if np.sum(m) >= 20:
        q90 = np.nanquantile(hs[m], 0.90)
        q95 = np.nanquantile(hs[m], 0.95)
        w[m & (hs >= q90)] = 2.0
        w[m & (hs >= q95)] = 3.0

    return w


def fit(df):
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from e

    cfg = get_method_settings("xgboost")

    work = df.copy()
    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        work = work.sort_values(TIME).reset_index(drop=True)

    work = prepare_ml_dataframe(work)
    features = resolve_feature_columns(work, cfg.get("features", []))

    y = pd.to_numeric(work[HS_OBS], errors="coerce").values - pd.to_numeric(
        work[HS_MODEL], errors="coerce"
    ).values

    valid_target = np.isfinite(y)
    valid_idx = np.flatnonzero(valid_target)

    min_train = int(cfg.get("min_train_samples", 50))
    if len(valid_idx) < min_train:
        raise ValueError("Too few valid samples for XGBoost.")

    X_all, fill = _prepare_features(work, features)
    weights_all = _build_sample_weights(work)

    X_train = X_all[valid_idx]
    y_train = y[valid_idx]
    w_train = weights_all[valid_idx]

    model = XGBRegressor(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        subsample=float(cfg.get("subsample", 0.8)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        random_state=int(cfg.get("random_state", 1)),
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    return {
        "features": features,
        "fill": fill,
        "model": model,
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
    hs = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values
    prepared[HS_MODEL] = clip_nonnegative(hs + residual)

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared