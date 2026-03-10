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

    return X.values.astype(float), fill


def fit(df):
    try:
        from pygam import LinearGAM, s
    except ImportError as e:
        raise ImportError(
            "pyGAM is required for GAM. Install it with: pip install pygam"
        ) from e

    cfg = get_method_settings("gam")

    work = df.copy()
    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        work = work.sort_values(TIME).reset_index(drop=True)

    work = prepare_ml_dataframe(work)
    features = resolve_feature_columns(work, cfg.get("features", []))

    X, fill = _prepare_features(work, features)
    y = pd.to_numeric(work[HS_OBS], errors="coerce").values - pd.to_numeric(
        work[HS_MODEL], errors="coerce"
    ).values

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if np.sum(valid) < 30:
        raise ValueError("Too few valid samples for GAM.")

    terms = s(0, n_splines=int(cfg.get("n_splines", 10)), spline_order=int(cfg.get("spline_order", 3)))
    for i in range(1, X.shape[1]):
        terms = terms + s(
            i,
            n_splines=int(cfg.get("n_splines", 10)),
            spline_order=int(cfg.get("spline_order", 3)),
        )

    gam = LinearGAM(terms)

    lam_grid = cfg.get("lam_grid", [0.1, 1.0, 10.0])
    gam.gridsearch(X[valid], y[valid], lam=lam_grid, progress=False)

    return {
        "features": features,
        "fill": fill,
        "model": gam,
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
    hs = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(float)
    prepared[HS_MODEL] = clip_nonnegative(hs + residual)

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared