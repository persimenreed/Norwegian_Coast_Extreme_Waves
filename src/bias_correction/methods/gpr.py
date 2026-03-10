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


def _fit_fill_and_scaler(df, feature_cols):
    fill = {}
    mean = {}
    std = {}

    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        med = float(np.nanmedian(vals))
        if not np.isfinite(med):
            med = 0.0
        fill[col] = med

        vals = np.where(np.isfinite(vals), vals, med)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        mean[col] = mu
        std[col] = sigma

    return fill, mean, std


def _transform_features(df, feature_cols, fill, mean, std):
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float64)

    for j, col in enumerate(feature_cols):
        vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        vals = np.where(np.isfinite(vals), vals, fill[col])
        vals = (vals - mean[col]) / std[col]
        X[:, j] = vals

    return X


def fit(df):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            RBF,
            RationalQuadratic,
            WhiteKernel,
        )
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for GPR. Install it with: pip install scikit-learn"
        ) from e

    cfg = get_method_settings("gpr")

    work = df.copy()
    if TIME in work.columns:
        work[TIME] = pd.to_datetime(work[TIME], errors="coerce")
        work = work.sort_values(TIME).reset_index(drop=True)

    work = prepare_ml_dataframe(work)
    features = resolve_feature_columns(work, cfg.get("features", []))

    y = pd.to_numeric(work[HS_OBS], errors="coerce").values - pd.to_numeric(
        work[HS_MODEL], errors="coerce"
    ).values
    valid = np.isfinite(y)

    if np.sum(valid) < 30:
        raise ValueError("Too few valid samples for GPR.")

    fit_df = work.loc[valid].reset_index(drop=True)
    y_fit = y[valid].astype(float)

    fill, mean, std = _fit_fill_and_scaler(fit_df, features)
    X_fit = _transform_features(fit_df, features, fill, mean, std)

    max_train = int(cfg.get("max_train_samples", 1500))
    if len(X_fit) > max_train:
        idx = np.linspace(0, len(X_fit) - 1, max_train).astype(int)
        X_fit = X_fit[idx]
        y_fit = y_fit[idx]

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * (RBF(length_scale=np.ones(X_fit.shape[1])) + RationalQuadratic())
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=int(cfg.get("n_restarts_optimizer", 2)),
        random_state=int(cfg.get("random_state", 1)),
    )

    model.fit(X_fit, y_fit)

    return {
        "features": features,
        "fill": fill,
        "mean": mean,
        "std": std,
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
    X = _transform_features(
        prepared,
        bundle["features"],
        bundle["fill"],
        bundle["mean"],
        bundle["std"],
    )

    residual = bundle["model"].predict(X)
    hs = pd.to_numeric(prepared[HS_MODEL], errors="coerce").values.astype(float)
    prepared[HS_MODEL] = clip_nonnegative(hs + residual)

    prepared = prepared.sort_values("_orig_index").drop(columns=["_orig_index"])
    prepared.index = range(len(prepared))
    return prepared