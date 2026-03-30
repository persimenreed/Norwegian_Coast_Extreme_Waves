import numpy as np

from src.bias_correction.methods.common import (
    HS_MODEL,
    HS_OBS,
    clip_nonnegative,
    feature_matrix,
    fit_standard_scaler,
    numeric_values,
    prepare_ml_dataframe,
    resolve_feature_columns,
    restore_frame_order,
    sort_frame,
)

GPR_CONFIG = {
    "max_train_samples": 1500,
    "random_state": 1,
    "n_restarts_optimizer": 2,
}


def fit(df, settings_name=None):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            RBF,
            RationalQuadratic,
            WhiteKernel,
        )
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for GPR. Install it with: pip install scikit-learn"
        ) from error

    work = prepare_ml_dataframe(sort_frame(df))
    features = resolve_feature_columns(work)

    residual = numeric_values(work, HS_OBS) - numeric_values(work, HS_MODEL)
    valid = np.isfinite(residual)
    if np.sum(valid) < 30:
        raise ValueError("Too few valid samples for GPR.")

    fit_df = work.loc[valid].reset_index(drop=True)
    fill, mean, std = fit_standard_scaler(fit_df, features)
    X_fit, _ = feature_matrix(fit_df, features, fill=fill, mean=mean, std=std, dtype=np.float64)
    y_fit = residual[valid].astype(float)

    if len(X_fit) > GPR_CONFIG["max_train_samples"]:
        idx = np.linspace(0, len(X_fit) - 1, GPR_CONFIG["max_train_samples"]).astype(int)
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
        n_restarts_optimizer=GPR_CONFIG["n_restarts_optimizer"],
        random_state=GPR_CONFIG["random_state"],
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
    prepared = prepare_ml_dataframe(sort_frame(df, preserve_order=True))
    X, _ = feature_matrix(
        prepared,
        bundle["features"],
        fill=bundle["fill"],
        mean=bundle["mean"],
        std=bundle["std"],
        dtype=np.float64,
    )
    prepared[HS_MODEL] = clip_nonnegative(
        numeric_values(prepared, HS_MODEL) + bundle["model"].predict(X)
    )
    return restore_frame_order(prepared)
