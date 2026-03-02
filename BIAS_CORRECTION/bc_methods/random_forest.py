import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _resolve_feature_columns(df, candidate_cols):
    cols = []
    for col in candidate_cols:
        if col in df.columns:
            cols.append(col)
    if not cols:
        raise ValueError("No RF feature columns available in dataframe.")
    return cols


def _prepare_features(df, feature_cols, fill_values=None):
    x_df = df[feature_cols].copy()

    if fill_values is None:
        fill_values = {}
        for col in feature_cols:
            median = float(np.nanmedian(x_df[col].values))
            if not np.isfinite(median):
                median = 0.0
            fill_values[col] = median

    for col in feature_cols:
        x_df[col] = x_df[col].fillna(fill_values[col])

    return x_df.values, fill_values


def fit_rf_residual(
    df,
    hs_model,
    hs_obs,
    feature_cols=None,
    random_state=42
):
    if feature_cols is None:
        feature_cols = [hs_model]

    feature_cols = _resolve_feature_columns(df, feature_cols)
    x, fill_values = _prepare_features(df, feature_cols)
    y = (df[hs_obs].values - df[hs_model].values)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(x, y)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "fill_values": fill_values
    }


def apply_rf_residual(df, rf_bundle, hs_model, out_col="hs_rf"):
    out = df.copy()
    x, _ = _prepare_features(
        out,
        rf_bundle["feature_cols"],
        fill_values=rf_bundle["fill_values"]
    )
    residual = rf_bundle["model"].predict(x)
    out[out_col] = out[hs_model].values + residual
    return out


def run_rf(df, hs_model, hs_obs, train_ratio=0.7):
    df = df.copy()
    n = len(df)
    split = int(n * train_ratio)

    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()

    rf_bundle = fit_rf_residual(df_train, hs_model, hs_obs, feature_cols=[hs_model])
    df_test = apply_rf_residual(df_test, rf_bundle, hs_model, out_col="hs_ml")

    return df_test, rf_bundle["model"]
