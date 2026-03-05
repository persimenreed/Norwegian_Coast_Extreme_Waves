import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.settings import get_columns

_COLUMNS = get_columns()

HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")

RF_FEATURES = [
    "hs",
    "tp",
    "tm2",
    "wind_speed_10m",
    "wind_speed_20m",
    "Pdir",
    "month_sin",
    "month_cos",
]


def _resolve_features(df):

    cols = [c for c in RF_FEATURES if c in df.columns]

    if not cols:
        raise ValueError("No RF features found")

    return cols


def _prepare_features(df, feature_cols, fill=None):

    X = df[feature_cols].copy()

    if fill is None:

        fill = {}

        for col in feature_cols:

            m = float(np.nanmedian(X[col].values))

            if not np.isfinite(m):
                m = 0.0

            fill[col] = m

    for col in feature_cols:
        X[col] = X[col].fillna(fill[col])

    return X.values, fill


def fit(df):

    features = _resolve_features(df)

    X, fill = _prepare_features(df, features)

    y = df[HS_OBS].values - df[HS_MODEL].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return {
        "model": model,
        "features": features,
        "fill": fill
    }


def apply(df, bundle):

    out = df.copy()

    X, _ = _prepare_features(
        out,
        bundle["features"],
        bundle["fill"]
    )

    residual = bundle["model"].predict(X)

    out[HS_MODEL] = out[HS_MODEL].values + residual

    return out