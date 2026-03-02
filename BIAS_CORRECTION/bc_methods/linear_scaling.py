import numpy as np


def fit_linear_scaling(df, hs_model, hs_obs):
    x = df[hs_model].values
    y = df[hs_obs].values

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        raise ValueError("Too few valid samples to fit linear scaling.")

    slope, intercept = np.polyfit(x[m], y[m], 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def apply_linear_scaling(df, linear_model, hs_model, out_col="hs_linear"):
    out = df.copy()
    out[out_col] = (
        linear_model["intercept"] + linear_model["slope"] * out[hs_model].values
    )
    return out
