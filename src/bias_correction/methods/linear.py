import numpy as np
from src.bias_correction.methods.common import (
    HS_MODEL,
    HS_OBS,
    finite_pair_mask,
    clip_nonnegative,
)


def fit(df):
    x = df[HS_MODEL].values
    y = df[HS_OBS].values

    m = finite_pair_mask(x, y)

    if m.sum() < 20:
        raise ValueError("Too few valid samples to fit linear scaling.")

    slope, intercept = np.polyfit(x[m], y[m], 1)

    return {"slope": float(slope), "intercept": float(intercept)}


def apply(df, model):
    out = df.copy()
    corrected = model["intercept"] + model["slope"] * out[HS_MODEL].values
    out[HS_MODEL] = clip_nonnegative(corrected)
    return out