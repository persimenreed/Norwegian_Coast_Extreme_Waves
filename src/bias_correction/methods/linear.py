import numpy as np
from src.settings import get_columns

_COLUMNS = get_columns()

HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")


def fit(df):

    x = df[HS_MODEL].values
    y = df[HS_OBS].values

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 20:
        raise ValueError("Too few valid samples to fit linear scaling.")

    slope, intercept = np.polyfit(x[m], y[m], 1)

    return {"slope": float(slope), "intercept": float(intercept)}


def apply(df, model):

    out = df.copy()

    corrected = model["intercept"] + model["slope"] * out[HS_MODEL].values

    out[HS_MODEL] = corrected

    return out