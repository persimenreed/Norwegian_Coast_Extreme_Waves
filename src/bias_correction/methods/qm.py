import numpy as np
from scipy import stats
from src.settings import get_columns

EPS = 1e-12

_COLUMNS = get_columns()

HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")
PER_MODEL = _COLUMNS.get("per_model", "tm2")
PER_OBS = _COLUMNS.get("per_obs", "Wave_Mean_Period_Tm02")


def _clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def fit_dist(data, dist_name):

    data = _clean_positive(data)

    if len(data) < 50:
        raise ValueError("Too few samples for distribution fit")

    if dist_name == "weibull":
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
        return stats.weibull_min(c, loc=loc, scale=scale)

    if dist_name == "lognormal":
        s, loc, scale = stats.lognorm.fit(data, floc=0)
        return stats.lognorm(s, loc=loc, scale=scale)

    raise ValueError("Unsupported distribution")


def apply_quantile_map(values, source_dist, target_dist):

    q = source_dist.cdf(values)
    q = np.clip(q, EPS, 1 - EPS)

    return target_dist.ppf(q)


# --------------------------------------------------
# ORIGINAL QM MODEL (unchanged)
# --------------------------------------------------

def fit_qm(df):

    hs_source = fit_dist(df[HS_MODEL].values, "weibull")
    hs_target = fit_dist(df[HS_OBS].values, "weibull")

    model = {
        "hs_source_dist": hs_source,
        "hs_target_dist": hs_target,
    }

    if PER_MODEL in df.columns and PER_OBS in df.columns:

        per_source = fit_dist(df[PER_MODEL].values, "lognormal")
        per_target = fit_dist(df[PER_OBS].values, "lognormal")

        model["tm2_source_dist"] = per_source
        model["tm2_target_dist"] = per_target

    return model


def apply_qm(df, model):

    out = df.copy()

    corrected = apply_quantile_map(
        out[HS_MODEL].values,
        model["hs_source_dist"],
        model["hs_target_dist"],
    )

    out[HS_MODEL] = corrected

    return out


# --------------------------------------------------
# WRAPPER FUNCTIONS USED BY PIPELINE
# --------------------------------------------------

def fit(df):
    return fit_qm(df)


def apply(df, model):
    return apply_qm(df, model)