import warnings
import numpy as np
from scipy import stats

from src.bias_correction.methods.common import (
    HS_MODEL,
    HS_OBS,
    clip_nonnegative,
)

EPS = 1e-10


def _clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def _fit_candidate(data, dist_name):
    data = _clean_positive(data)

    if len(data) < 40:
        raise ValueError("Too few positive samples for parametric fit.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if dist_name == "weibull":
            c, loc, scale = stats.weibull_min.fit(data, floc=0)
            dist = stats.weibull_min(c, loc=loc, scale=scale)

        elif dist_name == "lognormal":
            s, loc, scale = stats.lognorm.fit(data, floc=0)
            dist = stats.lognorm(s, loc=loc, scale=scale)

        elif dist_name == "gamma":
            a, loc, scale = stats.gamma.fit(data, floc=0)
            dist = stats.gamma(a, loc=loc, scale=scale)

        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

    logpdf = dist.logpdf(data)
    ll = np.sum(logpdf[np.isfinite(logpdf)])

    # 3 params for all families here
    k = 3
    aic = 2 * k - 2 * ll

    return dist, aic


def _select_best_dist(data, candidates):
    best = None

    for name in candidates:
        try:
            dist, aic = _fit_candidate(data, name)
        except Exception:
            continue

        if best is None or aic < best["aic"]:
            best = {"name": name, "dist": dist, "aic": aic}

    if best is None:
        raise ValueError("Could not fit any candidate distribution.")

    return best


def _apply_quantile_map(values, source_dist, target_dist):
    x = np.asarray(values, float)
    out = np.full_like(x, np.nan, dtype=float)

    m = np.isfinite(x) & (x > 0)
    if not np.any(m):
        return out

    q = source_dist.cdf(x[m])
    q = np.clip(q, EPS, 1 - EPS)
    out[m] = target_dist.ppf(q)
    return out


def _fit_variable(df, source_col, target_col, candidates):
    source_best = _select_best_dist(df[source_col].values, candidates)
    target_best = _select_best_dist(df[target_col].values, candidates)

    return {
        "source_name": source_best["name"],
        "source_dist": source_best["dist"],
        "target_name": target_best["name"],
        "target_dist": target_best["dist"],
    }


def fit(df):
    return {
        "hs": _fit_variable(df, HS_MODEL, HS_OBS, ["weibull", "gamma", "lognormal"])
    }


def apply(df, model):
    out = df.copy()

    hs_corr = _apply_quantile_map(
        out[HS_MODEL].values,
        model["hs"]["source_dist"],
        model["hs"]["target_dist"],
    )
    out[HS_MODEL] = clip_nonnegative(hs_corr)

    return out