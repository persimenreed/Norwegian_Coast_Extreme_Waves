import numpy as np
from scipy import stats

EPS = 1e-12


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


def quantile_map(model_values, target_values, dist_name):
    m_fit = fit_dist(model_values, dist_name)
    t_fit = fit_dist(target_values, dist_name)
    return apply_quantile_map(model_values, m_fit, t_fit)


def fit_qm(df, hs_model, hs_obs, per_model=None, per_obs=None):
    hs_source = fit_dist(df[hs_model].values, "weibull")
    hs_target = fit_dist(df[hs_obs].values, "weibull")

    model = {
        "hs_source_dist": hs_source,
        "hs_target_dist": hs_target,
    }

    if per_model and per_obs:
        per_source = fit_dist(df[per_model].values, "lognormal")
        per_target = fit_dist(df[per_obs].values, "lognormal")
        model["tm2_source_dist"] = per_source
        model["tm2_target_dist"] = per_target

    return model


def apply_qm(df, model, hs_model, per_model=None):
    out = df.copy()
    out["hs_qm"] = apply_quantile_map(
        out[hs_model].values,
        model["hs_source_dist"],
        model["hs_target_dist"]
    )

    if per_model and "tm2_source_dist" in model and "tm2_target_dist" in model:
        out["tm2_qm"] = apply_quantile_map(
            out[per_model].values,
            model["tm2_source_dist"],
            model["tm2_target_dist"]
        )

    return out


def run_qm(df, hs_model, hs_obs, per_model, per_obs):
    model = fit_qm(df, hs_model, hs_obs, per_model, per_obs)
    return apply_qm(df, model, hs_model, per_model)
