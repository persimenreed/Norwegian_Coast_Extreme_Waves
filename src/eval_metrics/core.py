import numpy as np


def _clean(model, obs):
    model = np.asarray(model, float)
    obs = np.asarray(obs, float)
    m = np.isfinite(model) & np.isfinite(obs)
    return model[m], obs[m]


def bias(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan
    return float(np.nanmean(model - obs))


def rmse(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((model - obs) ** 2)))


def mae(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan
    return float(np.nanmean(np.abs(model - obs)))


def corrcoef(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) < 3:
        return np.nan
    if np.nanstd(model) == 0 or np.nanstd(obs) == 0:
        return np.nan
    return float(np.corrcoef(model, obs)[0, 1])


def scatter_index(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan
    mean_obs = float(np.nanmean(obs))
    if mean_obs == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((model - obs) ** 2)) / mean_obs)


def twrmse(model, obs):
    model, obs = _clean(model, obs)

    if len(obs) < 20:
        return np.nan

    mean_obs = np.nanmean(obs)
    if not np.isfinite(mean_obs) or mean_obs == 0:
        return np.nan

    w = obs / mean_obs
    return float(np.sqrt(np.nanmean(w * (model - obs) ** 2)))


def quantile_score(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan

    e = obs - model
    return float(np.nanmean(np.maximum(q * e, (q - 1) * e)))


def tail_rmse(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) < 20:
        return np.nan

    thr = np.nanquantile(obs, q)
    mask = obs >= thr

    if np.sum(mask) < 20:
        return np.nan

    return rmse(model[mask], obs[mask])


def q_bias(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) < 20:
        return np.nan

    return float(
        np.nanquantile(model, q) -
        np.nanquantile(obs, q)
    )


def exceed_rate_bias(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) < 20:
        return np.nan

    thr = np.nanquantile(obs, q)

    a = np.mean(model >= thr)
    b = np.mean(obs >= thr)

    return float(a - b)


def compute_metrics(name, model, obs):
    model, obs = _clean(model, obs)

    return {
        "method": name,
        "n": int(len(obs)),

        "bias": bias(model, obs),
        "mae": mae(model, obs),
        "rmse": rmse(model, obs),
        "corr": corrcoef(model, obs),
        "scatter_index": scatter_index(model, obs),

        "tail_rmse_95": tail_rmse(model, obs, 0.95),
        "tail_rmse_99": tail_rmse(model, obs, 0.99),

        "twrmse": twrmse(model, obs),

        "q95_bias": q_bias(model, obs, 0.95),
        "q99_bias": q_bias(model, obs, 0.99),
        "q995_bias": q_bias(model, obs, 0.995),

        "exceed_rate_bias_q95": exceed_rate_bias(model, obs, 0.95),
        "exceed_rate_bias_q99": exceed_rate_bias(model, obs, 0.99),

        "quantile_score_95": quantile_score(model, obs, 0.95),
        "quantile_score_99": quantile_score(model, obs, 0.99),
        "quantile_score_995": quantile_score(model, obs, 0.995),
    }