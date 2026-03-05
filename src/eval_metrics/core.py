import numpy as np

def twrmse(model, obs):
    """
    Threshold-weighted RMSE.
    Gives larger weight to larger observed values.
    """

    model = np.asarray(model, float)
    obs = np.asarray(obs, float)

    m = np.isfinite(model) & np.isfinite(obs)

    if np.sum(m) < 20:
        return np.nan

    model = model[m]
    obs = obs[m]

    w = obs / np.nanmean(obs)

    return float(
        np.sqrt(
            np.nanmean(w * (model - obs) ** 2)
        )
    )

def quantile_score(model, obs, q):

    model = np.asarray(model, float)
    obs = np.asarray(obs, float)

    e = obs - model

    return float(np.nanmean(
        np.maximum(q * e, (q - 1) * e)
    ))

def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))

def mae(a, b):
    return float(np.nanmean(np.abs(a - b)))


def tail_rmse(model, obs, q):

    thr = np.nanquantile(obs, q)
    mask = obs >= thr

    if mask.sum() < 20:
        return np.nan

    return rmse(model[mask], obs[mask])


def q_bias(model, obs, q):

    if len(obs) < 20:
        return np.nan

    return float(
        np.nanquantile(model, q) -
        np.nanquantile(obs, q)
    )


def exceed_rate_bias(model, obs, q):

    thr = np.nanquantile(obs, q)

    a = np.mean(model >= thr)
    b = np.mean(obs >= thr)

    return float(a - b)


def compute_metrics(name, model, obs):

    model = np.asarray(model, float)
    obs = np.asarray(obs, float)

    m = np.isfinite(model) & np.isfinite(obs)

    model = model[m]
    obs = obs[m]

    return {
        "method": name,

        "mae": mae(model, obs),
        "rmse": rmse(model, obs),

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