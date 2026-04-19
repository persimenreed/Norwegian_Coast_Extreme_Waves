import numpy as np


RMSE_QUANTILES = (0.25, 0.50, 0.75, 0.95, 0.99, 0.995)


def _clean(model, obs):
    model = np.asarray(model, float)
    obs = np.asarray(obs, float)
    m = np.isfinite(model) & np.isfinite(obs)
    return model[m], obs[m]


def rmse(model, obs):
    model, obs = _clean(model, obs)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((model - obs) ** 2)))


def quantile_rmse(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) < 20:
        return np.nan

    thr = np.nanquantile(obs, q)
    mask = obs >= thr

    if np.sum(mask) < 20:
        return np.nan

    return rmse(model[mask], obs[mask])


def exceed_rate_bias(model, obs, q):
    model, obs = _clean(model, obs)
    if len(obs) < 20:
        return np.nan

    thr = np.nanquantile(obs, q)

    a = np.mean(model >= thr)
    b = np.mean(obs >= thr)

    return float(a - b)


def _quantile_suffix(q: float) -> str:
    scaled = int(round(float(q) * 1000))
    return str(scaled // 10) if scaled % 10 == 0 else str(scaled)


def compute_metrics(name, model, obs):
    model, obs = _clean(model, obs)

    metrics = {
        "method": name,
        "n": int(len(obs)),
        "rmse": rmse(model, obs),
    }

    for q in RMSE_QUANTILES:
        suffix = _quantile_suffix(q)
        metrics[f"rmse_q{suffix}"] = quantile_rmse(model, obs, q)

    metrics["exceed_rate_bias_q95"] = exceed_rate_bias(model, obs, 0.95)
    metrics["exceed_rate_bias_q99"] = exceed_rate_bias(model, obs, 0.99)

    return metrics
