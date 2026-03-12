from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from src.settings import get_methods, format_path

TIME = "time"
OBS = "Significant_Wave_Height_Hm0"
MODEL = "hs"
MODEL_CORR = "hs_corrected"


# -------------------------------------------------------
# utilities
# -------------------------------------------------------

def _validation_path(location, method):
    return Path(format_path("validation", location=location, corr_method=f"pooled_{method}"))


def _corrected_path(location, method):
    return Path(format_path("corrected", location=location, corr_method=f"pooled_{method}"))


def _load_validation_table(location, methods):

    merged = None

    for m in tqdm(methods, desc=f"Loading pooled validation members ({location})"):

        path = _validation_path(location, m)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df[TIME] = pd.to_datetime(df[TIME], errors="coerce")

        part = df[[TIME, OBS, MODEL_CORR]].rename(
            columns={
                OBS: "obs",
                MODEL_CORR: m,
            }
        )

        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[[TIME, m]], on=TIME)

    return merged.dropna()


# -------------------------------------------------------
# objective
# -------------------------------------------------------

def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _twrmse(pred, obs):
    mean_obs = np.mean(obs)
    if mean_obs == 0:
        return np.nan
    w = obs / mean_obs
    return np.sqrt(np.mean(w * (pred - obs) ** 2))


def _tail_rmse(pred, obs, q=0.95):
    thr = np.quantile(obs, q)
    m = obs >= thr
    if np.sum(m) < 20:
        return np.nan
    return _rmse(pred[m], obs[m])


def _q_bias(pred, obs, q=0.95):
    return np.quantile(pred, q) - np.quantile(obs, q)


def _loss(w, X, y):
    pred = X @ w
    rmse = _rmse(pred, y)
    twrmse = _twrmse(pred, y)
    tail = _tail_rmse(pred, y)
    qbias = abs(_q_bias(pred, y))
    # normalize to avoid scale issues
    mean_obs = np.mean(y)
    q95_obs = np.quantile(y, 0.95)

    rmse /= mean_obs
    twrmse /= mean_obs
    tail = tail / q95_obs if np.isfinite(tail) else rmse
    qbias /= q95_obs

    return (
        0.35 * rmse
        + 0.35 * twrmse
        + 0.20 * tail
        + 0.10 * qbias
    )


# -------------------------------------------------------
# weight fitting
# -------------------------------------------------------

def fit_weights(location, methods):

    df = _load_validation_table(location, methods)

    X = df[methods].values
    y = df["obs"].values

    n = len(methods)

    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    res = minimize(_loss, w0, args=(X, y), bounds=bounds, constraints=cons)

    w = res.x / np.sum(res.x)

    return dict(zip(methods, w))


# -------------------------------------------------------
# application
# -------------------------------------------------------

def apply(location, methods, weights):

    merged = None
    base = None

    for m in tqdm(methods, desc=f"Loading pooled hindcast members ({location})"):

        path = _corrected_path(location, m)

        df = pd.read_csv(path)
        df[TIME] = pd.to_datetime(df[TIME], errors="coerce")

        if base is None:
            base = df.copy()

        part = df[[TIME, MODEL]].rename(columns={MODEL: m})

        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=TIME)

    X = merged[methods].values
    w = np.array([weights[m] for m in methods])

    ensemble = X @ w

    out = base.merge(
        merged[[TIME]].assign(hs_ensemble=ensemble),
        on=TIME,
    )

    out[MODEL] = out["hs_ensemble"]
    out = out.drop(columns="hs_ensemble")

    return out


# -------------------------------------------------------
# public API
# -------------------------------------------------------

def run(location="vestfjorden", methods=None):

    if methods is None:
        methods = [m for m in get_methods() if m != "ensemble"]

    weights = fit_weights(location, methods)

    df = apply(location, methods, weights)

    out_path = Path(format_path("corrected", location=location, corr_method="ensemble"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)

    return {
        "location": location,
        "weights": weights,
        "hindcast_path": str(out_path),
        "training_mode": "pooled_validation",
        "application_mode": "pooled_member_level",
    }