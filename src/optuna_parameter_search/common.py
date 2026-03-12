import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contextlib
import copy
import numpy as np

from src.bias_correction.data import load_pairs
from src.settings import (
    get_core_buoy_locations,
    get_columns,
    load_settings
)

_COLUMNS = get_columns()

HS_MODEL = _COLUMNS.get("hs_model", "hs")
HS_OBS = _COLUMNS.get("hs_obs", "Significant_Wave_Height_Hm0")


# ---------------------------------------------------------
# Dataset cache (avoid reloading every Optuna trial)
# ---------------------------------------------------------

_core = get_core_buoy_locations()

if len(_core) != 2:
    raise RuntimeError(
        "Expected exactly two core buoys for spatial CV."
    )

_BUOY_A, _BUOY_B = _core

_DF_A = load_pairs(_BUOY_A)
_DF_B = load_pairs(_BUOY_B)


# ---------------------------------------------------------
# Temporary override of settings.yaml hyperparameters
# ---------------------------------------------------------

@contextlib.contextmanager
def override_method_settings(method_name, params):

    settings = load_settings()

    original = copy.deepcopy(settings["ml"].get(method_name, {}))

    settings["ml"][method_name].update(params)

    try:
        yield
    finally:
        settings["ml"][method_name] = original


# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------

def compute_rmse(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(m) == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


# ---------------------------------------------------------
# Quantile loss (Pinball loss)
# ---------------------------------------------------------

def compute_quantile_loss(y_true, y_pred, q=0.95):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(m) == 0:
        return np.nan

    err = y_true[m] - y_pred[m]

    loss = np.maximum(q * err, (q - 1) * err)

    return float(np.mean(loss))


# ---------------------------------------------------------
# Combined metric emphasizing extremes
# ---------------------------------------------------------

def compute_extreme_metric(y_true, y_pred):

    rmse = compute_rmse(y_true, y_pred)
    qloss = compute_quantile_loss(y_true, y_pred, q=0.95)

    return 0.5 * rmse + 0.5 * qloss


# ---------------------------------------------------------
# Spatial cross-validation (Fedjeosen ↔ Fauskane)
# ---------------------------------------------------------

def evaluate_pooled_cv(method_name, method_module, params, trial=None):

    scores = []

    splits = [
        (_DF_A, _DF_B),
        (_DF_B, _DF_A),
    ]

    for df_train, df_valid in splits:

        with override_method_settings(method_name, params):

            if trial is not None:
                model = method_module.fit(df_train, trial=trial)
            else:
                model = method_module.fit(df_train)

            df_pred = method_module.apply(
                df_valid.copy(),
                model
            )

        score = compute_extreme_metric(
            df_valid[HS_OBS].values,
            df_pred[HS_MODEL].values
        )

        scores.append(score)

    return float(np.mean(scores))


# ---------------------------------------------------------
# Early stopping callback for Optuna
# ---------------------------------------------------------

import optuna

class EarlyStoppingCallback:

    def __init__(self, patience=200):
        self.patience = patience
        self.best_trial = None

    def __call__(self, study, trial):

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            return

        best_trial = min(completed_trials, key=lambda t: t.value)

        if self.best_trial is None:
            self.best_trial = best_trial.number
            return

        if best_trial.number != self.best_trial:
            self.best_trial = best_trial.number

        if trial.number - self.best_trial >= self.patience:
            print(
                f"\nEarly stopping triggered: "
                f"No improvement in {self.patience} trials."
            )
            study.stop()