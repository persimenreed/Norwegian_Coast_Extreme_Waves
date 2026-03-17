import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contextlib
import copy
import inspect
import numpy as np

from src.bias_correction.data import load_pairs
from src.bias_correction.validation import iter_local_cv_splits
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
def override_method_settings(method_name, params, settings_name=None):

    settings = load_settings()
    settings_key = settings_name or method_name

    if settings_key not in settings["ml"]:
        settings["ml"][settings_key] = copy.deepcopy(
            settings["ml"].get(method_name, {})
        )

    original = copy.deepcopy(settings["ml"].get(settings_key, {}))

    settings["ml"][settings_key].update(params)

    try:
        yield
    finally:
        settings["ml"][settings_key] = original


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


def compute_tail_rmse(y_true, y_pred, q=0.95):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(m) < 20:
        return np.nan

    thr = np.nanquantile(y_true[m], q)
    tail = m & (y_true >= thr)

    if np.sum(tail) < 20:
        return np.nan

    return compute_rmse(y_true[tail], y_pred[tail])


# ---------------------------------------------------------
# Combined metric emphasizing extremes
# ---------------------------------------------------------

def compute_extreme_metric(y_true, y_pred):

    rmse = compute_rmse(y_true, y_pred)
    qloss = compute_quantile_loss(y_true, y_pred, q=0.95)
    tail_rmse = compute_tail_rmse(y_true, y_pred, q=0.95)

    parts = [
        (0.35, rmse),
        (0.35, qloss),
        (0.30, tail_rmse),
    ]

    num = sum(w * v for w, v in parts if np.isfinite(v))
    den = sum(w for w, v in parts if np.isfinite(v))

    if den == 0:
        return np.nan

    return float(num / den)


def _fit_with_optional_kwargs(
    method_module,
    df_train,
    trial=None,
    trial_step_offset=0,
    settings_name=None,
):
    fit_params = inspect.signature(method_module.fit).parameters
    extra_fit_kwargs = {}

    if "trial_step_offset" in fit_params:
        extra_fit_kwargs["trial_step_offset"] = trial_step_offset
    if "settings_name" in fit_params and settings_name is not None:
        extra_fit_kwargs["settings_name"] = settings_name

    if trial is not None and "trial" in fit_params:
        return method_module.fit(
            df_train,
            trial=trial,
            **extra_fit_kwargs,
        )

    return method_module.fit(df_train, **extra_fit_kwargs)


def evaluate_cv(method_name, method_module, params, trial=None, source=None, settings_name=None):
    fit_params = inspect.signature(method_module.fit).parameters
    has_internal_trial_reporting = trial is not None and "trial" in fit_params

    if source is None:
        splits = [
            (0, _DF_A, _DF_B, 1000),
            (1, _DF_B, _DF_A, 1001),
        ]
    else:
        df_source = load_pairs(source)
        splits = [
            (
                split["fold"],
                df_source.iloc[split["train_idx"]].copy(),
                df_source.iloc[split["test_idx"]].copy(),
                split["fold"] + 1,
            )
            for split in iter_local_cv_splits(df_source)
        ]
        if not splits:
            raise ValueError(f"No local CV folds were generated for source '{source}'.")

    scores = []
    for fold_id, df_train, df_valid, report_step in splits:
        with override_method_settings(
            method_name,
            params,
            settings_name=settings_name,
        ):
            model = _fit_with_optional_kwargs(
                method_module,
                df_train,
                trial=trial,
                trial_step_offset=fold_id * 1000,
                settings_name=settings_name,
            )
            df_pred = method_module.apply(df_valid.copy(), model)

        score = compute_extreme_metric(
            df_valid[HS_OBS].values,
            df_pred[HS_MODEL].values,
        )
        scores.append(score)

        if trial is not None and not has_internal_trial_reporting:
            trial.report(float(np.mean(scores)), report_step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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
