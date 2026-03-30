import inspect
import sys
from pathlib import Path

import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_profiles import override_profile
from src.settings import get_core_buoy_locations

HS_MODEL = "hs"
HS_OBS = "Significant_Wave_Height_Hm0"
CORE_BUOYS = get_core_buoy_locations()
if len(CORE_BUOYS) != 2:
    raise RuntimeError("Expected exactly two core buoys for spatial CV.")


def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if np.any(mask) else np.nan


def compute_quantile_loss(y_true, y_pred, q=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    err = y_true[mask] - y_pred[mask]
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def compute_tail_rmse(y_true, y_pred, q=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 20:
        return np.nan

    tail = mask & (y_true >= np.nanquantile(y_true[mask], q))
    return compute_rmse(y_true[tail], y_pred[tail]) if np.sum(tail) >= 20 else np.nan


def compute_extreme_metric(y_true, y_pred):
    parts = [
        (0.35, compute_rmse(y_true, y_pred)),
        (0.35, compute_quantile_loss(y_true, y_pred, q=0.95)),
        (0.30, compute_tail_rmse(y_true, y_pred, q=0.95)),
    ]
    total_weight = sum(weight for weight, value in parts if np.isfinite(value))
    if total_weight == 0:
        return np.nan
    return float(sum(weight * value for weight, value in parts if np.isfinite(value)) / total_weight)


def _fit_with_optional_kwargs(method_module, df_train, trial=None, trial_step_offset=0, profile_name=None):
    params = inspect.signature(method_module.fit).parameters
    fit_kwargs = {}

    if "trial_step_offset" in params:
        fit_kwargs["trial_step_offset"] = trial_step_offset
    if "settings_name" in params and profile_name is not None:
        fit_kwargs["settings_name"] = profile_name

    if trial is not None and "trial" in params:
        fit_kwargs["trial"] = trial
    return method_module.fit(df_train, **fit_kwargs)


def _cv_splits(source=None):
    from src.bias_correction.data import load_pairs
    from src.bias_correction.validation import iter_local_cv_splits

    if source is None:
        return [
            (fold_id, load_pairs(train_source), load_pairs(valid_source), 1000 + fold_id)
            for fold_id, (train_source, valid_source) in enumerate(
                ((CORE_BUOYS[0], CORE_BUOYS[1]), (CORE_BUOYS[1], CORE_BUOYS[0]))
            )
        ]

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
    return splits


def evaluate_cv(method_module, params, trial=None, source=None, profile_name=None):
    params_signature = inspect.signature(method_module.fit).parameters
    has_internal_trial_reporting = trial is not None and "trial" in params_signature
    profile_name = profile_name or method_module.__name__.rsplit(".", 1)[-1]

    scores = []
    for fold_id, df_train, df_valid, report_step in _cv_splits(source):
        with override_profile(profile_name, params):
            model = _fit_with_optional_kwargs(
                method_module,
                df_train,
                trial=trial,
                trial_step_offset=fold_id * 1000,
                profile_name=profile_name,
            )
            df_pred = method_module.apply(df_valid.copy(), model)

        score = compute_extreme_metric(df_valid[HS_OBS].to_numpy(), df_pred[HS_MODEL].to_numpy())
        scores.append(score)

        if trial is not None and not has_internal_trial_reporting:
            trial.report(float(np.mean(scores)), report_step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))


def create_study(study_name, storage, startup_trials=40, warmup_steps=1, interval_steps=1):
    return optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=startup_trials,
            seed=1,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=startup_trials,
            n_warmup_steps=warmup_steps,
            interval_steps=interval_steps,
        ),
    )


def print_best_trial(study, **details):
    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    for key, value in details.items():
        print(f"{key}:", value)


class EarlyStoppingCallback:
    def __init__(self, patience=200):
        self.patience = patience
        self.best_trial = None

    def __call__(self, study, trial):
        completed_trials = [
            candidate
            for candidate in study.trials
            if candidate.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            return

        best_trial = min(completed_trials, key=lambda candidate: candidate.value)
        if self.best_trial is None:
            self.best_trial = best_trial.number
            return

        if best_trial.number != self.best_trial:
            self.best_trial = best_trial.number

        if trial.number - self.best_trial >= self.patience:
            print(f"\nEarly stopping triggered: No improvement in {self.patience} trials.")
            study.stop()
