import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import contextlib
import copy

import numpy as np
import optuna

from src.ensemble.common import (
    OBS,
    grouped_time_folds,
    load_training_validation_specs,
    normalize_methods,
)
from src.ensemble.xgboost_ensemble_transfer import (
    _combined_training_specs,
    _single_source_training_specs,
    _training_members,
)
from src.ensemble.xgboost_core import (
    fit_state_corrected_ensemble,
    predict_state_corrected_ensemble,
)
from src.optuna_parameter_search.common import (
    EarlyStoppingCallback,
    compute_extreme_metric,
)
from src.settings import get_core_buoy_locations, get_method_settings


# ---------------------------------------------------------
# Temporary override of ensemble XGBoost hyperparameters
# ---------------------------------------------------------

@contextlib.contextmanager
def override_xgboost_settings(params, settings_name="ensemble_xgboost"):
    cfg = get_method_settings(settings_name) or get_method_settings("ensemble_xgboost")
    original = copy.deepcopy(cfg)

    cfg.update(params)

    try:
        yield
    finally:
        cfg.clear()
        cfg.update(original)


# ---------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------

def _training_setup(methods, source=None):
    if source is None:
        return {
            "training_specs": _combined_training_specs(methods),
            "training_members": _training_members(methods, combined=True),
            "settings_name": "ensemble_xgboost",
            "study_name": "ensemble_combined_spatial_cv",
        }

    return {
        "training_specs": _single_source_training_specs(source, methods),
        "training_members": _training_members(methods, source=source, combined=False),
        "settings_name": f"ensemble_xgboost_{source}",
        "study_name": f"ensemble_{source}_spatial_cv",
    }


def make_objective(df, training_members, n_splits, settings_name):
    y_true = df[OBS].values

    def objective(trial):
        tail_weight_q90 = trial.suggest_categorical(
            "tail_weight_q90",
            [1.0, 1.5, 2.0, 2.5, 3.0],
        )
        tail_weight_q95 = tail_weight_q90 + trial.suggest_categorical(
            "tail_weight_q95_extra",
            [0.0, 0.5, 1.0, 2.0, 3.0],
        )
        tail_weight_q99 = tail_weight_q95 + trial.suggest_categorical(
            "tail_weight_q99_extra",
            [1.0, 2.0, 4.0, 6.0],
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.1, 10.0, log=True
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 10.0, log=True
            ),
            "tail_weight_q90": tail_weight_q90,
            "tail_weight_q95": tail_weight_q95,
            "tail_weight_q99": tail_weight_q99,
            "tail_aware": trial.suggest_categorical("tail_aware", [True]),
            "tail_strength_q95": trial.suggest_float(
                "tail_strength_q95", 0.0, 0.4
            ),
            "tail_strength_q99": trial.suggest_float(
                "tail_strength_q99", 0.1, 0.8
            ),
            "random_state": 1,
        }

        try:
            with override_xgboost_settings(params, settings_name=settings_name):
                folds = grouped_time_folds(df, n_splits)
                if not folds:
                    raise ValueError("No grouped time folds available for Optuna.")

                pred = np.full(len(df), np.nan, dtype=float)

                for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
                    bundle = fit_state_corrected_ensemble(
                        df.iloc[train_idx].copy().reset_index(drop=True),
                        training_members,
                        settings_name=settings_name,
                    )
                    pred[test_idx] = predict_state_corrected_ensemble(
                        df.iloc[test_idx].copy().reset_index(drop=True),
                        bundle,
                    )

                    trial.report(
                        compute_extreme_metric(y_true[test_idx], pred[test_idx]),
                        fold_idx,
                    )
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if not np.all(np.isfinite(pred)):
                    missing = int(np.sum(~np.isfinite(pred)))
                    raise ValueError(
                        f"Failed to generate OOF predictions for {missing} rows."
                    )

            score = compute_extreme_metric(y_true, pred)
            return score

        except (RuntimeError, ValueError):
            raise optuna.exceptions.TrialPruned()

    return objective


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna search for the raw MoE XGBoost ensemble using the same "
            "training setup as the ensemble runner."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Optional ensemble members to include. "
            "Default: all non-ensemble methods from settings."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of grouped time OOF folds.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_ensemble.db",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Optional transfer source to tune separately, e.g. 'fedjeosen' or "
            "'fauskane'. Default: tune the combined ensemble setup."
        ),
    )

    args = parser.parse_args()

    methods = normalize_methods(args.methods)
    if args.source is not None and args.source not in get_core_buoy_locations():
        raise ValueError(
            f"Unknown core-buoy source '{args.source}'. "
            f"Available: {get_core_buoy_locations()}"
        )

    training_setup = _training_setup(methods, source=args.source)
    training_specs = training_setup["training_specs"]
    training_members = training_setup["training_members"]
    settings_name = training_setup["settings_name"]

    if not training_specs or not training_members:
        raise ValueError(
            f"No ensemble Optuna training specs were found for source '{args.source}'."
        )

    df = load_training_validation_specs(training_specs).reset_index(drop=True)
    study_name = args.study_name or training_setup["study_name"]

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=40,
            seed=1,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=40,
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    early_stop = EarlyStoppingCallback(patience=150)

    study.optimize(
        make_objective(
            df,
            training_members,
            args.folds,
            settings_name=settings_name,
        ),
        n_trials=args.trials,
        callbacks=[early_stop],
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    print("Settings key:", settings_name)
    print("Source:", args.source or "combined")
    print("Methods:", methods)
    print("Training members:", training_members)
    print("Training labels:", [spec["label"] for spec in training_specs])


if __name__ == "__main__":
    main()
