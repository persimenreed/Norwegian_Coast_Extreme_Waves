import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import contextlib
import copy

import optuna

from src.ensemble.common import (
    OBS,
    build_oof_predictions,
    default_training_specs,
    load_training_validation_specs,
    normalize_methods,
)
from src.ensemble.xgboost_core import (
    fit_state_corrected_ensemble,
    predict_state_corrected_ensemble,
)
from src.optuna_parameter_search.common import (
    EarlyStoppingCallback,
    compute_extreme_metric,
)
from src.settings import get_method_settings


# ---------------------------------------------------------
# Temporary override of ensemble XGBoost hyperparameters
# ---------------------------------------------------------

@contextlib.contextmanager
def override_xgboost_settings(params):
    cfg = get_method_settings("ensemble_xgboost")
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

def make_objective(df, methods, n_splits):
    y_true = df[OBS].values

    def objective(trial):
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
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 10.0, log=True
            ),
            "random_state": 1,
        }

        try:
            with override_xgboost_settings(params):
                pred = build_oof_predictions(
                    df,
                    fit_fn=lambda train_df: fit_state_corrected_ensemble(
                        train_df,
                        methods,
                    ),
                    predict_fn=predict_state_corrected_ensemble,
                    n_splits=n_splits,
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
            "Optuna search for the pooled XGBoost-gated ensemble using "
            "the default fair transfer-training setup."
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
        default=1000,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of grouped time OOF folds.",
    )
    parser.add_argument(
        "--study-name",
        default="ensemble_pooling_spatial_cv",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_ensemble.db",
    )

    args = parser.parse_args()

    methods = normalize_methods(args.methods)
    training_specs = default_training_specs(methods)
    df = load_training_validation_specs(training_specs, methods).reset_index(drop=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            seed=1,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=40,
        ),
    )

    early_stop = EarlyStoppingCallback(patience=200)

    study.optimize(
        make_objective(df, methods, args.folds),
        n_trials=args.trials,
        callbacks=[early_stop],
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    print("Methods:", methods)
    print("Training labels:", [spec["label"] for spec in training_specs])


if __name__ == "__main__":
    main()
