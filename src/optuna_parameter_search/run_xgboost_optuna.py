import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import optuna

from src.bias_correction.methods import xgboost
from src.optuna_parameter_search.common import (
    evaluate_cv,
    EarlyStoppingCallback
)
from src.settings import get_core_buoy_locations


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------

def make_objective(source, settings_name):

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),

            "max_depth": trial.suggest_int("max_depth", 3, 8),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),

            "subsample": trial.suggest_float("subsample", 0.6, 1.0),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
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

            "random_state": 1
        }

        return evaluate_cv(
            "xgboost",
            xgboost,
            params,
            trial=trial,
            source=source,
            settings_name=settings_name,
        )

    return objective


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_xgboost.db",
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Source buoy to tune for strict transfer, e.g. "
            "'fedjeosen' or 'fauskane'."
        ),
    )

    args = parser.parse_args()

    if args.source is not None and args.source not in get_core_buoy_locations():
        raise ValueError(
            f"Unknown core-buoy source '{args.source}'. "
            f"Available: {get_core_buoy_locations()}"
        )

    settings_name = f"xgboost_{args.source}"
    study_name = f"xgboost_{args.source}_local_cv"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=40,
            seed=1
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=40,
            n_warmup_steps=1,
            interval_steps=1
        )
    )

    early_stop = EarlyStoppingCallback(patience=150)

    study.optimize(
        make_objective(source=args.source, settings_name=settings_name),
        n_trials=args.trials,
        callbacks=[early_stop],
        show_progress_bar=True
    )

    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    print("Settings key:", settings_name)
    print("Source:", args.source)


if __name__ == "__main__":
    main()
