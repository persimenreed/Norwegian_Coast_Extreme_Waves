import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import get_core_buoy_locations


def make_objective(source, profile_name, evaluate_cv):
    from src.bias_correction.methods import xgboost

    def objective(trial):
        tail_weight_q90 = trial.suggest_categorical("tail_weight_q90", [1.0, 1.5, 2.0, 2.5, 3.0])
        tail_weight_q95 = tail_weight_q90 + trial.suggest_categorical(
            "tail_weight_q95_extra",
            [0.0, 0.5, 1.0, 2.0, 3.0],
        )
        tail_weight_q99 = tail_weight_q95 + trial.suggest_categorical(
            "tail_weight_q99_extra",
            [1.0, 2.0, 4.0, 6.0, 8.0],
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "quantile_bias_mode": "additive",
            "target_eps": trial.suggest_categorical("target_eps", [1e-6, 1e-5, 1e-4, 1e-3]),
            "tail_weight_q90": tail_weight_q90,
            "tail_weight_q95": tail_weight_q95,
            "tail_weight_q99": tail_weight_q99,
            "random_state": 1,
        }

        return evaluate_cv(
            xgboost,
            params,
            trial=trial,
            source=source,
            profile_name=profile_name,
        )

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna search for the bias-correction XGBoost model.")
    parser.add_argument("--source", required=True, help="Core buoy source, e.g. 'fedjeosen' or 'fauskane'.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--storage", default="sqlite:///optuna_xgboost.db")
    args = parser.parse_args()

    if args.source not in get_core_buoy_locations():
        raise ValueError(
            f"Unknown core-buoy source '{args.source}'. Available: {get_core_buoy_locations()}"
        )

    from src.optuna_parameter_search.common import (
        EarlyStoppingCallback,
        create_study,
        evaluate_cv,
        print_best_trial,
    )

    profile_name = f"xgboost_{args.source}"
    study = create_study(
        study_name=f"xgboost_{args.source}_local_cv",
        storage=args.storage,
        startup_trials=40,
        warmup_steps=1,
        interval_steps=1,
    )

    study.optimize(
        make_objective(args.source, profile_name, evaluate_cv),
        n_trials=args.trials,
        callbacks=[EarlyStoppingCallback(patience=150)],
        show_progress_bar=True,
    )

    print_best_trial(
        study,
        profile=profile_name,
        source=args.source,
        target_transform="quantile_residual",
        quantile_bias_mode="additive",
    )


if __name__ == "__main__":
    main()
