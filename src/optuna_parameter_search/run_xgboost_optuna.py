import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import optuna

from src.bias_correction.methods import xgboost
from src.optuna_parameter_search.common import (
    evaluate_pooled_cv,
    EarlyStoppingCallback
)


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------

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

    score = evaluate_pooled_cv(
        "xgboost",
        xgboost,
        params
    )

    return score


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1500)

    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        study_name="xgboost_spatial_cv",
        storage="sqlite:///optuna_xgboost.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            seed=1
        ),
        pruner=optuna.pruners.MedianPruner()
    )

    early_stop = EarlyStoppingCallback(patience=200)

    study.optimize(
        objective,
        n_trials=args.trials,
        callbacks=[early_stop],
        show_progress_bar=True
    )

    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)


if __name__ == "__main__":
    main()