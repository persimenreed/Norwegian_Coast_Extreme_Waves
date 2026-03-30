import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_ensemble import ENSEMBLE_MODELS

def make_objective(
    df,
    training_members,
    profile_name,
    optuna,
    grouped_time_folds,
    compute_extreme_metric,
    fit_state_corrected_ensemble,
    predict_state_corrected_ensemble,
    override_profile,
    folds,
    obs_column,
):
    y_true = df[obs_column].to_numpy()

    def objective(trial):
        tail_weight_q90 = trial.suggest_categorical("tail_weight_q90", [1.0, 1.5, 2.0, 2.5, 3.0])
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
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tail_weight_q90": tail_weight_q90,
            "tail_weight_q95": tail_weight_q95,
            "tail_weight_q99": tail_weight_q99,
            "tail_aware": True,
            "tail_strength_q95": trial.suggest_float("tail_strength_q95", 0.0, 0.4),
            "tail_strength_q99": trial.suggest_float("tail_strength_q99", 0.1, 0.8),
            "random_state": 1,
        }

        try:
            with override_profile(profile_name, params):
                time_folds = grouped_time_folds(df, folds)
                if not time_folds:
                    raise ValueError("No grouped time folds available for Optuna.")

                pred = np.full(len(df), np.nan, dtype=float)
                for fold_idx, (train_idx, test_idx) in enumerate(time_folds, start=1):
                    bundle = fit_state_corrected_ensemble(
                        df.iloc[train_idx].copy().reset_index(drop=True),
                        training_members,
                        profile_name=profile_name,
                    )
                    pred[test_idx] = predict_state_corrected_ensemble(
                        df.iloc[test_idx].copy().reset_index(drop=True),
                        bundle,
                    )

                    trial.report(compute_extreme_metric(y_true[test_idx], pred[test_idx]), fold_idx)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if not np.all(np.isfinite(pred)):
                    raise ValueError(f"Failed to generate OOF predictions for {int(np.sum(~np.isfinite(pred)))} rows.")

            return compute_extreme_metric(y_true, pred)
        except (RuntimeError, ValueError):
            raise optuna.exceptions.TrialPruned()

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna search for the XGBoost ensemble model.")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional transfer source to tune separately. Default: tune the combined ensemble.",
    )
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--storage", default="sqlite:///optuna_ensemble.db")
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args()

    import optuna

    from src.ensemble.common import OBS, grouped_time_folds, load_training_validation_data
    from src.ensemble.xgboost_core import (
        fit_state_corrected_ensemble,
        predict_state_corrected_ensemble,
    )
    from src.ensemble.xgboost_ensemble_transfer import ENSEMBLE_OOF_FOLDS, build_training_setup
    from src.model_profiles import override_profile
    from src.optuna_parameter_search.common import (
        EarlyStoppingCallback,
        compute_extreme_metric,
        create_study,
        print_best_trial,
    )

    setup = build_training_setup(source=args.source, methods=ENSEMBLE_MODELS)
    training_specs = setup["training_specs"]
    training_members = setup["training_members"]
    profile_name = setup["profile_name"]
    study_name = args.study_name or setup["study_name"]

    df = load_training_validation_data(training_specs).reset_index(drop=True)
    study = create_study(
        study_name=study_name,
        storage=args.storage,
        startup_trials=40,
        warmup_steps=1,
        interval_steps=1,
    )

    study.optimize(
        make_objective(
            df,
            training_members,
            profile_name,
            optuna,
            grouped_time_folds,
            compute_extreme_metric,
            fit_state_corrected_ensemble,
            predict_state_corrected_ensemble,
            override_profile,
            ENSEMBLE_OOF_FOLDS,
            OBS,
        ),
        n_trials=args.trials,
        callbacks=[EarlyStoppingCallback(patience=150)],
        show_progress_bar=True,
    )

    print_best_trial(
        study,
        profile=profile_name,
        source=args.source or "combined",
        training_members=training_members,
        training_labels=[spec["label"] for spec in training_specs],
    )


if __name__ == "__main__":
    main()
