import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import get_core_buoy_locations


def make_objective(source, profile_name, evaluate_cv):
    import optuna
    import torch

    from src.bias_correction.methods import transformer

    def objective(trial):
        model_config = trial.suggest_categorical(
            "model_config",
            ["16-2", "32-2", "32-4", "64-2", "64-4", "64-8", "128-4", "128-8"],
        )
        d_model, nhead = map(int, model_config.split("-"))
        ff_mult = trial.suggest_categorical("ff_mult", [2, 3, 4])

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
            "sequence_length": trial.suggest_int("sequence_length", 12, 96, log=True),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": d_model * ff_mult,
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "quantile_bias_mode": "additive",
            "target_eps": trial.suggest_categorical("target_eps", [1e-6, 1e-5, 1e-4, 1e-3]),
            "tail_weight_q90": tail_weight_q90,
            "tail_weight_q95": tail_weight_q95,
            "tail_weight_q99": tail_weight_q99,
            "random_state": 1,
        }

        try:
            return evaluate_cv(
                transformer,
                params,
                trial=trial,
                source=source,
                profile_name=profile_name,
            )
        except ValueError:
            raise optuna.exceptions.TrialPruned()
        except RuntimeError as error:
            if "CUDA" in str(error) or "cublas" in str(error).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna search for the bias-correction transformer model.")
    parser.add_argument("--source", required=True, help="Core buoy source, e.g. 'fedjeosen' or 'fauskane'.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--storage", default="sqlite:///optuna_transformer.db")
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

    profile_name = f"transformer_{args.source}"
    study = create_study(
        study_name=f"transformer_{args.source}_local_cv",
        storage=args.storage,
        startup_trials=12,
        warmup_steps=2,
        interval_steps=1,
    )

    study.optimize(
        make_objective(args.source, profile_name, evaluate_cv),
        n_trials=args.trials,
        callbacks=[EarlyStoppingCallback(patience=80)],
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
