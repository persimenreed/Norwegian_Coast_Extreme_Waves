import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import optuna
import torch

from src.bias_correction.methods import transformer
from src.optuna_parameter_search.common import (
    evaluate_cv,
    EarlyStoppingCallback
)
from src.settings import get_core_buoy_locations


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------

def make_objective(source, settings_name, target_transform):

    def objective(trial):

        model_config = trial.suggest_categorical(
            "model_config",
            [
                "16-2",
                "32-2",
                "32-4",
                "64-2",
                "64-4",
                "64-8",
                "128-4",
                "128-8",
            ]
        )

        d_model, nhead = map(int, model_config.split("-"))

        ff_mult = trial.suggest_categorical("ff_mult", [2, 3, 4])

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
            [1.0, 2.0, 4.0, 6.0, 8.0],
        )
        target_space_loss_weight = trial.suggest_categorical(
            "target_space_loss_weight",
            [0.25, 0.5, 0.75],
        )
        physical_space_loss_weight = trial.suggest_categorical(
            "physical_space_loss_weight",
            [1.0, 1.25, 1.5, 2.0],
        )

        params = {
            "sequence_length": trial.suggest_int(
                "sequence_length", 12, 96, log=True
            ),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": d_model * ff_mult,
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 2e-3, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            ),
            "target_transform": target_transform,
            "target_eps": trial.suggest_categorical(
                "target_eps", [1e-6, 1e-5, 1e-4, 1e-3]
            ),
            "tail_weight_q90": tail_weight_q90,
            "tail_weight_q95": tail_weight_q95,
            "tail_weight_q99": tail_weight_q99,
            "target_space_loss_weight": target_space_loss_weight,
            "physical_space_loss_weight": physical_space_loss_weight,
            "random_state": 1,
        }

        try:
            return evaluate_cv(
                "transformer",
                transformer,
                params,
                trial=trial,
                source=source,
                settings_name=settings_name,
            )

        except ValueError:
            raise optuna.exceptions.TrialPruned()

        except RuntimeError as e:
            if "CUDA" in str(e) or "cublas" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

            raise

    return objective


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=12,
        help="Number of completed trials before Optuna pruning starts.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of reported steps before the pruner becomes active within a trial.",
    )
    parser.add_argument(
        "--interval-steps",
        type=int,
        default=1,
        help="How often Optuna checks pruning after warmup.",
    )
    parser.add_argument(
        "--study-patience",
        type=int,
        default=80,
        help="Stop the whole study after this many non-improving completed trials.",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_transformer.db",
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Source buoy to tune for strict transfer, e.g. "
            "'fedjeosen' or 'fauskane'."
        ),
    )
    parser.add_argument(
        "--target-transform",
        default="log_ratio",
        choices=["log_ratio", "additive_residual"],
        help=(
            "Training target transform to optimize. "
            "Use 'log_ratio' for the new tail-focused setup."
        ),
    )

    args = parser.parse_args()

    if args.source is not None and args.source not in get_core_buoy_locations():
        raise ValueError(
            f"Unknown core-buoy source '{args.source}'. "
            f"Available: {get_core_buoy_locations()}"
        )

    settings_name = f"transformer_{args.source}"
    study_name = f"transformer_{args.source}_local_cv"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=args.startup_trials,
            seed=1
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.startup_trials,
            n_warmup_steps=args.warmup_steps,
            interval_steps=args.interval_steps
        )
    )

    early_stop = EarlyStoppingCallback(patience=args.study_patience)

    study.optimize(
        make_objective(
            source=args.source,
            settings_name=settings_name,
            target_transform=args.target_transform,
        ),
        n_trials=args.trials,
        callbacks=[early_stop],
        show_progress_bar=True
    )

    print("\nBest trial:")
    print("Score:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    print("Settings key:", settings_name)
    print("Source:", args.source)
    print("Target transform:", args.target_transform)


if __name__ == "__main__":
    main()
