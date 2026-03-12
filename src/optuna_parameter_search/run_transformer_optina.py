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
    evaluate_pooled_cv,
    EarlyStoppingCallback
)


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------

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

    ff_mult = trial.suggest_categorical("ff_mult", [2,3,4])

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
            "weight_decay", 1e-5, 1e-3, log=True
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size", [32,64,128,256]
        ),

        "random_state": 1
    }

    try:

        score = evaluate_pooled_cv(
            "transformer",
            transformer,
            params,
            trial=trial
        )

        return score

    except RuntimeError as e:

        if "CUDA" in str(e) or "cublas" in str(e).lower():
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        raise


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=600)

    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        study_name="transformer_spatial_cv",
        storage="sqlite:///optuna_transformer.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            seed=1
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=40,
            n_warmup_steps=10,
            interval_steps=2
        )
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