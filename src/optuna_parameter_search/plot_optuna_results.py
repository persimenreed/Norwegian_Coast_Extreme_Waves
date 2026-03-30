from pathlib import Path

import matplotlib.pyplot as plt
import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_LABEL = "Extreme Metric"


def _plot_trials(study, title, cumulative=False):
    trials = study.trials_dataframe()
    values = trials["value"].cummin() if cumulative else trials["value"]

    plt.figure(figsize=(8, 5))
    if cumulative:
        plt.plot(values)
    else:
        plt.scatter(trials["number"], values)

    plt.xlabel("Trial")
    plt.ylabel(OBJECTIVE_LABEL)
    plt.title(f"{title} — {'Best' if cumulative else 'Trial'} {OBJECTIVE_LABEL}")
    plt.grid()

    suffix = "best_metric" if cumulative else "trial_metric"
    save_path = SCRIPT_DIR / f"{title}_{suffix}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("Saved:", save_path)
    plt.show()


def main():
    db_files = sorted(REPO_ROOT.glob("optuna*.db"))
    if not db_files:
        print("No Optuna DB files found in:", REPO_ROOT)
        return

    print("Found DBs:")
    for db in db_files:
        print(" ", db.name)

    for db in db_files:
        storage = f"sqlite:///{db}"
        try:
            summaries = optuna.get_all_study_summaries(storage)
        except Exception as error:
            print(f"\nSkipping {db.name}: {error}")
            continue

        for summary in summaries:
            study_name = summary.study_name
            print(f"\nLoading study: {study_name} ({db.name})")
            study = optuna.load_study(study_name=study_name, storage=storage)
            _plot_trials(study, study_name, cumulative=False)
            _plot_trials(study, study_name, cumulative=True)


if __name__ == "__main__":
    main()
