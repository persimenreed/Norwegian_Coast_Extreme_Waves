import optuna
import matplotlib.pyplot as plt
from pathlib import Path


# project root (where the .db files live)
REPO_ROOT = Path(__file__).resolve().parents[2]

# folder where this script lives (where plots will be saved)
SCRIPT_DIR = Path(__file__).resolve().parent


def plot_rmse(study, title):

    trials = study.trials_dataframe()

    plt.figure(figsize=(8, 5))

    plt.scatter(
        trials["number"],
        trials["value"]
    )

    plt.xlabel("Trial")
    plt.ylabel("RMSE")
    plt.title(f"{title} — Trial RMSE")
    plt.grid()

    save_path = SCRIPT_DIR / f"{title}_trial_rmse.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    print("Saved:", save_path)

    plt.show()


def plot_best(study, title):

    trials = study.trials_dataframe()

    best = trials["value"].cummin()

    plt.figure(figsize=(8, 5))

    plt.plot(best)

    plt.xlabel("Trial")
    plt.ylabel("Best RMSE")
    plt.title(f"{title} — Best RMSE over trials")
    plt.grid()

    save_path = SCRIPT_DIR / f"{title}_best_rmse.png"
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
        except Exception as e:
            print(f"\nSkipping {db.name}: {e}")
            continue

        for summary in summaries:

            study_name = summary.study_name

            print(f"\nLoading study: {study_name} ({db.name})")

            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )

            plot_rmse(study, study_name)
            plot_best(study, study_name)


if __name__ == "__main__":
    main()