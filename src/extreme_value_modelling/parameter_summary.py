import pandas as pd
from pathlib import Path
from src.settings import get_path_template

SUMMARY_FILE = Path(get_path_template("evt_results_root")) / "evt_parameter_summary.csv"


def update_parameter_summary(row: dict):
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])

    if SUMMARY_FILE.exists():
        df = pd.read_csv(SUMMARY_FILE)
        df = df.dropna(how="all")

        mask = (
            (df["location"] == row["location"]) &
            (df["dataset"] == row["dataset"]) &
            (df["model"] == row["model"])
        )
        df = df[~mask]
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
    else:
        df = df_new

    df.to_csv(SUMMARY_FILE, index=False)
    print(f"Updated parameter summary: {SUMMARY_FILE}")