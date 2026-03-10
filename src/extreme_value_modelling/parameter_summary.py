import pandas as pd
from pathlib import Path
from src.settings import get_path_template


def _summary_file(location: str) -> Path:
    root = Path(get_path_template("evt_results_root"))
    return root / location / "evt_parameter_summary.csv"


def update_parameter_summary(row: dict):
    if "location" not in row or not row["location"]:
        raise ValueError("Parameter summary row must contain 'location'.")

    summary_file = _summary_file(str(row["location"]))
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            df = df.dropna(how="all")
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        if not df.empty:
            required = {"location", "dataset", "model"}
            missing = required - set(df.columns)
            if missing:
                df = pd.DataFrame()
            else:
                mask = (
                    (df["location"] == row["location"]) &
                    (df["dataset"] == row["dataset"]) &
                    (df["model"] == row["model"])
                )
                df = df[~mask]

            df = pd.concat([df, df_new], ignore_index=True, sort=False)
        else:
            df = df_new
    else:
        df = df_new

    df.to_csv(summary_file, index=False)
    print(f"Updated parameter summary: {summary_file}")