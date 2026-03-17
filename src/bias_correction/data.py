import pandas as pd
from src.settings import (
    format_path,
)


def load_pairs(location):
    path = format_path("pairs", location=location)
    df = pd.read_csv(path, comment="#")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


def load_hindcast(location):
    path = format_path("hindcast_raw", location=location)
    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)
