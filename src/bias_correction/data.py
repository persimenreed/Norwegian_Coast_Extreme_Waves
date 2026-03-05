import pandas as pd
from src.settings import format_path, get_buoy_locations


def load_pairs(location):

    path = format_path("pairs", location=location)
    df = pd.read_csv(path, comment="#")
    df["time"] = pd.to_datetime(df["time"])

    return df.sort_values("time")


def load_hindcast(location):

    path = format_path("hindcast_raw", location=location)
    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df.sort_values("time")


def load_pooled_pairs():

    buoys = get_buoy_locations()
    dfs = []

    for loc in buoys:
        df = load_pairs(loc)
        df["source"] = loc
        dfs.append(df)

    return pd.concat(dfs).sort_values("time")