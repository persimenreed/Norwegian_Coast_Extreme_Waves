import pandas as pd
from src.settings import (
    format_path,
    get_core_buoy_locations,
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


def load_pooled_pairs(exclude_locations=None, include_only_core=True):
    exclude_locations = set(exclude_locations or [])

    if include_only_core:
        buoys = get_core_buoy_locations()
    else:
        raise NotImplementedError("Only core-buoy pooling is supported in this thesis pipeline.")

    dfs = []
    for loc in buoys:
        if loc in exclude_locations:
            continue
        df = load_pairs(loc)
        df["source"] = loc
        dfs.append(df)

    if not dfs:
        raise ValueError("No buoy datasets available for pooled training.")

    return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)