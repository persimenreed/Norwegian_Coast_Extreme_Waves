import numpy as np
import pandas as pd
from src.settings import get_validation_settings

_TIME_COL = "time"


def _valid_rows(df):
    return df[_TIME_COL].notna().values


def _group_labels(df, mode="month"):
    time = pd.to_datetime(df[_TIME_COL], errors="coerce")

    if mode == "month":
        return time.dt.to_period("M").astype(str)

    if mode == "season":
        month = time.dt.month
        year = time.dt.year

        season = np.where(
            month.isin([12, 1, 2]), "DJF",
            np.where(month.isin([3, 4, 5]), "MAM",
                     np.where(month.isin([6, 7, 8]), "JJA", "SON"))
        )

        # Put December into next year's DJF
        season_year = year.copy()
        season_year = np.where((month == 12), year + 1, year)

        return pd.Series([f"{y}_{s}" if pd.notna(y) else None
                          for y, s in zip(season_year, season)], index=df.index)

    raise ValueError(f"Unsupported cv_time_group: {mode}")


def iter_local_cv_splits(df):
    cfg = get_validation_settings()
    n_folds = int(cfg.get("local_cv_folds", 4))
    min_train = int(cfg.get("local_cv_min_train_samples", 100))
    min_test = int(cfg.get("local_cv_min_test_samples", 20))
    group_mode = cfg.get("cv_time_group", "month")

    df = df.reset_index(drop=True)
    valid_mask = _valid_rows(df)

    if valid_mask.sum() < (min_train + min_test):
        raise ValueError("Too few valid rows for local cross-validation.")

    groups = _group_labels(df, mode=group_mode)
    valid_groups = pd.Series(groups[valid_mask]).dropna().unique().tolist()

    if len(valid_groups) < 2:
        raise ValueError("Too few time groups for cross-validation.")

    # Round-robin assignment of time groups to folds.
    fold_groups = [[] for _ in range(min(n_folds, len(valid_groups)))]
    for i, g in enumerate(valid_groups):
        fold_groups[i % len(fold_groups)].append(g)

    for fold_id, test_groups in enumerate(fold_groups):
        test_mask = valid_mask & groups.isin(test_groups).values
        train_mask = valid_mask & (~groups.isin(test_groups)).values

        if train_mask.sum() < min_train:
            continue
        if test_mask.sum() < min_test:
            continue

        yield {
            "fold": fold_id,
            "train_idx": np.flatnonzero(train_mask),
            "test_idx": np.flatnonzero(test_mask),
            "test_groups": list(test_groups),
        }