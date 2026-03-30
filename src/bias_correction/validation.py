import numpy as np
import pandas as pd

from src.bias_correction.methods.common import TIME

LOCAL_CV_FOLDS = 4
LOCAL_CV_MIN_TRAIN = 100
LOCAL_CV_MIN_TEST = 20


def iter_local_cv_splits(df):
    df = df.reset_index(drop=True)
    valid_rows = df[TIME].notna().to_numpy()

    if valid_rows.sum() < LOCAL_CV_MIN_TRAIN + LOCAL_CV_MIN_TEST:
        raise ValueError("Too few valid rows for local cross-validation.")

    groups = pd.to_datetime(df[TIME], errors="coerce").dt.to_period("M").astype(str)
    valid_groups = pd.Series(groups[valid_rows]).dropna().unique().tolist()

    if len(valid_groups) < 2:
        raise ValueError("Too few time groups for cross-validation.")

    fold_groups = [[] for _ in range(min(LOCAL_CV_FOLDS, len(valid_groups)))]
    for index, group in enumerate(valid_groups):
        fold_groups[index % len(fold_groups)].append(group)

    for fold_id, test_groups in enumerate(fold_groups):
        in_test = groups.isin(test_groups).to_numpy()
        test_mask = valid_rows & in_test
        train_mask = valid_rows & ~in_test

        if train_mask.sum() < LOCAL_CV_MIN_TRAIN or test_mask.sum() < LOCAL_CV_MIN_TEST:
            continue

        yield {
            "fold": fold_id,
            "train_idx": np.flatnonzero(train_mask),
            "test_idx": np.flatnonzero(test_mask),
            "test_groups": list(test_groups),
        }
