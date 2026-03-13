from pathlib import Path

import numpy as np
import pandas as pd

from src.bias_correction.data import load_hindcast
from src.settings import (
    get_core_buoy_locations,
    format_path,
    get_external_validation_buoys,
    get_methods,
    get_study_area_locations,
)

TIME = "time"
OBS = "Significant_Wave_Height_Hm0"
MODEL = "hs"
MODEL_CORR = "hs_corrected"
LOCATION = "ensemble_location"


def member_column(method):
    return f"member_{method}"


def normalize_methods(methods=None):
    available = [m for m in get_methods() if not str(m).startswith("ensemble")]

    if methods is None:
        return available

    selected = []
    for method in methods:
        if str(method).startswith("ensemble"):
            continue
        if method not in available:
            raise ValueError(
                f"Unknown ensemble member '{method}'. Available methods: {available}"
            )
        if method not in selected:
            selected.append(method)

    if not selected:
        raise ValueError("No valid ensemble members were selected.")

    return selected


def default_target_locations():
    out = []
    for location in get_external_validation_buoys() + get_study_area_locations():
        if location not in out:
            out.append(location)
    return out


def validation_path(location, member_family, method):
    return Path(
        format_path(
            "validation",
            location=location,
            corr_method=f"{member_family}_{method}",
        )
    )


def corrected_path(location, member_family, method):
    return Path(
        format_path(
            "corrected",
            location=location,
            corr_method=f"{member_family}_{method}",
        )
    )


def output_validation_path(location, output_name):
    return Path(format_path("validation", location=location, corr_method=output_name))


def output_corrected_path(location, output_name):
    return Path(format_path("corrected", location=location, corr_method=output_name))


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if TIME in df.columns:
        df[TIME] = pd.to_datetime(df[TIME], errors="coerce")
        df = df.sort_values(TIME).reset_index(drop=True)
    return df


def _strip_validation_columns(df):
    keep = [
        col for col in df.columns
        if col != "corr_method" and not str(col).endswith("_corrected")
    ]
    return df[keep].copy()


def _merge_member_predictions(parts):
    merged = None

    for part in parts:
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=TIME, how="inner")

    if merged is None:
        raise ValueError("No ensemble member tables were loaded.")

    return merged.sort_values(TIME).reset_index(drop=True)


def load_validation_member_dataset(
    location,
    methods,
    member_family="pooled",
    group_label=None,
):
    base = None
    parts = []

    for method in methods:
        path = validation_path(location, member_family, method)
        df = _read_csv(path)

        if MODEL_CORR not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{MODEL_CORR}'.")

        if OBS not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{OBS}'.")

        if base is None:
            base = _strip_validation_columns(df)

        parts.append(
            df[[TIME, MODEL_CORR]].rename(columns={MODEL_CORR: member_column(method)})
        )

    merged = _merge_member_predictions(parts)
    data = base.merge(merged, on=TIME, how="inner")
    member_cols = [member_column(method) for method in methods]
    data = data.dropna(subset=[OBS] + member_cols).reset_index(drop=True)
    data[LOCATION] = group_label or location
    return data


def load_hindcast_member_dataset(location, methods, member_family="pooled"):
    base = load_hindcast(location)
    parts = []

    for method in methods:
        path = corrected_path(location, member_family, method)
        df = _read_csv(path)

        if MODEL not in df.columns:
            raise ValueError(f"Corrected file {path} is missing '{MODEL}'.")

        parts.append(
            df[[TIME, MODEL]].rename(columns={MODEL: member_column(method)})
        )

    merged = _merge_member_predictions(parts)
    data = base.merge(merged, on=TIME, how="inner")
    data[LOCATION] = location
    return data.sort_values(TIME).reset_index(drop=True)


def load_training_validation_data(locations, methods, member_family="pooled"):
    frames = [
        load_validation_member_dataset(
            location=location,
            methods=methods,
            member_family=member_family,
        )
        for location in locations
    ]

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values([LOCATION, TIME]).reset_index(drop=True)


def load_training_validation_specs(specs, methods):
    frames = [
        load_validation_member_dataset(
            location=spec["location"],
            methods=methods,
            member_family=spec["member_family"],
            group_label=spec.get("group_label"),
        )
        for spec in specs
    ]

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values([LOCATION, TIME]).reset_index(drop=True)


def has_validation_members(location, methods, member_family="pooled"):
    return all(validation_path(location, member_family, method).exists() for method in methods)


def default_training_locations(methods, member_family="pooled"):
    out = [
        location
        for location in get_external_validation_buoys()
        if has_validation_members(location, methods, member_family)
    ]

    if not out:
        raise ValueError(
            "No training locations with matching validation members were found. "
            f"Checked external validation buoys for member family '{member_family}'."
        )

    return out


def default_transfer_training_specs(methods):
    specs = []
    core_buoys = get_core_buoy_locations()

    for target in core_buoys:
        for source in core_buoys:
            if source == target:
                continue

            member_family = f"transfer_{source}"
            if not has_validation_members(target, methods, member_family):
                continue

            specs.append(
                {
                    "location": target,
                    "member_family": member_family,
                    "group_label": f"{source}_to_{target}",
                    "label": f"{source}->{target}",
                }
            )

    if not specs:
        raise ValueError(
            "No transfer validation datasets were found for the core buoys. "
            "Run the transfer bias-correction stage first."
        )

    return specs


def unique_locations(locations):
    out = []
    for location in locations:
        if location not in out:
            out.append(location)
    return out


def grouped_time_folds(df, n_splits):
    if int(n_splits) < 2:
        return []

    group_splits = {}

    for _, group in df.groupby(LOCATION, sort=False):
        idx = group.index.to_numpy()
        splits = [split for split in np.array_split(idx, int(n_splits)) if len(split) > 0]
        if splits:
            group_splits[str(group[LOCATION].iloc[0])] = splits

    if not group_splits:
        return []

    all_idx = np.arange(len(df))
    max_folds = max(len(splits) for splits in group_splits.values())
    folds = []

    for fold_id in range(max_folds):
        test_parts = [
            splits[fold_id]
            for splits in group_splits.values()
            if fold_id < len(splits)
        ]

        if not test_parts:
            continue

        test_idx = np.concatenate(test_parts)
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        folds.append((train_idx, test_idx))

    return folds


def build_oof_predictions(df, fit_fn, predict_fn, n_splits):
    folds = grouped_time_folds(df, n_splits)

    if not folds:
        return None

    pred = np.full(len(df), np.nan, dtype=float)

    for train_idx, test_idx in folds:
        bundle = fit_fn(df.iloc[train_idx].copy().reset_index(drop=True))
        pred[test_idx] = predict_fn(
            df.iloc[test_idx].copy().reset_index(drop=True),
            bundle,
        )

    if not np.all(np.isfinite(pred)):
        missing = int(np.sum(~np.isfinite(pred)))
        raise ValueError(f"Failed to generate OOF predictions for {missing} rows.")

    return pred


def _drop_internal_columns(df):
    drop_cols = [LOCATION]
    drop_cols.extend(col for col in df.columns if str(col).startswith("member_"))
    return df.drop(columns=drop_cols, errors="ignore").copy()


def save_validation_output(
    location,
    df,
    prediction,
    output_name,
    train_locations,
    member_family,
    methods,
    validation_type,
):
    out = _drop_internal_columns(df)
    out["corr_method"] = output_name
    out[f"{OBS}_corrected"] = prediction
    out[f"{MODEL}_corrected"] = prediction
    out["validation_type"] = validation_type
    out["fold"] = -1
    out["test_groups"] = ""
    out["train_source"] = "|".join(train_locations)
    out["apply_target"] = location
    out["ensemble_member_family"] = member_family
    out["ensemble_members"] = "|".join(methods)

    path = output_validation_path(location, output_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def save_hindcast_output(location, df, prediction, output_name):
    out = _drop_internal_columns(df)
    out[MODEL] = prediction

    path = output_corrected_path(location, output_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)
