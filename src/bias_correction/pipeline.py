from pathlib import Path
import pandas as pd

from src.settings import (
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_buoy_locations,
    get_study_area_locations,
    get_methods,
    format_path,
)

from src.bias_correction.data import (
    load_pairs,
    load_hindcast,
    load_pooled_pairs,
)

from src.bias_correction.registry import get_method
from src.bias_correction.validation import iter_local_cv_splits


def _ensure_parent(path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def _save_df(df, path_str):
    _ensure_parent(path_str)
    df.to_csv(path_str, index=False)
    return path_str


def _selected_methods(method=None):
    methods = list(get_methods())

    if method is None:
        return methods

    if method not in methods:
        raise ValueError(
            f"Unknown bias-correction method '{method}'. Available methods: {methods}"
        )

    return [method]


def _validation_subset_to_output(df_base, df_corrected, method_name, meta):
    out = df_base.copy()
    out["corr_method"] = method_name

    for col in df_corrected.columns:
        if col == "time":
            continue
        if col in out.columns:
            out[f"{col}_corrected"] = df_corrected[col].values

    for k, v in meta.items():
        out[k] = v

    return out


def _fit_apply_save(method_name, df_train, df_hind, location, prefix):
    method = get_method(method_name)
    model = method.fit(df_train)
    df_pred = method.apply(df_hind.copy(), model)

    out_path = format_path(
        "corrected",
        location=location,
        corr_method=f"{prefix}{method_name}",
    )
    _save_df(df_pred, out_path)
    return out_path


def _fit_apply_validation(method_name, df_train, df_valid, location, prefix, meta):
    method = get_method(method_name)
    model = method.fit(df_train)
    df_pred = method.apply(df_valid.copy(), model)

    out = _validation_subset_to_output(
        df_base=df_valid,
        df_corrected=df_pred,
        method_name=f"{prefix}{method_name}",
        meta=meta,
    )

    out_path = format_path(
        "validation",
        location=location,
        corr_method=f"{prefix}{method_name}",
    )
    _save_df(out, out_path)
    return out_path


def _run_local_cv(location, df_pairs, df_hind, saved, methods):
    for name in methods:
        method = get_method(name)

        oof_parts = []
        used_folds = 0

        for split in iter_local_cv_splits(df_pairs):
            df_train = df_pairs.iloc[split["train_idx"]].copy()
            df_test = df_pairs.iloc[split["test_idx"]].copy()

            model = method.fit(df_train)
            df_pred = method.apply(df_test.copy(), model)

            out = _validation_subset_to_output(
                df_base=df_test,
                df_corrected=df_pred,
                method_name=f"localcv_{name}",
                meta={
                    "validation_type": "local_cv",
                    "fold": split["fold"],
                    "test_groups": "|".join(split["test_groups"]),
                    "train_source": location,
                    "apply_target": location,
                },
            )
            oof_parts.append(out)
            used_folds += 1

        if used_folds == 0:
            raise ValueError(f"No valid CV folds generated for local correction at {location}.")

        df_oof = pd.concat(oof_parts, ignore_index=True).sort_values("time")
        val_path = format_path(
            "validation",
            location=location,
            corr_method=f"localcv_{name}",
        )
        _save_df(df_oof, val_path)
        saved[f"localcv_{name}"] = val_path

        corr_path = _fit_apply_save(
            method_name=name,
            df_train=df_pairs,
            df_hind=df_hind,
            location=location,
            prefix="local_",
        )
        saved[f"local_{name}"] = corr_path


def _run_transfer(location, df_hind, df_pairs_target_or_none, saved, methods):
    core_buoys = get_core_buoy_locations()

    if location not in get_buoy_locations():
        return

    for source in core_buoys:
        if source == location:
            continue

        df_train = load_pairs(source)

        for name in methods:
            prefix = f"transfer_{source}_"

            corr_path = _fit_apply_save(
                method_name=name,
                df_train=df_train,
                df_hind=df_hind,
                location=location,
                prefix=prefix,
            )
            saved[f"{prefix}{name}"] = corr_path

            if df_pairs_target_or_none is not None:
                val_path = _fit_apply_validation(
                    method_name=name,
                    df_train=df_train,
                    df_valid=df_pairs_target_or_none,
                    location=location,
                    prefix=prefix,
                    meta={
                        "validation_type": "spatial_transfer",
                        "fold": -1,
                        "test_groups": "",
                        "train_source": source,
                        "apply_target": location,
                    },
                )
                saved[f"validation_{prefix}{name}"] = val_path


def _run_pooled(location, df_hind, df_pairs_target_or_none, saved, methods):
    df_train = load_pooled_pairs(exclude_locations=[], include_only_core=True)

    for name in methods:
        prefix = "pooled_"

        corr_path = _fit_apply_save(
            method_name=name,
            df_train=df_train,
            df_hind=df_hind,
            location=location,
            prefix=prefix,
        )
        saved[f"{prefix}{name}"] = corr_path

        if df_pairs_target_or_none is not None:
            val_path = _fit_apply_validation(
                method_name=name,
                df_train=df_train,
                df_valid=df_pairs_target_or_none,
                location=location,
                prefix=prefix,
                meta={
                    "validation_type": "external_pooled_validation",
                    "fold": -1,
                    "test_groups": "",
                    "train_source": "core_buoys_pooled",
                    "apply_target": location,
                },
            )
            saved[f"validation_{prefix}{name}"] = val_path


def run_bias_correction(location, method=None):
    core_buoys = set(get_core_buoy_locations())
    external_buoys = set(get_external_validation_buoys())
    study_areas = set(get_study_area_locations())
    buoy_locations = set(get_buoy_locations())

    methods = _selected_methods(method)

    df_hind = load_hindcast(location)
    saved = {}

    df_pairs_target = None
    if location in buoy_locations:
        df_pairs_target = load_pairs(location)

    if location in core_buoys:
        print(f"Running LOCAL CV + final local correction for {location}")
        _run_local_cv(location, df_pairs_target, df_hind, saved, methods)

        print(f"Running TRANSFER correction for {location}")
        _run_transfer(location, df_hind, df_pairs_target, saved, methods)

    elif location in external_buoys:
        print(f"Running TRANSFER correction for external validation buoy {location}")
        _run_transfer(location, df_hind, df_pairs_target, saved, methods)

        print(f"Running POOLED correction for external validation buoy {location}")
        _run_pooled(location, df_hind, df_pairs_target, saved, methods)

    elif location in study_areas:
        print(f"Running POOLED correction for study area {location}")
        _run_pooled(location, df_hind, None, saved, methods)

    else:
        raise ValueError(f"Unknown location role for '{location}'.")

    print("\nSaved outputs:")
    for k in sorted(saved):
        print(f"  {k}: {saved[k]}")

    return saved