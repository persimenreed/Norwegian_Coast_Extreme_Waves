from pathlib import Path

import pandas as pd

from src.settings import (
    format_path,
    get_buoy_locations,
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_methods,
    get_study_area_locations,
)
from src.bias_correction.data import load_hindcast, load_pairs
from src.bias_correction.methods.common import TIME
from src.bias_correction.registry import get_method
from src.bias_correction.validation import iter_local_cv_splits


def _save_df(df, path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_str, index=False)
    return path_str


def _selected_methods(method=None):
    methods = list(get_methods())
    if method is None:
        return methods
    if method not in methods:
        raise ValueError(f"Unknown bias-correction method '{method}'. Available methods: {methods}")
    return [method]


def _profile_name(method_name, source):
    return f"{method_name}_{source}"


def _fit_model(method_name, df_train, source):
    method = get_method(method_name)
    model = method.fit(df_train, settings_name=_profile_name(method_name, source))
    return method, model


def _save_corrected(location, output_name, df_corrected):
    return _save_df(
        df_corrected,
        format_path("corrected", location=location, corr_method=output_name),
    )


def _save_validation(location, output_name, df_base, df_corrected, meta):
    return _save_df(
        _validation_frame(df_base, df_corrected, output_name, meta),
        format_path("validation", location=location, corr_method=output_name),
    )


def _save_local_feature_importance(location, method_name, model):
    importance = model.get("feature_importance") if isinstance(model, dict) else None
    if method_name != "xgboost" or importance is None or location not in set(get_core_buoy_locations()):
        return None

    path = Path("results") / "bias_correction" / location / "feature_importance_local_xgboost.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return str(path)


def _save_local_training_history(location, method_name, model):
    history = model.get("training_history") if isinstance(model, dict) else None
    if history is None or history.empty or location not in set(get_core_buoy_locations()):
        return None

    out_dir = Path("results") / "bias_correction" / location
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"training_history_local_{method_name}.csv"
    history.to_csv(csv_path, index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return str(csv_path)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    if "val_loss" in history and history["val_loss"].notna().any():
        ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{method_name} training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"training_history_local_{method_name}.png", dpi=200)
    plt.close(fig)
    return str(csv_path)


def _run_local_cv(location, df_pairs, df_hind, saved, methods):
    for method_name in methods:
        oof_parts = []

        for split in iter_local_cv_splits(df_pairs):
            df_train = df_pairs.iloc[split["train_idx"]].copy()
            df_test = df_pairs.iloc[split["test_idx"]].copy()
            method, model = _fit_model(method_name, df_train, location)
            df_pred = method.apply(df_test.copy(), model)
            oof_parts.append(
                _validation_frame(
                    df_base=df_test,
                    df_corrected=df_pred,
                    output_name=f"localcv_{method_name}",
                    meta={
                        "validation_type": "local_cv",
                        "fold": split["fold"],
                        "test_groups": "|".join(split["test_groups"]),
                        "train_source": location,
                        "apply_target": location,
                    },
                )
            )

        if not oof_parts:
            raise ValueError(f"No valid CV folds generated for local correction at {location}.")

        saved[f"localcv_{method_name}"] = _save_df(
            pd.concat(oof_parts, ignore_index=True).sort_values(TIME),
            format_path("validation", location=location, corr_method=f"localcv_{method_name}"),
        )

        method, model = _fit_model(method_name, df_pairs, location)
        saved[f"local_{method_name}"] = _save_corrected(
            location,
            f"local_{method_name}",
            method.apply(df_hind.copy(), model),
        )
        _save_local_feature_importance(location, method_name, model)
        _save_local_training_history(location, method_name, model)


def _validation_frame(df_base, df_corrected, output_name, meta):
    out = df_base.copy()
    out["corr_method"] = output_name

    for column in df_corrected.columns:
        if column != TIME and column in out.columns:
            out[f"{column}_corrected"] = df_corrected[column].to_numpy()

    for key, value in meta.items():
        out[key] = value

    return out


def _run_transfer(location, df_hind, df_pairs_target, saved, methods):
    known_transfer_targets = set(get_buoy_locations()) | set(get_study_area_locations())
    if location not in known_transfer_targets:
        return

    for source in get_core_buoy_locations():
        if source == location:
            continue

        df_train = load_pairs(source)
        for method_name in methods:
            output_name = f"transfer_{source}_{method_name}"
            method, model = _fit_model(method_name, df_train, source)

            saved[output_name] = _save_corrected(
                location,
                output_name,
                method.apply(df_hind.copy(), model),
            )

            if df_pairs_target is not None:
                saved[f"validation_{output_name}"] = _save_validation(
                    location,
                    output_name,
                    df_pairs_target,
                    method.apply(df_pairs_target.copy(), model),
                    {
                        "validation_type": "spatial_transfer",
                        "fold": -1,
                        "test_groups": "",
                        "train_source": source,
                        "apply_target": location,
                    },
                )


def run_bias_correction(location, method=None):
    core_buoys = set(get_core_buoy_locations())
    external_buoys = set(get_external_validation_buoys())
    study_areas = set(get_study_area_locations())
    buoy_locations = set(get_buoy_locations())

    methods = _selected_methods(method)
    df_hind = load_hindcast(location)
    df_pairs_target = load_pairs(location) if location in buoy_locations else None
    saved = {}

    if location in core_buoys:
        print(f"Running LOCAL CV + final local correction for {location}")
        _run_local_cv(location, df_pairs_target, df_hind, saved, methods)
        print(f"Running TRANSFER correction for {location}")
        _run_transfer(location, df_hind, df_pairs_target, saved, methods)
    elif location in external_buoys:
        print(f"Running TRANSFER correction for external validation buoy {location}")
        _run_transfer(location, df_hind, df_pairs_target, saved, methods)
    elif location in study_areas:
        print(f"Running TRANSFER correction for study area {location}")
        _run_transfer(location, df_hind, None, saved, methods)
    else:
        raise ValueError(f"Unknown location role for '{location}'.")

    print("\nSaved outputs:")
    for key in sorted(saved):
        print(f"  {key}: {saved[key]}")

    return saved
