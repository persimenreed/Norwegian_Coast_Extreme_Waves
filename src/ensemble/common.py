from pathlib import Path

import numpy as np
import pandas as pd

from src.bias_correction.data import load_hindcast
from src.settings import (
    get_buoy_locations,
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


def build_member_specs(member_families, methods, include_family_in_label=False):
    specs = []

    for member_family in member_families:
        for method in methods:
            label = method
            if include_family_in_label:
                label = f"{member_family}_{method}"

            specs.append(
                {
                    "member_family": member_family,
                    "method": method,
                    "label": label,
                }
            )

    return specs


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


def default_transfer_target_locations(methods):
    out = []

    for location in get_external_validation_buoys() + get_study_area_locations():
        if available_transfer_member_families(location, methods, require_validation=False):
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


def _member_columns_from_specs(member_specs):
    return [member_column(spec.get("label", spec["method"])) for spec in member_specs]


def load_validation_member_dataset_specs(
    location,
    member_specs,
    group_label=None,
):
    specs = list(member_specs or [])
    if not specs:
        raise ValueError("No member specs were provided for validation loading.")

    base = None
    parts = []

    for spec in specs:
        member_family = spec["member_family"]
        method = spec["method"]
        label = spec.get("label", method)

        path = validation_path(location, member_family, method)
        df = _read_csv(path)

        if MODEL_CORR not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{MODEL_CORR}'.")

        if OBS not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{OBS}'.")

        if base is None:
            base = _strip_validation_columns(df)

        parts.append(
            df[[TIME, MODEL_CORR]].rename(columns={MODEL_CORR: member_column(label)})
        )

    merged = _merge_member_predictions(parts)
    data = base.merge(merged, on=TIME, how="inner")
    member_cols = _member_columns_from_specs(specs)
    data = data.dropna(subset=[OBS] + member_cols).reset_index(drop=True)
    data[LOCATION] = group_label or location
    return data


def _aggregate_member_frames(base, member_frames, methods):
    if not member_frames:
        raise ValueError("No ensemble member tables were loaded.")

    if len(member_frames) == 1:
        data = base.merge(member_frames[0], on=TIME, how="inner")
        return data.sort_values(TIME).reset_index(drop=True)

    merged = base.copy()

    for family_idx, frame in enumerate(member_frames):
        rename_map = {
            member_column(method): f"{member_column(method)}__family_{family_idx}"
            for method in methods
        }
        merged = merged.merge(
            frame.rename(columns=rename_map),
            on=TIME,
            how="inner",
        )

    temp_cols = []
    for method in methods:
        family_cols = [
            f"{member_column(method)}__family_{family_idx}"
            for family_idx in range(len(member_frames))
        ]
        temp_cols.extend(family_cols)
        merged[member_column(method)] = merged[family_cols].apply(
            pd.to_numeric,
            errors="coerce",
        ).mean(axis=1, skipna=True)

    return (
        merged.drop(columns=temp_cols, errors="ignore")
        .sort_values(TIME)
        .reset_index(drop=True)
    )


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


def load_validation_member_dataset_families(
    location,
    methods,
    member_families,
    group_label=None,
):
    families = list(member_families or [])
    if not families:
        raise ValueError("No member families were provided for validation loading.")

    base = None
    member_frames = []

    for member_family in families:
        df = load_validation_member_dataset(
            location=location,
            methods=methods,
            member_family=member_family,
            group_label=group_label,
        )

        if base is None:
            base = df.drop(
                columns=[member_column(method) for method in methods],
                errors="ignore",
            ).copy()

        member_frames.append(df[[TIME] + [member_column(method) for method in methods]])

    data = _aggregate_member_frames(base, member_frames, methods)
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


def load_hindcast_member_dataset_specs(location, member_specs):
    specs = list(member_specs or [])
    if not specs:
        raise ValueError("No member specs were provided for hindcast loading.")

    base = load_hindcast(location)
    parts = []

    for spec in specs:
        member_family = spec["member_family"]
        method = spec["method"]
        label = spec.get("label", method)

        path = corrected_path(location, member_family, method)
        df = _read_csv(path)

        if MODEL not in df.columns:
            raise ValueError(f"Corrected file {path} is missing '{MODEL}'.")

        parts.append(
            df[[TIME, MODEL]].rename(columns={MODEL: member_column(label)})
        )

    merged = _merge_member_predictions(parts)
    data = base.merge(merged, on=TIME, how="inner")
    data[LOCATION] = location
    return data.sort_values(TIME).reset_index(drop=True)


def load_hindcast_member_dataset_families(location, methods, member_families):
    families = list(member_families or [])
    if not families:
        raise ValueError("No member families were provided for hindcast loading.")

    base = load_hindcast(location)
    member_frames = []

    for member_family in families:
        df = load_hindcast_member_dataset(
            location=location,
            methods=methods,
            member_family=member_family,
        )
        member_frames.append(df[[TIME] + [member_column(method) for method in methods]])

    data = _aggregate_member_frames(base, member_frames, methods)
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


def load_training_validation_specs(specs, methods=None):
    frames = []

    for spec in specs:
        if "member_specs" in spec:
            frames.append(
                load_validation_member_dataset_specs(
                    location=spec["location"],
                    member_specs=spec["member_specs"],
                    group_label=spec.get("group_label"),
                )
            )
            continue

        if methods is None:
            raise ValueError("methods must be provided when specs do not define member_specs.")

        frames.append(
            load_validation_member_dataset(
                location=spec["location"],
                methods=methods,
                member_family=spec["member_family"],
                group_label=spec.get("group_label"),
            )
        )

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values([LOCATION, TIME]).reset_index(drop=True)


def has_validation_members(location, methods, member_family="pooled"):
    return all(validation_path(location, member_family, method).exists() for method in methods)


def has_validation_member_specs(location, member_specs):
    return all(
        validation_path(location, spec["member_family"], spec["method"]).exists()
        for spec in member_specs
    )


def has_corrected_members(location, methods, member_family="pooled"):
    return all(corrected_path(location, member_family, method).exists() for method in methods)


def has_corrected_member_specs(location, member_specs):
    return all(
        corrected_path(location, spec["member_family"], spec["method"]).exists()
        for spec in member_specs
    )


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


def default_training_specs(methods):
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
            "No cross-location validation datasets were found for the core buoys. "
            "Run the transfer bias-correction stage first."
        )

    return specs


def available_transfer_member_families(location, methods, require_validation=True):
    families = []

    for source in get_core_buoy_locations():
        if source == location:
            continue

        member_family = f"transfer_{source}"
        path_check = (
            has_validation_members(location, methods, member_family)
            if require_validation
            else has_corrected_members(location, methods, member_family)
        )
        if path_check:
            families.append(member_family)

    return families


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
    member_families=None,
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
    out["ensemble_input_families"] = "|".join(member_families or [member_family])

    path = output_validation_path(location, output_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def save_hindcast_output(location, df, prediction, output_name, member_families=None):
    out = _drop_internal_columns(df)
    out[MODEL] = prediction
    if member_families:
        out["ensemble_input_families"] = "|".join(member_families)

    path = output_corrected_path(location, output_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def save_ensemble_report(
    output_name,
    training_labels,
    member_family,
    methods,
    class_counts,
    top_features,
    contributions,
):
    out_dir = Path("results/ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"name: {output_name}",
        f"training_cases: {' | '.join(training_labels)}",
        f"application_member_family: {member_family}",
        f"members: {' | '.join(methods)}",
        "",
        "winner_counts:",
    ]

    for method in methods:
        lines.append(f"  {method}: {int(class_counts.get(method, 0))}")

    lines.extend(["", "top_features:"])
    if isinstance(top_features, dict):
        for section, values in top_features.items():
            lines.append(f"  {section}:")
            if values:
                for feature, score in values:
                    lines.append(f"    {feature}: {float(score):.6f}")
            else:
                lines.append("    none")
    elif top_features:
        for feature, score in top_features:
            lines.append(f"  {feature}: {float(score):.6f}")
    else:
        lines.append("  none")

    for location in sorted(contributions):
        stats = contributions[location]
        lines.extend(["", f"location: {location}"])

        if "input_families" in stats:
            lines.append(f"input_families: {' | '.join(stats['input_families'])}")

        for label in ["validation_mean_weights", "hindcast_mean_weights"]:
            if label not in stats:
                continue

            lines.append(f"{label}:")
            for method in methods:
                value = float(stats[label].get(method, 0.0))
                lines.append(f"  {method}: {value:.6f}")

    path = out_dir / f"{output_name}_summary.txt"
    block = "\n".join(lines) + "\n"
    if path.exists():
        with path.open("a", encoding="ascii") as fh:
            fh.write("\n" + ("=" * 60) + "\n\n")
            fh.write(block)
    else:
        path.write_text(block, encoding="ascii")
    return str(path)
