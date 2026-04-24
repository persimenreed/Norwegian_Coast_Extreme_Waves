from pathlib import Path

import numpy as np
import pandas as pd

from src.bias_correction.data import load_hindcast
from src.settings import format_path

TIME = "time"
OBS = "Significant_Wave_Height_Hm0"
MODEL = "hs"
MODEL_CORR = "hs_corrected"
LOCATION = "ensemble_location"


def member_column(label):
    return f"member_{label}"


def unique_locations(locations):
    out = []
    for location in locations:
        if location not in out:
            out.append(location)
    return out


def _member_path(location, spec, validation):
    return Path(
        format_path(
            "validation" if validation else "corrected",
            location=location,
            corr_method=f"{spec['member_family']}_{spec['method']}",
        )
    )


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if TIME in df.columns:
        df[TIME] = pd.to_datetime(df[TIME], errors="coerce")
        df = df.sort_values(TIME).reset_index(drop=True)
    return df


def member_specs_exist(location, member_specs, validation):
    return all(_member_path(location, spec, validation).exists() for spec in member_specs)


def _merge_member_predictions(parts):
    if not parts:
        raise ValueError("No ensemble member tables were loaded.")

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=TIME, how="inner")
    return merged.sort_values(TIME).reset_index(drop=True)


def _base_validation_frame(df):
    keep = [
        column
        for column in df.columns
        if column != "corr_method" and not str(column).endswith("_corrected")
    ]
    return df[keep].copy()


def load_validation_dataset(location, member_specs, group_label=None):
    member_specs = list(member_specs or [])
    if not member_specs:
        raise ValueError("No member specs were provided for validation loading.")

    base = None
    parts = []
    member_columns = []

    for spec in member_specs:
        path = _member_path(location, spec, validation=True)
        df = _read_csv(path)

        if MODEL_CORR not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{MODEL_CORR}'.")
        if OBS not in df.columns:
            raise ValueError(f"Validation file {path} is missing '{OBS}'.")

        if base is None:
            base = _base_validation_frame(df)

        label = spec.get("label", spec["method"])
        column = member_column(label)
        member_columns.append(column)
        parts.append(df[[TIME, MODEL_CORR]].rename(columns={MODEL_CORR: column}))

    data = base.merge(_merge_member_predictions(parts), on=TIME, how="inner")
    data = data.dropna(subset=[OBS] + member_columns).reset_index(drop=True)
    data[LOCATION] = group_label or location
    return data


def load_hindcast_dataset(location, member_specs):
    member_specs = list(member_specs or [])
    if not member_specs:
        raise ValueError("No member specs were provided for hindcast loading.")

    parts = []
    for spec in member_specs:
        path = _member_path(location, spec, validation=False)
        df = _read_csv(path)
        if MODEL not in df.columns:
            raise ValueError(f"Corrected file {path} is missing '{MODEL}'.")

        label = spec.get("label", spec["method"])
        parts.append(df[[TIME, MODEL]].rename(columns={MODEL: member_column(label)}))

    data = load_hindcast(location).merge(_merge_member_predictions(parts), on=TIME, how="inner")
    data[LOCATION] = location
    return data.sort_values(TIME).reset_index(drop=True)


def load_training_validation_data(training_specs):
    frames = [
        load_validation_dataset(
            location=spec["location"],
            member_specs=spec["member_specs"],
            group_label=spec.get("group_label"),
        )
        for spec in training_specs
    ]
    return pd.concat(frames, ignore_index=True).sort_values([LOCATION, TIME]).reset_index(drop=True)


def grouped_time_folds(df, n_splits):
    if int(n_splits) < 2:
        return []

    by_group = {}
    for _, group in df.groupby(LOCATION, sort=False):
        idx = group.index.to_numpy()
        splits = [split for split in np.array_split(idx, int(n_splits)) if len(split)]
        if splits:
            by_group[str(group[LOCATION].iloc[0])] = splits

    if not by_group:
        return []

    all_idx = np.arange(len(df))
    folds = []
    for fold_id in range(max(len(splits) for splits in by_group.values())):
        test_parts = [splits[fold_id] for splits in by_group.values() if fold_id < len(splits)]
        if not test_parts:
            continue

        test_idx = np.concatenate(test_parts)
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]

        if len(train_idx) and len(test_idx):
            folds.append((train_idx, test_idx))

    return folds


def build_oof_predictions(df, fit_fn, predict_fn, n_splits):
    folds = grouped_time_folds(df, n_splits)
    if not folds:
        return None

    pred = np.full(len(df), np.nan, dtype=float)
    for train_idx, test_idx in folds:
        bundle = fit_fn(df.iloc[train_idx].copy().reset_index(drop=True))
        pred[test_idx] = predict_fn(df.iloc[test_idx].copy().reset_index(drop=True), bundle)

    if not np.all(np.isfinite(pred)):
        raise ValueError(f"Failed to generate OOF predictions for {int(np.sum(~np.isfinite(pred)))} rows.")

    return pred


def _output_path(location, output_name, validation):
    return Path(
        format_path(
            "validation" if validation else "corrected",
            location=location,
            corr_method=output_name,
        )
    )


def _drop_internal_columns(df):
    drop = [LOCATION]
    drop.extend(column for column in df.columns if str(column).startswith("member_"))
    return df.drop(columns=drop, errors="ignore").copy()


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

    path = _output_path(location, output_name, validation=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def save_hindcast_output(location, df, prediction, output_name, member_families=None):
    out = _drop_internal_columns(df)
    out[MODEL] = prediction
    if member_families:
        out["ensemble_input_families"] = "|".join(member_families)

    path = _output_path(location, output_name, validation=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path)


def _season_from_month(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    return np.nan


def _wind_sector(direction):
    if not np.isfinite(direction):
        return np.nan

    direction = float(direction) % 360.0
    if direction >= 315.0 or direction < 45.0:
        return "North"
    if direction < 135.0:
        return "East"
    if direction < 225.0:
        return "South"
    return "West"


def _hs_percentile_bin(values):
    hs = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=hs.index, dtype=object)
    finite = hs[hs.notna()]
    if finite.empty:
        return out

    q50, q75, q95 = np.nanquantile(finite.to_numpy(dtype=float), [0.50, 0.75, 0.95])
    out.loc[hs <= q50] = "0-50"
    out.loc[(hs > q50) & (hs <= q75)] = "50-75"
    out.loc[(hs > q75) & (hs <= q95)] = "75-95"
    out.loc[hs > q95] = "95-100"
    return out


def _weight_summary_path(application_label):
    out_dir = Path("results") / "ensemble" / "weight_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{application_label}_weight_summary.csv"


def _expert_label(expert):
    text = str(expert)
    replacements = {
        "transfer_fauskane_": "Fauskane ",
        "transfer_fedjeosen_": "Fedjeosen ",
        "linear": "Linear",
        "pqm": "PQM",
        "dagqm": "DAGQM",
        "gpr": "GPR",
        "xgboost": "XGBoost",
        "transformer": "Transformer",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ").strip()


def save_weight_summary(
    location,
    df,
    weights,
    output_name,
    methods,
    application_label,
    member_families=None,
):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")
    if len(df) != weights.shape[0]:
        raise ValueError(
            f"Length mismatch between data frame ({len(df)}) and weights ({weights.shape[0]})."
        )
    if len(methods) != weights.shape[1]:
        raise ValueError(
            f"Length mismatch between methods ({len(methods)}) and weights ({weights.shape[1]})."
        )

    meta = pd.DataFrame()
    meta[TIME] = pd.to_datetime(df[TIME], errors="coerce")
    meta["season"] = meta[TIME].dt.month.map(_season_from_month)

    direction_column = next(
        (column for column in ("wind_direction_10m", "wind_direction_20m", "Pdir", "thq") if column in df.columns),
        None,
    )
    if direction_column is not None:
        meta["wind_sector"] = pd.to_numeric(df[direction_column], errors="coerce").map(_wind_sector)
    else:
        meta["wind_sector"] = np.nan

    hs_column = OBS if OBS in df.columns else MODEL if MODEL in df.columns else None
    if hs_column is not None:
        meta["hs_percentile"] = _hs_percentile_bin(df[hs_column])
    else:
        meta["hs_percentile"] = np.nan

    grouping_specs = [
        ("wind_sector", ["North", "East", "South", "West"]),
        ("season", ["DJF", "MAM", "JJA", "SON"]),
        ("hs_percentile", ["0-50", "50-75", "75-95", "95-100"]),
    ]

    rows = []
    input_families = "|".join(member_families or [])
    for grouping, labels in grouping_specs:
        groups = meta[grouping].astype(object)
        for label in labels:
            mask = groups == label
            n = int(mask.sum())
            if n <= 0:
                continue
            group_weights = weights[mask.to_numpy(), :]
            for idx, method in enumerate(methods):
                values = group_weights[:, idx]
                rows.append(
                    {
                        "application": application_label,
                        "location": location,
                        "output_name": output_name,
                        "dataset_kind": "hindcast",
                        "input_families": input_families,
                        "grouping": grouping,
                        "bin": label,
                        "n": n,
                        "expert": method,
                        "expert_label": _expert_label(method),
                        "mean_weight": float(np.nanmean(values)),
                    }
                )

    out = pd.DataFrame(rows)
    if member_families:
        out["input_families"] = "|".join(member_families)

    path = _weight_summary_path(application_label)
    out.to_csv(path, index=False)
    return str(path)


def save_ensemble_report(
    output_name,
    training_labels,
    member_family,
    methods,
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
    ]

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

        for label in ("validation_mean_weights", "hindcast_mean_weights"):
            if label not in stats:
                continue
            lines.append(f"{label}:")
            for method in methods:
                lines.append(f"  {method}: {float(stats[label].get(method, 0.0)):.6f}")

    path = out_dir / f"{output_name}_summary.txt"
    block = "\n".join(lines) + "\n"
    if path.exists():
        with path.open("a", encoding="ascii") as handle:
            handle.write("\n" + ("=" * 60) + "\n\n")
            handle.write(block)
    else:
        path.write_text(block, encoding="ascii")
    return str(path)
