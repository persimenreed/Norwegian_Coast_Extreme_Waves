from pathlib import Path

import numpy as np

from src.ensemble.common import (
    build_oof_predictions,
    load_hindcast_dataset,
    load_training_validation_data,
    load_validation_dataset,
    member_specs_exist,
    save_ensemble_report,
    save_hindcast_output,
    save_validation_output,
    unique_locations,
)
from src.ensemble.xgboost_core import (
    fit_state_corrected_ensemble,
    predict_state_corrected_ensemble,
)
from src.settings import (
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_methods,
    get_study_area_locations,
)

ENSEMBLE_OOF_FOLDS = 4


def _selected_methods(methods=None):
    available = list(get_methods())
    if methods is None:
        return available

    selected = []
    for method in methods:
        if method not in available:
            raise ValueError(f"Unknown ensemble member '{method}'. Available methods: {available}")
        if method not in selected:
            selected.append(method)

    if not selected:
        raise ValueError("No valid ensemble members were selected.")
    return selected


def _member_specs(member_family, methods, label_prefix=None):
    return [
        {
            "member_family": member_family,
            "method": method,
            "label": f"{label_prefix}_{method}" if label_prefix else method,
        }
        for method in methods
    ]


def _available_transfer_families(location, methods, validation):
    families = []
    for source in get_core_buoy_locations():
        if source == location:
            continue
        member_family = f"transfer_{source}"
        if member_specs_exist(
            location,
            _member_specs(member_family, methods, label_prefix=member_family),
            validation=validation,
        ):
            families.append(member_family)
    return families


def _single_source_training_specs(source, methods):
    member_specs = _member_specs("localcv", methods)
    if not member_specs_exist(source, member_specs, validation=True):
        raise ValueError(
            f"No local CV validation datasets were found for ensemble source '{source}'. "
            "Run the local bias-correction stage first."
        )

    return [
        {
            "location": source,
            "member_specs": member_specs,
            "group_label": source,
            "label": source,
        }
    ]


def _combined_training_specs(methods):
    specs = []

    for source in get_core_buoy_locations():
        member_specs = _member_specs("localcv", methods, label_prefix=f"transfer_{source}")
        if member_specs_exist(source, member_specs, validation=True):
            specs.append(
                {
                    "location": source,
                    "member_specs": member_specs,
                    "group_label": source,
                    "label": source,
                }
            )

    if not specs:
        raise ValueError(
            "No core-buoy local CV validation datasets were found for the combined ensemble. "
            "Run the local bias-correction stage first."
        )

    return specs


def build_training_setup(source=None, methods=None):
    methods = _selected_methods(methods)

    if source is None:
        training_specs = _combined_training_specs(methods)
        return {
            "methods": methods,
            "source": None,
            "combined": True,
            "training_specs": training_specs,
            "training_members": [
                spec["label"]
                for training_spec in training_specs
                for spec in training_spec["member_specs"]
            ],
            "profile_name": "ensemble_xgboost",
            "study_name": "ensemble_combined_spatial_cv",
            "default_output_name": "ensemble_combined",
        }

    if source not in get_core_buoy_locations():
        raise ValueError(
            f"Unknown core-buoy source '{source}'. Available: {get_core_buoy_locations()}"
        )

    return {
        "methods": methods,
        "source": source,
        "combined": False,
        "training_specs": _single_source_training_specs(source, methods),
        "training_members": list(methods),
        "profile_name": f"ensemble_xgboost_{source}",
        "study_name": f"ensemble_{source}_spatial_cv",
        "default_output_name": f"ensemble_{source}",
    }


def _target_member_specs(location, methods, source=None, combined=False, validation=False):
    if combined:
        specs = []
        for family in _available_transfer_families(location, methods, validation=validation):
            specs.extend(_member_specs(family, methods, label_prefix=family))
        return specs

    if source is None:
        raise ValueError("source must be provided for a single-source ensemble.")

    member_family = "localcv" if validation else "local"
    if source != location:
        member_family = f"transfer_{source}"
    specs = _member_specs(member_family, methods)
    return specs if member_specs_exist(location, specs, validation=validation) else []


def _default_target_locations(methods, source=None, combined=False):
    locations = (
        get_core_buoy_locations()
        + get_external_validation_buoys()
        + get_study_area_locations()
    )
    return unique_locations(
        location
        for location in locations
        if _target_member_specs(location, methods, source=source, combined=combined, validation=False)
    )


def _application_family(location, source, combined):
    if combined:
        return "transfer_combined"
    return "local" if location is not None and location == source else f"transfer_{source}"


def _mean_weight_map(weights, members):
    return {member: float(weights[:, idx].mean()) for idx, member in enumerate(members)}


def _save_feature_importance(location, output_name, bundle):
    importance = bundle.get("feature_importance") if isinstance(bundle, dict) else None
    if importance is None:
        return None

    out_dir = Path("results") / "bias_correction" / location
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"feature_importance_moe_{output_name}.csv"
    importance.to_csv(path, index=False)
    return str(path)


def run(location=None, methods=None, source=None, combined=False, output_name=None):
    setup = build_training_setup(source=None if combined else source, methods=methods)
    methods = setup["methods"]
    source = setup["source"]
    combined = setup["combined"]
    training_specs = setup["training_specs"]
    training_members = setup["training_members"]
    profile_name = setup["profile_name"]
    output_name = output_name or setup["default_output_name"]

    target_locations = unique_locations(
        [location] if location else _default_target_locations(methods, source=source, combined=combined)
    )
    training_labels = [spec["label"] for spec in training_specs]

    train_df = load_training_validation_data(training_specs)
    bundle = fit_state_corrected_ensemble(
        train_df,
        training_members,
        profile_name=profile_name,
    )
    oof_pred = build_oof_predictions(
        train_df,
        fit_fn=lambda df_fold: fit_state_corrected_ensemble(
            df_fold,
            training_members,
            profile_name=profile_name,
        ),
        predict_fn=predict_state_corrected_ensemble,
        n_splits=ENSEMBLE_OOF_FOLDS,
    )

    saved_validation = {}
    saved_hindcast = {}
    contributions = {}
    training_targets = set(train_df["apply_target"].dropna().astype(str).tolist()) if "apply_target" in train_df.columns else set()

    for target_location in target_locations:
        hindcast_specs = _target_member_specs(
            target_location,
            methods,
            source=source,
            combined=combined,
            validation=False,
        )
        if not hindcast_specs:
            continue

        input_families = unique_locations(spec["member_family"] for spec in hindcast_specs)
        family_label = "|".join(input_families)
        contributions.setdefault(target_location, {})["input_families"] = input_families
        _save_feature_importance(target_location, output_name, bundle)

        validation_specs = _target_member_specs(
            target_location,
            methods,
            source=source,
            combined=combined,
            validation=True,
        )

        if target_location in training_targets:
            mask = train_df["apply_target"].astype(str) == target_location
            df_val = train_df.loc[mask].copy().reset_index(drop=True)
            saved_validation[target_location] = save_validation_output(
                location=target_location,
                df=df_val,
                prediction=np.asarray(oof_pred[mask.to_numpy()], dtype=float),
                output_name=output_name,
                train_locations=training_labels,
                member_family=family_label,
                member_families=input_families,
                methods=training_members,
                validation_type="ensemble_oof",
            )
        elif validation_specs:
            df_val = load_validation_dataset(
                location=target_location,
                member_specs=validation_specs,
            )
            pred, weights = predict_state_corrected_ensemble(df_val, bundle, return_weights=True)
            saved_validation[target_location] = save_validation_output(
                location=target_location,
                df=df_val,
                prediction=pred,
                output_name=output_name,
                train_locations=training_labels,
                member_family=family_label,
                member_families=input_families,
                methods=training_members,
                validation_type="ensemble_external_apply",
            )
            contributions[target_location]["validation_mean_weights"] = _mean_weight_map(
                weights,
                training_members,
            )

        df_hind = load_hindcast_dataset(target_location, hindcast_specs)
        pred, weights = predict_state_corrected_ensemble(df_hind, bundle, return_weights=True)
        saved_hindcast[target_location] = save_hindcast_output(
            location=target_location,
            df=df_hind,
            prediction=pred,
            output_name=output_name,
            member_families=input_families,
        )
        contributions[target_location]["hindcast_mean_weights"] = _mean_weight_map(
            weights,
            training_members,
        )

    report_path = save_ensemble_report(
        output_name=output_name,
        training_labels=training_labels,
        member_family=_application_family(location, source, combined),
        methods=training_members,
        class_counts=bundle["class_counts"],
        top_features=bundle["top_features"],
        contributions=contributions,
    )

    return {
        "name": output_name,
        "training_labels": training_labels,
        "target_locations": target_locations,
        "training_member_families": unique_locations(
            spec["member_family"]
            for training_spec in training_specs
            for spec in training_spec["member_specs"]
        ),
        "application_member_family": _application_family(location, source, combined),
        "class_counts": bundle["class_counts"],
        "top_features": bundle["top_features"],
        "validation_paths": saved_validation,
        "hindcast_paths": saved_hindcast,
        "report_path": report_path,
    }
