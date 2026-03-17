import numpy as np

from src.ensemble.common import (
    available_transfer_member_families,
    build_member_specs,
    build_oof_predictions,
    has_corrected_member_specs,
    has_validation_member_specs,
    load_hindcast_member_dataset_specs,
    load_training_validation_specs,
    load_validation_member_dataset_specs,
    normalize_methods,
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
    get_study_area_locations,
    get_validation_settings,
)


def _combined_member_labels(methods):
    core_buoys = get_core_buoy_locations()
    return [
        f"transfer_{source}_{method}"
        for source in core_buoys
        for method in methods
    ]


def _single_source_training_specs(source, methods):
    member_specs = build_member_specs(["localcv"], methods)
    if not has_validation_member_specs(source, member_specs):
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
        member_specs = [
            {
                "member_family": "localcv",
                "method": method,
                "label": f"transfer_{source}_{method}",
            }
            for method in methods
        ]
        if not has_validation_member_specs(source, member_specs):
            continue

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


def _target_member_specs(location, methods, source=None, combined=False, require_validation=False):
    if combined:
        families = available_transfer_member_families(
            location,
            methods,
            require_validation=require_validation,
        )
        return build_member_specs(
            families,
            methods,
            include_family_in_label=True,
        )

    if source is None:
        raise ValueError("source must be provided for a single-source ensemble.")

    member_specs = build_member_specs([f"transfer_{source}"], methods)
    exists = (
        has_validation_member_specs(location, member_specs)
        if require_validation
        else has_corrected_member_specs(location, member_specs)
    )
    if not exists:
        return []

    return member_specs


def _default_target_locations(methods, source=None, combined=False):
    candidates = (
        get_core_buoy_locations()
        + get_external_validation_buoys()
        + get_study_area_locations()
    )

    out = []
    for location in candidates:
        member_specs = _target_member_specs(
            location,
            methods,
            source=source,
            combined=combined,
            require_validation=False,
        )
        if member_specs:
            out.append(location)

    return unique_locations(out)


def _training_members(methods, source=None, combined=False):
    if combined:
        return _combined_member_labels(methods)
    return list(methods)


def _mean_weight_map(weights, members):
    return {
        member: float(weights[:, idx].mean())
        for idx, member in enumerate(members)
    }


def run(
    location=None,
    methods=None,
    source=None,
    combined=False,
    output_name=None,
):
    methods = normalize_methods(methods)
    training_members = _training_members(methods, source=source, combined=combined)
    settings_name = (
        "ensemble_xgboost"
        if combined or source is None
        else f"ensemble_xgboost_{source}"
    )

    if combined:
        training_specs = _combined_training_specs(methods)
        apply_member_family = "transfer_combined"
        output_name = output_name or "ensemble_combined"
    else:
        if source is None:
            raise ValueError("source must be provided when combined=False.")
        training_specs = _single_source_training_specs(source, methods)
        apply_member_family = f"transfer_{source}"
        output_name = output_name or f"ensemble_{source}"

    target_locations = unique_locations(
        [location] if location else _default_target_locations(methods, source=source, combined=combined)
    )
    training_labels = [spec["label"] for spec in training_specs]

    train_df = load_training_validation_specs(training_specs)
    bundle = fit_state_corrected_ensemble(
        train_df,
        training_members,
        settings_name=settings_name,
    )

    n_splits = int(get_validation_settings().get("local_cv_folds", 4))
    oof_pred = build_oof_predictions(
        train_df,
        fit_fn=lambda fold_df: fit_state_corrected_ensemble(
            fold_df,
            training_members,
            settings_name=settings_name,
        ),
        predict_fn=predict_state_corrected_ensemble,
        n_splits=n_splits,
    )

    saved_validation = {}
    saved_hindcast = {}
    contributions = {}

    training_targets = set()
    if "apply_target" in train_df.columns:
        training_targets = set(train_df["apply_target"].dropna().astype(str).tolist())

    for target_location in target_locations:
        hindcast_member_specs = _target_member_specs(
            target_location,
            methods,
            source=source,
            combined=combined,
            require_validation=False,
        )
        if not hindcast_member_specs:
            continue

        input_families = unique_locations(
            [spec["member_family"] for spec in hindcast_member_specs]
        )
        contributions.setdefault(target_location, {})["input_families"] = input_families

        validation_member_specs = _target_member_specs(
            target_location,
            methods,
            source=source,
            combined=combined,
            require_validation=True,
        )

        if target_location in training_targets:
            mask = train_df["apply_target"].astype(str) == target_location
            df_val = train_df.loc[mask].copy().reset_index(drop=True)
            pred = np.asarray(oof_pred[mask.to_numpy()], dtype=float)
            saved_validation[target_location] = save_validation_output(
                location=target_location,
                df=df_val,
                prediction=pred,
                output_name=output_name,
                train_locations=training_labels,
                member_family=apply_member_family,
                member_families=input_families,
                methods=training_members,
                validation_type="ensemble_oof",
            )
        elif validation_member_specs:
            df_val = load_validation_member_dataset_specs(
                location=target_location,
                member_specs=validation_member_specs,
            )
            pred, weights = predict_state_corrected_ensemble(
                df_val,
                bundle,
                return_weights=True,
            )
            saved_validation[target_location] = save_validation_output(
                location=target_location,
                df=df_val,
                prediction=pred,
                output_name=output_name,
                train_locations=training_labels,
                member_family=apply_member_family,
                member_families=input_families,
                methods=training_members,
                validation_type="ensemble_external_apply",
            )
            contributions[target_location]["validation_mean_weights"] = _mean_weight_map(
                weights,
                training_members,
            )

        df_hind = load_hindcast_member_dataset_specs(
            target_location,
            hindcast_member_specs,
        )
        pred, weights = predict_state_corrected_ensemble(
            df_hind,
            bundle,
            return_weights=True,
        )
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
        member_family=apply_member_family,
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
            [spec["member_family"] for training_spec in training_specs for spec in training_spec["member_specs"]]
        ),
        "application_member_family": apply_member_family,
        "class_counts": bundle["class_counts"],
        "top_features": bundle["top_features"],
        "validation_paths": saved_validation,
        "hindcast_paths": saved_hindcast,
        "report_path": report_path,
    }
