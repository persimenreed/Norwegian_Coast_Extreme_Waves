import numpy as np

from src.ensemble.common import (
    available_transfer_member_families,
    build_oof_predictions,
    default_training_specs,
    default_transfer_target_locations,
    load_hindcast_member_dataset_families,
    load_training_validation_specs,
    load_validation_member_dataset_families,
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
from src.settings import get_validation_settings


def run(
    location=None,
    methods=None,
    output_name="ensemble_transfer",
):
    methods = normalize_methods(methods)
    target_locations = unique_locations(
        [location] if location else default_transfer_target_locations(methods)
    )
    training_specs = default_training_specs(methods)
    training_labels = [spec["label"] for spec in training_specs]

    train_df = load_training_validation_specs(training_specs, methods)
    bundle = fit_state_corrected_ensemble(train_df, methods)

    n_splits = int(get_validation_settings().get("local_cv_folds", 4))
    oof_pred = build_oof_predictions(
        train_df,
        fit_fn=lambda fold_df: fit_state_corrected_ensemble(fold_df, methods),
        predict_fn=predict_state_corrected_ensemble,
        n_splits=n_splits,
    )

    saved_validation = {}
    saved_hindcast = {}
    contributions = {}
    apply_member_family = "transfer_mean"

    training_targets = set()
    if "apply_target" in train_df.columns:
        training_targets = set(train_df["apply_target"].dropna().astype(str).tolist())

    for target_location in target_locations:
        member_families = available_transfer_member_families(
            target_location,
            methods,
            require_validation=False,
        )
        if not member_families:
            continue

        validation_member_families = available_transfer_member_families(
            target_location,
            methods,
            require_validation=True,
        )

        contributions.setdefault(target_location, {})["input_families"] = member_families

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
                member_families=member_families,
                methods=methods,
                validation_type="ensemble_transfer_oof",
            )
        else:
            if not validation_member_families:
                df_val = None
            else:
                df_val = load_validation_member_dataset_families(
                    location=target_location,
                    methods=methods,
                    member_families=validation_member_families,
                )
            if df_val is not None:
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
                    member_families=validation_member_families,
                    methods=methods,
                    validation_type="ensemble_transfer_external_apply",
                )
                contributions[target_location]["validation_mean_weights"] = {
                    method: float(weights[:, idx].mean())
                    for idx, method in enumerate(methods)
                }

        df_hind = load_hindcast_member_dataset_families(
            target_location,
            methods,
            member_families,
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
            member_families=member_families,
        )
        contributions[target_location]["hindcast_mean_weights"] = {
            method: float(weights[:, idx].mean())
            for idx, method in enumerate(methods)
        }

    report_path = save_ensemble_report(
        output_name=output_name,
        training_labels=training_labels,
        member_family=apply_member_family,
        methods=methods,
        class_counts=bundle["class_counts"],
        top_features=bundle["top_features"],
        contributions=contributions,
    )

    return {
        "name": output_name,
        "training_labels": training_labels,
        "target_locations": target_locations,
        "training_member_families": [spec["member_family"] for spec in training_specs],
        "application_member_family": apply_member_family,
        "class_counts": bundle["class_counts"],
        "top_features": bundle["top_features"],
        "validation_paths": saved_validation,
        "hindcast_paths": saved_hindcast,
        "report_path": report_path,
    }
