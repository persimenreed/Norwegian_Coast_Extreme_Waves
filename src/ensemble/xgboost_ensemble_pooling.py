from src.ensemble.common import (
    default_target_locations,
    default_training_specs,
    has_validation_members,
    load_hindcast_member_dataset,
    load_training_validation_specs,
    load_validation_member_dataset,
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


def run(
    location=None,
    methods=None,
    output_name="ensemble_pooling",
):
    methods = normalize_methods(methods)
    target_locations = unique_locations(
        [location] if location else default_target_locations()
    )
    training_specs = default_training_specs(methods)
    training_labels = [spec["label"] for spec in training_specs]

    train_df = load_training_validation_specs(training_specs, methods)
    bundle = fit_state_corrected_ensemble(train_df, methods)

    saved_validation = {}
    saved_hindcast = {}
    contributions = {}
    apply_member_family = "pooled"

    for target_location in target_locations:
        if has_validation_members(target_location, methods, apply_member_family):
            df_val = load_validation_member_dataset(
                target_location,
                methods,
                apply_member_family,
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
                methods=methods,
                validation_type="ensemble_pooling_external_apply",
            )
            contributions.setdefault(target_location, {})["validation_mean_weights"] = {
                method: float(weights[:, idx].mean())
                for idx, method in enumerate(methods)
            }

        df_hind = load_hindcast_member_dataset(
            target_location,
            methods,
            apply_member_family,
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
        )
        contributions.setdefault(target_location, {})["hindcast_mean_weights"] = {
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
