import numpy as np
import pandas as pd

from src.ensemble.common import (
    LOCATION,
    OBS,
    build_oof_predictions,
    default_target_locations,
    default_training_locations,
    has_validation_members,
    load_hindcast_member_dataset,
    load_training_validation_data,
    load_validation_member_dataset,
    member_column,
    normalize_methods,
    save_hindcast_output,
    save_validation_output,
    unique_locations,
)
from src.eval_metrics.core import compute_metrics

RANK_METRICS = [
    "rmse",
    "mae",
    "twrmse",
    "tail_rmse_95",
    "quantile_score_95",
    "abs_q95_bias",
]


def score_members(df, methods):
    obs = pd.to_numeric(df[OBS], errors="coerce").to_numpy(dtype=float)
    rows = []

    for method in methods:
        pred = pd.to_numeric(
            df[member_column(method)],
            errors="coerce",
        ).to_numpy(dtype=float)

        row = compute_metrics(method, pred, obs)
        row["abs_q95_bias"] = abs(row["q95_bias"]) if np.isfinite(row["q95_bias"]) else np.inf
        rows.append(row)

    metrics = pd.DataFrame(rows).set_index("method")

    for metric in RANK_METRICS:
        metrics[f"rank_{metric}"] = metrics[metric].rank(method="average")

    rank_cols = [f"rank_{metric}" for metric in RANK_METRICS]
    metrics["rank_score"] = metrics[rank_cols].mean(axis=1)
    metrics = metrics.sort_values(["rank_score", "rmse", "tail_rmse_95", "twrmse"])
    return metrics


def fit_simple_average(df, methods, top_k):
    metrics = score_members(df, methods)
    top_k = max(1, min(int(top_k), len(methods)))
    selected = metrics.index[:top_k].tolist()

    return {
        "methods": list(methods),
        "selected_methods": selected,
        "weights": {
            method: (1.0 / len(selected) if method in selected else 0.0)
            for method in methods
        },
        "member_metrics": metrics,
    }


def predict_simple_average(df, bundle):
    selected_cols = [member_column(method) for method in bundle["selected_methods"]]
    pred = df[selected_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    if pred.isna().any():
        fallback_cols = [member_column(method) for method in bundle["methods"]]
        fallback = df[fallback_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        pred = pred.fillna(fallback)

    values = pred.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        missing = int(np.sum(~np.isfinite(values)))
        raise ValueError(f"Unable to produce simple ensemble predictions for {missing} rows.")

    return np.clip(values, 0.0, None)


def run(
    train_locations=None,
    target_locations=None,
    methods=None,
    top_k=3,
    cv_folds=5,
    output_name="ensemble",
    member_family="pooled",
):
    methods = normalize_methods(methods)
    train_locations = unique_locations(
        train_locations or default_training_locations(methods, member_family)
    )
    target_locations = unique_locations(
        target_locations or default_target_locations()
    )

    train_df = load_training_validation_data(
        locations=train_locations,
        methods=methods,
        member_family=member_family,
    )
    bundle = fit_simple_average(train_df, methods, top_k)

    saved_validation = {}
    oof_pred = build_oof_predictions(
        train_df,
        fit_fn=lambda frame: fit_simple_average(frame, methods, top_k),
        predict_fn=predict_simple_average,
        n_splits=cv_folds,
    )

    if oof_pred is not None:
        for location in target_locations:
            mask = train_df[LOCATION] == location
            if not np.any(mask):
                continue
            saved_validation[location] = save_validation_output(
                location=location,
                df=train_df.loc[mask].reset_index(drop=True),
                prediction=oof_pred[mask.to_numpy()],
                output_name=output_name,
                train_locations=train_locations,
                member_family=member_family,
                methods=methods,
                validation_type="ensemble_oof",
            )

    for location in target_locations:
        if location in saved_validation:
            continue
        if not has_validation_members(location, methods, member_family):
            continue

        df_val = load_validation_member_dataset(location, methods, member_family)
        pred = predict_simple_average(df_val, bundle)
        saved_validation[location] = save_validation_output(
            location=location,
            df=df_val,
            prediction=pred,
            output_name=output_name,
            train_locations=train_locations,
            member_family=member_family,
            methods=methods,
            validation_type="ensemble_apply",
        )

    saved_hindcast = {}
    for location in target_locations:
        df_hind = load_hindcast_member_dataset(location, methods, member_family)
        pred = predict_simple_average(df_hind, bundle)
        saved_hindcast[location] = save_hindcast_output(
            location=location,
            df=df_hind,
            prediction=pred,
            output_name=output_name,
        )

    return {
        "name": output_name,
        "train_locations": train_locations,
        "target_locations": target_locations,
        "member_family": member_family,
        "selected_methods": bundle["selected_methods"],
        "weights": bundle["weights"],
        "member_metrics": bundle["member_metrics"],
        "validation_paths": saved_validation,
        "hindcast_paths": saved_hindcast,
    }
