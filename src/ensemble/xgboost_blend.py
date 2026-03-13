import numpy as np
import pandas as pd

from src.bias_correction.methods.common import (
    HS_MODEL as RAW_MODEL,
    clip_nonnegative,
    prepare_ml_dataframe,
    resolve_feature_columns,
)
from src.ensemble.common import (
    OBS,
    default_target_locations,
    default_training_specs,
    has_validation_members,
    load_hindcast_member_dataset,
    load_training_validation_specs,
    load_validation_member_dataset,
    member_column,
    normalize_methods,
    save_ensemble_report,
    save_hindcast_output,
    save_validation_output,
    unique_locations,
)
from src.settings import get_method_settings


def _sample_weights(obs):
    obs = np.asarray(obs, dtype=float)
    weights = np.ones(len(obs), dtype=float)
    valid = np.isfinite(obs)

    if np.sum(valid) < 20:
        return weights

    q90 = np.nanquantile(obs[valid], 0.90)
    q95 = np.nanquantile(obs[valid], 0.95)
    weights[obs >= q90] = 2.0
    weights[obs >= q95] = 3.0
    return weights


def _feature_frame(df, methods, state_features=None, fill=None):
    prepared = prepare_ml_dataframe(df.copy())
    feature_cols = list(state_features or resolve_feature_columns(prepared, []))

    X = pd.DataFrame(index=prepared.index)

    for col in feature_cols:
        X[col] = pd.to_numeric(prepared[col], errors="coerce")

    raw_hs = pd.to_numeric(prepared.get(RAW_MODEL), errors="coerce")
    member_frames = []

    for method in methods:
        col = member_column(method)
        values = pd.to_numeric(prepared[col], errors="coerce")
        X[col] = values
        if RAW_MODEL in prepared.columns:
            X[f"delta_{method}"] = values - raw_hs
        member_frames.append(values.rename(col))

    members = pd.concat(member_frames, axis=1)
    X["member_mean"] = members.mean(axis=1, skipna=True)
    X["member_std"] = members.std(axis=1, ddof=0, skipna=True)
    X["member_min"] = members.min(axis=1, skipna=True)
    X["member_max"] = members.max(axis=1, skipna=True)
    X["member_spread"] = X["member_max"] - X["member_min"]

    if fill is None:
        fill = {}
        for col in X.columns:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            med = float(np.nanmedian(values))
            if not np.isfinite(med):
                med = 0.0
            fill[col] = med

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill[col])

    return X.to_numpy(dtype=np.float32), fill, list(X.columns)


def _member_matrix(df, methods):
    cols = [member_column(method) for method in methods]
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _build_targets(df, methods):
    obs = pd.to_numeric(df[OBS], errors="coerce").to_numpy(dtype=float)
    members = _member_matrix(df, methods)

    valid = np.isfinite(obs) & np.all(np.isfinite(members), axis=1)
    if np.sum(valid) < 50:
        raise ValueError("Too few valid samples to train the XGBoost ensemble.")

    errors = np.abs(members[valid] - obs[valid, None])
    winners = np.argmin(errors, axis=1).astype(int)
    return valid, winners, obs


def fit_state_gated_ensemble(df, methods, state_features=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "XGBoost is not installed in the active environment."
        ) from e

    valid_mask, winners, obs = _build_targets(df, methods)
    state_feature_names = list(state_features or resolve_feature_columns(prepare_ml_dataframe(df.copy()), []))
    X_all, fill, feature_names = _feature_frame(
        df,
        methods,
        state_features=state_feature_names,
    )

    present_classes = np.unique(winners)
    counts = np.bincount(winners, minlength=len(methods))

    if len(present_classes) == 1:
        chosen = int(present_classes[0])
        return {
            "methods": list(methods),
            "constant_class": chosen,
            "present_classes": present_classes.tolist(),
            "fill": fill,
            "feature_names": feature_names,
            "state_feature_names": state_feature_names,
            "class_counts": {
                methods[idx]: int(counts[idx])
                for idx in range(len(methods))
                if counts[idx] > 0
            },
            "top_features": [],
        }

    cfg = get_method_settings("ensemble_xgboost")

    objective = "binary:logistic" if len(present_classes) == 2 else "multi:softprob"
    eval_metric = "logloss" if len(present_classes) == 2 else "mlogloss"

    model = XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        subsample=float(cfg.get("subsample", 0.8)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        gamma=float(cfg.get("gamma", 0.0)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        random_state=int(cfg.get("random_state", 1)),
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
        n_jobs=-1,
    )

    label_map = {cls: idx for idx, cls in enumerate(present_classes)}
    y = np.array([label_map[cls] for cls in winners], dtype=np.int32)
    weights = _sample_weights(obs[valid_mask])

    model_kwargs = {}
    if objective == "multi:softprob":
        model_kwargs["num_class"] = len(present_classes)

    model.set_params(**model_kwargs)
    model.fit(
        X_all[valid_mask],
        y,
        sample_weight=weights,
        verbose=False,
    )

    feature_importance = sorted(
        zip(feature_names, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "methods": list(methods),
        "model": model,
        "present_classes": present_classes.tolist(),
        "fill": fill,
        "feature_names": feature_names,
        "state_feature_names": state_feature_names,
        "class_counts": {
            methods[idx]: int(counts[idx])
            for idx in range(len(methods))
            if counts[idx] > 0
        },
        "top_features": feature_importance[:10],
    }


def predict_state_gated_ensemble(df, bundle, return_weights=False):
    methods = bundle["methods"]
    members = _member_matrix(df, methods)
    n_rows = len(df)

    if "constant_class" in bundle:
        weights = np.zeros((n_rows, len(methods)), dtype=float)
        weights[:, int(bundle["constant_class"])] = 1.0
    else:
        X, _, _ = _feature_frame(
            df,
            methods,
            state_features=bundle["state_feature_names"],
            fill=bundle["fill"],
        )

        probs = bundle["model"].predict_proba(X)
        if probs.ndim == 1:
            probs = probs[:, None]

        weights = np.zeros((n_rows, len(methods)), dtype=float)
        for col_idx, method_idx in enumerate(bundle["present_classes"]):
            weights[:, int(method_idx)] = probs[:, col_idx]

    finite_members = np.isfinite(members)
    weights = np.where(finite_members, weights, 0.0)
    row_sum = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(
        weights,
        row_sum,
        out=np.zeros_like(weights),
        where=row_sum > 0,
    )

    pred = np.nansum(normalized * members, axis=1)
    fallback = np.nanmean(members, axis=1)

    missing = ~np.isfinite(pred)
    pred[missing] = fallback[missing]

    if not np.all(np.isfinite(pred)):
        missing_rows = int(np.sum(~np.isfinite(pred)))
        raise ValueError(f"Unable to predict ensemble values for {missing_rows} rows.")

    pred = clip_nonnegative(pred)

    if return_weights:
        return pred, normalized

    return pred


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
    bundle = fit_state_gated_ensemble(train_df, methods)

    saved_validation = {}
    contributions = {}
    apply_member_family = "pooled"

    for target_location in target_locations:
        if not has_validation_members(target_location, methods, apply_member_family):
            continue

        df_val = load_validation_member_dataset(
            target_location,
            methods,
            apply_member_family,
        )
        pred, weights = predict_state_gated_ensemble(
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

    saved_hindcast = {}
    for target_location in target_locations:
        df_hind = load_hindcast_member_dataset(
            target_location,
            methods,
            apply_member_family,
        )
        pred, weights = predict_state_gated_ensemble(
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
