import numpy as np
import pandas as pd

from src.bias_correction.methods.common import (
    HS_MODEL as RAW_MODEL,
    clip_nonnegative,
    prepare_ml_dataframe,
    resolve_feature_columns,
)
from src.ensemble.common import OBS, member_column
from src.settings import get_method_settings


def _sample_weights(obs):
    obs = np.asarray(obs, dtype=float)
    weights = np.ones(len(obs), dtype=float)
    valid = np.isfinite(obs)

    if np.sum(valid) < 20:
        return weights

    q90 = np.nanquantile(obs[valid], 0.90)
    q95 = np.nanquantile(obs[valid], 0.95)
    q99 = np.nanquantile(obs[valid], 0.99)
    weights[obs >= q90] = 2.0
    weights[obs >= q95] = 3.0
    weights[obs >= q99] = 4.0
    return weights


def _feature_frame(
    df,
    methods,
    state_features=None,
    fill=None,
    blended_prediction=None,
):
    prepared = prepare_ml_dataframe(df.copy())
    feature_cols = list(state_features or resolve_feature_columns(prepared, []))

    X = pd.DataFrame(index=prepared.index)

    for col in feature_cols:
        X[col] = pd.to_numeric(prepared[col], errors="coerce")

    raw_hs = pd.to_numeric(prepared.get(RAW_MODEL), errors="coerce")
    if RAW_MODEL not in X.columns:
        X[RAW_MODEL] = raw_hs

    member_frames = []
    for method in methods:
        col = member_column(method)
        values = pd.to_numeric(prepared.get(col), errors="coerce")
        X[col] = values
        X[f"delta_{method}"] = values - raw_hs
        member_frames.append(values.rename(col))

    members = pd.concat(member_frames, axis=1)
    X["member_mean"] = members.mean(axis=1, skipna=True)
    X["member_median"] = members.median(axis=1, skipna=True)
    X["member_std"] = members.std(axis=1, ddof=0, skipna=True)
    X["member_min"] = members.min(axis=1, skipna=True)
    X["member_max"] = members.max(axis=1, skipna=True)
    X["member_spread"] = X["member_max"] - X["member_min"]
    X["member_mean_minus_raw"] = X["member_mean"] - raw_hs
    X["member_min_minus_raw"] = X["member_min"] - raw_hs
    X["member_max_minus_raw"] = X["member_max"] - raw_hs

    if blended_prediction is not None:
        blend = pd.Series(blended_prediction, index=prepared.index, dtype=float)
        X["blend_prediction"] = blend
        X["blend_minus_raw"] = blend - raw_hs
        X["blend_minus_member_mean"] = blend - X["member_mean"]
        for method in methods:
            col = member_column(method)
            X[f"{col}_minus_blend"] = X[col] - blend

    if fill is None:
        fill = {}
        for col in X.columns:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            med = float(np.nanmedian(values))
            if not np.isfinite(med):
                med = 0.0
            fill[col] = med

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill.get(col, 0.0))

    return X.to_numpy(dtype=np.float32), fill, list(X.columns)


def _member_matrix(df, methods):
    cols = [member_column(method) for method in methods]
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _build_targets(df, methods):
    obs = pd.to_numeric(df[OBS], errors="coerce").to_numpy(dtype=float)
    members = _member_matrix(df, methods)

    finite_members = np.isfinite(members)
    valid = np.isfinite(obs) & np.any(finite_members, axis=1)
    if np.sum(valid) < 50:
        raise ValueError("Too few valid samples to train the XGBoost ensemble.")

    errors = np.abs(members[valid] - obs[valid, None])
    errors[~finite_members[valid]] = np.inf
    winners = np.argmin(errors, axis=1).astype(int)
    return valid, winners, obs


def _common_xgb_params():
    cfg = get_method_settings("ensemble_xgboost")
    return {
        "n_estimators": int(cfg.get("n_estimators", 300)),
        "max_depth": int(cfg.get("max_depth", 4)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "gamma": float(cfg.get("gamma", 0.0)),
        "min_child_weight": float(cfg.get("min_child_weight", 1.0)),
        "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
        "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
        "random_state": int(cfg.get("random_state", 1)),
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }


def _predict_gate(df, bundle, return_weights=False):
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
            fill=bundle["gate_fill"],
        )

        probs = bundle["gate_model"].predict_proba(X)
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
        raise ValueError(
            f"Unable to predict ensemble blend values for {missing_rows} rows."
        )

    if return_weights:
        return pred, normalized

    return pred


def fit_state_corrected_ensemble(df, methods, state_features=None):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed in the active environment."
        ) from exc

    valid_mask, winners, obs = _build_targets(df, methods)
    prepared = prepare_ml_dataframe(df.copy())
    state_feature_names = list(
        state_features or resolve_feature_columns(prepared, [])
    )

    X_all, gate_fill, gate_feature_names = _feature_frame(
        df,
        methods,
        state_features=state_feature_names,
    )

    present_classes = np.unique(winners)
    counts = np.bincount(winners, minlength=len(methods))

    bundle = {
        "methods": list(methods),
        "present_classes": present_classes.tolist(),
        "state_feature_names": state_feature_names,
        "gate_fill": gate_fill,
        "gate_feature_names": gate_feature_names,
        "class_counts": {
            methods[idx]: int(counts[idx])
            for idx in range(len(methods))
            if counts[idx] > 0
        },
        "top_features": {"gate": [], "residual": []},
    }

    if len(present_classes) == 1:
        bundle["constant_class"] = int(present_classes[0])
    else:
        objective = "binary:logistic" if len(present_classes) == 2 else "multi:softprob"
        eval_metric = "logloss" if len(present_classes) == 2 else "mlogloss"
        gate_model = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            **_common_xgb_params(),
        )

        label_map = {cls: idx for idx, cls in enumerate(present_classes)}
        y = np.array([label_map[cls] for cls in winners], dtype=np.int32)

        if objective == "multi:softprob":
            gate_model.set_params(num_class=len(present_classes))

        gate_model.fit(
            X_all[valid_mask],
            y,
            sample_weight=_sample_weights(obs[valid_mask]),
            verbose=False,
        )
        bundle["gate_model"] = gate_model
        bundle["top_features"]["gate"] = sorted(
            zip(gate_feature_names, gate_model.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

    blended_train = _predict_gate(df.iloc[valid_mask].copy().reset_index(drop=True), bundle)
    residual_target = obs[valid_mask] - blended_train

    if len(residual_target) < 50:
        return bundle

    X_resid, residual_fill, residual_feature_names = _feature_frame(
        df.iloc[valid_mask].copy().reset_index(drop=True),
        methods,
        state_features=state_feature_names,
        blended_prediction=blended_train,
    )

    residual_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        **_common_xgb_params(),
    )
    residual_model.fit(
        X_resid,
        residual_target,
        sample_weight=_sample_weights(obs[valid_mask]),
        verbose=False,
    )

    bundle["residual_model"] = residual_model
    bundle["residual_fill"] = residual_fill
    bundle["residual_feature_names"] = residual_feature_names
    bundle["top_features"]["residual"] = sorted(
        zip(residual_feature_names, residual_model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return bundle


def predict_state_corrected_ensemble(df, bundle, return_weights=False):
    blended, normalized = _predict_gate(df, bundle, return_weights=True)

    residual = np.zeros(len(df), dtype=float)
    if "residual_model" in bundle:
        X_resid, _, _ = _feature_frame(
            df,
            bundle["methods"],
            state_features=bundle["state_feature_names"],
            fill=bundle["residual_fill"],
            blended_prediction=blended,
        )
        residual = np.asarray(bundle["residual_model"].predict(X_resid), dtype=float)
        residual[~np.isfinite(residual)] = 0.0

    pred = clip_nonnegative(blended + residual)

    if return_weights:
        return pred, normalized

    return pred
