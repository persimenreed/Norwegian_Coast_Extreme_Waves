import numpy as np
import pandas as pd

from src.bias_correction.methods.common import (
    HS_MODEL as RAW_MODEL,
    clip_nonnegative,
    prepare_ml_dataframe,
    resolve_feature_columns,
)
from src.ensemble.common import OBS, member_column
from src.model_profiles import resolve_profile

DEFAULT_ENSEMBLE_XGBOOST_CONFIG = {
    "tail_weight_q90": 2.0,
    "tail_weight_q95": 3.0,
    "tail_weight_q99": 4.0,
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tail_aware": True,
    "tail_strength_q95": 0.20,
    "tail_strength_q99": 0.50,
    "random_state": 1,
}


def _config(profile_name="ensemble_xgboost"):
    return resolve_profile(DEFAULT_ENSEMBLE_XGBOOST_CONFIG, "ensemble_xgboost", profile_name)


def _sample_weights(obs, profile_name="ensemble_xgboost"):
    cfg = _config(profile_name)
    obs = np.asarray(obs, dtype=float)
    weights = np.ones(len(obs), dtype=float)
    valid = np.isfinite(obs)

    if np.sum(valid) < 20:
        return weights

    q90 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q90_quantile", 0.90)))
    q95 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q95_quantile", 0.95)))
    q99 = np.nanquantile(obs[valid], float(cfg.get("tail_weight_q99_quantile", 0.99)))
    weights[obs >= q90] = float(cfg.get("tail_weight_q90", 2.0))
    weights[obs >= q95] = float(cfg.get("tail_weight_q95", 3.0))
    weights[obs >= q99] = float(cfg.get("tail_weight_q99", 4.0))
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

    finite_errors = np.where(np.isfinite(errors), errors, np.nan)
    row_scale = np.nanmedian(finite_errors, axis=1, keepdims=True)
    row_scale[~np.isfinite(row_scale)] = 1.0
    row_scale = np.maximum(row_scale, 1e-3)

    scores = np.exp(-errors / row_scale)
    scores[~np.isfinite(scores)] = 0.0
    score_sum = scores.sum(axis=1, keepdims=True)
    soft_targets = np.divide(
        scores,
        score_sum,
        out=np.zeros_like(scores),
        where=score_sum > 0,
    )

    dominant = np.argmax(soft_targets, axis=1).astype(int)
    return valid, soft_targets, dominant, obs


def _common_xgb_params(profile_name="ensemble_xgboost"):
    cfg = _config(profile_name)
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


def _tail_aware_config(df, profile_name="ensemble_xgboost"):
    cfg = _config(profile_name)

    raw = pd.to_numeric(df.get(RAW_MODEL), errors="coerce").to_numpy(dtype=float)
    valid = raw[np.isfinite(raw)]
    if len(valid) < 50:
        return {
            "enabled": False,
            "raw_q95": np.nan,
            "raw_q99": np.nan,
            "strength_q95": 0.0,
            "strength_q99": 0.0,
        }

    return {
        "enabled": bool(cfg.get("tail_aware", True)),
        "raw_q95": float(np.nanquantile(valid, float(cfg.get("tail_q95", 0.95)))),
        "raw_q99": float(np.nanquantile(valid, float(cfg.get("tail_q99", 0.99)))),
        "strength_q95": float(cfg.get("tail_strength_q95", 0.20)),
        "strength_q99": float(cfg.get("tail_strength_q99", 0.50)),
    }


def _apply_tail_aware_weighting(df, bundle, weights, members):
    cfg = bundle.get("tail_aware")
    if not cfg or not cfg.get("enabled", False):
        return weights

    raw = pd.to_numeric(df.get(RAW_MODEL), errors="coerce").to_numpy(dtype=float)
    if raw.ndim != 1 or len(raw) != len(weights):
        return weights

    strength = np.zeros(len(raw), dtype=float)
    q95 = float(cfg.get("raw_q95", np.nan))
    q99 = float(cfg.get("raw_q99", np.nan))

    if np.isfinite(q95):
        strength[raw >= q95] = float(cfg.get("strength_q95", 0.0))
    if np.isfinite(q99):
        strength[raw >= q99] = float(cfg.get("strength_q99", 0.0))

    if not np.any(strength > 0):
        return weights

    finite_members = np.isfinite(members)
    safe_min = np.where(finite_members, members, np.inf)
    safe_max = np.where(finite_members, members, -np.inf)
    member_min = safe_min.min(axis=1, keepdims=True)
    member_max = safe_max.max(axis=1, keepdims=True)
    member_min[~np.isfinite(member_min)] = 0.0
    member_max[~np.isfinite(member_max)] = 0.0
    spread = np.maximum(member_max - member_min, 1e-6)
    relative_level = np.clip((members - member_min) / spread, 0.0, 1.0)

    factors = 1.0 + strength[:, None] * relative_level
    adjusted = np.where(finite_members, weights * factors, 0.0)
    return adjusted


def _predict_gate(df, bundle, return_weights=False):
    methods = bundle["methods"]
    members = _member_matrix(df, methods)
    n_rows = len(df)

    if "gate_models" in bundle or "constant_scores" in bundle:
        X, _, _ = _feature_frame(
            df,
            methods,
            state_features=bundle["state_feature_names"],
            fill=bundle["gate_fill"],
        )

        weights = np.zeros((n_rows, len(methods)), dtype=float)
        fallback = np.asarray(
            bundle.get(
                "mean_target_weights",
                np.full(len(methods), 1.0 / max(len(methods), 1), dtype=float),
            ),
            dtype=float,
        )

        for idx, method in enumerate(methods):
            if method in bundle.get("gate_models", {}):
                pred = bundle["gate_models"][method].predict(X)
                weights[:, idx] = np.asarray(pred, dtype=float)
            else:
                weights[:, idx] = float(
                    bundle.get("constant_scores", {}).get(method, fallback[idx])
                )

        weights = np.clip(weights, a_min=0.0, a_max=None)
        zero_rows = weights.sum(axis=1) <= 0
        if np.any(zero_rows):
            weights[zero_rows] = fallback
    elif "constant_class" in bundle:
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
    weights = _apply_tail_aware_weighting(df, bundle, weights, members)
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


def fit_state_corrected_ensemble(
    df,
    methods,
    state_features=None,
    profile_name="ensemble_xgboost",
):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed in the active environment."
        ) from exc

    valid_mask, soft_targets, dominant, obs = _build_targets(df, methods)
    prepared = prepare_ml_dataframe(df.copy())
    state_feature_names = list(
        state_features or resolve_feature_columns(prepared, [])
    )

    X_all, gate_fill, gate_feature_names = _feature_frame(
        df,
        methods,
        state_features=state_feature_names,
    )

    counts = np.bincount(dominant, minlength=len(methods))

    bundle = {
        "methods": list(methods),
        "state_feature_names": state_feature_names,
        "gate_fill": gate_fill,
        "gate_feature_names": gate_feature_names,
        "tail_aware": _tail_aware_config(df, profile_name=profile_name),
        "mean_target_weights": soft_targets.mean(axis=0).tolist(),
        "class_counts": {
            methods[idx]: int(counts[idx])
            for idx in range(len(methods))
            if counts[idx] > 0
        },
        "top_features": {"gate": [], "residual": []},
        "gate_models": {},
        "constant_scores": {},
    }

    X_train = X_all[valid_mask]
    row_weights = _sample_weights(obs[valid_mask], profile_name=profile_name)

    importance_sum = np.zeros(len(gate_feature_names), dtype=float)
    importance_weight = 0.0

    for idx, method in enumerate(methods):
        y = soft_targets[:, idx].astype(float)

        if np.allclose(y, y[0]):
            bundle["constant_scores"][method] = float(y[0])
            continue

        gate_model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **_common_xgb_params(profile_name=profile_name),
        )

        gate_model.fit(
            X_train,
            y,
            sample_weight=row_weights,
            verbose=False,
        )

        bundle["gate_models"][method] = gate_model
        avg_weight = max(float(np.mean(y)), 1e-6)
        importance_sum += avg_weight * gate_model.feature_importances_
        importance_weight += avg_weight

    if importance_weight > 0:
        bundle["top_features"]["gate"] = sorted(
            zip(gate_feature_names, (importance_sum / importance_weight).tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

    return bundle


def predict_state_corrected_ensemble(df, bundle, return_weights=False):
    blended, normalized = _predict_gate(df, bundle, return_weights=True)
    pred = clip_nonnegative(blended)

    if return_weights:
        return pred, normalized

    return pred
