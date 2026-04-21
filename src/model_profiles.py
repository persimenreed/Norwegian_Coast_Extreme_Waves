from contextlib import contextmanager
from copy import deepcopy

MODEL_PROFILES = {
    "xgboost_fedjeosen": {
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 2.5,
        "tail_weight_q99": 10.5,
        "n_estimators": 302,
        "max_depth": 3,
        "learning_rate": 0.055648505307237334,
        "subsample": 0.8590790280785099,
        "colsample_bytree": 0.9266453746666847,
        "gamma": 1.8431490345172308,
        "min_child_weight": 7.5570866902026,
        "reg_alpha": 0.0001732350939124616,
        "reg_lambda": 0.0003012596609143149,
        "target_eps": 1e-06,
    },
    "xgboost_fauskane": {
        "tail_weight_q90": 2.0,
        "tail_weight_q95": 5.0,
        "tail_weight_q99": 6.0,
        "n_estimators": 482,
        "max_depth": 3,
        "learning_rate": 0.1845867116962881,
        "subsample": 0.9712328555228449,
        "colsample_bytree": 0.7708220375703201,
        "gamma": 4.6037471104722885,
        "min_child_weight": 0.4094728813120213,
        "reg_alpha": 3.77004822318677,
        "reg_lambda": 0.010874540763084285,
        "target_eps": 1e-06,
    },
    "transformer_fedjeosen": {
        "sequence_length": 13,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.02950424974999888,
        "learning_rate": 0.0016068898193273805,
        "weight_decay": 0.0004937076774357371,
        "batch_size": 128,
        "target_eps": 0.0001,
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 5.5,
        "tail_weight_q99": 11.5,
    },
    "transformer_fauskane": {
        "sequence_length": 12,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.14393197309462152,
        "learning_rate": 0.0003450912283403252,
        "weight_decay": 0.0005778932603207226,
        "batch_size": 64,
        "target_eps": 0.001,
        "tail_weight_q90": 1.5,
        "tail_weight_q95": 1.5,
        "tail_weight_q99": 9.5,
    },
    "ensemble_xgboost": {
        "tail_weight_q90": 2.5,
        "tail_weight_q95": 5.5,
        "tail_weight_q99": 9.5,
        "n_estimators": 326,
        "max_depth": 8,
        "learning_rate": 0.04443178435373279,
        "subsample": 0.9577428006276985,
        "colsample_bytree": 0.8305900144461262,
        "gamma": 0.11863996683465117,
        "min_child_weight": 2.0852025789784348,
        "reg_alpha": 1.1973457391452267,
        "reg_lambda": 0.002968010050332768,
        "tail_aware": True,
        "tail_strength_q95": 0.33611163794275056,
        "tail_strength_q99": 0.6699129786383526,
        "random_state": 1,
    },
    "ensemble_xgboost_fedjeosen": {
        "tail_weight_q90": 1.0,
        "tail_weight_q95": 3.0,
        "tail_weight_q99": 7.0,
        "n_estimators": 203,
        "max_depth": 6,
        "learning_rate": 0.045105563239769116,
        "subsample": 0.9204069712677262,
        "colsample_bytree": 0.6905187685481068,
        "gamma": 0.17410976689852775,
        "min_child_weight": 3.29837216676429,
        "reg_alpha": 0.682853526578858,
        "reg_lambda": 0.0027191393757147494,
        "tail_aware": True,
        "tail_strength_q95": 0.37205592896274886,
        "tail_strength_q99": 0.41891871116803375,
        "random_state": 1,
    },
    "ensemble_xgboost_fauskane": {
        "tail_weight_q90": 2.0,
        "tail_weight_q95": 2.0,
        "tail_weight_q99": 3.0,
        "n_estimators": 577,
        "max_depth": 6,
        "learning_rate": 0.08698689908266935,
        "subsample": 0.924234228391637,
        "colsample_bytree": 0.9119428660632724,
        "gamma": 0.0456826719980786,
        "min_child_weight": 2.018360660429937,
        "reg_alpha": 0.058967272475367774,
        "reg_lambda": 1.5314966501648426,
        "tail_aware": True,
        "tail_strength_q95": 0.03787464893301723,
        "tail_strength_q99": 0.7350280658145332,
        "random_state": 1,
    },
}


def resolve_profile(defaults, *profile_names):
    profile = deepcopy(defaults)
    for name in profile_names:
        if name:
            profile.update(MODEL_PROFILES.get(str(name), {}))
    return profile


@contextmanager
def override_profile(profile_name, params):
    profile_name = str(profile_name)
    existed = profile_name in MODEL_PROFILES
    original = deepcopy(MODEL_PROFILES.get(profile_name, {}))

    MODEL_PROFILES.setdefault(profile_name, {})
    MODEL_PROFILES[profile_name].update(params)

    try:
        yield
    finally:
        if existed:
            MODEL_PROFILES[profile_name] = original
        else:
            MODEL_PROFILES.pop(profile_name, None)
